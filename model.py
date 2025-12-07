"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from oracle_teacher import OracleAnnotations, OracleTeacher
from oracle_controls import OracleCurriculumController, OracleLossController, OracleTeacherController
from stopping import (
    AttentiveStoppingStrategy,
    DefaultStoppingStrategy,
    HardAttentiveStoppingStrategy,
    LearnedStoppingStrategy,
    OracleAttentiveStoppingStrategy,
    OracleGuidedStoppingStrategy,
    StickyDropoutStrategy,
    StoppingContext,
    StoppingController,
)
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization without bias."""

    def __init__(self, ndim, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_normalized


def gate_act_fn_clamp(x: torch.Tensor) -> torch.Tensor:
    # Helper for gating experiments: interpret x as a stop signal and produce a
    # "continue" probability in [0, 1]. Higher values -> more compute contribution.
    return 1 - torch.clamp(x, 0, 1)


def apply_score_mod(mask: torch.Tensor | None, gate: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Turns a gate in (0, 1] into an additive attention bias. We clamp to `eps` so
    zero gates stay finite, avoiding NaNs inside the softmax while still letting
    downstream residual gating zero the activations.
    """
    squeezed_gate = gate.squeeze(-1).to(torch.float32)
    safe_gate = squeezed_gate.clamp(eps, 1.0)
    score_mod = torch.log(safe_gate)  # b q_idx
    if mask is None:
        return score_mod[:, :, None].unsqueeze(1)  # b h q_idx kv_idx
    mask = mask[None, :, :] + score_mod[:, :, None]  # b q_idx kv_idx
    mask = mask.unsqueeze(1)  # b h q_idx kv_idx
    return mask


@dataclass
class StopHeadOutput:
    stop_logits: torch.Tensor
    difficulty_logits: Optional[torch.Tensor] = None


class StopHead(nn.Module):
    """
    Shared head used by all stopping strategies. Optionally emits a second logit
    for token-difficulty supervision when oracle stopping is active.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        predict_difficulty: bool = False,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.predict_difficulty = predict_difficulty
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.stop_out = nn.Linear(hidden_dim, 1)
        if predict_difficulty:
            self.difficulty_out = nn.Linear(hidden_dim, 1)
        else:
            self.difficulty_out = None

        if zero_init:
            nn.init.zeros_(self.stop_out.weight)
            nn.init.zeros_(self.stop_out.bias)
        if self.difficulty_out is not None:
            nn.init.zeros_(self.difficulty_out.weight)
            nn.init.zeros_(self.difficulty_out.bias)

    @property
    def preferred_dtype(self) -> torch.dtype:
        return self.stop_out.weight.dtype

    def forward(self, features: torch.Tensor) -> StopHeadOutput:
        hidden = self.trunk(features)
        stop_logits = self.stop_out(hidden).squeeze(-1)
        difficulty_logits = (
            self.difficulty_out(hidden).squeeze(-1) if self.difficulty_out is not None else None
        )
        return StopHeadOutput(stop_logits=stop_logits, difficulty_logits=difficulty_logits)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        attentive = bool(getattr(config, 'attentive_stopping', False))
        hard_attentive = bool(getattr(config, 'hard_attentive_stopping', False))
        # hard attentive reuses the same gating path, so force math attention in both modes
        self._disable_flash_with_gate = attentive or hard_attentive
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        elif self._disable_flash_with_gate:
            print("WARNING: attentive/hard attentive stopping disables Flash Attention; using math attention instead.")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, gate=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        use_flash = self.flash and not (self._disable_flash_with_gate and gate is not None)
        if use_flash:
            # efficient attention using Flash Attention CUDA kernels
            attn_mask = None
            if gate is not None:
                # Build additive bias from the gate ("continue" prob). Smaller gates reduce effective
                # attention scores, tapering compute. Expand to (B, H, T, T) with contiguous last dim.
                base_bias = apply_score_mod(None, gate)  # shape (B, 1, 1, 1) or (B, 1, L, S)
                attn_mask = base_bias.to(q.dtype).expand(B, self.n_head, q.size(-2), k.size(-2)).contiguous()
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            if not hasattr(self, "bias"):
                # create causal mask lazily for cases where flash would normally be used
                mask = torch.tril(torch.ones(self.block_size, self.block_size, device=x.device))
                self.register_buffer("bias", mask.view(1, 1, self.block_size, self.block_size), persistent=False)
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            if gate is not None:
                # Gate is the "continue" probability; convert to additive bias in attention space.
                gate_bias = apply_score_mod(None, gate).to(att.dtype)
                att = att + gate_bias
            att = F.softmax(att, dim=-1)
            if gate is not None:
                # Guard against any pathological rows that might still introduce NaNs numerically.
                att = torch.nan_to_num(att, nan=0.0)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, mlp_override=None, use_rmsnorm=False):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.sandwich_norm = getattr(config, 'sandwich_norm', False)
        self.apply_gate_to_residual = bool(getattr(config, 'attentive_stopping', False))
        if self.use_rmsnorm:
            self.norm_attn_in = RMSNorm(config.n_embd)
            self.norm_attn_out = RMSNorm(config.n_embd)
            self.norm_mlp_in = RMSNorm(config.n_embd)
            self.norm_mlp_out = RMSNorm(config.n_embd)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        if self.sandwich_norm:
            if self.use_rmsnorm:
                self.ln_1_post = RMSNorm(config.n_embd)
                self.ln_2_post = RMSNorm(config.n_embd)
            else:
                self.ln_1_post = LayerNorm(config.n_embd, bias=config.bias)
                self.ln_2_post = LayerNorm(config.n_embd, bias=config.bias)

        self.attn = CausalSelfAttention(config)
        if mlp_override:
            self.mlp = mlp_override
        elif config.moe:
            self.mlp = MoE(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, gate=None):
        if self.use_rmsnorm:
            attn_in = self.norm_attn_in(x)
            attn_out = self.norm_attn_out(self.attn(attn_in, gate=gate))
            if self.apply_gate_to_residual and gate is not None:
                attn_out = attn_out * gate.to(attn_out.dtype)
            x = x + attn_out
            if self.sandwich_norm:
                x = self.ln_1_post(x)

            mlp_in = self.norm_mlp_in(x)
            mlp_out = self.norm_mlp_out(self.mlp(mlp_in))
            if self.apply_gate_to_residual and gate is not None:
                mlp_out = mlp_out * gate.to(mlp_out.dtype)
            x = x + mlp_out
            if self.sandwich_norm:
                x = self.ln_2_post(x)
            return x

        attn_out = self.attn(self.ln_1(x), gate=gate)
        if self.apply_gate_to_residual and gate is not None:
            # Scale residual contribution by the "continue" probability from the stopping controller.
            attn_out = attn_out * gate.to(attn_out.dtype)
        x = x + attn_out
        if self.sandwich_norm:
            x = self.ln_1_post(x)

        mlp_out = self.mlp(self.ln_2(x))
        if self.apply_gate_to_residual and gate is not None:
            # Scale MLP residual by the same gate so both paths taper consistently.
            mlp_out = mlp_out * gate.to(mlp_out.dtype)
        x = x + mlp_out
        if self.sandwich_norm:
            x = self.ln_2_post(x)
        return x

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.hard_routing = config.moe_hard_routing

        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])
        self.dummy_expert = nn.Identity()
        self.router = nn.Linear(config.n_embd, self.num_experts + 1) # +1 for dummy expert

    def forward(self, x):
        router_logits = self.router(x)
        
        if self.hard_routing:
            # Hard routing: select one expert
            expert_indices = torch.argmax(router_logits, dim=-1)
            output = torch.zeros_like(x)
            for i in range(self.num_experts + 1):
                mask = (expert_indices == i)
                if mask.any():
                    if i == self.num_experts:
                        output[mask] = self.dummy_expert(x[mask])
                    else:
                        output[mask] = self.experts[i](x[mask]).to(x.dtype)
        else:
            # Soft routing: weighted average of top-k experts
            router_probs = F.softmax(router_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
            
            output = torch.zeros_like(x)
            for i in range(self.top_k):
                expert_indices = top_k_indices[..., i]
                expert_probs = top_k_probs[..., i]

                for j in range(self.num_experts + 1):
                    mask = (expert_indices == j)
                    if mask.any():
                        if j == self.num_experts:
                            expert_output = self.dummy_expert(x[mask])
                        else:
                            expert_output = self.experts[j](x[mask])
                        output[mask] += expert_output * expert_probs[mask].unsqueeze(-1)

        # This is a hack to make DDP happy. It ensures all expert parameters are used in the forward pass.
        if self.training:
            dummy_val = sum(p.sum() for p in self.router.parameters()) * 0.0
            for expert in self.experts:
                for p in expert.parameters():
                    dummy_val += p.sum() * 0.0
            output = output + dummy_val

        return output

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # share one Transformer block's weights across all layers (ALBERT-style)
    share_parameters_across_layers: bool = False
    # experiment: recurrent shared weights
    recurrent_shared_weights: bool = False
    recurrent_depth: int = 32 # default depth for recurrent shared weights
    recurrent_shared_num_blocks: int = 1
    bp_truncate_depth: int = 0
    # MoE parameters
    moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_hard_routing: bool = False
    share_moe_experts: bool = False
    scale_loss_by_n_layer: bool = False
    # new experiments
    sticky_dropout: float = 0.0
    learned_stopping: bool = False
    learned_stopping_warmup_steps: int = 0
    learned_stopping_controller_weight: float = 0.0
    learned_stopping_entropy_weight: float = 0.0
    learned_stopping_target_depth: Optional[float] = None
    learned_stopping_temperature: float = 1.0
    learned_stopping_min_prob: float = 1e-4
    learned_stopping_use_threshold: bool = False
    learned_stopping_threshold: float = 0.5
    oracle_stopping: bool = False
    oracle_dummy: bool = False
    oracle_bootstrap_checkpoint: Optional[str] = None
    oracle_update_interval: int = 1000
    oracle_max_depth: Optional[int] = None
    oracle_stop_weight: float = 1.0
    oracle_difficulty_weight: float = 1.0
    oracle_stop_backward: bool = False
    oracle_temperature: float = 1.0
    oracle_min_prob: float = 1e-4
    oracle_use_threshold: bool = False
    oracle_threshold: float = 0.5
    oracle_teacher_use_ema: bool = False
    oracle_teacher_ema_decay: float = 0.999
    oracle_teacher_ema_decay_min: float = 0.99
    oracle_teacher_ema_decay_schedule: int = 0
    oracle_adaptive_update_interval: bool = False
    oracle_update_interval_min: int = 1
    oracle_update_interval_max: int = 1000
    oracle_update_interval_shrink: float = 0.8
    oracle_update_interval_growth: float = 1.2
    oracle_update_interval_tolerance: float = 0.05
    oracle_confidence_weighting: bool = False
    oracle_confidence_floor: float = 0.05
    oracle_confidence_exponent: float = 1.0
    oracle_confidence_ceiling: float = 1.0
    oracle_stop_adv_clip: float = 0.0
    oracle_curriculum_depth_start: Optional[int] = None
    oracle_curriculum_depth_warmup_steps: int = 0
    oracle_temperature_final: Optional[float] = None
    oracle_temperature_schedule_steps: int = 0
    # attentive stopping
    attentive_stopping: bool = False
    attentive_stopping_warmup_steps: int = 0
    attentive_stopping_controller_weight: float = 0.0
    attentive_stopping_entropy_weight: float = 0.0
    attentive_stopping_target_depth: Optional[float] = None
    attentive_stopping_temperature: float = 1.0
    attentive_stopping_min_prob: float = 1e-4
    attentive_stopping_use_threshold: bool = False
    attentive_stopping_threshold: float = 0.5
    stop_use_cumsum_pooling: bool = False
    stop_disable_pooled_features: bool = True
    hard_attentive_stopping: bool = False
    hard_attentive_stopping_threshold: float = 0.5
    sandwich_norm: bool = False
    stopping_tokenwise: bool = False
    fixed_edge_blocks: bool = False
    n_layers_prelude: int = 1
    n_layers_coda: int = 1
    use_rmsnorm: bool = False
    recurrent_noise_mode: str = 'none'  # options: 'none', 'add', 'concat'
    recurrent_noise_std: float = 0.0
    recurrent_noise_concat_dim: Optional[int] = None
    recurrent_extra_layernorm: bool = False
    recurrent_prelude_injection: bool = False
    recurrent_prelude_injection_mode: str = 'add'
    attentive_stopping: bool = False
    attentive_stopping_warmup_steps: int = 0
    attentive_stopping_controller_weight: float = 0.0
    attentive_stopping_entropy_weight: float = 0.0
    attentive_stopping_target_depth: float = None
    attentive_stopping_temperature: float = 1.0
    attentive_stopping_min_prob: float = 1e-4
    attentive_stopping_use_threshold: bool = False
    attentive_stopping_threshold: float = 0.5
    stop_use_cumsum_pooling: bool = False

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.bp_truncate_depth = max(0, int(getattr(config, 'bp_truncate_depth', 0) or 0))
        self.ce_loss = None
        self.total_loss = None

        # new attributes for experiments
        self.sticky_dropout = config.sticky_dropout
        self.stopping_tokenwise = bool(getattr(config, 'stopping_tokenwise', False))
        self.learned_stopping = config.learned_stopping
        self.learned_stopping_warmup_steps = config.learned_stopping_warmup_steps
        self.learned_stopping_controller_weight = config.learned_stopping_controller_weight
        self.learned_stopping_entropy_weight = config.learned_stopping_entropy_weight
        self.learned_stopping_target_depth = config.learned_stopping_target_depth
        self.learned_stopping_temperature = config.learned_stopping_temperature
        self.learned_stopping_min_prob = config.learned_stopping_min_prob
        self.learned_stopping_use_threshold = config.learned_stopping_use_threshold
        self.learned_stopping_threshold = config.learned_stopping_threshold

        self.oracle_stopping = bool(getattr(config, 'oracle_stopping', False))
        self.oracle_dummy = bool(getattr(config, 'oracle_dummy', False))
        self.oracle_bootstrap_checkpoint = getattr(config, 'oracle_bootstrap_checkpoint', None) or None
        self.oracle_update_interval = int(getattr(config, 'oracle_update_interval', 0) or 0)
        self.oracle_max_depth = getattr(config, 'oracle_max_depth', None)
        self.oracle_stop_weight = getattr(config, 'oracle_stop_weight', 1.0)
        self.oracle_difficulty_weight = getattr(config, 'oracle_difficulty_weight', 1.0)
        self.oracle_stop_backward = bool(getattr(config, 'oracle_stop_backward', False))
        self.oracle_temperature = getattr(config, 'oracle_temperature', 1.0)
        self.oracle_min_prob = getattr(config, 'oracle_min_prob', 1e-4)
        self.oracle_use_threshold = getattr(config, 'oracle_use_threshold', False)
        self.oracle_threshold = getattr(config, 'oracle_threshold', 0.5)
        self.oracle_teacher_controller = OracleTeacherController(config) if self.oracle_stopping else None
        self.oracle_loss_controller = OracleLossController.from_config(config) if self.oracle_stopping else None
        self.oracle_curriculum_controller = OracleCurriculumController(config) if self.oracle_stopping else None
        self._oracle_runtime_temperature = self.oracle_temperature

        self.attentive_stopping = config.attentive_stopping
        self.attentive_stopping_warmup_steps = config.attentive_stopping_warmup_steps
        self.attentive_stopping_controller_weight = config.attentive_stopping_controller_weight
        self.attentive_stopping_entropy_weight = config.attentive_stopping_entropy_weight
        self.attentive_stopping_target_depth = config.attentive_stopping_target_depth
        self.attentive_stopping_temperature = config.attentive_stopping_temperature
        self.attentive_stopping_min_prob = config.attentive_stopping_min_prob
        self.attentive_stopping_use_threshold = config.attentive_stopping_use_threshold
        self.attentive_stopping_threshold = config.attentive_stopping_threshold
        self.hard_attentive_stopping = getattr(config, 'hard_attentive_stopping', False)
        self.hard_attentive_stopping_threshold = getattr(config, 'hard_attentive_stopping_threshold', 0.5)
        self.oracle_teacher: Optional[OracleTeacher] = None
        self.oracle_metrics = None
        self._oracle_last_update_step = -1
        self.stop_use_cumsum_pooling = bool(getattr(config, "stop_use_cumsum_pooling", False))
        self.stop_disable_pooled_features = bool(getattr(config, "stop_disable_pooled_features", True))
        needs_stop_head = self.learned_stopping or self.attentive_stopping or self.oracle_stopping
        if needs_stop_head:
            pooled_factor = 0 if self.stop_disable_pooled_features else 1
            feature_dim = config.n_embd * (1 + pooled_factor)
            zero_init = self.attentive_stopping
            self.stop_predictor = StopHead(
                feature_dim,
                config.n_embd,
                predict_difficulty=self.oracle_stopping,
                zero_init=zero_init,
            )
        else:
            self.stop_predictor = None

        self.stopping_metrics = {} if self.learned_stopping else None
        self._stopping_step = 0
        self.attentive_stopping_metrics = {} if self.attentive_stopping else None
        self._attentive_stopping_step = 0
        self.hard_attentive_stopping_metrics = {} if self.hard_attentive_stopping else None
        self._hard_attentive_stopping_step = 0
        self._oracle_step = 0

        self.fixed_edge_blocks = bool(config.share_parameters_across_layers and config.fixed_edge_blocks)
        if config.fixed_edge_blocks and not config.share_parameters_across_layers:
            print("WARNING: fixed_edge_blocks requested without shared parameters; ignoring fixed_edge_blocks.")
        self.use_rmsnorm = bool(getattr(config, 'use_rmsnorm', False))

        self.recurrent_noise_mode = config.recurrent_noise_mode
        if self.recurrent_noise_mode not in ('none', 'add', 'concat'):
            raise ValueError(f"Unsupported recurrent_noise_mode: {self.recurrent_noise_mode}")
        self.recurrent_noise_std = config.recurrent_noise_std
        self.recurrent_noise_concat_dim = None
        self.recurrent_noise_proj = None
        if self.recurrent_noise_mode == 'concat':
            concat_dim = config.recurrent_noise_concat_dim or config.n_embd
            if concat_dim <= 0:
                raise ValueError("recurrent_noise_concat_dim must be a positive integer when using 'concat' noise mode.")
            self.recurrent_noise_concat_dim = concat_dim
            self.recurrent_noise_proj = nn.Linear(config.n_embd + concat_dim, config.n_embd, bias=config.bias)
        else:
            self.recurrent_noise_concat_dim = config.recurrent_noise_concat_dim
        self.recurrent_extra_layernorm = config.recurrent_extra_layernorm
        self.recurrent_prelude_injection = config.recurrent_prelude_injection
        self.recurrent_prelude_injection_mode = getattr(config, 'recurrent_prelude_injection_mode', 'add')
        if self.recurrent_prelude_injection_mode not in ('add', 'concat'):
            raise ValueError(f"Unsupported recurrent_prelude_injection_mode: {self.recurrent_prelude_injection_mode}")
        if self.recurrent_prelude_injection_mode == 'concat':
            self.recurrent_prelude_proj = nn.Linear(config.n_embd * 2, config.n_embd, bias=config.bias)
        else:
            self.recurrent_prelude_proj = None
        self.attentive_stopping = config.attentive_stopping
        self.attentive_stopping_warmup_steps = config.attentive_stopping_warmup_steps
        self.attentive_stopping_controller_weight = config.attentive_stopping_controller_weight
        self.attentive_stopping_entropy_weight = config.attentive_stopping_entropy_weight
        self.attentive_stopping_target_depth = config.attentive_stopping_target_depth
        self.attentive_stopping_temperature = config.attentive_stopping_temperature
        self.attentive_stopping_min_prob = config.attentive_stopping_min_prob
        self.attentive_stopping_use_threshold = config.attentive_stopping_use_threshold
        self.attentive_stopping_threshold = config.attentive_stopping_threshold
        if self.recurrent_extra_layernorm:
            if self.use_rmsnorm:
                self.recurrent_extra_ln = RMSNorm(config.n_embd)
            else:
                self.recurrent_extra_ln = LayerNorm(config.n_embd, bias=config.bias)
        else:
            self.recurrent_extra_ln = None

        self.stopping_controller = StoppingController([
            StickyDropoutStrategy(),
            OracleAttentiveStoppingStrategy(),
            OracleGuidedStoppingStrategy(),
            LearnedStoppingStrategy(),
            HardAttentiveStoppingStrategy(),
            AttentiveStoppingStrategy(),
            DefaultStoppingStrategy(),
        ])

        mlp_override = None
        if config.moe and config.share_moe_experts:
            mlp_override = MoE(config)
        num_blocks = config.recurrent_shared_num_blocks if config.share_parameters_across_layers else config.n_layer
        if config.share_parameters_across_layers and num_blocks < 1:
            raise ValueError("recurrent_shared_num_blocks must be >= 1 when sharing parameters across layers.")
        h = nn.ModuleList([
            Block(config, mlp_override=mlp_override, use_rmsnorm=self.use_rmsnorm)
            for _ in range(num_blocks)
        ])
        if self.fixed_edge_blocks:
            self.fixed_head = nn.ModuleList([Block(config, mlp_override=mlp_override, use_rmsnorm=self.use_rmsnorm) for _ in range(config.n_layers_prelude)])
            self.fixed_tail = nn.ModuleList([Block(config, mlp_override=mlp_override, use_rmsnorm=self.use_rmsnorm) for _ in range(config.n_layers_coda)])
        else:
            self.fixed_head = None
            self.fixed_tail = None

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = h,
            ln_f = RMSNorm(config.n_embd) if self.use_rmsnorm else LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_recurrent_noise(self, x):
        if self.recurrent_noise_std <= 0 or self.recurrent_noise_mode == 'none':
            return x
        
        if self.recurrent_noise_mode == 'add':
            noise = torch.randn_like(x) * self.recurrent_noise_std
            return x + noise
        if self.recurrent_noise_mode == 'concat':
            noise_shape = x.shape[:-1] + (self.recurrent_noise_concat_dim,)
            noise = torch.randn(noise_shape, device=x.device, dtype=x.dtype) * self.recurrent_noise_std
            x_aug = torch.cat((x, noise), dim=-1)
            if self.recurrent_noise_proj is None:
                raise RuntimeError("recurrent_noise_proj must be initialized when using 'concat' noise mode.")
            return self.recurrent_noise_proj(x_aug)
        return x

    def _ensure_oracle_teacher(self, device: torch.device, dtype: torch.dtype) -> Optional[OracleTeacher]:
        if not self.oracle_stopping:
            return None
        if self.oracle_teacher is None:
            teacher_config = copy.deepcopy(self.config)
            teacher_config.oracle_stopping = False
            teacher_config.learned_stopping = False
            teacher_config.attentive_stopping = False
            teacher_config.oracle_bootstrap_checkpoint = None
            teacher_config.oracle_update_interval = 0
            teacher_model = type(self)(teacher_config)
            teacher_model.to(device=device, dtype=dtype)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad_(False)
            self.oracle_teacher = OracleTeacher(teacher_model, max_depth=self.oracle_max_depth)
            if self.oracle_bootstrap_checkpoint:
                try:
                    self.oracle_teacher.load_checkpoint(self.oracle_bootstrap_checkpoint)
                except FileNotFoundError:
                    print(f"Warning: oracle bootstrap checkpoint {self.oracle_bootstrap_checkpoint} not found; falling back to student weights.")
                    self._copy_student_to_oracle()
                self._oracle_last_update_step = 0
            else:
                self._copy_student_to_oracle()
                self._oracle_last_update_step = 0
            if self.oracle_teacher_controller:
                self.oracle_teacher_controller.last_update_step = self._oracle_last_update_step
        else:
            teacher_model = self.oracle_teacher.model
            teacher_model.to(device=device, dtype=dtype)
        return self.oracle_teacher

    def _copy_student_to_oracle(self):
        if not self.oracle_teacher:
            return
        teacher_model = self.oracle_teacher.model
        controller = self.oracle_teacher_controller
        if controller and controller.use_ema:
            controller._hard_copy(self, teacher_model)
        else:
            with torch.no_grad():
                teacher_model.load_state_dict(self.state_dict(), strict=False)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad_(False)

    def maybe_update_oracle(self, step: int):
        if not self.oracle_stopping:
            return
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self._ensure_oracle_teacher(device, dtype)
        if self.oracle_teacher is None:
            return
        controller = self.oracle_teacher_controller
        if controller:
            updated = controller.maybe_update(self, self.oracle_teacher.model, step, self.oracle_metrics)
            if updated:
                self._oracle_last_update_step = step
            return
        interval = max(1, int(self.oracle_update_interval) if self.oracle_update_interval is not None else 1)
        if self._oracle_last_update_step < 0 or (step - self._oracle_last_update_step) >= interval:
            self._copy_student_to_oracle()
            self._oracle_last_update_step = step

    def get_oracle_state(self):
        if not self.oracle_stopping or self.oracle_teacher is None:
            return None
        from collections import OrderedDict
        state_dict = OrderedDict((k, v.cpu()) for k, v in self.oracle_teacher.state_dict().items())
        return {
            "state_dict": state_dict,
            "last_update_step": self._oracle_last_update_step,
            "controller": self.oracle_teacher_controller.state_dict() if self.oracle_teacher_controller else None,
        }

    def load_oracle_state(self, state: Optional[dict]):
        if not self.oracle_stopping or state is None:
            return
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        teacher = self._ensure_oracle_teacher(device, dtype)
        if teacher is None:
            return
        teacher.load_state_dict(state.get("state_dict", {}))
        self._oracle_last_update_step = state.get("last_update_step", -1)
        controller_state = state.get("controller", None)
        if controller_state and self.oracle_teacher_controller:
            self.oracle_teacher_controller.load_state_dict(controller_state)

    def _get_oracle_teacher(self, device: torch.device, dtype: torch.dtype) -> OracleTeacher:
        teacher = self._ensure_oracle_teacher(device, dtype)
        if teacher is None:
            raise RuntimeError("oracle_stopping is disabled but oracle teacher requested.")
        return teacher

    def _forward_shared_block(self, block, x, gate=None, prelude_output=None):
        x_in = self._apply_recurrent_noise(x)
        if prelude_output is not None:
            x_in = self._inject_prelude(x_in, prelude_output)
        out = block(x_in, gate=gate)
        if self.recurrent_extra_layernorm:
            out = self.recurrent_extra_ln(out)
        return out

    def _inject_prelude(self, x, prelude_output):
        if self.recurrent_prelude_injection_mode == 'add':
            return x + prelude_output
        if self.recurrent_prelude_injection_mode == 'concat':
            if self.recurrent_prelude_proj is None:
                raise RuntimeError("Prelude projection is uninitialized for 'concat' injection mode.")
            x_aug = torch.cat([x, prelude_output], dim=-1)
            return self.recurrent_prelude_proj(x_aug)
        raise ValueError(f"Unsupported recurrent_prelude_injection_mode: {self.recurrent_prelude_injection_mode}")

    def forward(self, idx, targets=None, n=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        num_expanded_layers = None

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        if self.config.share_parameters_across_layers:
            prelude_output = None
            shared_blocks = self.transformer.h
            num_shared_blocks = len(shared_blocks)
            if num_shared_blocks == 0:
                raise RuntimeError("No shared blocks available for recurrent forwarding.")
            def run_shared_block(tensor, gate, step_idx):
                block = shared_blocks[step_idx % num_shared_blocks]
                if self.recurrent_prelude_injection:
                    return self._forward_shared_block(block, tensor, gate=gate, prelude_output=prelude_output)
                return self._forward_shared_block(block, tensor, gate=gate)
            if self.fixed_edge_blocks and self.fixed_head is not None:
                for block in self.fixed_head:
                    x = block(x)
                if self.recurrent_prelude_injection:
                    prelude_output = x.clone()
            step_state = {"i": 0}
            def shared_block_fn(tensor, gate=None):
                step_idx = step_state["i"]
                step_state["i"] += 1
                if self.bp_truncate_depth > 0 and step_idx < self.bp_truncate_depth:
                    with torch.no_grad():
                        return run_shared_block(tensor, gate, step_idx)
                return run_shared_block(tensor, gate, step_idx)

            # Determine the number of recurrent steps
            if self.config.recurrent_shared_weights:
                # During training, n is sampled and passed. During inference, it can be user-specified.
                num_layers = n if n is not None else self.config.recurrent_depth
            else:
                num_layers = self.config.n_layer
            if self.oracle_curriculum_controller:
                num_layers = self.oracle_curriculum_controller.depth(num_layers, self._oracle_step)
                self._oracle_runtime_temperature = self.oracle_curriculum_controller.temperature(
                    self.oracle_temperature, self._oracle_step
                )
            else:
                self._oracle_runtime_temperature = self.oracle_temperature

            oracle_annotations = None
            if self.oracle_stopping and not self.oracle_dummy and targets is not None:
                oracle_annotations = self._get_oracle_teacher(x.device, x.dtype).annotate(
                    idx, targets, num_layers, self.stopping_tokenwise
                )

            ctx = StoppingContext(
                x=x,
                shared_block_fn=shared_block_fn,
                num_layers=num_layers,
                targets=targets,
                aux_loss=aux_loss,
                oracle=oracle_annotations,
            )
            stopping_result = self.stopping_controller.run(self, ctx)
            x = stopping_result.x
            aux_loss = stopping_result.aux_loss
            num_expanded_layers = stopping_result.num_expanded_layers
            if self.fixed_edge_blocks and self.fixed_tail is not None:
                for block in self.fixed_tail:
                    x = block(x)
        else: # Baseline
            for block in self.transformer.h:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            self.ce_loss = loss.item()
            aux_backward_enabled = self.training and (
                self.learned_stopping
                or self.attentive_stopping
                or (self.oracle_stopping and self.oracle_stop_backward)
            )
            if aux_backward_enabled:
                loss = loss + aux_loss.to(loss.dtype)
            self.total_loss = loss.item()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, num_expanded_layers

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, n=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond, n=n)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
