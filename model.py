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
from dataclasses import dataclass
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerDropout(torch.nn.Module):
    """
    A module that applies layer dropout to the input tensor of an underlying module.
    It drops a portion of an input tensor, applies the underlying module on the
    remaining parts of the tensor, and then concatenates with the dropped portion of the tensor.
    When applied during training, it can have a regularization effect, and can potentially speedup training.

    Args:
        prob (float): The probability of dropping an input. Defaults to 0.0.
        dim (Optional[int]): The dimension of input tensor along which to drop layers. Defaults to 0 (i.e., batch size).
        disable_on_eval (Optional[bool]): Whether to disable layer dropout during evaluation. Defaults to True.
        seed (Optional[int]): The seed for the random number generator. Defaults to None.
    """

    def __init__(
        self,
        prob: float = 0.0,
        dim: Optional[int] = 0,
        disable_on_eval: Optional[bool] = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.prob: float = prob
        self.dim = dim
        self.disable_on_eval: bool = disable_on_eval
        self.generator = torch.Generator(device="cpu")
        self.inferred: float = None

        if seed is not None:
            self.generator.manual_seed(seed)

    def forward(
        self,
        function: Union[Callable, torch.nn.Module],
        input: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply layer dropout to the input tensor.

        Args:
            function (Union[Callable, torch.nn.Module]): The function or module to apply to the input tensor.
            input (torch.Tensor): The input tensor.
            *args: Additional positional arguments passed to the function.
            **kwargs: Additional keyword arguments passed to the function.
        Returns:
            torch.Tensor: The output tensor after applying layer dropout.
        """
        n = input.shape[self.dim]

        if self.prob == 0 or (self.disable_on_eval and self.training is False):
            self.inferred = 1.0
            return function(input, *args, **kwargs)

        skip = (
            torch.bernoulli(torch.Tensor((n) * [self.prob]), generator=self.generator)
            .to(input.device)
            .to(input.dtype)
        )
        self.inferred = 1 - torch.mean(skip)
        ind_selected = (skip == 0).nonzero().squeeze()

        if ind_selected.numel() > 0:
            x_selected = torch.index_select(input, self.dim, ind_selected)
            out_selected = function(x_selected, *args, **kwargs)

        out = input.clone()
        assert (
            self.dim == 0
        ), "Currently only supporting dropping elements along the 0th dimension"
        if ind_selected.numel() > 0:
            out[ind_selected] = out_selected
        return out

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
    return 1 - torch.clamp(x, 0, 1)


def apply_score_mod(mask: torch.Tensor | None, gate: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    score_mod = torch.log(gate.clamp(eps, 1)).squeeze(-1)  # b q_idx
    if mask is None:
        return score_mod[:, :, None].unsqueeze(1)  # b h q_idx kv_idx
    mask = mask[None, :, :] + score_mod[:, :, None]  # b q_idx kv_idx
    mask = mask.unsqueeze(1)  # b h q_idx kv_idx
    return mask

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
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
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
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            attn_mask = None
            if gate is not None:
                attn_mask = apply_score_mod(None, gate)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            if gate is not None:
                att = att + apply_score_mod(None, gate)
            att = F.softmax(att, dim=-1)
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
        if self.use_rmsnorm:
            self.norm_attn_in = RMSNorm(config.n_embd)
            self.norm_attn_out = RMSNorm(config.n_embd)
            self.norm_mlp_in = RMSNorm(config.n_embd)
            self.norm_mlp_out = RMSNorm(config.n_embd)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
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
            x = x + self.norm_attn_out(self.attn(attn_in, gate=gate))


            mlp_in = self.norm_mlp_in(x)
            x = x + self.norm_mlp_out(self.mlp(mlp_in))
            return x

        x = x + self.attn(self.ln_1(x), gate=gate)
        x = x + self.mlp(self.ln_2(x))
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
    # MoE parameters
    moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_hard_routing: bool = False
    share_moe_experts: bool = False
    scale_loss_by_n_layer: bool = False
    # new experiments
    layer_dropout: float = 0.0
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
    fixed_edge_blocks: bool = False
    use_rmsnorm: bool = False
    recurrent_noise_mode: str = 'none'  # options: 'none', 'add', 'concat'
    recurrent_noise_std: float = 0.0
    recurrent_noise_concat_dim: Optional[int] = None
    recurrent_extra_layernorm: bool = False

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # new attributes for experiments
        self.layer_dropout = config.layer_dropout
        if self.layer_dropout > 0:
            self.layer_dropout_module = LayerDropout(prob=self.layer_dropout)
        self.sticky_dropout = config.sticky_dropout
        self.learned_stopping = config.learned_stopping
        self.learned_stopping_warmup_steps = config.learned_stopping_warmup_steps
        self.learned_stopping_controller_weight = config.learned_stopping_controller_weight
        self.learned_stopping_entropy_weight = config.learned_stopping_entropy_weight
        self.learned_stopping_target_depth = config.learned_stopping_target_depth
        self.learned_stopping_temperature = config.learned_stopping_temperature
        self.learned_stopping_min_prob = config.learned_stopping_min_prob
        self.learned_stopping_use_threshold = config.learned_stopping_use_threshold
        self.learned_stopping_threshold = config.learned_stopping_threshold
        if self.learned_stopping:
            feature_dim = config.n_embd * 2
            self.stop_predictor = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, 1),
            )
            self.stopping_metrics = {}
            self._stopping_step = 0
        else:
            self.stop_predictor = None
            self.stopping_metrics = None
            self._stopping_step = 0

        self.attentive_stopping = config.attentive_stopping
        self.attentive_stopping_warmup_steps = config.attentive_stopping_warmup_steps
        self.attentive_stopping_controller_weight = config.attentive_stopping_controller_weight
        self.attentive_stopping_entropy_weight = config.attentive_stopping_entropy_weight
        self.attentive_stopping_target_depth = config.attentive_stopping_target_depth
        self.attentive_stopping_temperature = config.attentive_stopping_temperature
        self.attentive_stopping_min_prob = config.attentive_stopping_min_prob
        self.attentive_stopping_use_threshold = config.attentive_stopping_use_threshold
        self.attentive_stopping_threshold = config.attentive_stopping_threshold
        if self.attentive_stopping:
            feature_dim = config.n_embd * 2
            self.stop_predictor = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, 1),
            )
            self.attentive_stopping_metrics = {}
            self._attentive_stopping_step = 0
        else:
            self.stop_predictor = None
            self.attentive_stopping_metrics = None
            self._attentive_stopping_step = 0

        self.fixed_edge_blocks = bool(config.share_parameters_across_layers and config.fixed_edge_blocks)
        if config.fixed_edge_blocks and not config.share_parameters_across_layers:
            print("WARNING: fixed_edge_blocks requested without shared parameters; ignoring fixed_edge_blocks.")
        self.use_rmsnorm = bool(getattr(config, 'use_rmsnorm', False))
        if self.use_rmsnorm and not self.fixed_edge_blocks:
            raise ValueError("use_rmsnorm=True requires fixed_edge_blocks=True with shared parameters.")
        self.recurrent_noise_mode = config.recurrent_noise_mode
        if self.recurrent_noise_mode not in ('none', 'add', 'concat'):
            raise ValueError(f"Unsupported recurrent_noise_mode: {self.recurrent_noise_mode}")
        self.recurrent_noise_std = config.recurrent_noise_std
        self.recurrent_noise_concat_dim = None
        if self.recurrent_noise_mode == 'concat':
            concat_dim = config.recurrent_noise_concat_dim or config.n_embd
            if concat_dim <= 0:
                raise ValueError("recurrent_noise_concat_dim must be a positive integer when using 'concat' noise mode.")
            self.recurrent_noise_concat_dim = concat_dim
            self.recurrent_noise_proj = nn.Linear(config.n_embd + concat_dim, config.n_embd, bias=config.bias)
        else:
            self.recurrent_noise_concat_dim = config.recurrent_noise_concat_dim
            self.recurrent_noise_proj = None
        self.recurrent_extra_layernorm = config.recurrent_extra_layernorm
        if self.recurrent_extra_layernorm:
            if self.use_rmsnorm:
                self.recurrent_extra_ln = RMSNorm(config.n_embd)
            else:
                self.recurrent_extra_ln = LayerNorm(config.n_embd, bias=config.bias)
        else:
            self.recurrent_extra_ln = None

        mlp_override = None
        if config.moe and config.share_moe_experts:
            mlp_override = MoE(config)
        num_blocks = 1 if config.share_parameters_across_layers else config.n_layer
        h = nn.ModuleList([
            Block(config, mlp_override=mlp_override, use_rmsnorm=self.use_rmsnorm)
            for _ in range(num_blocks)
        ])
        if self.fixed_edge_blocks:
            self.fixed_head = Block(config, mlp_override=mlp_override, use_rmsnorm=self.use_rmsnorm)
            self.fixed_tail = Block(config, mlp_override=mlp_override, use_rmsnorm=self.use_rmsnorm)
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

    def _forward_shared_block(self, block, x, gate=None):
        x_in = self._apply_recurrent_noise(x)
        out = block(x_in, gate=gate)
        if self.recurrent_extra_layernorm:
            out = self.recurrent_extra_ln(out)
        return out

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
            blk = self.transformer.h[0]
            shared_block_fn = lambda tensor, gate=None: self._forward_shared_block(blk, tensor, gate=gate)
            if self.fixed_edge_blocks and self.fixed_head is not None:
                x = self.fixed_head(x)
            
            # Determine the number of recurrent steps
            if self.config.recurrent_shared_weights:
                # During training, n is sampled and passed. During inference, it can be user-specified.
                num_layers = n if n is not None else self.config.recurrent_depth
            else:
                num_layers = self.config.n_layer

            if self.sticky_dropout > 0 and self.training:
                B = x.shape[0]
                active_mask = torch.ones(B, device=x.device, dtype=torch.bool)
                num_expanded_layers = 0
                for _ in range(num_layers):
                    active_indices = active_mask.nonzero(as_tuple=True)[0]
                    if active_indices.numel() == 0:
                        break
                    
                    x_active = x[active_indices]
                    
                    if self.layer_dropout > 0:
                        x_active = self.layer_dropout_module(shared_block_fn, x_active)
                    else:
                        x_active = shared_block_fn(x_active)
                    
                    x[active_indices] = x_active
                    num_expanded_layers += active_indices.numel() / B

                    # update mask for sticky dropout
                    drop_probs = torch.full((active_indices.numel(),), self.sticky_dropout, device=x.device)
                    drops = torch.bernoulli(drop_probs).bool()
                    active_mask.scatter_(0, active_indices[drops], False)

            elif self.learned_stopping:
                B = x.shape[0]
                device = x.device
                dtype = x.dtype
                prob_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
                num_expanded_layers_tensor = torch.zeros(B, device=device, dtype=prob_dtype)
                continue_probs = torch.ones(B, device=device, dtype=prob_dtype)
                stop_masses = []

                if self.training:
                    self._stopping_step += 1

                for _ in range(num_layers):
                    x_in = x
                    if self.layer_dropout > 0 and self.training:
                        x_transformed = self.layer_dropout_module(shared_block_fn, x_in)
                    else:
                        x_transformed = shared_block_fn(x_in)

                    pooled = x_in.mean(dim=1)
                    last_token = x_in[:, -1, :]
                    stop_features = torch.cat((last_token, pooled), dim=-1)
                    predictor_dtype = self.stop_predictor[0].weight.dtype if isinstance(self.stop_predictor, nn.Sequential) else stop_features.dtype
                    stop_features = stop_features.to(predictor_dtype)
                    stop_logits = self.stop_predictor(stop_features).squeeze(-1)
                    if self.learned_stopping_temperature != 1.0:
                        stop_logits = stop_logits / self.learned_stopping_temperature
                    stop_prob = torch.sigmoid(stop_logits).to(prob_dtype)

                    if self.training and self._stopping_step <= self.learned_stopping_warmup_steps:
                        stop_prob = torch.zeros_like(stop_prob)
                    elif not self.training and self.learned_stopping_use_threshold:
                        stop_prob = (stop_prob > self.learned_stopping_threshold).to(prob_dtype)

                    if self.training or not self.learned_stopping_use_threshold:
                        stop_prob = stop_prob.clamp(self.learned_stopping_min_prob, 1.0 - self.learned_stopping_min_prob)

                    stop_mass = continue_probs * stop_prob
                    stop_masses.append(stop_mass)

                    continue_gate = 1.0 - stop_prob
                    stop_prob_mix = stop_prob.to(dtype)
                    continue_gate_mix = continue_gate.to(dtype)
                    x = stop_prob_mix.view(B, 1, 1) * x_in + continue_gate_mix.view(B, 1, 1) * x_transformed

                    num_expanded_layers_tensor = num_expanded_layers_tensor + (continue_probs * continue_gate)
                    continue_probs = continue_probs * continue_gate

                residual_mass = continue_probs
                stop_distribution = torch.stack(stop_masses + [residual_mass], dim=1).to(prob_dtype)
                stop_distribution = stop_distribution / stop_distribution.sum(dim=1, keepdim=True)

                depth_mean = num_expanded_layers_tensor.mean()
                target_depth = self.learned_stopping_target_depth
                if target_depth is None:
                    if self.config.recurrent_shared_weights:
                        reference = float(num_layers)
                    else:
                        reference = float(self.config.n_layer)
                    target_depth = 0.5 * (reference + 1.0)

                depth_target_tensor = depth_mean.new_tensor(target_depth)
                depth_delta = depth_mean - depth_target_tensor
                controller_loss = depth_delta ** 2
                controller_active = not (self.training and self._stopping_step <= self.learned_stopping_warmup_steps)
                controller_loss = controller_loss * self.learned_stopping_controller_weight if (controller_active and self.learned_stopping_controller_weight > 0) else controller_loss * 0.0

                entropy = -(stop_distribution * torch.log(stop_distribution + 1e-8)).sum(dim=1).mean()
                entropy_term = -self.learned_stopping_entropy_weight * entropy if (controller_active and self.learned_stopping_entropy_weight > 0) else entropy * 0.0

                if self.training and targets is not None:
                    aux_loss = (controller_loss + entropy_term).to(dtype)
                    self.stopping_metrics = {
                        "mean_depth": depth_mean.detach().item(),
                        "controller_loss": controller_loss.detach().item(),
                        "entropy": entropy.detach().item(),
                        "target_depth": float(target_depth),
                        "depth_delta": depth_delta.detach().item(),
                        "controller_active": 1.0 if controller_active else 0.0,
                    }
                else:
                    aux_loss = torch.tensor(0.0, device=device, dtype=dtype)
                    self.stopping_metrics = None

                num_expanded_layers = depth_mean.item()

            elif self.attentive_stopping:
                B = x.shape[0]
                device = x.device
                dtype = x.dtype
                prob_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
                num_expanded_layers_tensor = torch.zeros(B, device=device, dtype=prob_dtype)
                continue_probs = torch.ones(B, device=device, dtype=prob_dtype)
                stop_masses = []

                if self.training:
                    self._attentive_stopping_step += 1

                for _ in range(num_layers):
                    pooled = x.mean(dim=1)
                    last_token = x[:, -1, :]
                    stop_features = torch.cat((last_token, pooled), dim=-1)
                    predictor_dtype = self.stop_predictor[0].weight.dtype if isinstance(self.stop_predictor, nn.Sequential) else stop_features.dtype
                    stop_features = stop_features.to(predictor_dtype)
                    stop_logits = self.stop_predictor(stop_features).squeeze(-1)
                    if self.attentive_stopping_temperature != 1.0:
                        stop_logits = stop_logits / self.attentive_stopping_temperature
                    stop_prob = torch.sigmoid(stop_logits).to(prob_dtype)

                    if self.training and self._attentive_stopping_step <= self.attentive_stopping_warmup_steps:
                        stop_prob = torch.zeros_like(stop_prob)
                    elif not self.training and self.attentive_stopping_use_threshold:
                        stop_prob = (stop_prob > self.attentive_stopping_threshold).to(prob_dtype)
                    
                    if self.training or not self.attentive_stopping_use_threshold:
                        stop_prob = stop_prob.clamp(self.attentive_stopping_min_prob, 1.0 - self.attentive_stopping_min_prob)

                    stop_mass = continue_probs * stop_prob
                    stop_masses.append(stop_mass)

                    continue_gate = 1.0 - stop_prob
                    x = shared_block_fn(x, gate=continue_gate.to(dtype).view(B, 1, 1))

                    num_expanded_layers_tensor = num_expanded_layers_tensor + (continue_probs * continue_gate)
                    continue_probs = continue_probs * continue_gate

                residual_mass = continue_probs
                stop_distribution = torch.stack(stop_masses + [residual_mass], dim=1).to(prob_dtype)
                stop_distribution = stop_distribution / stop_distribution.sum(dim=1, keepdim=True)

                depth_mean = num_expanded_layers_tensor.mean()
                target_depth = self.attentive_stopping_target_depth
                if target_depth is None:
                    if self.config.recurrent_shared_weights:
                        reference = float(num_layers)
                    else:
                        reference = float(self.config.n_layer)
                    target_depth = 0.5 * (reference + 1.0)
                
                depth_target_tensor = depth_mean.new_tensor(target_depth)
                depth_delta = depth_mean - depth_target_tensor
                controller_loss = depth_delta ** 2
                controller_active = not (self.training and self._attentive_stopping_step <= self.attentive_stopping_warmup_steps)
                controller_loss = controller_loss * self.attentive_stopping_controller_weight if (controller_active and self.attentive_stopping_controller_weight > 0) else controller_loss * 0.0

                entropy = -(stop_distribution * torch.log(stop_distribution + 1e-8)).sum(dim=1).mean()
                entropy_term = -self.attentive_stopping_entropy_weight * entropy if (controller_active and self.attentive_stopping_entropy_weight > 0) else entropy * 0.0

                if self.training and targets is not None:
                    aux_loss = (controller_loss + entropy_term).to(dtype)
                    self.attentive_stopping_metrics = {
                        "mean_depth": depth_mean.detach().item(),
                        "controller_loss": controller_loss.detach().item(),
                        "entropy": entropy.detach().item(),
                        "target_depth": float(target_depth),
                        "depth_delta": depth_delta.detach().item(),
                        "controller_active": 1.0 if controller_active else 0.0,
                    }
                else:
                    aux_loss = torch.tensor(0.0, device=device, dtype=dtype)
                    self.attentive_stopping_metrics = None
                
                num_expanded_layers = depth_mean.item()

            else:
                # original logic + layer_dropout
                num_expanded_layers = num_layers
                for _ in range(num_layers):
                    if self.layer_dropout > 0 and self.training:
                        x = self.layer_dropout_module(shared_block_fn, x)
                    else:
                        x = shared_block_fn(x)
            if self.fixed_edge_blocks and self.fixed_tail is not None:
                x = self.fixed_tail(x)
        else: # Baseline
            for block in self.transformer.h:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if (self.learned_stopping or self.attentive_stopping) and self.training:
                loss = loss + aux_loss.to(loss.dtype)
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
