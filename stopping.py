from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn


@dataclass
class StoppingContext:
    x: torch.Tensor
    shared_block_fn: Callable[..., torch.Tensor]
    num_layers: int
    targets: Optional[torch.Tensor]
    aux_loss: torch.Tensor


@dataclass
class StoppingResult:
    x: torch.Tensor
    aux_loss: torch.Tensor
    num_expanded_layers: float


class StoppingStrategy:
    def is_active(self, model: Any, ctx: StoppingContext) -> bool:
        raise NotImplementedError

    def run(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        raise NotImplementedError


class StickyDropoutStrategy(StoppingStrategy):
    def is_active(self, model: Any, ctx: StoppingContext) -> bool:
        return model.sticky_dropout > 0 and model.training

    def run(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        if getattr(model, "stopping_tokenwise", False):
            return self._run_tokenwise(model, ctx)
        return self._run_sequencewise(model, ctx)

    def _run_sequencewise(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        x = ctx.x
        shared_block_fn = ctx.shared_block_fn
        num_layers = ctx.num_layers

        B = x.shape[0]
        active_mask = torch.ones(B, device=x.device, dtype=torch.bool)
        num_expanded_layers = 0.0

        for _ in range(num_layers):
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            if active_indices.numel() == 0:
                break

            x_active = x[active_indices]

            x_active = shared_block_fn(x_active)

            x[active_indices] = x_active
            num_expanded_layers += active_indices.numel() / B

            drop_probs = torch.full((active_indices.numel(),), model.sticky_dropout, device=x.device)
            drops = torch.bernoulli(drop_probs).bool()
            active_mask.scatter_(0, active_indices[drops], False)

        return StoppingResult(x=x, aux_loss=ctx.aux_loss, num_expanded_layers=num_expanded_layers)

    def _run_tokenwise(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        x = ctx.x
        shared_block_fn = ctx.shared_block_fn
        num_layers = ctx.num_layers

        B, T = x.shape[:2]
        device = x.device
        dtype = x.dtype

        active_mask = torch.ones(B, T, device=device, dtype=torch.bool)
        expansion_counts = torch.zeros(B, T, device=device, dtype=torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype)

        for _ in range(num_layers):
            if not active_mask.any():
                break

            x_in = x
            x_transformed = shared_block_fn(x_in)

            update_mask = active_mask.unsqueeze(-1)
            x = torch.where(update_mask, x_transformed, x_in)

            expansion_counts = expansion_counts + active_mask.to(expansion_counts.dtype)

            if model.sticky_dropout > 0:
                drop_probs = torch.full_like(active_mask, model.sticky_dropout, dtype=torch.float32, device=device)
                drop_mask = torch.bernoulli(drop_probs).bool()
                active_mask = active_mask & (~drop_mask)

        num_expanded_layers = (expansion_counts.sum() / (B * T)).item()
        return StoppingResult(x=x, aux_loss=ctx.aux_loss, num_expanded_layers=num_expanded_layers)


class LearnedStoppingStrategy(StoppingStrategy):
    """
    Learns a probabilistic early-exit controller ("stop head") that makes a soft, differentiable
    decision at each shared layer application. Instead of hard-stopping compute, we mix the current
    hidden state with the block's output using the stop probability:

        x <- stop_prob * x_in + (1 - stop_prob) * block(x_in)

    We also track the expected expanded depth by accumulating the remaining probability mass that
    has not stopped yet. A controller term nudges the mean expected depth toward a target, and an
    entropy term encourages a high-entropy exit distribution so the controller doesn't collapse
    prematurely. See comments below for which quantities affect loss vs. are logged for diagnostics.
    """
    def is_active(self, model: Any, ctx: StoppingContext) -> bool:
        return bool(model.learned_stopping)

    def run(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        if getattr(model, "stopping_tokenwise", False):
            return self._run_tokenwise(model, ctx)
        return self._run_sequencewise(model, ctx)

    def _run_sequencewise(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        """
        Applies learned stopping at the sequence level and tracks expected expanded depth by
        integrating continuation probabilities across steps.
        """
        x = ctx.x
        shared_block_fn = ctx.shared_block_fn
        num_layers = ctx.num_layers
        targets = ctx.targets
        aux_loss = ctx.aux_loss

        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        prob_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        # Expected depth accumulator and probability mass that is still "alive" (i.e., continuing)
        # at each layer. These directly affect the controller/entropy losses below.
        num_expanded_layers_tensor = torch.zeros(B, device=device, dtype=prob_dtype)
        continue_probs = torch.ones(B, device=device, dtype=prob_dtype)
        # Stores probability mass for halting exactly at each layer; later forms a categorical distribution
        # over "first stop at layer k" used to compute the entropy regularizer (affects loss) and also
        # for logging convenience.
        stop_masses: List[torch.Tensor] = []

        if model.training:
            model._stopping_step += 1

        for _ in range(num_layers):
            x_in = x
            x_transformed = shared_block_fn(x_in)

            # Use coarse sequence statistics so the predictor can understand current progress.
            # last_token ~ local signal; pooled ~ global progress summary.
            pooled = x_in.mean(dim=1)
            last_token = x_in[:, -1, :]
            stop_features = torch.cat((last_token, pooled), dim=-1)
            predictor_dtype = (
                model.stop_predictor[0].weight.dtype
                if isinstance(model.stop_predictor, nn.Sequential)
                else stop_features.dtype
            )
            # Cast to predictor dtype (e.g., fp32) for numerical stability on mixed precision.
            stop_features = stop_features.to(predictor_dtype)
            stop_logits = model.stop_predictor(stop_features).squeeze(-1)
            if model.learned_stopping_temperature != 1.0:
                stop_logits = stop_logits / model.learned_stopping_temperature
            stop_prob = torch.sigmoid(stop_logits).to(prob_dtype)

            # Warmup keeps the gate fully open (continue=1) for stability early in training.
            if model.training and model._stopping_step <= model.learned_stopping_warmup_steps:
                stop_prob = torch.zeros_like(stop_prob)
            # At eval we can snap to a hard decision if configured.
            elif not model.training and model.learned_stopping_use_threshold:
                stop_prob = (stop_prob > model.learned_stopping_threshold).to(prob_dtype)

            # During training (or when not using hard thresholding), clamp away from 0/1 to avoid
            # saturating gradients.
            if model.training or not model.learned_stopping_use_threshold:
                stop_prob = stop_prob.clamp(model.learned_stopping_min_prob, 1.0 - model.learned_stopping_min_prob)

            # stop_mass is the probability of stopping for the first time at this layer (affects entropy term).
            stop_mass = continue_probs * stop_prob
            stop_masses.append(stop_mass)

            continue_gate = 1.0 - stop_prob
            stop_prob_mix = stop_prob.to(dtype)
            continue_gate_mix = continue_gate.to(dtype)
            # Convex combination keeps the computation differentiable instead of hard-stopping activations.
            x = stop_prob_mix.view(B, 1, 1) * x_in + continue_gate_mix.view(B, 1, 1) * x_transformed

            # Only the probability mass that continues should count toward expected depth.
            num_expanded_layers_tensor = num_expanded_layers_tensor + (continue_probs * continue_gate)
            continue_probs = continue_probs * continue_gate

        # Any probability mass that never fired a stop forms the residual "run full depth" outcome.
        residual_mass = continue_probs
        # This distribution is used to compute the entropy regularizer (affects loss) and for logging.
        stop_distribution = torch.stack(stop_masses + [residual_mass], dim=1).to(prob_dtype)
        stop_distribution = stop_distribution / stop_distribution.sum(dim=1, keepdim=True)

        depth_mean = num_expanded_layers_tensor.mean()
        target_depth = model.learned_stopping_target_depth
        if target_depth is None:
            if model.config.recurrent_shared_weights:
                reference = float(num_layers)
            else:
                reference = float(model.config.n_layer)
            # Default target is the center-of-mass of a uniform distribution over {1..N}.
            target_depth = 0.5 * (reference + 1.0)

        # Controller nudges the mean depth toward the configured target depth (affects loss).
        depth_target_tensor = depth_mean.new_tensor(target_depth)
        depth_delta = depth_mean - depth_target_tensor
        controller_loss = depth_delta ** 2
        controller_active = not (model.training and model._stopping_step <= model.learned_stopping_warmup_steps)
        if controller_active and model.learned_stopping_controller_weight > 0:
            controller_loss = controller_loss * model.learned_stopping_controller_weight
        else:
            controller_loss = controller_loss * 0.0

        # Entropy regularizer prevents the controller from collapsing to a single layer too early (affects loss).
        entropy = -(stop_distribution * torch.log(stop_distribution + 1e-8)).sum(dim=1).mean()
        if controller_active and model.learned_stopping_entropy_weight > 0:
            entropy_term = -model.learned_stopping_entropy_weight * entropy
        else:
            entropy_term = entropy * 0.0

        if model.training and targets is not None:
            aux_loss = (controller_loss + entropy_term).to(dtype)
            # LOGGING-ONLY: the following dictionary is exported to dashboards; it does not change
            # the forward activations and is detached from autograd.
            model.stopping_metrics = {
                "mean_depth": depth_mean.detach().item(),
                "controller_loss": controller_loss.detach().item(),
                "entropy": entropy.detach().item(),
                "target_depth": float(target_depth),
                "depth_delta": depth_delta.detach().item(),
                "controller_active": 1.0 if controller_active else 0.0,
            }
        else:
            aux_loss = torch.tensor(0.0, device=device, dtype=dtype)
            model.stopping_metrics = None

        # Note: returned for logging/analysis in the training loop; does not affect loss directly.
        num_expanded_layers = depth_mean.item()

        return StoppingResult(x=x, aux_loss=aux_loss, num_expanded_layers=num_expanded_layers)

    def _run_tokenwise(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        """
        Extends learned stopping to per-token decisions so each position can halt independently.
        The convex mixing happens per token, allowing different positions to "stop" at different
        depths in a single forward.
        """
        x = ctx.x
        shared_block_fn = ctx.shared_block_fn
        num_layers = ctx.num_layers
        targets = ctx.targets
        aux_loss = ctx.aux_loss

        B, T = x.shape[:2]
        device = x.device
        dtype = x.dtype
        prob_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        # Track per-position expected depth and remaining continuation mass. These affect the
        # controller/entropy losses below and provide interpretable diagnostics.
        num_expanded_tokens = torch.zeros(B, T, device=device, dtype=prob_dtype)
        continue_probs = torch.ones(B, T, device=device, dtype=prob_dtype)
        stop_masses: List[torch.Tensor] = []

        if model.training:
            model._stopping_step += 1

        for _ in range(num_layers):
            x_in = x
            x_transformed = shared_block_fn(x_in)

            # Tokenwise variant gives each position its own features plus the pooled summary.
            pooled = x_in.mean(dim=1, keepdim=True).expand(-1, T, -1)
            stop_features = torch.cat((x_in, pooled), dim=-1)
            predictor_dtype = (
                model.stop_predictor[0].weight.dtype
                if isinstance(model.stop_predictor, nn.Sequential)
                else stop_features.dtype
            )
            # Cast to predictor dtype (e.g., fp32) for numerical stability on mixed precision.
            stop_features = stop_features.to(predictor_dtype)
            stop_logits = model.stop_predictor(stop_features).squeeze(-1)
            if model.learned_stopping_temperature != 1.0:
                stop_logits = stop_logits / model.learned_stopping_temperature
            stop_prob = torch.sigmoid(stop_logits).to(prob_dtype)

            # Warmup keeps the gate fully open (continue=1) for stability early in training.
            if model.training and model._stopping_step <= model.learned_stopping_warmup_steps:
                stop_prob = torch.zeros_like(stop_prob)
            # At eval we can snap to a hard decision if configured.
            elif not model.training and model.learned_stopping_use_threshold:
                stop_prob = (stop_prob > model.learned_stopping_threshold).to(prob_dtype)

            # During training (or when not using hard thresholding), clamp away from 0/1 to avoid
            # saturating gradients.
            if model.training or not model.learned_stopping_use_threshold:
                stop_prob = stop_prob.clamp(model.learned_stopping_min_prob, 1.0 - model.learned_stopping_min_prob)

            stop_mass = continue_probs * stop_prob
            stop_masses.append(stop_mass)

            continue_gate = 1.0 - stop_prob
            stop_prob_mix = stop_prob.to(dtype).unsqueeze(-1)
            continue_gate_mix = continue_gate.to(dtype).unsqueeze(-1)
            # Mix per-token activations so different positions can stop at different depths.
            x = stop_prob_mix * x_in + continue_gate_mix * x_transformed

            num_expanded_tokens = num_expanded_tokens + (continue_probs * continue_gate)
            continue_probs = continue_probs * continue_gate

        residual_mass = continue_probs
        # Used to compute entropy regularizer (affects loss) and for logging.
        stop_distribution = torch.stack(stop_masses + [residual_mass], dim=1).to(prob_dtype)
        stop_distribution = stop_distribution / (stop_distribution.sum(dim=1, keepdim=True) + 1e-8)

        depth_mean = num_expanded_tokens.mean()
        target_depth = model.learned_stopping_target_depth
        if target_depth is None:
            if model.config.recurrent_shared_weights:
                reference = float(num_layers)
            else:
                reference = float(model.config.n_layer)
            target_depth = 0.5 * (reference + 1.0)

        # Controller nudges the mean depth toward the configured target depth (affects loss).
        depth_target_tensor = depth_mean.new_tensor(target_depth)
        depth_delta = depth_mean - depth_target_tensor
        controller_loss = depth_delta ** 2
        controller_active = not (model.training and model._stopping_step <= model.learned_stopping_warmup_steps)
        if controller_active and model.learned_stopping_controller_weight > 0:
            controller_loss = controller_loss * model.learned_stopping_controller_weight
        else:
            controller_loss = controller_loss * 0.0

        # Entropy regularizer keeps stopping distribution from collapsing too early (affects loss).
        entropy = -(stop_distribution * torch.log(stop_distribution + 1e-8)).sum(dim=1).mean()
        if controller_active and model.learned_stopping_entropy_weight > 0:
            entropy_term = -model.learned_stopping_entropy_weight * entropy
        else:
            entropy_term = entropy * 0.0

        if model.training and targets is not None:
            aux_loss = (controller_loss + entropy_term).to(dtype)
            # LOGGING-ONLY: exported diagnostics; do not affect forward activations.
            model.stopping_metrics = {
                "mean_depth": depth_mean.detach().item(),
                "controller_loss": controller_loss.detach().item(),
                "entropy": entropy.detach().item(),
                "target_depth": float(target_depth),
                "depth_delta": depth_delta.detach().item(),
                "controller_active": 1.0 if controller_active else 0.0,
            }
        else:
            aux_loss = torch.tensor(0.0, device=device, dtype=dtype)
            model.stopping_metrics = None

        # Note: returned for logging/analysis in the training loop; does not affect loss directly.
        num_expanded_layers = depth_mean.item()

        return StoppingResult(x=x, aux_loss=aux_loss, num_expanded_layers=num_expanded_layers)


class AttentiveStoppingStrategy(StoppingStrategy):
    """
    Variant where the stop probability controls a gate that is fed into the shared block.
    The gate scales attention- and MLP-path contributions and can also bias the attention
    scores. This makes the compute taper off smoothly as the stop head "closes" the gate.

    Internally, we still track the expected expanded depth using continuation probabilities
    and apply the same controller/entropy regularization terms. See comments below for which
    quantities affect loss vs. are logged for diagnostics.
    """
    def is_active(self, model: Any, ctx: StoppingContext) -> bool:
        return bool(model.attentive_stopping)

    def run(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        if getattr(model, "stopping_tokenwise", False):
            return self._run_tokenwise(model, ctx)
        return self._run_sequencewise(model, ctx)

    def _run_sequencewise(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        """
        Sequence-level attentive stopping: one gate per sequence (broadcast across tokens). The
        gate is shaped as (B, 1, 1) and passed into the shared block so the block can reduce
        its contribution proportionally.
        """
        x = ctx.x
        shared_block_fn = ctx.shared_block_fn
        num_layers = ctx.num_layers
        targets = ctx.targets
        aux_loss = ctx.aux_loss

        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        prob_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        # Track expected depth and remaining continuation mass per sequence. These directly affect
        # the controller/entropy losses below.
        num_expanded_layers_tensor = torch.zeros(B, device=device, dtype=prob_dtype)
        continue_probs = torch.ones(B, device=device, dtype=prob_dtype)
        stop_masses: List[torch.Tensor] = []

        if model.training:
            model._attentive_stopping_step += 1

        for _ in range(num_layers):
            pooled = x.mean(dim=1)
            last_token = x[:, -1, :]
            stop_features = torch.cat((last_token, pooled), dim=-1)
            predictor_dtype = (
                model.stop_predictor[0].weight.dtype
                if isinstance(model.stop_predictor, nn.Sequential)
                else stop_features.dtype
            )
            # Cast to predictor dtype (e.g., fp32) for numerical stability on mixed precision.
            stop_features = stop_features.to(predictor_dtype)
            stop_logits = model.stop_predictor(stop_features).squeeze(-1)
            if model.attentive_stopping_temperature != 1.0:
                stop_logits = stop_logits / model.attentive_stopping_temperature
            stop_prob = torch.sigmoid(stop_logits).to(prob_dtype)

            # Warmup keeps the gate fully open (continue=1) for stability early in training.
            if model.training and model._attentive_stopping_step <= model.attentive_stopping_warmup_steps:
                stop_prob = torch.zeros_like(stop_prob)
            # At eval we can snap to a hard decision if configured.
            elif not model.training and model.attentive_stopping_use_threshold:
                stop_prob = (stop_prob > model.attentive_stopping_threshold).to(prob_dtype)

            # During training (or when not using hard thresholding), clamp away from 0/1 to avoid
            # saturating gradients.
            if model.training or not model.attentive_stopping_use_threshold:
                stop_prob = stop_prob.clamp(model.attentive_stopping_min_prob, 1.0 - model.attentive_stopping_min_prob)

            # Same stop-mass accounting as learned stopping; treats each layer as a categorical outcome.
            stop_mass = continue_probs * stop_prob
            stop_masses.append(stop_mass)

            continue_gate = 1.0 - stop_prob
            gate = continue_gate.to(dtype).view(B, 1, 1)
            # Feed gate into the shared block; internally attention logits/residuals are scaled accordingly.
            x = shared_block_fn(x, gate=gate)

            num_expanded_layers_tensor = num_expanded_layers_tensor + (continue_probs * continue_gate)
            continue_probs = continue_probs * continue_gate

        residual_mass = continue_probs
        # Used to compute entropy regularizer (affects loss) and for logging.
        stop_distribution = torch.stack(stop_masses + [residual_mass], dim=1).to(prob_dtype)
        stop_distribution = stop_distribution / stop_distribution.sum(dim=1, keepdim=True)

        depth_mean = num_expanded_layers_tensor.mean()
        target_depth = model.attentive_stopping_target_depth
        if target_depth is None:
            if model.config.recurrent_shared_weights:
                reference = float(num_layers)
            else:
                reference = float(model.config.n_layer)
            target_depth = 0.5 * (reference + 1.0)

        # Controller nudges the mean depth toward the configured target depth (affects loss).
        depth_target_tensor = depth_mean.new_tensor(target_depth)
        depth_delta = depth_mean - depth_target_tensor
        controller_loss = depth_delta ** 2
        controller_active = not (model.training and model._attentive_stopping_step <= model.attentive_stopping_warmup_steps)
        if controller_active and model.attentive_stopping_controller_weight > 0:
            controller_loss = controller_loss * model.attentive_stopping_controller_weight
        else:
            controller_loss = controller_loss * 0.0

        # Entropy regularizer keeps stopping distribution from collapsing too early (affects loss).
        entropy = -(stop_distribution * torch.log(stop_distribution + 1e-8)).sum(dim=1).mean()
        if controller_active and model.attentive_stopping_entropy_weight > 0:
            entropy_term = -model.attentive_stopping_entropy_weight * entropy
        else:
            entropy_term = entropy * 0.0

        if model.training and targets is not None:
            aux_loss = (controller_loss + entropy_term).to(dtype)
            # LOGGING-ONLY: the following dictionary is exported to dashboards; it does not change
            # the forward activations and is detached from autograd.
            model.attentive_stopping_metrics = {
                "mean_depth": depth_mean.detach().item(),
                "controller_loss": controller_loss.detach().item(),
                "entropy": entropy.detach().item(),
                "target_depth": float(target_depth),
                "depth_delta": depth_delta.detach().item(),
                "controller_active": 1.0 if controller_active else 0.0,
            }
        else:
            aux_loss = torch.tensor(0.0, device=device, dtype=dtype)
            model.attentive_stopping_metrics = None

        # Note: returned for logging/analysis in the training loop; does not affect loss directly.
        num_expanded_layers = depth_mean.item()

        return StoppingResult(x=x, aux_loss=aux_loss, num_expanded_layers=num_expanded_layers)

    def _run_tokenwise(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        """
        Token-level attentive stopping so individual positions can taper compute through gated
        shared blocks. The gate is shaped as (B, T, 1) and passed into the shared block.
        """
        x = ctx.x
        shared_block_fn = ctx.shared_block_fn
        num_layers = ctx.num_layers
        targets = ctx.targets
        aux_loss = ctx.aux_loss

        B, T = x.shape[:2]
        device = x.device
        dtype = x.dtype
        prob_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        # Track per-position expected depth and remaining continuation mass. These directly affect
        # the controller/entropy losses below.
        num_expanded_tokens = torch.zeros(B, T, device=device, dtype=prob_dtype)
        continue_probs = torch.ones(B, T, device=device, dtype=prob_dtype)
        stop_masses: List[torch.Tensor] = []

        if model.training:
            model._attentive_stopping_step += 1

        for _ in range(num_layers):
            pooled = x.mean(dim=1, keepdim=True).expand(-1, T, -1)
            stop_features = torch.cat((x, pooled), dim=-1)
            predictor_dtype = (
                model.stop_predictor[0].weight.dtype
                if isinstance(model.stop_predictor, nn.Sequential)
                else stop_features.dtype
            )
            # Cast to predictor dtype (e.g., fp32) for numerical stability on mixed precision.
            stop_features = stop_features.to(predictor_dtype)
            stop_logits = model.stop_predictor(stop_features).squeeze(-1)
            if model.attentive_stopping_temperature != 1.0:
                stop_logits = stop_logits / model.attentive_stopping_temperature
            stop_prob = torch.sigmoid(stop_logits).to(prob_dtype)

            # Warmup keeps the gate fully open (continue=1) for stability early in training.
            if model.training and model._attentive_stopping_step <= model.attentive_stopping_warmup_steps:
                stop_prob = torch.zeros_like(stop_prob)
            # At eval we can snap to a hard decision if configured.
            elif not model.training and model.attentive_stopping_use_threshold:
                stop_prob = (stop_prob > model.attentive_stopping_threshold).to(prob_dtype)

            # During training (or when not using hard thresholding), clamp away from 0/1 to avoid
            # saturating gradients.
            if model.training or not model.attentive_stopping_use_threshold:
                stop_prob = stop_prob.clamp(model.attentive_stopping_min_prob, 1.0 - model.attentive_stopping_min_prob)

            stop_mass = continue_probs * stop_prob
            stop_masses.append(stop_mass)

            continue_gate = 1.0 - stop_prob
            gate = continue_gate.to(dtype).unsqueeze(-1)
            # Each token receives a gate so self-attention/MLP work can taper off per position.
            x = shared_block_fn(x, gate=gate)

            num_expanded_tokens = num_expanded_tokens + (continue_probs * continue_gate)
            continue_probs = continue_probs * continue_gate

        residual_mass = continue_probs
        # Used to compute entropy regularizer (affects loss) and for logging.
        stop_distribution = torch.stack(stop_masses + [residual_mass], dim=1).to(prob_dtype)
        stop_distribution = stop_distribution / (stop_distribution.sum(dim=1, keepdim=True) + 1e-8)

        depth_mean = num_expanded_tokens.mean()
        target_depth = model.attentive_stopping_target_depth
        if target_depth is None:
            if model.config.recurrent_shared_weights:
                reference = float(num_layers)
            else:
                reference = float(model.config.n_layer)
            target_depth = 0.5 * (reference + 1.0)

        # Controller nudges the mean depth toward the configured target depth (affects loss).
        depth_target_tensor = depth_mean.new_tensor(target_depth)
        depth_delta = depth_mean - depth_target_tensor
        controller_loss = depth_delta ** 2
        controller_active = not (model.training and model._attentive_stopping_step <= model.attentive_stopping_warmup_steps)
        if controller_active and model.attentive_stopping_controller_weight > 0:
            controller_loss = controller_loss * model.attentive_stopping_controller_weight
        else:
            controller_loss = controller_loss * 0.0

        # Entropy regularizer keeps stopping distribution from collapsing too early (affects loss).
        entropy = -(stop_distribution * torch.log(stop_distribution + 1e-8)).sum(dim=1).mean()
        if controller_active and model.attentive_stopping_entropy_weight > 0:
            entropy_term = -model.attentive_stopping_entropy_weight * entropy
        else:
            entropy_term = entropy * 0.0

        if model.training and targets is not None:
            aux_loss = (controller_loss + entropy_term).to(dtype)
            # LOGGING-ONLY: exported diagnostics; do not affect forward activations.
            model.attentive_stopping_metrics = {
                "mean_depth": depth_mean.detach().item(),
                "controller_loss": controller_loss.detach().item(),
                "entropy": entropy.detach().item(),
                "target_depth": float(target_depth),
                "depth_delta": depth_delta.detach().item(),
                "controller_active": 1.0 if controller_active else 0.0,
            }
        else:
            aux_loss = torch.tensor(0.0, device=device, dtype=dtype)
            model.attentive_stopping_metrics = None

        # Note: returned for logging/analysis in the training loop; does not affect loss directly.
        num_expanded_layers = depth_mean.item()

        return StoppingResult(x=x, aux_loss=aux_loss, num_expanded_layers=num_expanded_layers)


class DefaultStoppingStrategy(StoppingStrategy):
    def is_active(self, model: Any, ctx: StoppingContext) -> bool:
        return True

    def run(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        x = ctx.x
        shared_block_fn = ctx.shared_block_fn
        num_layers = ctx.num_layers

        num_expanded_layers = num_layers
        for _ in range(num_layers):
            x = shared_block_fn(x)

        return StoppingResult(x=x, aux_loss=ctx.aux_loss, num_expanded_layers=num_expanded_layers)


class StoppingController:
    def __init__(self, strategies: List[StoppingStrategy]):
        self.strategies = strategies

    def run(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        for strategy in self.strategies:
            if strategy.is_active(model, ctx):
                return strategy.run(model, ctx)
        raise RuntimeError("No stopping strategy was applicable.")
