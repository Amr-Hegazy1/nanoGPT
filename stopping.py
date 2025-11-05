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


class LearnedStoppingStrategy(StoppingStrategy):
    def is_active(self, model: Any, ctx: StoppingContext) -> bool:
        return bool(model.learned_stopping)

    def run(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        x = ctx.x
        shared_block_fn = ctx.shared_block_fn
        num_layers = ctx.num_layers
        targets = ctx.targets
        aux_loss = ctx.aux_loss

        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        prob_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        num_expanded_layers_tensor = torch.zeros(B, device=device, dtype=prob_dtype)
        continue_probs = torch.ones(B, device=device, dtype=prob_dtype)
        stop_masses: List[torch.Tensor] = []

        if model.training:
            model._stopping_step += 1

        for _ in range(num_layers):
            x_in = x
            x_transformed = shared_block_fn(x_in)

            pooled = x_in.mean(dim=1)
            last_token = x_in[:, -1, :]
            stop_features = torch.cat((last_token, pooled), dim=-1)
            predictor_dtype = (
                model.stop_predictor[0].weight.dtype
                if isinstance(model.stop_predictor, nn.Sequential)
                else stop_features.dtype
            )
            stop_features = stop_features.to(predictor_dtype)
            stop_logits = model.stop_predictor(stop_features).squeeze(-1)
            if model.learned_stopping_temperature != 1.0:
                stop_logits = stop_logits / model.learned_stopping_temperature
            stop_prob = torch.sigmoid(stop_logits).to(prob_dtype)

            if model.training and model._stopping_step <= model.learned_stopping_warmup_steps:
                stop_prob = torch.zeros_like(stop_prob)
            elif not model.training and model.learned_stopping_use_threshold:
                stop_prob = (stop_prob > model.learned_stopping_threshold).to(prob_dtype)

            if model.training or not model.learned_stopping_use_threshold:
                stop_prob = stop_prob.clamp(model.learned_stopping_min_prob, 1.0 - model.learned_stopping_min_prob)

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
        target_depth = model.learned_stopping_target_depth
        if target_depth is None:
            if model.config.recurrent_shared_weights:
                reference = float(num_layers)
            else:
                reference = float(model.config.n_layer)
            target_depth = 0.5 * (reference + 1.0)

        depth_target_tensor = depth_mean.new_tensor(target_depth)
        depth_delta = depth_mean - depth_target_tensor
        controller_loss = depth_delta ** 2
        controller_active = not (model.training and model._stopping_step <= model.learned_stopping_warmup_steps)
        if controller_active and model.learned_stopping_controller_weight > 0:
            controller_loss = controller_loss * model.learned_stopping_controller_weight
        else:
            controller_loss = controller_loss * 0.0

        entropy = -(stop_distribution * torch.log(stop_distribution + 1e-8)).sum(dim=1).mean()
        if controller_active and model.learned_stopping_entropy_weight > 0:
            entropy_term = -model.learned_stopping_entropy_weight * entropy
        else:
            entropy_term = entropy * 0.0

        if model.training and targets is not None:
            aux_loss = (controller_loss + entropy_term).to(dtype)
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

        num_expanded_layers = depth_mean.item()

        return StoppingResult(x=x, aux_loss=aux_loss, num_expanded_layers=num_expanded_layers)


class AttentiveStoppingStrategy(StoppingStrategy):
    def is_active(self, model: Any, ctx: StoppingContext) -> bool:
        return bool(model.attentive_stopping)

    def run(self, model: Any, ctx: StoppingContext) -> StoppingResult:
        x = ctx.x
        shared_block_fn = ctx.shared_block_fn
        num_layers = ctx.num_layers
        targets = ctx.targets
        aux_loss = ctx.aux_loss

        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        prob_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

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
            stop_features = stop_features.to(predictor_dtype)
            stop_logits = model.stop_predictor(stop_features).squeeze(-1)
            if model.attentive_stopping_temperature != 1.0:
                stop_logits = stop_logits / model.attentive_stopping_temperature
            stop_prob = torch.sigmoid(stop_logits).to(prob_dtype)

            if model.training and model._attentive_stopping_step <= model.attentive_stopping_warmup_steps:
                stop_prob = torch.zeros_like(stop_prob)
            elif not model.training and model.attentive_stopping_use_threshold:
                stop_prob = (stop_prob > model.attentive_stopping_threshold).to(prob_dtype)

            if model.training or not model.attentive_stopping_use_threshold:
                stop_prob = stop_prob.clamp(model.attentive_stopping_min_prob, 1.0 - model.attentive_stopping_min_prob)

            stop_mass = continue_probs * stop_prob
            stop_masses.append(stop_mass)

            continue_gate = 1.0 - stop_prob
            gate = continue_gate.to(dtype).view(B, 1, 1)
            x = shared_block_fn(x, gate=gate)

            num_expanded_layers_tensor = num_expanded_layers_tensor + (continue_probs * continue_gate)
            continue_probs = continue_probs * continue_gate

        residual_mass = continue_probs
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

        depth_target_tensor = depth_mean.new_tensor(target_depth)
        depth_delta = depth_mean - depth_target_tensor
        controller_loss = depth_delta ** 2
        controller_active = not (model.training and model._attentive_stopping_step <= model.attentive_stopping_warmup_steps)
        if controller_active and model.attentive_stopping_controller_weight > 0:
            controller_loss = controller_loss * model.attentive_stopping_controller_weight
        else:
            controller_loss = controller_loss * 0.0

        entropy = -(stop_distribution * torch.log(stop_distribution + 1e-8)).sum(dim=1).mean()
        if controller_active and model.attentive_stopping_entropy_weight > 0:
            entropy_term = -model.attentive_stopping_entropy_weight * entropy
        else:
            entropy_term = entropy * 0.0

        if model.training and targets is not None:
            aux_loss = (controller_loss + entropy_term).to(dtype)
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
