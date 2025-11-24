"""
Helpers that keep oracle teacher updates, loss shaping, and curriculum logic modular.

The goal is to keep the oracle-stopping path flexible without crowding `model.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


def _safe_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    mask = mask.to(x.dtype)
    denom = mask.sum()
    if denom <= 0:
        return x.mean()
    return (x * mask).sum() / denom


class OracleTeacherController:
    """
    Encapsulates how the oracle teacher is refreshed.

    Supports:
    - Hard copy on interval (existing behavior).
    - EMA updates with an optional decay schedule.
    - Adaptive interval scaling based on oracle loss drift.
    """

    def __init__(self, config: Any) -> None:
        self.use_ema = bool(getattr(config, "oracle_teacher_use_ema", False))
        self.ema_decay = float(getattr(config, "oracle_teacher_ema_decay", 0.999))
        self.ema_decay_min = float(getattr(config, "oracle_teacher_ema_decay_min", self.ema_decay))
        self.ema_decay_schedule = int(getattr(config, "oracle_teacher_ema_decay_schedule", 0))

        self.interval = max(1, int(getattr(config, "oracle_update_interval", 1)))
        self.adaptive_interval = bool(getattr(config, "oracle_adaptive_update_interval", False))
        self.interval_min = max(1, int(getattr(config, "oracle_update_interval_min", self.interval)))
        self.interval_max = max(self.interval_min, int(getattr(config, "oracle_update_interval_max", self.interval)))
        self.interval_shrink = float(getattr(config, "oracle_update_interval_shrink", 0.8))
        self.interval_growth = float(getattr(config, "oracle_update_interval_growth", 1.2))
        self.interval_tolerance = float(getattr(config, "oracle_update_interval_tolerance", 0.05))

        self.last_update_step = -1
        self.prev_stop_loss: Optional[float] = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "ema_decay_min": self.ema_decay_min,
            "ema_decay_schedule": self.ema_decay_schedule,
            "interval": self.interval,
            "interval_min": self.interval_min,
            "interval_max": self.interval_max,
            "interval_shrink": self.interval_shrink,
            "interval_growth": self.interval_growth,
            "interval_tolerance": self.interval_tolerance,
            "last_update_step": self.last_update_step,
            "prev_stop_loss": self.prev_stop_loss,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_update_step = state.get("last_update_step", -1)
        self.prev_stop_loss = state.get("prev_stop_loss", None)
        # do not override constructor config; state carries runtime knobs only
        if "interval" in state:
            self.interval = int(state["interval"])

    def _current_decay(self, step: int) -> float:
        if not self.use_ema:
            return 1.0
        if self.ema_decay_schedule <= 0:
            return self.ema_decay
        progress = min(1.0, max(0.0, step / max(1, self.ema_decay_schedule)))
        return max(self.ema_decay_min, self.ema_decay - (self.ema_decay - self.ema_decay_min) * progress)

    def _ema_update(self, student: nn.Module, teacher: nn.Module, decay: float) -> None:
        with torch.no_grad():
            student_params = dict(student.named_parameters())
            for name, param_t in teacher.named_parameters():
                param_s = student_params.get(name, None)
                if param_s is None:
                    continue
                param_t.mul_(decay).add_(param_s, alpha=1.0 - decay)
            # Buffers (e.g., layer norm stats) are copied directly.
            student_buffers = dict(student.named_buffers())
            for name, buf_t in teacher.named_buffers():
                buf_s = student_buffers.get(name, None)
                if buf_s is not None:
                    buf_t.copy_(buf_s)

    def _hard_copy(self, student: nn.Module, teacher: nn.Module) -> None:
        with torch.no_grad():
            teacher.load_state_dict(student.state_dict(), strict=False)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)

    def _maybe_adjust_interval(self, metrics: Optional[dict[str, Any]]) -> None:
        if not self.adaptive_interval or metrics is None:
            return
        stop_loss = metrics.get("stop_loss", None)
        if stop_loss is None:
            return
        if self.prev_stop_loss is None:
            self.prev_stop_loss = stop_loss
            return
        prev = max(self.prev_stop_loss, 1e-8)
        rel_change = (stop_loss - prev) / prev
        if rel_change > self.interval_tolerance:
            scaled = int(math.ceil(self.interval * self.interval_growth))
            self.interval = min(self.interval_max, max(1, scaled))
        elif rel_change < -self.interval_tolerance:
            scaled = int(math.floor(self.interval * self.interval_shrink))
            self.interval = max(self.interval_min, scaled)
        self.prev_stop_loss = stop_loss

    def maybe_update(self, student: nn.Module, teacher: nn.Module, step: int, metrics: Optional[dict[str, Any]]) -> bool:
        if teacher is None:
            return False
        if self.use_ema and self.last_update_step == step:
            return False
        if self.use_ema:
            decay = self._current_decay(step)
            self._ema_update(student, teacher, decay)
            self.last_update_step = step
            return True

        should_copy = self.last_update_step < 0 or (step - self.last_update_step) >= self.interval
        if should_copy:
            self._hard_copy(student, teacher)
            self.last_update_step = step
        self._maybe_adjust_interval(metrics)
        return should_copy


@dataclass
class OracleLossController:
    confidence_weighting: bool
    confidence_floor: float
    confidence_exponent: float
    confidence_ceiling: float
    stop_adv_clip: float

    @classmethod
    def from_config(cls, config: Any) -> "OracleLossController":
        return cls(
            confidence_weighting=bool(getattr(config, "oracle_confidence_weighting", False)),
            confidence_floor=float(getattr(config, "oracle_confidence_floor", 0.05)),
            confidence_exponent=float(getattr(config, "oracle_confidence_exponent", 1.0)),
            confidence_ceiling=float(getattr(config, "oracle_confidence_ceiling", 1.0)),
            stop_adv_clip=float(getattr(config, "oracle_stop_adv_clip", 0.0)),
        )

    def _apply_confidence(self, loss: torch.Tensor, confidence: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.confidence_weighting or confidence is None:
            return loss
        weight = confidence.clamp(self.confidence_floor, self.confidence_ceiling)
        if self.confidence_exponent != 1.0:
            weight = weight.pow(self.confidence_exponent)
        return loss * weight

    def _apply_advantage_clip(self, loss: torch.Tensor, stop_prob: torch.Tensor, stop_target: torch.Tensor) -> torch.Tensor:
        if self.stop_adv_clip <= 0:
            return loss
        # Treat advantage as negative log-prob of the teacher-preferred event; clip its magnitude.
        target_prob = torch.where(stop_target > 0.5, stop_prob, 1.0 - stop_prob).clamp(1e-8, 1.0)
        advantage = (-torch.log(target_prob)).detach()
        clipped = torch.clamp(advantage, max=self.stop_adv_clip)
        scale = clipped / (advantage + 1e-8)
        return loss * scale

    def reduce_stop_loss(
        self,
        layer_loss: torch.Tensor,
        stop_prob: torch.Tensor,
        stop_target: torch.Tensor,
        confidence: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        weighted = self._apply_confidence(layer_loss, confidence)
        weighted = self._apply_advantage_clip(weighted, stop_prob, stop_target)
        return _safe_mean(weighted, mask)

    def reduce_difficulty_loss(
        self,
        difficulty_loss: torch.Tensor,
        confidence: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        weighted = self._apply_confidence(difficulty_loss, confidence)
        return _safe_mean(weighted, mask)


class OracleCurriculumController:
    """Linearly schedules oracle depth and temperature."""

    def __init__(self, config: Any) -> None:
        self.depth_start = getattr(config, "oracle_curriculum_depth_start", None)
        self.depth_warmup = int(getattr(config, "oracle_curriculum_depth_warmup_steps", 0) or 0)
        self.temperature_final = getattr(config, "oracle_temperature_final", None)
        self.temperature_schedule = int(getattr(config, "oracle_temperature_schedule_steps", 0) or 0)

    def depth(self, base_depth: int, step: int) -> int:
        if self.depth_start is None or self.depth_warmup <= 0:
            return base_depth
        start = int(self.depth_start)
        target = max(1, base_depth)
        progress = min(1.0, max(0.0, step / max(1, self.depth_warmup)))
        depth = int(round(start + (target - start) * progress))
        return max(1, min(depth, target))

    def temperature(self, base_temp: float, step: int) -> float:
        if self.temperature_final is None or self.temperature_schedule <= 0:
            return base_temp
        start = float(base_temp)
        end = float(self.temperature_final)
        progress = min(1.0, max(0.0, step / max(1, self.temperature_schedule)))
        return start + (end - start) * progress
