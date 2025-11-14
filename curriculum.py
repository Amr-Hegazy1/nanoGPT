"""
Curriculum strategies for recurrent depth scheduling.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Tuple, Type

import numpy as np

__all__ = [
    "RecurrentDepthStrategy",
    "create_recurrent_depth_strategy",
    "determine_recurrent_depth",
    "parse_schedule_options",
    "sample_recurrent_depth",
]


def sample_recurrent_depth(
    max_depth: int,
    *,
    scale_loss_by_n_layer: bool = False,
    min_depth: int = 1,
    peak: int = 32,
) -> int:
    """Sample the number of recurrent steps to expand."""
    if max_depth < min_depth:
        raise ValueError(f"max_depth ({max_depth}) must be >= min_depth ({min_depth})")
    if scale_loss_by_n_layer:
        return int(np.random.randint(min_depth, max_depth + 1))
    sigma = 0.435
    mu = math.log(peak) + sigma**2
    sample = np.random.lognormal(mu, sigma)
    return int(np.clip(round(sample), min_depth, max_depth))


class RecurrentDepthStrategy:
    """Base class for recurrent depth curriculum strategies."""

    name: str = "base"
    aliases: Tuple[str, ...] = ()

    def __init__(
        self,
        *,
        max_depth: int,
        min_depth: int = 1,
        interval: int = 1,
        peak: int = 32,
        scale_loss_by_n_layer: bool = False,
        **kwargs: Any,
    ) -> None:
        self.max_depth = max(int(max_depth), int(min_depth))
        self.min_depth = max(1, int(min_depth))
        self.interval = max(1, int(interval))
        self.peak = int(peak)
        self.scale_loss_by_n_layer = bool(scale_loss_by_n_layer)
        self.depth_span = max(0, self.max_depth - self.min_depth)
        self.extra_kwargs = dict(kwargs)

    def stage(self, step: int) -> int:
        return int(step) // self.interval

    def clamp(self, depth: float | int) -> int:
        return int(max(self.min_depth, min(self.max_depth, int(round(depth)))))

    def __call__(
        self,
        step: int,
        *,
        feedback_metric: float | None = None,
        difficulty_score: float | None = None,
    ) -> int:
        raise NotImplementedError


STRATEGY_REGISTRY: Dict[str, Type[RecurrentDepthStrategy]] = {}


def register_strategy(cls: Type[RecurrentDepthStrategy]) -> Type[RecurrentDepthStrategy]:
    names = {cls.name.lower(), *(alias.lower() for alias in cls.aliases)}
    for name in names:
        STRATEGY_REGISTRY[name] = cls
    return cls


@register_strategy
class RandomStrategy(RecurrentDepthStrategy):
    name = "random"
    aliases = ("default", "rand", "lognormal")

    def __call__(self, step: int, **_: Any) -> int:  # type: ignore[override]
        return sample_recurrent_depth(
            self.max_depth,
            scale_loss_by_n_layer=self.scale_loss_by_n_layer,
            min_depth=self.min_depth,
            peak=self.peak,
        )


@register_strategy
class AscendingStrategy(RecurrentDepthStrategy):
    name = "ascending"
    aliases = ("ascend", "increase", "increasing", "curriculum_up")

    def __call__(self, step: int, **_: Any) -> int:  # type: ignore[override]
        stage = self.stage(step)
        depth = self.min_depth + stage
        return self.clamp(depth)


@register_strategy
class DescendingStrategy(RecurrentDepthStrategy):
    name = "descending"
    aliases = ("descend", "decrease", "decreasing", "reverse", "curriculum_down")

    def __call__(self, step: int, **_: Any) -> int:  # type: ignore[override]
        stage = self.stage(step)
        depth = self.max_depth - stage
        return self.clamp(depth)


@register_strategy
class CyclicalStrategy(RecurrentDepthStrategy):
    name = "cyclical"
    aliases = ("cycle", "sawtooth")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        cycle_len = kwargs.get("cycle_length") or (self.depth_span + 1)
        self.cycle_length = max(1, int(cycle_len))

    def __call__(self, step: int, **_: Any) -> int:  # type: ignore[override]
        stage = self.stage(step) % self.cycle_length
        depth = self.min_depth + stage
        return self.clamp(depth)


@register_strategy
class StaircaseStrategy(RecurrentDepthStrategy):
    name = "staircase"
    aliases = ("plateau", "ladder")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stair_width = max(1, int(kwargs.get("stair_width", 2)))

    def __call__(self, step: int, **_: Any) -> int:  # type: ignore[override]
        stage = self.stage(step) // self.stair_width
        depth = self.min_depth + stage
        return self.clamp(depth)


@register_strategy
class ExponentialStrategy(RecurrentDepthStrategy):
    name = "exponential"
    aliases = ("log", "log_growth", "exp")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        growth = float(kwargs.get("growth_factor", 1.5))
        self.growth_factor = growth if growth > 1.0 else 1.05
        start = kwargs.get("start_depth", self.min_depth)
        self.current_depth = self.clamp(start)

    def __call__(self, step: int, **_: Any) -> int:  # type: ignore[override]
        stage = self.stage(step)
        depth = self.current_depth * (self.growth_factor ** stage)
        return self.clamp(depth)


@register_strategy
class RandomWalkStrategy(RecurrentDepthStrategy):
    name = "random_walk"
    aliases = ("jittered", "walk")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.step_size = max(1, int(kwargs.get("step_size", 1)))
        reset_prob = float(kwargs.get("reset_prob", 0.0))
        self.reset_prob = max(0.0, min(1.0, reset_prob))
        self.current_depth = self.clamp(kwargs.get("start_depth", self.min_depth))
        self._last_stage = None

    def __call__(self, step: int, **_: Any) -> int:  # type: ignore[override]
        stage = self.stage(step)
        if self._last_stage is None:
            self._last_stage = stage
            return self.current_depth
        if stage != self._last_stage:
            steps = stage - self._last_stage
            direction_steps = max(1, abs(steps))
            for _ in range(direction_steps):
                if np.random.rand() < self.reset_prob:
                    self.current_depth = self.min_depth
                delta = np.random.randint(-self.step_size, self.step_size + 1)
                self.current_depth = self.clamp(self.current_depth + delta)
            self._last_stage = stage
        return self.current_depth


@register_strategy
class TwoPhaseStrategy(RecurrentDepthStrategy):
    name = "two_phase"
    aliases = ("burst", "mix")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.steady_intervals = max(1, int(kwargs.get("steady_intervals", 5)))
        self.burst_intervals = max(1, int(kwargs.get("burst_intervals", 1)))
        self.steady_depth = self.clamp(kwargs.get("steady_depth", self.min_depth))
        self.burst_depth = self.clamp(kwargs.get("burst_depth", self.max_depth))
        self.cycle = self.steady_intervals + self.burst_intervals

    def __call__(self, step: int, **_: Any) -> int:  # type: ignore[override]
        stage = self.stage(step) % self.cycle
        if stage < self.steady_intervals:
            return self.steady_depth
        return self.burst_depth


@register_strategy
class PerformanceAwareStrategy(RecurrentDepthStrategy):
    name = "performance"
    aliases = ("metric", "adaptive")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tolerance = float(kwargs.get("tolerance", 1e-4))
        self.max_patience = max(1, int(kwargs.get("patience", 5)))
        self.warmup_intervals = max(0, int(kwargs.get("warmup_intervals", 0)))
        self.current_depth = self.clamp(kwargs.get("start_depth", self.min_depth))
        self.best_metric: float | None = None
        self._patience = 0

    def __call__(  # type: ignore[override]
        self,
        step: int,
        *,
        feedback_metric: float | None = None,
        **_: Any,
    ) -> int:
        stage = self.stage(step)
        if stage < self.warmup_intervals:
            self.current_depth = self.min_depth
            return self.current_depth

        metric = feedback_metric
        if metric is None or not math.isfinite(metric):
            return self.current_depth

        if self.best_metric is None or metric < self.best_metric - self.tolerance:
            self.best_metric = metric
            self.current_depth = self.clamp(self.current_depth + 1)
            self._patience = 0
            return self.current_depth

        self._patience += 1
        if self._patience >= self.max_patience:
            self.current_depth = self.clamp(self.current_depth - 1)
            self._patience = 0
        return self.current_depth


@register_strategy
class DifficultyAwareStrategy(RecurrentDepthStrategy):
    name = "difficulty"
    aliases = ("difficulty_aware", "adaptive_difficulty")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        floor = float(kwargs.get("difficulty_floor", 0.0))
        ceiling = float(kwargs.get("difficulty_ceiling", 1.0))
        if ceiling <= floor:
            ceiling = floor + 1.0
        self.difficulty_floor = floor
        self.difficulty_ceiling = ceiling
        self.invert = bool(kwargs.get("invert", False))

    def __call__(  # type: ignore[override]
        self,
        step: int,
        *,
        feedback_metric: float | None = None,
        difficulty_score: float | None = None,
    ) -> int:
        score = difficulty_score
        if score is None:
            if feedback_metric is None or not math.isfinite(feedback_metric):
                score = 0.5 * (self.difficulty_floor + self.difficulty_ceiling)
            else:
                score = feedback_metric
        score = max(self.difficulty_floor, min(self.difficulty_ceiling, float(score)))
        norm = (score - self.difficulty_floor) / (self.difficulty_ceiling - self.difficulty_floor)
        if self.invert:
            norm = 1.0 - norm
        depth = self.min_depth + round(norm * self.depth_span)
        return self.clamp(depth)


def parse_schedule_options(
    options: str | Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Parse schedule options from a dict or JSON/kv string."""
    if not options:
        return {}
    if isinstance(options, dict):
        return dict(options)
    text = str(options).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    result: Dict[str, Any] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
        else:
            key, value = part.strip(), "true"
        result[key] = _coerce_value(value)
    return result


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _freeze_options(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_options(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_options(v) for v in value)
    return value


def create_recurrent_depth_strategy(
    schedule: str,
    *,
    max_depth: int,
    min_depth: int,
    interval: int,
    peak: int,
    scale_loss_by_n_layer: bool,
    schedule_options: Dict[str, Any] | None = None,
) -> RecurrentDepthStrategy:
    """Instantiate a strategy from the registry."""
    name = (schedule or "random").lower()
    cls = STRATEGY_REGISTRY.get(name, STRATEGY_REGISTRY["random"])
    return cls(
        max_depth=max_depth,
        min_depth=min_depth,
        interval=interval,
        peak=peak,
        scale_loss_by_n_layer=scale_loss_by_n_layer,
        **(schedule_options or {}),
    )


_STRATEGY_CACHE: Dict[
    Tuple[Any, ...],
    RecurrentDepthStrategy,
] = {}


def determine_recurrent_depth(
    step: int,
    max_depth: int,
    *,
    schedule: str = "random",
    interval: int | float | None = None,
    min_depth: int = 1,
    resample_prob: float = 0.0,
    scale_loss_by_n_layer: bool = False,
    peak: int = 32,
    schedule_options: dict[str, Any] | str | None = None,
    feedback_metric: float | None = None,
    difficulty_score: float | None = None,
) -> int:
    """
    Determine the number of recurrent steps according to a curriculum schedule.
    """
    if max_depth < min_depth:
        max_depth = min_depth

    interval_val = 1 if interval is None else int(interval)
    interval_val = max(1, interval_val)

    resample_prob = max(0.0, min(1.0, float(resample_prob)))
    if resample_prob > 0.0 and np.random.rand() < resample_prob:
        return sample_recurrent_depth(
            max_depth,
            scale_loss_by_n_layer=scale_loss_by_n_layer,
            min_depth=min_depth,
            peak=peak,
        )

    options = parse_schedule_options(schedule_options)
    strategy = _get_or_create_strategy(
        schedule=schedule,
        max_depth=max_depth,
        min_depth=min_depth,
        interval=interval_val,
        peak=peak,
        scale_loss_by_n_layer=scale_loss_by_n_layer,
        schedule_options=options,
    )
    return strategy(
        step,
        feedback_metric=feedback_metric,
        difficulty_score=difficulty_score,
    )


def _get_or_create_strategy(
    *,
    schedule: str,
    max_depth: int,
    min_depth: int,
    interval: int,
    peak: int,
    scale_loss_by_n_layer: bool,
    schedule_options: Dict[str, Any],
) -> RecurrentDepthStrategy:
    cls = STRATEGY_REGISTRY.get((schedule or "random").lower(), STRATEGY_REGISTRY["random"])
    key = (
        cls.name,
        int(max_depth),
        int(min_depth),
        int(interval),
        int(peak),
        bool(scale_loss_by_n_layer),
        _freeze_options(schedule_options),
    )
    strategy = _STRATEGY_CACHE.get(key)
    if strategy is None:
        strategy = create_recurrent_depth_strategy(
            cls.name,
            max_depth=max_depth,
            min_depth=min_depth,
            interval=interval,
            peak=peak,
            scale_loss_by_n_layer=scale_loss_by_n_layer,
            schedule_options=schedule_options,
        )
        _STRATEGY_CACHE[key] = strategy
    return strategy
