"""Optimizer and scheduler planning helpers shared by training backends."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, cast

LRScalingPolicy = Literal["none", "sqrt", "linear"]
SchedulerStepScalePolicy = Literal["none", "sample"]


@dataclass(frozen=True)
class OptimizationPlan:
    policy: LRScalingPolicy
    batch_size: int
    accum_steps: int
    world_size: int
    effective_batch_size: int
    base_batch_size: int
    batch_ratio: float
    lr_scale: float
    base_learning_rate: float
    learning_rate: float
    base_warmup_steps: int
    warmup_steps: int
    scale_warmup_steps: bool
    scheduler_step_scale_policy: SchedulerStepScalePolicy
    scheduler_step_scale: float

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


def scaled_learning_rate(base_learning_rate: float, *, batch_ratio: float, policy: LRScalingPolicy) -> float:
    if policy == "none":
        return base_learning_rate
    if policy == "sqrt":
        return base_learning_rate * math.sqrt(batch_ratio)
    if policy == "linear":
        return base_learning_rate * batch_ratio
    raise ValueError(f"Unsupported lr_scaling_policy: {policy!r}")


def scaled_warmup_steps(base_warmup_steps: int, *, batch_ratio: float, enabled: bool) -> int:
    if not enabled:
        return max(1, base_warmup_steps)
    return max(1, round(base_warmup_steps / batch_ratio))


def resolve_optimization_plan(config: object, *, world_size: int = 1) -> OptimizationPlan:
    optimizer = getattr(config, "optimizer")
    scheduler = getattr(config, "scheduler")
    data = getattr(config, "data", None)
    training = getattr(config, "training")

    batch_size = max(1, int(getattr(data, "batch_size", 1)))
    accum_steps = max(1, int(getattr(training, "accum", 1)))
    resolved_world_size = max(1, int(world_size))
    effective_batch_size = batch_size * accum_steps * resolved_world_size

    base_batch_size_value = getattr(optimizer, "base_batch_size", None)
    base_batch_size = max(1, int(base_batch_size_value if base_batch_size_value is not None else batch_size))
    batch_ratio = effective_batch_size / base_batch_size

    policy_value = str(getattr(optimizer, "lr_scaling_policy", "none")).lower()
    if policy_value not in {"none", "sqrt", "linear"}:
        raise ValueError(f"Unsupported lr_scaling_policy: {policy_value!r}")
    policy = cast(LRScalingPolicy, policy_value)

    configured_lr = float(getattr(optimizer, "learning_rate"))
    base_lr_value = getattr(optimizer, "base_learning_rate", None)
    base_learning_rate = float(base_lr_value if base_lr_value is not None else configured_lr)
    learning_rate = scaled_learning_rate(base_learning_rate, batch_ratio=batch_ratio, policy=policy)

    configured_warmup = max(1, int(getattr(scheduler, "warmup_step", 1)))
    base_warmup_value = getattr(scheduler, "base_warmup_step", None)
    base_warmup_steps = max(1, int(base_warmup_value if base_warmup_value is not None else configured_warmup))
    scale_warmup = bool(getattr(scheduler, "scale_warmup_steps", False))
    step_scale_policy_value = str(getattr(scheduler, "step_scale_policy", "none")).lower()
    if step_scale_policy_value not in {"none", "sample"}:
        raise ValueError(f"Unsupported scheduler step_scale_policy: {step_scale_policy_value!r}")
    step_scale_policy = cast(SchedulerStepScalePolicy, step_scale_policy_value)
    scheduler_step_scale = batch_ratio if step_scale_policy == "sample" else 1.0
    warmup_steps = (
        base_warmup_steps
        if step_scale_policy == "sample"
        else scaled_warmup_steps(base_warmup_steps, batch_ratio=batch_ratio, enabled=scale_warmup)
    )

    return OptimizationPlan(
        policy=policy,
        batch_size=batch_size,
        accum_steps=accum_steps,
        world_size=resolved_world_size,
        effective_batch_size=effective_batch_size,
        base_batch_size=base_batch_size,
        batch_ratio=batch_ratio,
        lr_scale=learning_rate / base_learning_rate if base_learning_rate != 0 else 0.0,
        base_learning_rate=base_learning_rate,
        learning_rate=learning_rate,
        base_warmup_steps=base_warmup_steps,
        warmup_steps=warmup_steps,
        scale_warmup_steps=scale_warmup,
        scheduler_step_scale_policy=step_scale_policy,
        scheduler_step_scale=scheduler_step_scale,
    )


__all__ = [
    "LRScalingPolicy",
    "OptimizationPlan",
    "SchedulerStepScalePolicy",
    "resolve_optimization_plan",
    "scaled_learning_rate",
    "scaled_warmup_steps",
]
