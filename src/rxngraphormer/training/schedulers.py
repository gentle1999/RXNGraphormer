"""Learning-rate scheduler implementations used by all training paths."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def WarmupCosineAnnealingLR(
    cur_iter: int,
    warm_up_iter: int = 10,
    T_max: int = 50,
    lr_max: float = 0.1,
    lr_min: float = 1e-5,
) -> float:
    """
    From: https://blog.marquis.eu.org/posts/2e3746c6/
    Usage: scheduler = LambdaLR(optimizer, lr_lambda=WarmupCosineAnnealingLR)
    """
    if cur_iter < warm_up_iter:
        return (lr_max - lr_min) * (cur_iter / warm_up_iter) + lr_min
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi)
    )


def get_linear_scheduler_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
        )

    lr_lambdas: Callable[[int], float] = lr_lambda
    return LambdaLR(optimizer, lr_lambdas, last_epoch)


class NoamLR(LRScheduler):
    """
    Adapted from https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py

    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model.
    """

    def __init__(self, optimizer: Optimizer, model_size: int, warmup_steps: int, step_scale: float = 1.0) -> None:
        if step_scale <= 0:
            raise ValueError("step_scale must be positive")
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.step_scale = step_scale
        super().__init__(optimizer)

    def get_lr(self) -> list[float | torch.Tensor]:
        step = max(1.0, 1.0 + float(self._step_count - 1) * self.step_scale)
        scale = self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

        return [base_lr * scale for base_lr in self.base_lrs]


__all__ = ["NoamLR", "WarmupCosineAnnealingLR", "get_linear_scheduler_with_warmup"]
