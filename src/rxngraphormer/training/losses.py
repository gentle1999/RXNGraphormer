from __future__ import annotations

from typing import Protocol

import torch


class RegressionLossSettings(Protocol):
    loss: str
    huber_beta: float


class RegressionLossConfig(Protocol):
    training: RegressionLossSettings


def build_regression_loss(config: RegressionLossConfig) -> torch.nn.Module:
    loss_name = str(config.training.loss).lower()
    if loss_name in {"l1", "mae"}:
        return torch.nn.L1Loss()
    if loss_name in {"l2", "mse"}:
        return torch.nn.MSELoss()
    if loss_name in {"smooth_l1", "huber"}:
        beta = float(config.training.huber_beta)
        if beta <= 0.0:
            raise ValueError("training.huber_beta must be positive for smooth_l1/huber loss")
        return torch.nn.SmoothL1Loss(beta=beta)
    raise NotImplementedError(f"Loss function {config.training.loss} is not implemented for regression")
