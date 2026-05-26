from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from ..config import EvalConfig, load_config
from ..models import build_pretrained_classification_model, build_regression_model_from_config
from ..runtime import DeviceManager


class _PretrainModelSettings(Protocol):
    pretrained_model_path: object


class _PretrainConfig(Protocol):
    model: _PretrainModelSettings


@dataclass
class RegressionPretrainCompatibilitySettings:
    config_path: str | os.PathLike
    device: torch.device | str | None = None
    atol: float = 1e-6


@dataclass
class RegressionPretrainCompatibilityReport:
    config_path: str
    pretrained_model_path: str
    rct_encoder_keys: int
    pdt_encoder_keys: int
    max_rct_encoder_abs_diff: float
    max_pdt_encoder_abs_diff: float
    atol: float = 1e-6

    @property
    def ok(self) -> bool:
        return (
            self.max_rct_encoder_abs_diff <= self.atol
            and self.max_pdt_encoder_abs_diff <= self.atol
        )

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "config_path": self.config_path,
            "pretrained_model_path": self.pretrained_model_path,
            "rct_encoder_keys": self.rct_encoder_keys,
            "pdt_encoder_keys": self.pdt_encoder_keys,
            "max_rct_encoder_abs_diff": self.max_rct_encoder_abs_diff,
            "max_pdt_encoder_abs_diff": self.max_pdt_encoder_abs_diff,
            "atol": self.atol,
        }


def check_regression_pretrain_compatibility(
    settings: RegressionPretrainCompatibilitySettings,
) -> RegressionPretrainCompatibilityReport:
    config = cast(_PretrainConfig, _load_compat_config(settings.config_path))
    pretrained_path = getattr(config.model, "pretrained_model_path", "")
    if not pretrained_path:
        raise ValueError("config.model.pretrained_model_path is required")
    device_manager = DeviceManager.from_value(settings.device or "cpu")

    pretrained_model = device_manager.move_module(build_pretrained_classification_model(pretrained_path)).eval()
    regression_model = device_manager.move_module(build_regression_model_from_config(config)).eval()

    rct_diff, rct_count = _max_state_dict_abs_diff(
        regression_model.rct_encoder.state_dict(),
        pretrained_model.rct_encoder.state_dict(),
    )
    pdt_diff, pdt_count = _max_state_dict_abs_diff(
        regression_model.pdt_encoder.state_dict(),
        pretrained_model.pdt_encoder.state_dict(),
    )
    return RegressionPretrainCompatibilityReport(
        config_path=os.fspath(settings.config_path),
        pretrained_model_path=os.fspath(pretrained_path),
        rct_encoder_keys=rct_count,
        pdt_encoder_keys=pdt_count,
        max_rct_encoder_abs_diff=rct_diff,
        max_pdt_encoder_abs_diff=pdt_diff,
        atol=settings.atol,
    )


def _load_compat_config(config_path: str | os.PathLike) -> object:
    config = load_config(config_path)
    if isinstance(config, EvalConfig):
        raise TypeError(f"Expected train config for regression compatibility, got eval config: {config_path}")
    return config


def _max_state_dict_abs_diff(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> tuple[float, int]:
    if left.keys() != right.keys():
        missing = sorted(set(right) - set(left))
        unexpected = sorted(set(left) - set(right))
        raise RuntimeError(f"State dict keys differ: missing={missing[:10]}, unexpected={unexpected[:10]}")
    max_diff = 0.0
    for key in left:
        left_value = left[key].detach().cpu()
        right_value = right[key].detach().cpu()
        if left_value.shape != right_value.shape:
            raise RuntimeError(f"State dict shape differs for {key}: {tuple(left_value.shape)} != {tuple(right_value.shape)}")
        if torch.is_floating_point(left_value):
            diff = float((left_value.float() - right_value.float()).abs().max().item())
        else:
            diff = 0.0 if torch.equal(left_value, right_value) else float("inf")
        max_diff = max(max_diff, diff)
    return max_diff, len(left)
