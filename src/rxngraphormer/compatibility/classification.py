from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from ..config import EvalConfig, load_config, resolve_config_file
from ..evaluation import classification_batch_input
from ..lightning.module import RXNGraphormerLitModule
from ..models import build_classification_model
from ..runtime import DeviceManager
from ..serialization import DEFAULT_CHECKPOINT_FILE
from .checkpointing import CheckpointAdapter, CheckpointLoadReport, resolve_model_checkpoint


@dataclass
class ClassificationCheckpointCompatibilitySettings:
    model_path: str | os.PathLike
    ckpt_file: str = DEFAULT_CHECKPOINT_FILE
    config_path: str | os.PathLike | None = None
    checkpoint_path: str | os.PathLike | None = None
    device: torch.device | str | None = None
    atol: float = 1e-6


@dataclass
class ClassificationCheckpointCompatibilityReport:
    model_path: str
    config_path: str
    checkpoint_path: str
    bare_load: CheckpointLoadReport
    lightning_load: CheckpointLoadReport
    batch_size: int | None = None
    max_logits_abs_diff: float | None = None
    max_probabilities_abs_diff: float | None = None
    atol: float = 1e-6

    @property
    def ok(self) -> bool:
        if not self.bare_load.ok or not self.lightning_load.ok:
            return False
        if self.max_logits_abs_diff is not None and self.max_logits_abs_diff > self.atol:
            return False
        if self.max_probabilities_abs_diff is not None and self.max_probabilities_abs_diff > self.atol:
            return False
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "bare_loaded_keys": self.bare_load.loaded_key_count,
            "bare_source_keys": self.bare_load.source_key_count,
            "lightning_loaded_keys": self.lightning_load.loaded_key_count,
            "lightning_source_keys": self.lightning_load.source_key_count,
            "bare_missing_keys": len(self.bare_load.missing_keys),
            "bare_unexpected_keys": len(self.bare_load.unexpected_keys),
            "bare_shape_mismatches": len(self.bare_load.shape_mismatches),
            "lightning_missing_keys": len(self.lightning_load.missing_keys),
            "lightning_unexpected_keys": len(self.lightning_load.unexpected_keys),
            "lightning_shape_mismatches": len(self.lightning_load.shape_mismatches),
            "batch_size": self.batch_size,
            "max_logits_abs_diff": self.max_logits_abs_diff,
            "max_probabilities_abs_diff": self.max_probabilities_abs_diff,
            "atol": self.atol,
        }


def check_classification_checkpoint_compatibility(
    settings: ClassificationCheckpointCompatibilitySettings,
    *,
    batch: Any | None = None,
) -> ClassificationCheckpointCompatibilityReport:
    config_path = settings.config_path or resolve_config_file(settings.model_path)
    checkpoint_path = resolve_model_checkpoint(
        settings.model_path,
        ckpt_file=settings.ckpt_file,
        checkpoint_path=settings.checkpoint_path,
    )
    config = _load_compat_config(config_path)
    device_manager = DeviceManager.from_value(settings.device or "cpu")
    device = device_manager.device
    adapter = CheckpointAdapter()

    bare_model = device_manager.move_module(build_classification_model(config))
    bare_report = adapter.load_into_model(
        bare_model,
        checkpoint_path,
        map_location="cpu",
        mode="strict",
    )
    bare_model.eval()

    lit_module = RXNGraphormerLitModule(
        build_classification_model(config),
        config,
        task="classification",
    )
    device_manager.move_module(lit_module)
    lightning_report = adapter.load_into_model(
        lit_module.model,
        checkpoint_path,
        map_location="cpu",
        mode="strict",
    )
    lit_module.eval()

    batch_size = None
    max_logits_abs_diff = None
    max_probabilities_abs_diff = None
    if batch is not None:
        model_input, target = classification_batch_input(batch, device)
        batch_size = int(target.shape[0])
        with torch.no_grad():
            bare_logits = bare_model.logits(model_input)
            lightning_logits = lit_module._forward_logits(model_input)
            max_logits_abs_diff = float((bare_logits - lightning_logits).abs().max().detach().cpu())
            bare_probabilities = torch.softmax(bare_logits, dim=-1)
            lightning_probabilities = torch.softmax(lightning_logits, dim=-1)
            max_probabilities_abs_diff = float(
                (bare_probabilities - lightning_probabilities).abs().max().detach().cpu()
            )

    return ClassificationCheckpointCompatibilityReport(
        model_path=os.fspath(settings.model_path),
        config_path=os.fspath(config_path),
        checkpoint_path=os.fspath(checkpoint_path),
        bare_load=bare_report,
        lightning_load=lightning_report,
        batch_size=batch_size,
        max_logits_abs_diff=max_logits_abs_diff,
        max_probabilities_abs_diff=max_probabilities_abs_diff,
        atol=settings.atol,
    )


def _load_compat_config(config_path: str | os.PathLike) -> object:
    config = load_config(config_path)
    if isinstance(config, EvalConfig):
        raise TypeError(f"Expected train config for checkpoint compatibility, got eval config: {config_path}")
    return config
