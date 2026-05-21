from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    pl = None

from ..checkpointing import CheckpointAdapter, CheckpointLoadReport
from ..dataloader import DataLoaderSettings
from ..experiment import build_fit_manifest, best_checkpoint_path, run_post_fit_regression_eval, trainer_output_dir, write_fit_manifest
from .datamodule import RXNGraphormerDataModule
from .module import RXNGraphormerLitModule
from .trainer import build_trainer


@dataclass
class LightningFitSettings:
    default_root_dir: str | None = None
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str | None = None
    max_epochs: int | None = None
    num_workers: int | None = None
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    split_manifest: str | Path | None = None
    init_ckpt: str | Path | None = None
    resume_from_checkpoint: str | Path | None = None
    deterministic: bool = False
    compile_model: bool = False
    compile_mode: str | None = None
    early_stopping_patience: int | None = None
    config_path: str | Path | None = None
    write_manifest: bool = True
    eval_after_fit: bool = False
    eval_splits: tuple[str, ...] = ("test",)
    eval_batch_size: int | None = None
    eval_scale: float = 1.0
    eval_yield_constrain: bool = False
    eval_max_batches: int | None = None
    eval_specific_val: bool = False


@dataclass
class LightningFitArtifacts:
    config: Any
    datamodule: RXNGraphormerDataModule
    lit_module: RXNGraphormerLitModule
    trainer: Any
    root_dir: str
    init_checkpoint_report: CheckpointLoadReport | None = None
    manifest: dict[str, Any] | None = None
    manifest_paths: dict[str, str] | None = None
    eval_reports: dict[str, Any] | None = None


def prepare_lightning_config(config: Any, settings: LightningFitSettings) -> Any:
    if config.task != "regression":
        raise NotImplementedError("Lightning training entry currently supports regression configs only.")

    if settings.deterministic:
        config.runtime.deterministic = True
    if settings.compile_model:
        config.runtime.compile_model = True
    if settings.compile_mode is not None:
        config.runtime.compile_mode = settings.compile_mode
    if settings.early_stopping_patience is not None:
        config.runtime.early_stopping_patience = settings.early_stopping_patience

    if bool(config.runtime.deterministic):
        if pl is None:
            raise ImportError("lightning is required for deterministic Lightning training")
        pl.seed_everything(int(config.data.seed), workers=True)
        torch.use_deterministic_algorithms(True, warn_only=True)
    return config


def build_dataloader_settings(config: Any, settings: LightningFitSettings) -> DataLoaderSettings:
    return DataLoaderSettings(
        batch_size=int(config.data.batch_size),
        num_workers=settings.num_workers if settings.num_workers is not None else int(getattr(config.data, "num_workers", 0)),
        pin_memory=settings.pin_memory or bool(getattr(config.data, "pin_memory", False)),
        persistent_workers=settings.persistent_workers or bool(getattr(config.data, "persistent_workers", False)),
        prefetch_factor=settings.prefetch_factor if settings.prefetch_factor is not None else getattr(config.data, "prefetch_factor", None),
    )


def build_datamodule(config: Any, settings: LightningFitSettings) -> RXNGraphormerDataModule:
    return RXNGraphormerDataModule(
        config,
        split_manifest=settings.split_manifest,
        dataloader=build_dataloader_settings(config, settings),
    )


def build_lit_module(
    config: Any,
    settings: LightningFitSettings,
) -> tuple[RXNGraphormerLitModule, CheckpointLoadReport | None]:
    lit_module = RXNGraphormerLitModule.from_config(config)
    if settings.init_ckpt is None:
        return lit_module, None

    report = CheckpointAdapter().load_into_model(
        lit_module.model,
        settings.init_ckpt,
        map_location="cpu",
        mode="strict",
    )
    return lit_module, report


def build_fit_artifacts(config: Any, settings: LightningFitSettings) -> LightningFitArtifacts:
    config = prepare_lightning_config(config, settings)
    datamodule = build_datamodule(config, settings)
    lit_module, init_report = build_lit_module(config, settings)
    root_dir = settings.default_root_dir or config.model.save_dir
    trainer = build_trainer(
        config,
        default_root_dir=root_dir,
        accelerator=settings.accelerator,
        devices=settings.devices,
        precision=settings.precision,
        max_epochs=settings.max_epochs,
    )
    return LightningFitArtifacts(
        config=config,
        datamodule=datamodule,
        lit_module=lit_module,
        trainer=trainer,
        root_dir=root_dir,
        init_checkpoint_report=init_report,
    )


def format_checkpoint_report(report: CheckpointLoadReport) -> str:
    return (
        "Loaded initial checkpoint: "
        f"{report.loaded_key_count}/{report.source_key_count} keys, "
        f"missing={len(report.missing_keys)}, unexpected={len(report.unexpected_keys)}, "
        f"shape_mismatch={len(report.shape_mismatches)}"
    )


def fit_config(
    config: Any,
    settings: LightningFitSettings,
    *,
    print_initial_checkpoint: bool = True,
) -> LightningFitArtifacts:
    started_at = time.time()
    artifacts = build_fit_artifacts(config, settings)
    if print_initial_checkpoint and artifacts.init_checkpoint_report is not None:
        print(format_checkpoint_report(artifacts.init_checkpoint_report))
    artifacts.trainer.fit(
        artifacts.lit_module,
        datamodule=artifacts.datamodule,
        ckpt_path=settings.resume_from_checkpoint,
    )
    ended_at = time.time()

    output_dir = trainer_output_dir(artifacts.trainer, artifacts.root_dir)
    eval_reports: dict[str, Any] = {}
    if settings.eval_after_fit:
        checkpoint_path = best_checkpoint_path(artifacts.trainer)
        if not checkpoint_path:
            raise RuntimeError("eval_after_fit=True requires a checkpoint callback with a best_model_path")
        eval_reports["regression"] = run_post_fit_regression_eval(
            artifacts=artifacts,
            config_path=settings.config_path,
            checkpoint_path=checkpoint_path,
            splits=settings.eval_splits,
            batch_size=settings.eval_batch_size,
            scale=settings.eval_scale,
            yield_constrain=settings.eval_yield_constrain,
            max_batches=settings.eval_max_batches,
            specific_val=settings.eval_specific_val,
            output_dir=output_dir,
        )
    artifacts.eval_reports = eval_reports

    if settings.write_manifest:
        artifacts.manifest = build_fit_manifest(
            artifacts=artifacts,
            settings=settings,
            started_at=started_at,
            ended_at=ended_at,
            config_path=settings.config_path,
            eval_reports=eval_reports,
        )
        artifacts.manifest_paths = write_fit_manifest(
            artifacts.manifest,
            output_dir=output_dir,
            root_dir=artifacts.root_dir,
        )
    return artifacts
