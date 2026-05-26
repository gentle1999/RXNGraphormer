from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch

from ..compatibility.checkpointing import CheckpointAdapter, CheckpointLoadReport
from ..data.loader import DataLoaderSettings
from ..experiment import (
    best_checkpoint_path,
    build_fit_manifest,
    jsonable,
    run_post_fit_classification_eval,
    run_post_fit_regression_eval,
    trainer_output_dir,
    write_fit_manifest,
    write_json,
)
from ..runtime import configure_torch_runtime
from ..serialization import DEFAULT_CHECKPOINT_FILE, export_model_state_dict
from ..training.optimization import LRScalingPolicy, SchedulerStepScalePolicy
from ..training.step_budget import apply_max_step_epoch_estimate
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
    max_steps: int | None = None
    accum_steps: int | None = None
    num_workers: int | None = None
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    train_drop_last: bool | None = None
    split_manifest: str | Path | None = None
    init_ckpt: str | Path | None = None
    resume_from_checkpoint: str | Path | None = None
    deterministic: bool = False
    benchmark: bool | None = None
    float32_matmul_precision: str | None = None
    compile_model: bool = False
    compile_mode: str | None = None
    early_stopping_patience: int | None = None
    check_val_every_n_epoch: int | None = None
    val_check_interval: int | None = None
    num_sanity_val_steps: int | None = None
    enable_progress_bar: bool | None = None
    enable_model_summary: bool | None = None
    config_path: str | Path | None = None
    write_manifest: bool = True
    eval_after_fit: bool = False
    eval_splits: tuple[str, ...] = ("test",)
    eval_batch_size: int | None = None
    eval_scale: float = 1.0
    eval_yield_constrain: bool = False
    eval_max_batches: int | None = None
    eval_specific_val: bool = False
    lr_scaling_policy: LRScalingPolicy | None = None
    base_batch_size: int | None = None
    base_learning_rate: float | None = None
    scale_warmup_steps: bool | None = None
    base_warmup_step: int | None = None
    scheduler_step_scale_policy: SchedulerStepScalePolicy | None = None


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
    exported_checkpoint_path: str | None = None
    exported_config_path: str | None = None


def prepare_lightning_config(config: Any, settings: LightningFitSettings) -> Any:
    if config.task not in {"regression", "classification"}:
        raise NotImplementedError("Lightning training entry supports regression and classification configs.")

    if settings.deterministic:
        config.runtime.deterministic = True
    if settings.benchmark is not None:
        config.runtime.benchmark = settings.benchmark
    if settings.float32_matmul_precision is not None:
        config.runtime.float32_matmul_precision = settings.float32_matmul_precision
    if settings.compile_model:
        config.runtime.compile_model = True
    if settings.compile_mode is not None:
        config.runtime.compile_mode = settings.compile_mode
    if settings.early_stopping_patience is not None:
        config.runtime.early_stopping_patience = settings.early_stopping_patience
    if settings.accum_steps is not None:
        config.training.accum = max(1, int(settings.accum_steps))
    if settings.max_epochs is not None:
        config.training.epoch = settings.max_epochs
    if settings.max_steps is not None:
        config.training.max_steps = settings.max_steps
        try:
            apply_max_step_epoch_estimate(config, max_steps=settings.max_steps)
        except ValueError as exc:
            if "Cannot estimate train size" not in str(exc):
                raise
    if settings.check_val_every_n_epoch is not None:
        config.runtime.check_val_every_n_epoch = settings.check_val_every_n_epoch
    if settings.val_check_interval is not None:
        config.runtime.val_check_interval = settings.val_check_interval
    if settings.num_sanity_val_steps is not None:
        config.runtime.num_sanity_val_steps = settings.num_sanity_val_steps
    if settings.enable_progress_bar is not None:
        config.runtime.enable_progress_bar = settings.enable_progress_bar
    if settings.enable_model_summary is not None:
        config.runtime.enable_model_summary = settings.enable_model_summary
    if settings.lr_scaling_policy is not None:
        config.optimizer.lr_scaling_policy = settings.lr_scaling_policy
    if settings.base_batch_size is not None:
        config.optimizer.base_batch_size = settings.base_batch_size
    if settings.base_learning_rate is not None:
        config.optimizer.base_learning_rate = settings.base_learning_rate
    if settings.scale_warmup_steps is not None:
        config.scheduler.scale_warmup_steps = settings.scale_warmup_steps
    if settings.base_warmup_step is not None:
        config.scheduler.base_warmup_step = settings.base_warmup_step
    if settings.scheduler_step_scale_policy is not None:
        config.scheduler.step_scale_policy = settings.scheduler_step_scale_policy

    if bool(config.runtime.deterministic):
        pl.seed_everything(int(config.data.seed), workers=True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        config.runtime.benchmark = False
    configure_torch_runtime(config.runtime)
    return config


def build_dataloader_settings(config: Any, settings: LightningFitSettings) -> DataLoaderSettings:
    return DataLoaderSettings(
        batch_size=int(config.data.batch_size),
        num_workers=settings.num_workers if settings.num_workers is not None else int(getattr(config.data, "num_workers", 0)),
        pin_memory=settings.pin_memory or bool(getattr(config.data, "pin_memory", False)),
        persistent_workers=settings.persistent_workers or bool(getattr(config.data, "persistent_workers", False)),
        prefetch_factor=settings.prefetch_factor if settings.prefetch_factor is not None else getattr(config.data, "prefetch_factor", None),
        train_drop_last=(
            settings.train_drop_last
            if settings.train_drop_last is not None
            else bool(getattr(config.data, "train_drop_last", False))
        ),
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
        max_steps=settings.max_steps,
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


def export_canonical_lightning_checkpoint(
    artifacts: LightningFitArtifacts,
    *,
    output_dir: str | Path,
    trainer_checkpoint_path: str | Path | None = None,
) -> str:
    path = Path(output_dir) / "model" / DEFAULT_CHECKPOINT_FILE
    model = artifacts.lit_module.model
    if trainer_checkpoint_path is not None and Path(trainer_checkpoint_path).exists():
        CheckpointAdapter().load_into_model(
            model,
            trainer_checkpoint_path,
            map_location="cpu",
            mode="strict",
        )
    export_model_state_dict(model, path)
    return str(path)


def write_lightning_fit_config(config: Any, *, output_dir: str | Path) -> str:
    path = Path(output_dir) / "parameters.json"
    payload = jsonable(config)
    if not isinstance(payload, dict):
        payload = {"config": payload}
    write_json(path, payload)
    return str(path)


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
    trainer_checkpoint_path = best_checkpoint_path(artifacts.trainer)
    exported_checkpoint_path = export_canonical_lightning_checkpoint(
        artifacts,
        output_dir=output_dir,
        trainer_checkpoint_path=trainer_checkpoint_path,
    )
    artifacts.exported_checkpoint_path = exported_checkpoint_path
    artifacts.exported_config_path = write_lightning_fit_config(config, output_dir=output_dir)
    eval_reports: dict[str, Any] = {}
    if settings.eval_after_fit:
        if not trainer_checkpoint_path:
            raise RuntimeError("eval_after_fit=True requires a checkpoint callback with a best_model_path")
        checkpoint_path = exported_checkpoint_path
        if config.task == "classification":
            eval_reports["classification"] = run_post_fit_classification_eval(
                artifacts=artifacts,
                config_path=settings.config_path,
                checkpoint_path=checkpoint_path,
                splits=settings.eval_splits,
                batch_size=settings.eval_batch_size,
                max_batches=settings.eval_max_batches,
                specific_val=settings.eval_specific_val,
                output_dir=output_dir,
            )
        else:
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
