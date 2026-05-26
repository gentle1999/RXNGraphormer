from __future__ import annotations

from typing import Any, Literal, cast

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from .callbacks import EpochRuntimeMonitor

LightningPrecision = Literal[
    "64-true",
    "32-true",
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "transformer-engine",
    "transformer-engine-float16",
]


def precision_from_config(config: Any, precision: LightningPrecision | None = None) -> LightningPrecision:
    if precision is not None:
        return precision
    enable_amp = bool(getattr(config.runtime, "enable_amp", False))
    if not enable_amp:
        return "32-true"
    return "bf16-mixed" if getattr(config.runtime, "amp_dtype", "fp16") == "bf16" else "16-mixed"


def build_callbacks(config: Any):
    if getattr(config, "task", "regression") == "classification":
        monitor = "val_acc"
        mode = "max"
        filename = "epoch={epoch:03d}-val_acc={val_acc:.6f}"
    else:
        monitor = "val_mae"
        mode = "min"
        filename = "epoch={epoch:03d}-val_mae={val_mae:.6f}"
    callbacks = [
        ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            filename=filename,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        EpochRuntimeMonitor(
            log_epoch_time=bool(getattr(config.runtime, "log_epoch_time", True)),
            log_gpu_memory=bool(getattr(config.runtime, "log_gpu_memory", True)),
        ),
    ]
    early_stopping_patience = int(getattr(config.runtime, "early_stopping_patience", 0))
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=early_stopping_patience,
            )
        )
    return callbacks


def build_trainer(
    config: Any,
    *,
    default_root_dir: str,
    accelerator: str = "auto",
    devices: str = "auto",
    precision: str | None = None,
    max_epochs: int | None = None,
    max_steps: int | None = None,
):
    logger = TensorBoardLogger(save_dir=default_root_dir, name=getattr(config.others, "tag", "lightning"))
    resolved_max_steps = max_steps if max_steps is not None else getattr(config.training, "max_steps", None)
    resolved_max_epochs = max_epochs or config.training.epoch
    if resolved_max_steps is not None:
        resolved_max_epochs = -1
    val_check_interval = getattr(config.runtime, "val_check_interval", None)
    check_val_every_n_epoch: int | None = max(1, int(getattr(config.runtime, "check_val_every_n_epoch", 1)))
    if resolved_max_steps is not None and val_check_interval is not None:
        check_val_every_n_epoch = None
    return pl.Trainer(
        default_root_dir=default_root_dir,
        accelerator=accelerator,
        devices=devices,
        precision=precision_from_config(config, precision=cast_lightning_precision(precision)),
        max_epochs=resolved_max_epochs,
        max_steps=-1 if resolved_max_steps is None else max(1, int(resolved_max_steps)),
        accumulate_grad_batches=max(1, int(config.training.accum)),
        gradient_clip_val=float(config.training.clip_norm),
        deterministic=bool(config.runtime.deterministic),
        benchmark=bool(getattr(config.runtime, "benchmark", False)),
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=max(0, int(getattr(config.runtime, "num_sanity_val_steps", 2))),
        enable_progress_bar=bool(getattr(config.runtime, "enable_progress_bar", True)),
        enable_model_summary=bool(getattr(config.runtime, "enable_model_summary", True)),
        logger=logger,
        callbacks=build_callbacks(config),
        log_every_n_steps=max(1, int(getattr(config.training, "log_iter_step", 50))),
    )


def cast_lightning_precision(precision: str | None) -> LightningPrecision | None:
    if precision is None:
        return None
    allowed: set[str] = {
        "64-true",
        "32-true",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "transformer-engine",
        "transformer-engine-float16",
    }
    if precision not in allowed:
        raise ValueError(f"Unsupported Lightning precision: {precision}")
    return cast(LightningPrecision, precision)
