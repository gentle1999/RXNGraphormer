from __future__ import annotations

from typing import Any

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:  # pragma: no cover
    pl = None
    EarlyStopping = LearningRateMonitor = ModelCheckpoint = TensorBoardLogger = None

from .callbacks import EpochRuntimeMonitor


def precision_from_config(config: Any, precision: str | None = None) -> str:
    if precision is not None:
        return precision
    enable_amp = bool(getattr(config.runtime, "enable_amp", False))
    if not enable_amp:
        return "32-true"
    return "bf16-mixed" if getattr(config.runtime, "amp_dtype", "fp16") == "bf16" else "16-mixed"


def build_callbacks(config: Any):
    if ModelCheckpoint is None or LearningRateMonitor is None:
        raise ImportError("lightning is required to build callbacks")
    callbacks = [
        ModelCheckpoint(
            monitor="val_mae",
            mode="min",
            save_top_k=1,
            filename="epoch={epoch:03d}-val_mae={val_mae:.6f}",
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
                monitor="val_mae",
                mode="min",
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
):
    if pl is None or TensorBoardLogger is None:
        raise ImportError("lightning is required to build a Trainer")
    logger = TensorBoardLogger(save_dir=default_root_dir, name=getattr(config.others, "tag", "lightning"))
    return pl.Trainer(
        default_root_dir=default_root_dir,
        accelerator=accelerator,
        devices=devices,
        precision=precision_from_config(config, precision),
        max_epochs=max_epochs or config.training.epoch,
        accumulate_grad_batches=max(1, int(config.training.accum)),
        gradient_clip_val=float(config.training.clip_norm),
        deterministic=bool(config.runtime.deterministic),
        logger=logger,
        callbacks=build_callbacks(config),
        log_every_n_steps=max(1, int(getattr(config.training, "log_iter_step", 50))),
    )
