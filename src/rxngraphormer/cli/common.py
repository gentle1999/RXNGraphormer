from __future__ import annotations

import argparse
from typing import cast

from rxngraphormer.config import Config, EvalConfig
from rxngraphormer.training.optimization import LRScalingPolicy, SchedulerStepScalePolicy


def as_train_config(config: object) -> Config:
    if isinstance(config, EvalConfig):
        raise TypeError("Expected train config with model/data sections, got eval config")
    return cast(Config, config)


def as_eval_config(config: object) -> EvalConfig:
    if isinstance(config, EvalConfig):
        return config
    if not isinstance(config, Config):
        return cast(EvalConfig, config)
    return EvalConfig(
        task=config.task,
        model=config.model,
        data=config.data,
        training=config.training,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        runtime=config.runtime,
        infer=config.infer,
    )


def add_lightning_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--default_root_dir", type=str, default=None)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--train_drop_last", action="store_true", help="Drop incomplete batches only for the training dataloader.")
    parser.add_argument("--split_manifest", type=str, default=None)
    parser.add_argument(
        "--init_ckpt",
        type=str,
        default=None,
        help="Legacy/canonical checkpoint to load into LightningModule.model before training.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Lightning checkpoint used for Trainer resume.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--float32_matmul_precision", choices=["highest", "high", "medium"], default=None)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--compile_mode", choices=["default", "reduce-overhead", "max-autotune"], default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=None)
    parser.add_argument("--val_check_interval", type=int, default=None)
    parser.add_argument("--num_sanity_val_steps", type=int, default=None)
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--disable_model_summary", action="store_true")
    parser.add_argument("--no_manifest", action="store_true", help="Disable writing experiment manifest.json files.")
    parser.add_argument("--eval_after_fit", action="store_true", help="Evaluate the best checkpoint after training.")
    parser.add_argument("--eval_splits", nargs="+", choices=["train", "valid", "test"], default=["test"])
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_scale", type=float, default=1.0)
    parser.add_argument("--eval_yield_constrain", action="store_true")
    parser.add_argument("--eval_max_batches", type=int, default=None)
    parser.add_argument("--eval_specific_val", action="store_true")
    parser.add_argument("--lr_scaling_policy", choices=["none", "sqrt", "linear"], default=None)
    parser.add_argument("--base_batch_size", type=int, default=None)
    parser.add_argument("--base_learning_rate", type=float, default=None)
    parser.add_argument("--scale_warmup_steps", action="store_true")
    parser.add_argument("--base_warmup_step", type=int, default=None)
    parser.add_argument(
        "--scheduler_step_scale_policy",
        choices=["none", "sample"],
        default=None,
        help="For NoamLR, use sample to align scheduler progress to the base batch size.",
    )


def lightning_settings_from_args(args: argparse.Namespace):
    from rxngraphormer.lightning import LightningFitSettings

    return LightningFitSettings(
        default_root_dir=args.default_root_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accum_steps=args.accum_steps,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        train_drop_last=True if args.train_drop_last else None,
        split_manifest=getattr(args, "split_manifest", None),
        init_ckpt=args.init_ckpt,
        resume_from_checkpoint=args.resume_from_checkpoint,
        deterministic=args.deterministic,
        benchmark=args.benchmark if args.benchmark else None,
        float32_matmul_precision=args.float32_matmul_precision,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        early_stopping_patience=args.early_stopping_patience,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        num_sanity_val_steps=args.num_sanity_val_steps,
        enable_progress_bar=False if args.disable_progress_bar else None,
        enable_model_summary=False if args.disable_model_summary else None,
        config_path=args.config_path,
        write_manifest=not args.no_manifest,
        eval_after_fit=args.eval_after_fit,
        eval_splits=tuple(args.eval_splits),
        eval_batch_size=args.eval_batch_size,
        eval_scale=args.eval_scale,
        eval_yield_constrain=args.eval_yield_constrain,
        eval_max_batches=args.eval_max_batches,
        eval_specific_val=args.eval_specific_val,
        lr_scaling_policy=cast(LRScalingPolicy | None, args.lr_scaling_policy),
        base_batch_size=args.base_batch_size,
        base_learning_rate=args.base_learning_rate,
        scale_warmup_steps=True if args.scale_warmup_steps else None,
        base_warmup_step=args.base_warmup_step,
        scheduler_step_scale_policy=cast(SchedulerStepScalePolicy | None, args.scheduler_step_scale_policy),
    )
