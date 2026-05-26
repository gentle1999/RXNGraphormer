"""Training-framework import boundary for RXNGraphormer."""

__all__ = [
    "DataLoaderSettings",
    "EpochRuntimeMonitor",
    "LightningFitArtifacts",
    "LightningFitSettings",
    "get_lr",
    "grad_norm",
    "NoamLR",
    "param_count",
    "param_norm",
    "MaxStepEpochEstimate",
    "set_seed",
    "RXNGraphormerDataModule",
    "RXNGraphormerLitModule",
    "setup_logger",
    "SPLITClassifierTrainer",
    "SPLITRegressorTrainer",
    "SequenceTrainer",
    "WarmupCosineAnnealingLR",
    "build_datamodule",
    "build_dataloader_settings",
    "build_fit_artifacts",
    "build_lit_module",
    "build_trainer",
    "dataloader_kwargs",
    "dataloader_settings_from_config",
    "fit_config",
    "get_linear_scheduler_with_warmup",
    "apply_max_step_epoch_estimate",
    "estimate_max_step_epochs",
    "estimate_max_step_epochs_from_config",
    "estimate_train_size_from_config",
    "prepare_lightning_config",
    "precision_from_config",
]


def __getattr__(name: str) -> object:
    if name in {"get_lr", "grad_norm", "param_count", "param_norm", "set_seed", "setup_logger"}:
        from . import diagnostics

        return getattr(diagnostics, name)
    if name in {
        "DataLoaderSettings",
        "dataloader_kwargs",
        "dataloader_settings_from_config",
    }:
        from ..data import loader

        return getattr(loader, name)
    if name in {"NoamLR", "WarmupCosineAnnealingLR", "get_linear_scheduler_with_warmup"}:
        from . import schedulers

        return getattr(schedulers, name)
    if name in {
        "MaxStepEpochEstimate",
        "apply_max_step_epoch_estimate",
        "estimate_max_step_epochs",
        "estimate_max_step_epochs_from_config",
        "estimate_train_size_from_config",
    }:
        from . import step_budget

        return getattr(step_budget, name)
    if name in {"SPLITClassifierTrainer", "SPLITRegressorTrainer", "SequenceTrainer"}:
        from . import legacy

        return getattr(legacy, name)
    if name in {
        "EpochRuntimeMonitor",
        "LightningFitArtifacts",
        "LightningFitSettings",
        "RXNGraphormerDataModule",
        "RXNGraphormerLitModule",
        "build_datamodule",
        "build_dataloader_settings",
        "build_fit_artifacts",
        "build_lit_module",
        "build_trainer",
        "fit_config",
        "prepare_lightning_config",
        "precision_from_config",
    }:
        from .. import lightning

        return getattr(lightning, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
