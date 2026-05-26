"""Compatibility import boundary for legacy config/checkpoint helpers."""

from importlib import import_module

_CHECKPOINT_EXPORTS = {
    "CheckpointAdapter",
    "CheckpointLoadReport",
    "load_checkpoint",
}
_STATE_DICT_EXPORTS = {
    "update_state_dict_keys",
}
_CLASSIFICATION_EXPORTS = {
    "ClassificationCheckpointCompatibilityReport",
    "ClassificationCheckpointCompatibilitySettings",
    "check_classification_checkpoint_compatibility",
}
_REGRESSION_EXPORTS = {
    "RegressionPretrainCompatibilityReport",
    "RegressionPretrainCompatibilitySettings",
    "check_regression_pretrain_compatibility",
}

__all__ = [
    "ClassificationCheckpointCompatibilityReport",
    "ClassificationCheckpointCompatibilitySettings",
    "CheckpointAdapter",
    "CheckpointLoadReport",
    "RegressionPretrainCompatibilityReport",
    "RegressionPretrainCompatibilitySettings",
    "check_classification_checkpoint_compatibility",
    "check_regression_pretrain_compatibility",
    "load_checkpoint",
    "load_legacy_torch",
    "update_state_dict_keys",
    "torch_compat",
]


def __getattr__(name: str) -> object:
    if name == "torch_compat":
        return import_module(f"{__name__}.torch_compat")
    if name == "load_legacy_torch":
        return import_module(f"{__name__}.torch_compat").load_legacy_torch
    if name in _CHECKPOINT_EXPORTS:
        return getattr(import_module(f"{__name__}.checkpointing"), name)
    if name in _STATE_DICT_EXPORTS:
        return getattr(import_module(f"{__name__}.state_dict"), name)
    if name in _CLASSIFICATION_EXPORTS:
        return getattr(import_module(f"{__name__}.classification"), name)
    if name in _REGRESSION_EXPORTS:
        return getattr(import_module(f"{__name__}.regression"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
