"""Training-step and epoch budget helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Protocol, cast

from ..data.files import _matched_raw_files


class _StepDataConfig(Protocol):
    data_path: str
    rct_data_file: str
    train_rct_data_file: str
    rct_name_regrex: str
    data_trunck: int
    file_num_trunck: int
    train_ratio: float
    batch_size: int
    train_drop_last: bool


class _StepTrainingConfig(Protocol):
    epoch: int
    max_steps: int | None
    accum: int


class _StepBudgetConfig(Protocol):
    data: _StepDataConfig
    training: _StepTrainingConfig


@dataclass(frozen=True)
class MaxStepEpochEstimate:
    train_size: int
    batches_per_epoch: int
    steps_per_epoch: int
    max_epochs: int


def count_nonempty_rows(path: str | PathLike[str]) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_data_path(data_path: str, *, root: str | PathLike[str] | None = None) -> Path:
    path = Path(data_path)
    if path.is_absolute() or root is None:
        return path
    return Path(root) / path


def estimate_train_size_from_config(config: object, *, root: str | PathLike[str] | None = None) -> int:
    data = cast(_StepBudgetConfig, config).data
    data_path = resolve_data_path(str(getattr(data, "data_path", ".")), root=root)
    trunck = max(0, int(getattr(data, "data_trunck", 0) or 0))

    train_rct_file = str(getattr(data, "train_rct_data_file", "") or "")
    if train_rct_file:
        return _limited_row_count(data_path / train_rct_file, trunck=trunck)

    rct_file = str(getattr(data, "rct_data_file", "") or "")
    if rct_file:
        total_size = _limited_row_count(data_path / rct_file, trunck=trunck)
        return _ratio_train_size(total_size, float(getattr(data, "train_ratio", 0.8)))

    rct_name_regrex = str(getattr(data, "rct_name_regrex", "") or "")
    if rct_name_regrex:
        raw_files = _matched_raw_files(data_path, rct_name_regrex)
        file_num_trunck = max(0, int(getattr(data, "file_num_trunck", 0) or 0))
        if file_num_trunck:
            raw_files = raw_files[:file_num_trunck]
        total_size = sum(_limited_row_count(path, trunck=trunck) for path in raw_files)
        return _ratio_train_size(total_size, float(getattr(data, "train_ratio", 0.8)))

    raise ValueError("Cannot estimate train size without train_rct_data_file, rct_data_file, or rct_name_regrex.")


def estimate_max_step_epochs(
    *,
    train_size: int,
    batch_size: int,
    accum_steps: int,
    max_steps: int,
    drop_last: bool = False,
) -> MaxStepEpochEstimate:
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    resolved_batch_size = max(1, batch_size)
    resolved_accum_steps = max(1, accum_steps)
    if drop_last:
        batches_per_epoch = train_size // resolved_batch_size
        if batches_per_epoch <= 0:
            raise ValueError("drop_last=True would produce zero train batches")
    else:
        batches_per_epoch = max(1, math.ceil(train_size / resolved_batch_size))
    steps_per_epoch = max(1, math.ceil(batches_per_epoch / resolved_accum_steps))
    max_epochs = math.ceil(max_steps / steps_per_epoch)
    return MaxStepEpochEstimate(
        train_size=train_size,
        batches_per_epoch=batches_per_epoch,
        steps_per_epoch=steps_per_epoch,
        max_epochs=max_epochs,
    )


def estimate_max_step_epochs_from_config(
    config: object,
    *,
    max_steps: int,
    root: str | PathLike[str] | None = None,
) -> MaxStepEpochEstimate:
    resolved = cast(_StepBudgetConfig, config)
    train_size = estimate_train_size_from_config(config, root=root)
    return estimate_max_step_epochs(
        train_size=train_size,
        batch_size=int(getattr(resolved.data, "batch_size", 1)),
        accum_steps=int(getattr(resolved.training, "accum", 1)),
        max_steps=max_steps,
        drop_last=bool(getattr(resolved.data, "train_drop_last", False)),
    )


def apply_max_step_epoch_estimate(
    config: object,
    *,
    max_steps: int,
    root: str | PathLike[str] | None = None,
) -> MaxStepEpochEstimate:
    estimate = estimate_max_step_epochs_from_config(config, max_steps=max_steps, root=root)
    training = cast(_StepBudgetConfig, config).training
    training.max_steps = max_steps
    training.epoch = estimate.max_epochs
    return estimate


def _limited_row_count(path: str | PathLike[str], *, trunck: int) -> int:
    count = count_nonempty_rows(path)
    if trunck > 0:
        count = min(count, trunck)
    if count <= 0:
        raise ValueError(f"No training rows found in {path}")
    return count


def _ratio_train_size(total_size: int, train_ratio: float) -> int:
    if total_size <= 0:
        raise ValueError("total_size must be positive")
    return max(1, min(total_size, int(total_size * train_ratio)))


__all__ = [
    "MaxStepEpochEstimate",
    "apply_max_step_epoch_estimate",
    "count_nonempty_rows",
    "estimate_max_step_epochs",
    "estimate_max_step_epochs_from_config",
    "estimate_train_size_from_config",
    "resolve_data_path",
]
