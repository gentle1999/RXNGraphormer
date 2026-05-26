"""File and processed-block helpers for legacy PyG datasets."""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from os import PathLike
from typing import TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Data
from torch_geometric.data.separate import separate

DataT = TypeVar("DataT", bound=Data)
SliceDict: TypeAlias = Mapping[str, torch.Tensor]
ExtFeatureInput: TypeAlias = torch.Tensor | npt.NDArray[np.float64] | npt.NDArray[np.int_] | list[float] | list[int]


def _raw_file_sort_key(path: str | PathLike[str]) -> tuple[int, int | str, str | None]:
    stem = os.path.splitext(os.path.basename(path))[0]
    suffix = stem.rsplit("_", 1)[-1]
    return (0, int(suffix), stem) if suffix.isdigit() else (1, stem, None)


def _matched_raw_files(root: str | PathLike[str], name_regrex: str) -> list[str]:
    pattern = os.path.join(root, name_regrex)
    files = [path for path in glob.glob(pattern) if os.path.isfile(path)]
    if not files:
        raise FileNotFoundError(f"No raw data files matched pattern: {pattern}")
    return sorted(files, key=_raw_file_sort_key)


def _collated_data_count(slices: Mapping[str, torch.Tensor] | None) -> int:
    if slices is None:
        return 1
    first_slices = next(iter(slices.values()))
    return len(first_slices) - 1


def _separate_collated_data(data: DataT, slices: SliceDict | None, idx: int) -> DataT:
    if slices is None:
        if idx != 0:
            raise IndexError(
                f"Single-item processed block only supports idx=0, got {idx}"
            )
        return cast(DataT, data.clone()) if hasattr(data, "clone") else data
    return cast(DataT, separate(
        cls=data.__class__, batch=data, idx=idx, slice_dict=slices, decrement=False
    ))


def _ext_feat_tensor(ext_feat_desc: ExtFeatureInput) -> torch.Tensor:
    return torch.as_tensor(ext_feat_desc, dtype=torch.float)


__all__ = [
    "_collated_data_count",
    "_ext_feat_tensor",
    "_matched_raw_files",
    "_raw_file_sort_key",
    "_separate_collated_data",
]
