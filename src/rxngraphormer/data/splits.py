"""Dataset split helpers."""

from __future__ import annotations

import numpy as np
import torch


def get_idx_split(data_size: int, train_size: int, valid_size: int, seed: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(data_size).tolist()
    if abs(train_size + valid_size - data_size) < 2:
        train_idx, val_idx = (
            torch.tensor(ids[:train_size]),
            torch.tensor(ids[train_size:]),
        )
        test_idx = val_idx
    else:
        train_idx, val_idx, test_idx = (
            torch.tensor(ids[:train_size]),
            torch.tensor(ids[train_size : train_size + valid_size]),
            torch.tensor(ids[train_size + valid_size :]),
        )
    split_dict = {"train": train_idx, "valid": val_idx, "test": test_idx}
    return split_dict


__all__ = ["get_idx_split"]
