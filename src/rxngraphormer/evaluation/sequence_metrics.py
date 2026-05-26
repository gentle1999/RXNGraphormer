from __future__ import annotations

import torch


def sequence_exact_match(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int | None = None,
) -> torch.Tensor:
    result = predictions == targets
    if pad_idx is not None:
        result = result | ((predictions == pad_idx) & (targets == pad_idx))
    return result.all(dim=1).float()


__all__ = ["sequence_exact_match"]
