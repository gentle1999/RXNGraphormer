from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import torch

MapLocation: TypeAlias = str | torch.device | dict[str, str] | None


def load_legacy_torch(path: str | Path, *, map_location: MapLocation = "cpu") -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


__all__ = ["load_legacy_torch"]
