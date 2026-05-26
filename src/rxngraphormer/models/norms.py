from __future__ import annotations

from typing import cast

from torch import nn
from torch_geometric.nn import GraphNorm

from ..config import NormType

VALID_NORM_TYPES = {"batchnorm", "layernorm", "graphnorm", "none"}


def normalize_norm_type(norm_type: NormType | str | None, *, default: NormType = "batchnorm") -> NormType:
    if norm_type is None:
        return default
    normalized = str(norm_type).strip().lower()
    if normalized == "batch":
        normalized = "batchnorm"
    elif normalized == "layer":
        normalized = "layernorm"
    elif normalized == "graph":
        normalized = "graphnorm"
    if normalized not in VALID_NORM_TYPES:
        raise ValueError("norm_type must be one of 'batchnorm', 'layernorm', 'graphnorm', or 'none'")
    return cast(NormType, normalized)


def legacy_batch_norm_type(batch_norm: bool) -> NormType:
    return "batchnorm" if batch_norm else "none"


def make_graph_norm(norm_type: NormType | str | None, hidden_size: int) -> nn.Module:
    normalized = normalize_norm_type(norm_type)
    if normalized == "batchnorm":
        return nn.BatchNorm1d(hidden_size)
    if normalized == "layernorm":
        return nn.LayerNorm(hidden_size)
    if normalized == "graphnorm":
        return GraphNorm(hidden_size)
    return nn.Identity()


def make_head_norm(norm_type: NormType | str | None, hidden_size: int) -> nn.Module:
    normalized = normalize_norm_type(norm_type, default="none")
    if normalized == "graphnorm":
        raise ValueError("graphnorm is only supported in graph encoders")
    if normalized == "batchnorm":
        return nn.BatchNorm1d(hidden_size)
    if normalized == "layernorm":
        return nn.LayerNorm(hidden_size)
    return nn.Identity()


__all__ = [
    "VALID_NORM_TYPES",
    "legacy_batch_norm_type",
    "make_graph_norm",
    "make_head_norm",
    "normalize_norm_type",
]
