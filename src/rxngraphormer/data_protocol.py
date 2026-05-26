from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

GRAPH_FIELDS = ("x", "edge_index", "edge_attr", "mol_index")
TARGET_FIELD = "y"
SEQUENCE_FIELDS = ("tgt_token_ids", "tgt_lens")


@dataclass(frozen=True)
class ReactionDatasetSpec:
    task: str
    graph_fields: tuple[str, ...] = GRAPH_FIELDS
    target_field: str | None = TARGET_FIELD
    sequence_fields: tuple[str, ...] = SEQUENCE_FIELDS
    format_version: str = "rxngraphormer-data-v1"


@runtime_checkable
class ReactionDatasetProtocol(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> object:
        ...


def validate_graph_like(data: object, *, require_target: bool = False) -> None:
    missing = [field for field in GRAPH_FIELDS if not hasattr(data, field)]
    if require_target and not hasattr(data, TARGET_FIELD):
        missing.append(TARGET_FIELD)
    if missing:
        raise ValueError(f"Reaction graph data is missing required fields: {missing}")


def validate_graph_data(data: object, *, require_target: bool = False) -> None:
    validate_graph_like(data, require_target=require_target)


__all__ = [
    "GRAPH_FIELDS",
    "SEQUENCE_FIELDS",
    "TARGET_FIELD",
    "ReactionDatasetProtocol",
    "ReactionDatasetSpec",
    "validate_graph_data",
    "validate_graph_like",
]
