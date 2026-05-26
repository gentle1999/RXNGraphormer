"""PyG batch collation helpers."""

from __future__ import annotations

from collections.abc import Sequence

from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from .graph_data import as_reaction_graph_data


def single_collate_fn(data_list: Sequence[BaseData]) -> Batch:
    batch = Batch.from_data_list([as_reaction_graph_data(data) for data in data_list])
    return batch


def pair_collate_fn(data_list: Sequence[tuple[BaseData, BaseData]]) -> tuple[Batch, Batch]:
    batchA = Batch.from_data_list([as_reaction_graph_data(data[0]) for data in data_list])
    batchB = Batch.from_data_list([as_reaction_graph_data(data[1]) for data in data_list])
    return batchA, batchB


def triple_collate_fn(data_list: Sequence[tuple[BaseData, BaseData, BaseData]]) -> tuple[Batch, Batch, Batch]:
    batchA = Batch.from_data_list([as_reaction_graph_data(data[0]) for data in data_list])
    batchB = Batch.from_data_list([as_reaction_graph_data(data[1]) for data in data_list])
    batchC = Batch.from_data_list([as_reaction_graph_data(data[2]) for data in data_list])
    return batchA, batchB, batchC


__all__ = ["pair_collate_fn", "single_collate_fn", "triple_collate_fn"]
