"""PyG batch collation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from .graph_data import ReactionGraphData, as_reaction_graph_data


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


def fast_batch_from_in_memory(dataset, indices: Sequence[int]) -> Batch:
    """Build a PyG batch directly from an InMemoryDataset storage block."""
    if not indices:
        raise ValueError("fast_batch_from_in_memory requires at least one index")
    if hasattr(dataset, "indices"):
        dataset_indices = dataset.indices()
        indices = [int(dataset_indices[idx]) for idx in indices]
    data = dataset._data
    slices = dataset.slices
    batch = Batch(_base_cls=ReactionGraphData)
    node_counts: list[int] = []
    edge_counts: list[int] = []

    def cat_attr(key: str):
        values = []
        cat_dim = data.__cat_dim__(key, data[key])
        for idx in indices:
            start = int(slices[key][idx])
            end = int(slices[key][idx + 1])
            if cat_dim == 0:
                values.append(data[key][start:end])
            elif cat_dim == -1:
                values.append(data[key][..., start:end])
            else:
                raise ValueError(f"Unsupported fast batch cat_dim for {key}: {cat_dim}")
        return torch.cat(values, dim=cat_dim)

    if "x" in slices:
        batch.x = cat_attr("x")
        node_counts = [int(slices["x"][idx + 1] - slices["x"][idx]) for idx in indices]
        batch.num_nodes = int(sum(node_counts))
    if "atom_mass" in slices:
        batch.atom_mass = cat_attr("atom_mass")
    if "mol_index" in slices:
        batch.mol_index = cat_attr("mol_index")
    if "y" in slices:
        batch.y = cat_attr("y")
    if "ext_feat" in slices:
        batch.ext_feat = cat_attr("ext_feat")
    if "edge_attr" in slices:
        batch.edge_attr = cat_attr("edge_attr")
        edge_counts = [int(slices["edge_attr"][idx + 1] - slices["edge_attr"][idx]) for idx in indices]
    if "edge_index" in slices:
        edge_parts = []
        node_offset = 0
        for item_pos, idx in enumerate(indices):
            start = int(slices["edge_index"][idx])
            end = int(slices["edge_index"][idx + 1])
            edge_parts.append(data.edge_index[:, start:end] + node_offset)
            node_offset += node_counts[item_pos]
        batch.edge_index = torch.cat(edge_parts, dim=1)
        if not edge_counts:
            edge_counts = [part.shape[1] for part in edge_parts]

    device = batch.x.device if hasattr(batch, "x") else None
    if node_counts:
        batch.batch = torch.repeat_interleave(
            torch.arange(len(indices), device=device),
            torch.as_tensor(node_counts, device=device),
        )
        batch.ptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                torch.as_tensor(node_counts, dtype=torch.long, device=device).cumsum(0),
            ]
        )
    batch._num_graphs = len(indices)
    batch._slice_dict = {
        key: torch.as_tensor([0] + [int(slices[key][idx + 1] - slices[key][idx]) for idx in indices]).cumsum(0)
        for key in slices.keys()
        if key != "edge_index" or edge_counts
    }
    if "edge_index" in slices:
        batch._slice_dict["edge_index"] = torch.as_tensor([0] + edge_counts).cumsum(0)
    batch._inc_dict = {}
    return batch


def identity_collate(batch):
    return batch


__all__ = ["fast_batch_from_in_memory", "identity_collate", "pair_collate_fn", "single_collate_fn", "triple_collate_fn"]
