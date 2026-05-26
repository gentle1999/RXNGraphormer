"""Batch tensor helpers for graph encoders and legacy trainers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar, cast

import numpy as np
import torch
from numpy.typing import NDArray

from .constants import ATOM_DICT, ATOM_FEAT_DIMS
from .graph import gen_onehot


class SparseFeatureBatch(Protocol):
    x_oh: torch.Tensor
    edge_oh_attr: torch.Tensor
    a_graphs: torch.Tensor
    b_graphs: torch.Tensor


class DenseGraphBatch(Protocol):
    x: torch.Tensor
    edge_attr: torch.Tensor
    edge_index: torch.Tensor
    mol_index: object
    batch: torch.Tensor


TSparseFeatureBatch = TypeVar("TSparseFeatureBatch", bound=SparseFeatureBatch)
TDenseGraphBatch = TypeVar("TDenseGraphBatch", bound=DenseGraphBatch)


def pad_feat(feat: torch.Tensor, batch: torch.Tensor, num_features: int) -> torch.Tensor:
    device = feat.device
    batch = batch.to(device)
    if batch.numel() == 0:
        raise ValueError("pad_feat requires at least one batch entry")

    batch_size = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=batch_size)
    max_length = int(counts.max().item())

    padded_feat = feat.new_zeros((batch_size, max_length, num_features))

    first_indices = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(batch.numel(), device=device) - first_indices[batch]
    padded_feat[batch, positions] = feat

    return padded_feat


def update_batch_idx(
    mol_index: object,
    device: torch.device | str | int,
    *,
    atom_batch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(mol_index) and atom_batch is not None:
        mol_tensor = cast(torch.Tensor, mol_index).to(device=device, dtype=torch.long).view(-1)
        atom_batch = atom_batch.to(device=device, dtype=torch.long).view(-1)
        mol_tensors = _split_mol_index_tensor(mol_tensor, atom_batch)
    elif torch.is_tensor(mol_index):
        mol_tensors = [cast(torch.Tensor, mol_index).to(device=device, dtype=torch.long).view(-1)]
    else:
        mol_index_sequence = _as_sequence(mol_index)
        if len(mol_index_sequence) == 0:
            raise ValueError("update_batch_idx requires at least one molecule index block")

        first_block = mol_index_sequence[0]
        if not torch.is_tensor(first_block) and not isinstance(first_block, (list, tuple, np.ndarray)):
            mol_index_blocks = [mol_index]
        else:
            mol_index_blocks = list(mol_index_sequence)
        mol_tensors = [torch.as_tensor(m, dtype=torch.long, device=device).view(-1) for m in mol_index_blocks]

    atom_counts = torch.tensor([m.numel() for m in mol_tensors], dtype=torch.long, device=device)
    if torch.any(atom_counts == 0):
        raise ValueError("update_batch_idx requires every molecule index block to be non-empty")

    mol_tensors = [m - m.min() for m in mol_tensors]
    max_values = torch.stack([m.max() for m in mol_tensors])
    mol_counts = max_values + 1
    offsets = torch.cumsum(mol_counts, dim=0) - mol_counts

    graph_ids_per_atom = torch.repeat_interleave(
        torch.arange(len(mol_tensors), dtype=torch.long, device=device),
        atom_counts,
    )
    batch_mol_index = torch.cat(mol_tensors) + offsets[graph_ids_per_atom]
    batch = torch.repeat_interleave(
        torch.arange(len(mol_tensors), dtype=torch.long, device=device),
        mol_counts,
    )

    return batch_mol_index, batch


def add_empty_node_and_edge(batch_data: TSparseFeatureBatch) -> TSparseFeatureBatch:
    empty_node = [
        ATOM_DICT.get("*", ATOM_DICT["unk"]),
        0,  # Degree
        0,  # Formal Charge
        0,  # Hybridization
        0,  # Chiral Tag
        0,  # Is Aromatic
        0,  # Total Valence
        0,  # Total Num Hs
        0,  # RS Tag
    ]
    oh_empty_node = batch_data.x_oh.new_tensor(gen_onehot(empty_node, ATOM_FEAT_DIMS)).unsqueeze(0)
    oh_empty_edge = batch_data.edge_oh_attr.new_zeros((1, batch_data.edge_oh_attr.shape[1]))
    a_graphs_empty = batch_data.a_graphs.new_zeros((1, batch_data.a_graphs.shape[1]))
    b_graphs_empty = batch_data.b_graphs.new_zeros((1, batch_data.b_graphs.shape[1]))

    x_oh_merge_ = torch.cat([oh_empty_node, batch_data.x_oh], dim=0)
    shifted_edge_oh_attr = batch_data.edge_oh_attr.clone()
    shifted_edge_oh_attr[:, :2] = shifted_edge_oh_attr[:, :2] + 1
    edge_oh_attr_merge_ = torch.cat([oh_empty_edge, shifted_edge_oh_attr], dim=0)
    a_graphs_merge_ = torch.cat([a_graphs_empty, batch_data.a_graphs + 1], dim=0)
    b_graphs_merge_ = torch.cat([b_graphs_empty, batch_data.b_graphs + 1], dim=0)

    a_graphs_merge_[a_graphs_merge_ >= 999999999] = 0
    b_graphs_merge_[b_graphs_merge_ >= 999999999] = 0

    batch_data.x_oh = x_oh_merge_
    batch_data.edge_oh_attr = edge_oh_attr_merge_
    batch_data.a_graphs = a_graphs_merge_
    batch_data.b_graphs = b_graphs_merge_
    return batch_data


def add_dense_empty_node_edge(batch_data: TDenseGraphBatch) -> TDenseGraphBatch:
    empty_node = batch_data.x.new_tensor(
        [
            [
                ATOM_DICT.get("*", ATOM_DICT["unk"]),
                0,  # Degree
                0,  # Formal Charge
                0,  # Hybridization
                0,  # Chiral Tag
                0,  # Is Aromatic
                0,  # Total Valence
                0,  # Total Num Hs
                0,  # RS Tag
            ]
        ]
    )
    empty_edge = batch_data.edge_attr.new_zeros((1, batch_data.edge_attr.shape[1]))
    x_merge_ = torch.cat([empty_node, batch_data.x], dim=0)
    edge_merge_ = torch.cat([empty_edge, batch_data.edge_attr], dim=0)
    edge_index_ = torch.cat([batch_data.edge_index.new_zeros((2, 1)), batch_data.edge_index + 1], dim=1)

    batch_data.mol_index = _prepend_empty_mol_index(batch_data.mol_index)
    batch_data.x = x_merge_
    batch_data.edge_attr = edge_merge_
    batch_data.edge_index = edge_index_
    return batch_data


def _as_sequence(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        object_array = cast(NDArray[np.object_], value.astype(object))
        return cast(list[object], object_array.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sequence_value = cast(Sequence[object], value)
        return [item for item in sequence_value]
    raise TypeError("mol_index must be a tensor, numpy array, or sequence")


def _prepend_empty_mol_index(mol_index: object) -> object:
    if torch.is_tensor(mol_index):
        tensor_mol_index = cast(torch.Tensor, mol_index)
        return torch.cat([tensor_mol_index.new_zeros(1), tensor_mol_index.view(-1)])
    mol_index_sequence = _as_sequence(mol_index)
    return [[0], *list(mol_index_sequence)]


def _split_mol_index_tensor(mol_index: torch.Tensor, atom_batch: torch.Tensor) -> list[torch.Tensor]:
    if mol_index.numel() != atom_batch.numel():
        raise ValueError(
            "mol_index tensor and atom batch vector must have the same number of atoms "
            f"({mol_index.numel()} != {atom_batch.numel()})"
        )
    if atom_batch.numel() == 0:
        raise ValueError("update_batch_idx requires at least one molecule index block")
    batch_size = int(atom_batch.max().item()) + 1
    counts = torch.bincount(atom_batch, minlength=batch_size)
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).view(-1).detach().cpu().tolist()
        raise ValueError(f"atom batch vector contains empty graph ids: {missing}")
    split_sizes = [int(count.item()) for count in counts]
    return list(torch.split(mol_index, split_sizes))


__all__ = [
    "add_dense_empty_node_edge",
    "add_empty_node_and_edge",
    "pad_feat",
    "update_batch_idx",
]
