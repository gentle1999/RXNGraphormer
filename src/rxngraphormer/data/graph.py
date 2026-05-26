"""Molecule graph feature and distance helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
import torch

from .constants import (
    ATOM_DICT,
    BOND_DIR_LST,
    BOND_STEREO_LST,
    BOND_TYPE_LST,
    CHIRAL_TAG_DICT,
    FC_DICT,
    HYBRIDTYPE_DICT,
    MAX_NEIGHBORS,
    NUM_AROMATIC_NUM,
    NUM_ATOM_TYPE,
    NUM_BOND_DIRECTION,
    NUM_BOND_INRING,
    NUM_BOND_ISCONJ,
    NUM_BOND_STEREO,
    NUM_BOND_TYPE,
    NUM_CHIRAL_TYPE,
    NUM_DEGRESS_TYPE,
    NUM_FORMCHRG_TYPE,
    NUM_HYBRIDTYPE,
    NUM_RS_TPYE,
    NUM_VALENCE_TYPE,
    RS_TAG_DICT,
    VALENCE_DICT,
    NUM_Hs_DICT,
    NUM_Hs_TYPE,
)

GraphTask = Literal["forward_prediction", "retrosynthesis"]
IntArray: TypeAlias = npt.NDArray[np.int64] | npt.NDArray[np.int32] | npt.NDArray[np.int_]
MolGraphInfo: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
EdgeDict: TypeAlias = dict[tuple[int, int], int]


class AtomLike(Protocol):
    def GetSymbol(self) -> str:
        ...

    def GetDegree(self) -> int:
        ...

    def GetFormalCharge(self) -> int:
        ...

    def GetHybridization(self) -> object:
        ...

    def GetChiralTag(self) -> object:
        ...

    def GetIsAromatic(self) -> bool:
        ...

    def GetTotalValence(self) -> int:
        ...

    def GetTotalNumHs(self) -> int:
        ...

    def GetPropsAsDict(self) -> Mapping[str, object]:
        ...

    def GetMass(self) -> float:
        ...

    def SetAtomMapNum(self, map_num: int) -> None:
        ...

    def GetIdx(self) -> int:
        ...


class BondLike(Protocol):
    def GetBeginAtomIdx(self) -> int:
        ...

    def GetEndAtomIdx(self) -> int:
        ...

    def GetBondType(self) -> object:
        ...

    def GetBondDir(self) -> object:
        ...

    def GetStereo(self) -> object:
        ...

    def IsInRing(self) -> bool:
        ...

    def GetIsConjugated(self) -> bool:
        ...


class MolLike(Protocol):
    def GetAtoms(self) -> Iterable[AtomLike]:
        ...

    def GetBonds(self) -> Iterable[BondLike]:
        ...


def gen_onehot(features: Sequence[int], feature_dims: Sequence[int]) -> npt.NDArray[np.float64]:
    assert len(features) == len(feature_dims), (
        "size of 'features' and 'feature_dims' should be same"
    )
    onehot: list[npt.NDArray[np.float64]] = []
    for feat, feat_dim in zip(features, feature_dims):
        f_oh = np.zeros(feat_dim)
        f_oh[feat] = 1
        onehot.append(f_oh)
    return np.concatenate(onehot)


def mol2graphinfo(mol: MolLike) -> MolGraphInfo:
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    """
    # atoms
    atom_features_list: list[list[int]] = []
    atom_oh_features_list: list[npt.NDArray[np.float64]] = []
    atom_mass_list: list[float] = []

    atom_feat_dims = [
        NUM_ATOM_TYPE,
        NUM_DEGRESS_TYPE,
        NUM_FORMCHRG_TYPE,
        NUM_HYBRIDTYPE,
        NUM_CHIRAL_TYPE,
        NUM_AROMATIC_NUM,
        NUM_VALENCE_TYPE,
        NUM_Hs_TYPE,
        NUM_RS_TPYE,
    ]
    bond_feat_dims = [
        NUM_BOND_TYPE,
        NUM_BOND_DIRECTION,
        NUM_BOND_STEREO,
        NUM_BOND_INRING,
        NUM_BOND_ISCONJ,
    ]

    for atom in mol.GetAtoms():
        atom_feature = [
            ATOM_DICT.get(atom.GetSymbol(), ATOM_DICT["unk"]),
            min(atom.GetDegree(), MAX_NEIGHBORS),
            FC_DICT.get(atom.GetFormalCharge(), 4),
            HYBRIDTYPE_DICT.get(atom.GetHybridization(), 5),
            CHIRAL_TAG_DICT.get(atom.GetChiralTag(), 2),
            int(atom.GetIsAromatic()),
            VALENCE_DICT.get(atom.GetTotalValence(), 6),
            NUM_Hs_DICT.get(atom.GetTotalNumHs(), 4),
            RS_TAG_DICT.get(str(atom.GetPropsAsDict().get("_CIPCode", "None")), 2),
        ]
        atom_oh_feature = gen_onehot(atom_feature, atom_feat_dims)
        atom_mass = atom.GetMass()
        atom_features_list.append(atom_feature)
        atom_oh_features_list.append(atom_oh_feature)
        atom_mass_list.append(atom_mass)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    x_oh = torch.tensor(np.array(atom_oh_features_list), dtype=torch.long)
    atom_mass = torch.from_numpy(np.array(atom_mass_list))
    # bonds
    num_bond_features = (
        5  # bond type, bond direction, bond stereo, isinring, isconjugated
    )
    num_oh_bond_features = sum(bond_feat_dims)
    bonds = list(mol.GetBonds())
    if bonds:  # mol has bonds
        edges_list: list[tuple[int, int]] = []
        edge_features_list: list[list[int]] = []
        edge_oh_features_list: list[npt.NDArray[np.float64]] = []
        for bond in bonds:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                BOND_TYPE_LST.index(bond.GetBondType()),
                BOND_DIR_LST.index(bond.GetBondDir()),
                BOND_STEREO_LST.index(bond.GetStereo()),
                int(bond.IsInRing()),
                int(bond.GetIsConjugated()),
            ]
            edge_oh_feature = gen_onehot(edge_feature, bond_feat_dims)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edge_oh_features_list.append(edge_oh_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
            edge_oh_features_list.append(edge_oh_feature)

        edge_index = np.array(edges_list).T

        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
        edge_oh_attr = torch.tensor(np.array(edge_oh_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        edge_oh_attr = torch.empty((0, num_oh_bond_features), dtype=torch.long)
    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    a_graphs, edge_dict = get_agraph(len(x), edge_index)
    b_graphs = get_bgraphs(edge_index, edge_dict)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return x, edge_index, edge_attr, atom_mass, x_oh, edge_oh_attr, a_graphs, b_graphs


def calc_graph_distance(
    atom_feat: IntArray | torch.Tensor,
    edge_index: IntArray,
    task: GraphTask = "retrosynthesis",
) -> npt.NDArray[np.int32]:
    """
    task : "forward_prediction" or "retrosynthesis".
    g_dist = torch.from_numpy(calc_graph_distance(x_merge,edge_index_merge,task=task))
    """
    assert task in ["forward_prediction", "retrosynthesis"], (
        "task must be 'forward_prediction' or 'retrosynthesis'"
    )
    a_length = atom_feat.shape[0]
    unreachable = 1_000_000
    distance = np.full((a_length, a_length), unreachable, dtype=np.int32)
    np.fill_diagonal(distance, 0)

    for u, v in edge_index.T:
        distance[u, v] = 1

    for k in range(a_length):
        distance = np.minimum(distance, distance[:, k : k + 1] + distance[k : k + 1, :])

    # bucket
    unreachable_mask = distance >= unreachable
    distance[(distance > 8) & (distance < 15)] = 8
    distance[distance >= 15] = 9
    if task == "forward_prediction":
        distance[unreachable_mask] = 10

    # reset diagonal
    np.fill_diagonal(distance, 0)

    return distance


def calc_batch_graph_distance(
    batch: torch.Tensor,
    edge_index: torch.Tensor,
    task: GraphTask,
    max_nodes_per_graph: int | None = 512,
) -> torch.Tensor:
    ## adapted from https://github.com/coleygroup/Graph2SMILES
    assert task in ["forward_prediction", "retrosynthesis"], (
        "task must be 'forward_prediction' or 'retrosynthesis'"
    )
    num_graphs = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=num_graphs)
    max_len = int(counts.max())
    if max_nodes_per_graph is not None and max_len > max_nodes_per_graph:
        raise ValueError(
            f"attentionxl graph distance requires dense per-graph matrices; "
            f"got max_nodes={max_len}, limit={max_nodes_per_graph}. "
            "Use graph_pooling='attention' or increase max_nodes_per_graph explicitly."
        )

    distances: list[torch.Tensor] = []
    offsets = torch.cumsum(counts, dim=0) - counts
    for i in range(num_graphs):
        graph_len = int(counts[i])
        start = int(offsets[i])
        end = start + graph_len
        unreachable = 1_000_000
        dist = torch.full(
            (graph_len, graph_len), unreachable, dtype=torch.int32, device=batch.device
        )
        dist.fill_diagonal_(0)

        edge_mask = (
            (edge_index[0] >= start)
            & (edge_index[0] < end)
            & (edge_index[1] >= start)
            & (edge_index[1] < end)
        )
        local_src = edge_index[0, edge_mask] - start
        local_dst = edge_index[1, edge_mask] - start
        if local_src.numel() > 0:
            dist[local_src, local_dst] = 1

        # Floyd-Warshall on each graph avoids allocating an O(N_total^2) batch matrix.
        for k in range(graph_len):
            dist = torch.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])

        # Apply task-specific transformations
        unreachable_mask = dist >= unreachable
        dist[(dist > 8) & (dist < 15)] = (
            8  # Adjust these numbers based on your bucketing
        )
        dist[dist >= 15] = 9
        if task == "forward_prediction":
            dist[unreachable_mask] = 10
        dist.fill_diagonal_(0)

        # Padding to maximum size
        padded_dist = torch.full(
            (max_len, max_len),
            11 if task == "forward_prediction" else 10,
            dtype=torch.int32,
            device=dist.device,
        )
        actual_size = dist.size(0)
        padded_dist[:actual_size, :actual_size] = dist
        distances.append(padded_dist)

    distances_tensor = torch.stack(distances)
    return distances_tensor


def get_agraph(node_num: int, edge_index: IntArray) -> tuple[torch.Tensor, EdgeDict]:
    # edge_index : numpy.ndarray
    a_graphs: list[list[int]] = [[] for _ in range(node_num)]
    edge_dict: EdgeDict = {}
    # edge iteration to get (dense) bond features
    for (
        u,
        v,
    ) in edge_index.T:
        u_int = int(u)
        v_int = int(v)
        eid = len(edge_dict)
        edge_dict[(u_int, v_int)] = eid
        a_graphs[v_int].append(eid)
    for a_graph in a_graphs:
        while len(a_graph) < 11:
            a_graph.append(1_000_000_000)
    a_graph_tensor = torch.tensor(a_graphs).long()
    return a_graph_tensor, edge_dict


def get_bgraphs(edge_index: IntArray, edge_dict: EdgeDict) -> torch.Tensor:
    # edge_index : numpy.ndarray
    src_tgt_lst_map: dict[int, list[int]] = {}
    for src, tgt in edge_index.T:
        src_int = int(src)
        tgt_int = int(tgt)
        if src_int not in src_tgt_lst_map:
            src_tgt_lst_map[src_int] = [tgt_int]
        else:
            src_tgt_lst_map[src_int].append(tgt_int)

    # second edge iteration to get neighboring edges (after edge_dict is updated fully)
    b_graphs: list[list[int]] = [[] for _ in range(len(edge_index.T))]
    for (
        u,
        v,
    ) in edge_index.T:
        u = int(u)
        v = int(v)
        eid = edge_dict[(u, v)]

        for w in src_tgt_lst_map[u]:
            if not w == v:
                b_graphs[eid].append(edge_dict[(w, u)])

    for b_graph in b_graphs:
        while len(b_graph) < 11:
            b_graph.append(1_000_000_000)
    b_graph_tensor = torch.tensor(b_graphs).long()
    return b_graph_tensor


__all__ = [
    "calc_batch_graph_distance",
    "calc_graph_distance",
    "gen_onehot",
    "get_agraph",
    "get_bgraphs",
    "mol2graphinfo",
]
