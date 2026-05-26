"""Reaction-level graph construction helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Literal, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import torch
from rdkit import Chem
from tqdm import tqdm

from .features import ext_feat_gen
from .graph import MolLike, mol2graphinfo
from .tokenization import get_token_ids

TaskName = Literal["regression", "classification"]
ExtFeatParams: TypeAlias = Mapping[str, int | bool | float | str]
ExtFeature: TypeAlias = torch.Tensor | npt.NDArray[np.float64] | npt.NDArray[np.int_]
GraphInfo: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    ExtFeature,
]
PerformanceGraphInfo: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    ExtFeature,
]
SequenceGraphInfo: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def get_cpd_rxn_info(
    cpd_smi: str,
    ext: bool,
    ext_param: ExtFeatParams,
    ext_type: str,
    mul_ext_readout: str,
) -> GraphInfo | None:

    rxn_mol = Chem.MolFromSmiles(cpd_smi)
    if rxn_mol is None:
        return None
    ext_feat: ExtFeature
    if ext:
        ext_feat = ext_feat_gen(
            rxn_mol, params=ext_param, desc_type=ext_type, multi_readout=mul_ext_readout
        )
    else:
        ext_feat = torch.tensor([0.0]).float()
    smi_blk_lst = cpd_smi.split(".")
    x_edge_index_attr_lst: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    failed = False
    for smi in smi_blk_lst:
        rdkit_mol = Chem.MolFromSmiles(smi)
        if rdkit_mol is None:
            failed = True
            break
        x_edge_index_attr_lst.append(mol2graphinfo(rdkit_mol))

    if failed:
        return None

    atom_num_lst = [len(item[0]) for item in x_edge_index_attr_lst]
    atom_num_start = [0]
    mol_index_values: list[int] = []
    for num in atom_num_lst[:-1]:
        atom_num_start.append(atom_num_start[-1] + num)
    for i, num in enumerate(atom_num_lst):
        mol_index_values += [i] * num
    if len(mol_index_values) == 0:
        # fix for empty molecule
        x_merge = torch.zeros([1, 9], dtype=torch.int64)
        atom_mass_merge = torch.tensor([0], dtype=torch.float64)
        mol_index = torch.tensor([0], dtype=torch.long)
    else:
        x_merge = torch.cat([item[0] for item in x_edge_index_attr_lst])
        atom_mass_merge = torch.cat([item[3] for item in x_edge_index_attr_lst])
        mol_index = torch.tensor(mol_index_values, dtype=torch.long)
    edge_index_merge = torch.cat(
        [item[1] + num for item, num in zip(x_edge_index_attr_lst, atom_num_start)],
        dim=1,
    )
    edge_attr_merge = torch.cat([item[2] for item in x_edge_index_attr_lst])

    x_oh_merge = torch.cat([item[4] for item in x_edge_index_attr_lst])
    edge_oh_attr_merge = torch.cat([item[5] for item in x_edge_index_attr_lst])
    a_graphs_merge = torch.cat([item[6] for item in x_edge_index_attr_lst])
    b_graphs_merge = torch.cat([item[7] for item in x_edge_index_attr_lst])

    return (
        x_merge,
        edge_index_merge,
        edge_attr_merge,
        atom_mass_merge,
        x_oh_merge,
        edge_oh_attr_merge,
        a_graphs_merge,
        b_graphs_merge,
        mol_index,
        ext_feat,
    )


def get_rxn_pfm_info(
    rxn_smi_tgt_task_ens: tuple[str, str, bool, str, ExtFeatParams, str],
) -> PerformanceGraphInfo | None:
    rxn_smi_tgt, task, ext, ext_type, ext_param, mul_ext_readout = rxn_smi_tgt_task_ens
    task = task.lower()
    assert task in ["regression", "classification"], (
        "task must be regression or classification"
    )
    rxn_smi, tgt_ = rxn_smi_tgt.split(",")
    if task.lower() == "regression":
        tgt_ = torch.tensor([float(tgt_)]).float()
    elif task.lower() == "classification":
        tgt_ = torch.tensor([int(tgt_)]).long()
    else:
        raise ValueError("task must be regression or classification")
    rxn_inf = get_cpd_rxn_info(rxn_smi, ext, ext_param, ext_type, mul_ext_readout)
    if rxn_inf is None:
        return None
    (
        x_merge,
        edge_index_merge,
        edge_attr_merge,
        atom_mass_merge,
        x_oh_merge,
        edge_oh_attr_merge,
        a_graphs_merge,
        b_graphs_merge,
        mol_index,
        ext_feat,
    ) = rxn_inf
    return (
        x_merge,
        edge_index_merge,
        edge_attr_merge,
        atom_mass_merge,
        x_oh_merge,
        edge_oh_attr_merge,
        a_graphs_merge,
        b_graphs_merge,
        mol_index,
        tgt_,
        ext_feat,
    )


def get_rxn_seq_info(input_: tuple[str, str, Mapping[str, int], int]) -> SequenceGraphInfo | None:
    src_line, tgt_line, vocab, max_length = input_
    src_smi = "".join(src_line.strip().split())
    tgt_tokens = tgt_line.strip().split()
    tgt_token_ids, tgt_lens = get_token_ids(tgt_tokens, vocab, max_length)
    tgt_token_ids = torch.tensor([tgt_token_ids], dtype=torch.long)
    tgt_lens = torch.tensor([tgt_lens], dtype=torch.long)

    ## Molecular Graph for Reactant Molecules
    src_smi_blk_lst = src_smi.split(".")
    x_edge_index_attr_lst: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    failed = False
    for smi in src_smi_blk_lst:
        rdkit_mol = Chem.MolFromSmiles(smi)
        if rdkit_mol is None:
            failed = True
            break
        x_edge_index_attr_lst.append(mol2graphinfo(rdkit_mol))
    if failed:
        return None
    atom_num_lst = [len(item[0]) for item in x_edge_index_attr_lst]
    atom_num_start = [0]
    mol_index_values: list[int] = []
    for num in atom_num_lst[:-1]:
        atom_num_start.append(atom_num_start[-1] + num)
    for i, num in enumerate(atom_num_lst):
        mol_index_values += [i] * num

    x_merge = torch.cat([item[0] for item in x_edge_index_attr_lst])
    edge_index_merge = torch.cat(
        [item[1] + num for item, num in zip(x_edge_index_attr_lst, atom_num_start)],
        dim=1,
    )
    edge_attr_merge = torch.cat([item[2] for item in x_edge_index_attr_lst])
    atom_mass_merge = torch.cat([item[3] for item in x_edge_index_attr_lst])
    x_oh_merge = torch.cat([item[4] for item in x_edge_index_attr_lst])
    edge_oh_attr_merge = torch.cat([item[5] for item in x_edge_index_attr_lst])
    a_graphs_merge = torch.cat([item[6] for item in x_edge_index_attr_lst])
    b_graphs_merge = torch.cat([item[7] for item in x_edge_index_attr_lst])
    if len(mol_index_values) == 0:
        return None
    mol_index = torch.tensor(mol_index_values, dtype=torch.long)
    return (
        x_merge,
        edge_index_merge,
        edge_attr_merge,
        mol_index,
        atom_mass_merge,
        x_oh_merge,
        edge_oh_attr_merge,
        a_graphs_merge,
        b_graphs_merge,
        tgt_token_ids,
        tgt_lens,
    )


def generate_regression_dataset(
    dataset_file: str | PathLike[str],
    rct_cols: Sequence[str] = ("Reactant1", "Reactant2", "Solvents", "Reagents"),
    pdt_cols: Sequence[str] = ("Product",),
    tgt_inf: str | None = None,
) -> tuple[list[str], list[str]]:
    import pandas as pd

    raw_dataset = pd.read_csv(dataset_file)
    rct_inf_lst = raw_dataset[rct_cols].to_numpy()
    pdt_inf_lst = raw_dataset[pdt_cols].to_numpy()
    if tgt_inf is None:
        rct_parts = [
            f"{Chem.MolToSmiles(Chem.MolFromSmiles('.'.join(rct_inf)))},0.0"
            for rct_inf in tqdm(rct_inf_lst)
        ]
        pdt_parts = [
            f"{Chem.MolToSmiles(Chem.MolFromSmiles('.'.join(pdt_inf)))},0.0"
            for pdt_inf in tqdm(pdt_inf_lst)
        ]
    else:
        tgt_lst = raw_dataset[tgt_inf].to_list()
        rct_parts = [
            f"{Chem.MolToSmiles(Chem.MolFromSmiles('.'.join(rct_inf)))},{tgt}"
            for rct_inf, tgt in tqdm(zip(rct_inf_lst, tgt_lst))
        ]
        pdt_parts = [
            f"{Chem.MolToSmiles(Chem.MolFromSmiles('.'.join(pdt_inf)))},{tgt}"
            for pdt_inf, tgt in tqdm(zip(pdt_inf_lst, tgt_lst))
        ]
    return rct_parts, pdt_parts


def gen_mol_in_rxn(rxn_smiles: str, show_atom_map: bool = True) -> tuple[list[MolLike], bool]:
    smi_blk_lst = rxn_smiles.split(".")
    mol_lst: list[MolLike] = []
    failed = False
    for smi in smi_blk_lst:
        rdkit_mol = Chem.MolFromSmiles(smi)
        if rdkit_mol is None:
            failed = True
            break
        mol = cast(MolLike, rdkit_mol)
        if show_atom_map:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx() + 1)
        mol_lst.append(mol)
    return mol_lst, failed


__all__ = [
    "gen_mol_in_rxn",
    "generate_regression_dataset",
    "get_cpd_rxn_info",
    "get_rxn_pfm_info",
    "get_rxn_seq_info",
]
