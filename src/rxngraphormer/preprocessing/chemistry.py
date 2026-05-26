from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

from ._rdkit_protocols import ReactionProtocol

BondMap = tuple[int, int]
AtomLocation = list[int]
BondLocation = list[AtomLocation]


def _bond_map_tuple(bond: frozenset[int]) -> BondMap:
    left, right = tuple(bond)
    return int(left), int(right)


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return ""

    original_smi = smiles
    viewed_smi = {original_smi: 1}
    while original_smi != (
        canonical_smi := Chem.CanonSmiles(original_smi, useChiral=True)
    ) and (canonical_smi not in viewed_smi or viewed_smi[canonical_smi] < 2):
        original_smi = canonical_smi
        if original_smi not in viewed_smi:
            viewed_smi[original_smi] = 1
        else:
            viewed_smi[original_smi] += 1
    else:
        return original_smi


def inchi_to_smiles(inchi: str) -> str | None:
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            return canonical_smiles(Chem.MolToSmiles(mol))
        return None
    except Exception:
        return None


def mod_mol(
    mol1: Chem.Mol,
    mol2: Chem.Mol | None = None,
    type: str = "add",
    bond_idx: Sequence[int] = (0, 1),
    bond_type: Chem.BondType = Chem.BondType.SINGLE,
) -> Chem.Mol:
    mode = type.lower()
    if mode == "add":
        if mol2 is None:
            raise ValueError("mol2 is required when adding a bond between molecules")
        rwmol = Chem.RWMol(mol1)
        rwmol.InsertMol(mol2)
        rwmol.AddBond(bond_idx[0], bond_idx[1] + mol1.GetNumAtoms(), bond_type)
        new_mol = rwmol.GetMol()
    elif mode == "remove":
        rwmol = Chem.RWMol(mol1)
        rwmol.RemoveBond(bond_idx[0], bond_idx[1])
        new_mol = rwmol.GetMol()
    else:
        raise ValueError("type must be 'add' or 'remove'")
    return new_mol


def gen_mid_mols(rct_blks: Sequence[str], pdt_blks: Sequence[str]) -> list[Chem.Mol]:
    rct_mols = [Chem.MolFromSmiles(item) for item in rct_blks]
    pdt_mols = [Chem.MolFromSmiles(item) for item in pdt_blks]

    reaction_smiles = ".".join(rct_blks) + ">>" + ".".join(pdt_blks)
    reaction = _rd_reaction_from_smarts(reaction_smiles)
    reactants = [reaction.GetReactantTemplate(i) for i in range(reaction.GetNumReactantTemplates())]
    products = [reaction.GetProductTemplate(i) for i in range(reaction.GetNumProductTemplates())]

    reactant_bonds: set[frozenset[int]] = set()
    rct_mol_atom_idx_map: dict[int, AtomLocation] = {}
    for rct_idx, reactant in enumerate(reactants):
        atoms = [reactant.GetAtomWithIdx(atom_idx) for atom_idx in range(reactant.GetNumAtoms())]
        rct_mol_atom_idx_map.update({atom.GetAtomMapNum(): [rct_idx, atom.GetIdx()] for atom in atoms})
        for bond in _mol_bonds(reactant):
            begin_idx = bond.GetBeginAtom().GetAtomMapNum()
            end_idx = bond.GetEndAtom().GetAtomMapNum()
            if begin_idx and end_idx:
                reactant_bonds.add(frozenset([begin_idx, end_idx]))

    product_bonds: set[frozenset[int]] = set()
    pdt_mol_atom_idx_map: dict[int, AtomLocation] = {}
    for pdt_idx, product in enumerate(products):
        atoms = [product.GetAtomWithIdx(atom_idx) for atom_idx in range(product.GetNumAtoms())]
        pdt_mol_atom_idx_map.update({atom.GetAtomMapNum(): [pdt_idx, atom.GetIdx()] for atom in atoms})
        for bond in _mol_bonds(product):
            begin_idx = bond.GetBeginAtom().GetAtomMapNum()
            end_idx = bond.GetEndAtom().GetAtomMapNum()
            if begin_idx and end_idx:
                product_bonds.add(frozenset([begin_idx, end_idx]))

    broken_bonds: list[BondMap] = [_bond_map_tuple(bond) for bond in reactant_bonds - product_bonds]
    new_bonds: list[BondMap] = [_bond_map_tuple(bond) for bond in product_bonds - reactant_bonds]

    broken_bonds_in_rct: list[BondLocation] = [[rct_mol_atom_idx_map[bond[0]], rct_mol_atom_idx_map[bond[1]]] for bond in broken_bonds]
    broken_bonds_in_pdt: list[BondLocation] = [[pdt_mol_atom_idx_map[bond[0]], pdt_mol_atom_idx_map[bond[1]]] for bond in broken_bonds]
    new_bonds_in_rct = [
        [rct_mol_atom_idx_map[bond[0]], rct_mol_atom_idx_map[bond[1]]]
        for bond in new_bonds
        if bond[0] in rct_mol_atom_idx_map and bond[1] in rct_mol_atom_idx_map
    ]
    new_bonds_in_pdt = [
        [pdt_mol_atom_idx_map[bond[0]], pdt_mol_atom_idx_map[bond[1]]]
        for bond in new_bonds
        if bond[0] in pdt_mol_atom_idx_map and bond[1] in pdt_mol_atom_idx_map
    ]

    all_mid_mols: list[Chem.Mol] = []
    for bond in new_bonds_in_rct:
        try:
            left = rct_mols[bond[0][0]]
            right = rct_mols[bond[1][0]]
            if left is None or right is None:
                continue
            mid_mol = mod_mol(left, right, "add", [bond[0][1], bond[1][1]])
        except Exception:
            continue
        all_mid_mols.append(mid_mol)
    for bond in new_bonds_in_pdt:
        try:
            mol = pdt_mols[bond[0][0]]
            if mol is None:
                continue
            mid_mol = mod_mol(mol, None, "remove", [bond[0][1], bond[1][1]])
        except Exception:
            continue
        all_mid_mols.append(mid_mol)

    for bond in broken_bonds_in_rct:
        try:
            mol = rct_mols[bond[0][0]]
            if mol is None:
                continue
            mid_mol = mod_mol(mol, None, "remove", [bond[0][1], bond[1][1]])
        except Exception:
            continue
        all_mid_mols.append(mid_mol)
    for bond in broken_bonds_in_pdt:
        try:
            left = pdt_mols[bond[0][0]]
            right = pdt_mols[bond[1][0]]
            if left is None or right is None:
                continue
            mid_mol = mod_mol(left, right, "add", [bond[0][1], bond[1][1]])
        except Exception:
            continue
        all_mid_mols.append(mid_mol)
    return all_mid_mols


def get_random_shuffle_smiles(
    init_smi: str,
    rxn_template: str = "[C,N,O:1]-[*:2]~[*:3]-[*:4]>>[*:4]-[*:2]~[*:3]-[C,N,O:1]",
    sel_num: int = 2,
    random_seed: int = 42,
) -> list[str]:
    """
    get random shuffle smiles with given rxn_template and init_smi
    init_smi : initial smiles, e.g. "CCc1ncNccc1O"
    """
    np.random.seed(random_seed)
    rxn1 = _reaction_from_smarts(rxn_template)
    init_mol = Chem.MolFromSmiles(init_smi)
    if init_mol is None:
        return []
    rxn_res = rxn1.RunReactants((_add_hs(init_mol),))
    if len(rxn_res) > 0:
        rxn_smi_res: list[str] = []
        for item in rxn_res:
            try:
                _smi = Chem.MolToSmiles(Chem.RemoveHs(item[0]))
                if _smi == init_smi:
                    continue
                rxn_smi_res.append(_smi)
            except Exception:
                continue
        rxn_smi_res = list(set(rxn_smi_res))
        sel_idx = np.random.choice(
            len(rxn_smi_res),
            min(sel_num, len(rxn_smi_res)),
            replace=False,
        )
        sel_smi = [rxn_smi_res[idx] for idx in sel_idx]
    else:
        sel_smi = []
    return sel_smi


def gen_truth_false_rxn_smi(
    rct_pdt_smi__split: tuple[str, str, bool],
    rxn_template: str = "[C,N,O:1]-[*:2]~[*:3]-[*:4]>>[*:4]-[*:2]~[*:3]-[C,N,O:1]",
    sel_num: int = 2,
    random_seed: int = 42,
) -> tuple[list[str], list[str]]:
    # default split=False
    rct_smi, pdt_smi, split = rct_pdt_smi__split
    false_pdt_smi_lst = get_random_shuffle_smiles(
        pdt_smi,
        rxn_template=rxn_template,
        sel_num=sel_num,
        random_seed=random_seed,
    )
    if not split:
        truth_rxn_smi = canonical_smiles(f"{rct_smi}.{pdt_smi}")
        false_rxn_smi_lst = []
        for f_pdt_smi in false_pdt_smi_lst:
            false_rxn_smi = canonical_smiles(f"{rct_smi}.{f_pdt_smi}")
            false_rxn_smi_lst.append(false_rxn_smi)
    else:
        rct_smi = canonical_smiles(rct_smi)
        pdt_smi = canonical_smiles(pdt_smi)
        truth_rxn_smi = f"{rct_smi}>>{pdt_smi}"
        false_rxn_smi_lst = []
        for f_pdt_smi in false_pdt_smi_lst:
            f_pdt_smi = canonical_smiles(f_pdt_smi)
            false_rxn_smi = f"{rct_smi}>>{f_pdt_smi}"
            false_rxn_smi_lst.append(false_rxn_smi)
    return [truth_rxn_smi], false_rxn_smi_lst


def _reaction_from_smarts(smarts: str) -> ReactionProtocol:
    reaction_from_smarts = cast(Callable[[str], object], getattr(AllChem, "ReactionFromSmarts"))
    return cast(ReactionProtocol, reaction_from_smarts(smarts))


def _rd_reaction_from_smarts(smarts: str) -> ReactionProtocol:
    reaction_from_smarts = cast(Callable[[str], object], rdChemReactions.ReactionFromSmarts)
    return cast(ReactionProtocol, reaction_from_smarts(smarts))


def _add_hs(mol: Chem.Mol) -> Chem.Mol:
    add_hs = cast(Callable[..., Chem.Mol], getattr(AllChem, "AddHs"))
    return add_hs(mol)


def _mol_bonds(mol: Chem.Mol) -> list[Chem.Bond]:
    return [mol.GetBondWithIdx(bond_idx) for bond_idx in range(mol.GetNumBonds())]
