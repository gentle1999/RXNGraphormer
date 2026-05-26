import os
import re

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Mol, rdmolops

import rxngraphormer
from rxngraphormer.midgen.MechFinder import MechFinder
from rxngraphormer.preprocessing.chemistry import canonical_smiles

finder = MechFinder(collection_dir=f"{os.path.dirname(rxngraphormer.__file__)}/midgen/collections")
mapper = None
rxn_mapper = None


def get_localmapper():
    global mapper
    if mapper is None:
        from localmapper import localmapper

        mapper = localmapper("cpu")  ## avoid some bugs
    return mapper


def _new_cpu_rxn_mapper():
    from rxnmapper import RXNMapper

    # RXNMapper chooses cuda automatically and has no device argument.
    cuda_is_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        mapper_ = RXNMapper()
    finally:
        torch.cuda.is_available = cuda_is_available
    mapper_.device = torch.device("cpu")
    mapper_.model.to(mapper_.device)
    return mapper_


def get_rxn_mapper():
    global rxn_mapper
    if rxn_mapper is None:
        rxn_mapper = _new_cpu_rxn_mapper()
    return rxn_mapper


### Mid molecule generation
# Extract key information and store it as a collection of the form (atom1 map, atom2 map)
def get_bond_set(molecules: list[Mol], allowed_at_idx_lst: list[int]) -> set[tuple[int, int]]:
    bond_set = set()
    for mol in molecules:
        for bond in mol.GetBonds():  # type: ignore
            a1 = bond.GetBeginAtom().GetAtomMapNum()
            a2 = bond.GetEndAtom().GetAtomMapNum()
            if a1 not in allowed_at_idx_lst or a2 not in allowed_at_idx_lst:
                continue
            if a1 > a2:  # Ensure that smaller mapping numbers are placed first
                a1, a2 = a2, a1
            bond_set.add((a1, a2))
    return bond_set


def analyze_reaction_bonds(
    reaction_smiles: str, allowed_at_idx_lst: list[int]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    # Decompose reaction SMILES string into reactants and products
    reactants_smiles, products_smiles = reaction_smiles.split(">>")

    # Use RDKit to build molecules of reactants and products
    reactants = [Chem.MolFromSmiles(smile) for smile in reactants_smiles.split(".")]
    products = [Chem.MolFromSmiles(smile) for smile in products_smiles.split(".")]

    reactant_bonds = get_bond_set(reactants, allowed_at_idx_lst)
    product_bonds = get_bond_set(products, allowed_at_idx_lst)

    # Find broken bonds and newly formed bonds
    broken_bonds = reactant_bonds - product_bonds
    formed_bonds = product_bonds - reactant_bonds

    return broken_bonds, formed_bonds


def get_atommap_atomidx_map(smiles: str, allowed_at_idx_lst: list[int]) -> dict[int, int]:
    mol = Chem.MolFromSmiles(smiles)
    atommap_atomidx_map = {}
    for atom in mol.GetAtoms():  # type: ignore
        if atom.GetAtomMapNum() in allowed_at_idx_lst:
            atommap_atomidx_map[atom.GetAtomMapNum()] = atom.GetIdx()
    return atommap_atomidx_map


def ex_atmap_inf(smiles: str) -> dict[int, str]:
    # Regular expression matching pattern
    pattern = r"\[([A-Za-z0-9+-]+?):(\d+)\]"
    matches = re.findall(pattern, smiles)
    atom_mappings = {}
    for atom, map_num in matches:
        atom_mappings[int(map_num)] = atom

    return atom_mappings


def break_bond(smiles: str, idx1: int, idx2: int) -> tuple[Mol, str]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string")

    if idx1 < 0 or idx1 >= mol.GetNumAtoms() or idx2 < 0 or idx2 >= mol.GetNumAtoms():
        raise ValueError("Atom index out of allowed range")

    bond = mol.GetBondBetweenAtoms(idx1, idx2)
    if not bond:
        print("If the specified key does not exist, the original molecule is returned.")
        # Fix: Ensure consistent return type (Tuple) to avoid unpacking errors in caller
        return mol, Chem.MolToSmiles(mol)

    bond_type = bond.GetBondType()
    elec_num = 0
    if bond_type == Chem.BondType.SINGLE:
        elec_num = 1
    elif bond_type == Chem.BondType.DOUBLE:
        elec_num = 2
    elif bond_type == Chem.BondType.AROMATIC:
        elec_num = 1
    else:
        raise ValueError("The specified bond is not a single, double or aromatic bond")

    editable_mol = Chem.EditableMol(mol)
    editable_mol.RemoveBond(idx1, idx2)
    modified_mol = editable_mol.GetMol()

    modified_mol.GetAtomWithIdx(idx1).SetNumRadicalElectrons(
        modified_mol.GetAtomWithIdx(idx1).GetNumRadicalElectrons() + elec_num
    )
    modified_mol.GetAtomWithIdx(idx2).SetNumRadicalElectrons(
        modified_mol.GetAtomWithIdx(idx2).GetNumRadicalElectrons() + elec_num
    )

    rdmolops.SanitizeMol(modified_mol)

    modified_smiles = Chem.MolToSmiles(modified_mol)
    return modified_mol, modified_smiles


def remove_atmmap(smiles: str) -> str:
    cleaned_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(re.sub(r":\d+", "", smiles)))

    return cleaned_smiles


def get_mid_smi_from_rxn(updated_reaction: str) -> list[str]:
    reactants_part, products_part = updated_reaction.split(">>")
    atmap_inf = ex_atmap_inf(updated_reaction)
    broken_bonds, formed_bonds = analyze_reaction_bonds(updated_reaction, list(atmap_inf.keys()))
    # print(f"broken bonds: {broken_bonds}, formed bonds {formed_bonds}")
    all_mid_mols = []
    all_mid_smis = []
    for smiles in reactants_part.split("."):
        atommap_idx_map = get_atommap_atomidx_map(smiles, list(atmap_inf.keys()))
        for broken_bond in broken_bonds:
            if broken_bond[0] in atommap_idx_map and broken_bond[1] in atommap_idx_map:
                modified_mol, modified_smi = break_bond(
                    smiles,
                    atommap_idx_map[broken_bond[0]],
                    atommap_idx_map[broken_bond[1]],
                )
                all_mid_mols.append(modified_mol)
                all_mid_smis.append(modified_smi)

    for smiles in products_part.split("."):
        atommap_idx_map = get_atommap_atomidx_map(smiles, list(atmap_inf.keys()))
        for formed_bond in formed_bonds:
            if formed_bond[0] in atommap_idx_map and formed_bond[1] in atommap_idx_map:
                modified_mol, modified_smi = break_bond(
                    smiles,
                    atommap_idx_map[formed_bond[0]],
                    atommap_idx_map[formed_bond[1]],
                )
                all_mid_mols.append(modified_mol)
                all_mid_smis.append(modified_smi)

    concat_mid_smis = []
    for mid_smis in all_mid_smis:
        concat_mid_smis += [remove_atmmap(smi) for smi in mid_smis.split(".")]
    return concat_mid_smis


def gen_mech_mid_smi(task: tuple[str, str], confidence_threshold: float = 0.002) -> tuple[str, str, str]:
    rct_line, pdt_line = task
    rct_smi_lst = rct_line.split(".")
    pdt_smi_lst = pdt_line.split(".")
    rxn_smiles = f"{rct_line}>>{pdt_line}"
    # atmap_rxn = mapper.get_atom_map(rxn_smiles)
    atmap_rxn_res = mapper.get_atom_map(rxn_smiles, return_dict=True)  # type: ignore
    atmap_rxn = atmap_rxn_res["mapped_rxn"]
    confident = atmap_rxn_res["confident"]
    if not confident:
        results = rxn_mapper.get_attention_guided_atom_maps([rxn_smiles])  # type: ignore
        atmap_rxn = results[0]["mapped_rxn"]
        if results[0]["confidence"] < confidence_threshold:
            print("confidence too low")
            atmap_rxn = ""
    if atmap_rxn != "":
        updated_reaction, LRT, MT_class, electron_path = finder.get_electron_path(atmap_rxn)
        mid_smi_lst = get_mid_smi_from_rxn(updated_reaction)
        pot_mech_smi_lst = np.concatenate(
            [remove_atmmap(smi).split(".") for smi in updated_reaction.split(">>")]
        ).tolist()
        mech_mid_smi_lst = [
            smi for smi in pot_mech_smi_lst + mid_smi_lst if smi not in rct_smi_lst and smi not in pdt_smi_lst
        ]
        mech_mid_smi = canonical_smiles(".".join(list(set(mech_mid_smi_lst))))
    else:
        mid_smi_lst = []
        mech_mid_smi = ""
        updated_reaction = rxn_smiles

    return mech_mid_smi, atmap_rxn, updated_reaction
