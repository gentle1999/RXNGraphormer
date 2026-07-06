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


def _mech_mid_smi_from_mapped_reaction(rct_line: str, pdt_line: str, atmap_rxn: str) -> tuple[str, str, str]:
    rct_smi_lst = rct_line.split(".")
    pdt_smi_lst = pdt_line.split(".")
    rxn_smiles = f"{rct_line}>>{pdt_line}"
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


def gen_mech_mid_smi(task: tuple[str, str], confidence_threshold: float = 0.002) -> tuple[str, str, str]:
    result = gen_mech_mid_smis([task], confidence_threshold=confidence_threshold)[0]
    if isinstance(result, Exception):
        raise result
    return result


def gen_mech_mid_smis(
    tasks: list[tuple[str, str]],
    confidence_threshold: float = 0.002,
) -> list[tuple[str, str, str] | Exception]:
    global mapper, rxn_mapper
    if not tasks:
        return []
    mapper = mapper or get_localmapper()

    rxn_smiles = [f"{rct_line}>>{pdt_line}" for rct_line, pdt_line in tasks]
    atmap_rxns: list[str | Exception] = _localmapper_atom_maps(rxn_smiles)

    missing_indices = [
        idx for idx, atmap_rxn in enumerate(atmap_rxns) if not isinstance(atmap_rxn, Exception) and not atmap_rxn
    ]
    if missing_indices:
        rxn_mapper = rxn_mapper or get_rxn_mapper()
        fallback_rxns = [rxn_smiles[idx] for idx in missing_indices]
        fallback_results = _rxnmapper_atom_maps(fallback_rxns, confidence_threshold=confidence_threshold)
        for idx, fallback_result in zip(missing_indices, fallback_results, strict=True):
            atmap_rxns[idx] = fallback_result

    results: list[tuple[str, str, str] | Exception] = []
    for (rct_line, pdt_line), atmap_rxn in zip(tasks, atmap_rxns, strict=True):
        if isinstance(atmap_rxn, Exception):
            results.append(atmap_rxn)
            continue
        try:
            results.append(_mech_mid_smi_from_mapped_reaction(rct_line, pdt_line, atmap_rxn))
        except Exception as exc:
            results.append(exc)
    return results


def _localmapper_atom_maps(rxn_smiles: list[str]) -> list[str | Exception]:
    try:
        mapped_rxns = mapper.get_atom_map(rxn_smiles)  # type: ignore[union-attr]
        if isinstance(mapped_rxns, str):
            mapped_rxns = [mapped_rxns]
        return list(mapped_rxns)
    except Exception:
        results: list[str | Exception] = []
        for rxn in rxn_smiles:
            try:
                results.append(mapper.get_atom_map(rxn))  # type: ignore[union-attr]
            except Exception as exc:
                results.append(exc)
        return results


def _rxnmapper_atom_maps(
    rxn_smiles: list[str],
    *,
    confidence_threshold: float,
) -> list[str | Exception]:
    try:
        results = rxn_mapper.get_attention_guided_atom_maps(rxn_smiles)  # type: ignore[union-attr]
    except Exception:
        mapped_rxns: list[str | Exception] = []
        for rxn in rxn_smiles:
            try:
                result = rxn_mapper.get_attention_guided_atom_maps([rxn])[0]  # type: ignore[union-attr]
            except Exception as exc:
                mapped_rxns.append(exc)
                continue
            mapped_rxns.append(_rxnmapper_result_to_mapped_rxn(result, confidence_threshold=confidence_threshold))
        return mapped_rxns
    return [_rxnmapper_result_to_mapped_rxn(result, confidence_threshold=confidence_threshold) for result in results]


def _rxnmapper_result_to_mapped_rxn(result: dict[str, object], *, confidence_threshold: float) -> str:
    confidence = result["confidence"]
    if not isinstance(confidence, int | float | str):
        raise TypeError(f"Unexpected rxnmapper confidence type: {type(confidence).__name__}")
    if float(confidence) < confidence_threshold:
        print("confidence too low")
        return ""
    return str(result["mapped_rxn"])
