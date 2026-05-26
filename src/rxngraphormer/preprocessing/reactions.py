from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem.MolStandardize import rdMolStandardize

from ._rdkit_protocols import ReactionProtocol


@dataclass(frozen=True)
class ParsedReactionSmiles:
    reactants: str
    products: str
    agents: str = ""
    has_reactant_product_mapping: bool = False


def split_reaction_smiles(rxn_smiles: str) -> tuple[str, str]:
    parsed = parse_reaction_smiles(rxn_smiles)
    return parsed.reactants, parsed.products


def canonicalize_reaction_side(smiles: str) -> str:
    side_text = str(smiles).strip()
    if not side_text:
        raise ValueError("Reaction side SMILES must not be empty")

    mols: list[Chem.Mol] = []
    for fragment in side_text.split("."):
        if not fragment:
            raise ValueError(f"Reaction side contains an empty fragment: {smiles!r}")
        mol = Chem.MolFromSmiles(fragment, sanitize=False)
        if mol is None:
            raise ValueError(f"Invalid reaction side SMILES fragment: {fragment!r}")
        mols.append(_clean_mol(mol))
    return _mols_to_smiles(mols)


def reaction_has_atom_mapping(rxn_smiles: str) -> bool:
    return parse_reaction_smiles(rxn_smiles).has_reactant_product_mapping


def parse_reaction_smiles(rxn_smiles: str) -> ParsedReactionSmiles:
    rxn, sanitize_templates = _reaction_from_smiles(rxn_smiles)
    if sanitize_templates:
        _sanitize_reaction(rxn)
    else:
        _initialize_reaction(rxn)

    reactants = [_clean_mol(rxn.GetReactantTemplate(idx)) for idx in range(rxn.GetNumReactantTemplates())]
    agents = [_clean_mol(rxn.GetAgentTemplate(idx)) for idx in range(rxn.GetNumAgentTemplates())]
    products = [_clean_mol(rxn.GetProductTemplate(idx)) for idx in range(rxn.GetNumProductTemplates())]
    if not reactants or not products:
        raise ValueError(
            f"Reaction SMILES must contain at least one reactant and one product: {rxn_smiles!r}"
        )

    has_mapping = _sides_have_shared_atom_maps(reactants, products)
    if has_mapping:
        max_map = max(_atom_maps(reactants + products), default=0)
        agents = _assign_agent_maps_by_atom_order(agents, start=max_map)
    else:
        reactants = [_without_atom_maps(mol) for mol in reactants]
        agents = [_without_atom_maps(mol) for mol in agents]
        products = [_without_atom_maps(mol) for mol in products]

    agent_side = _mols_to_smiles(agents)
    return ParsedReactionSmiles(
        reactants=_mols_to_smiles(reactants + agents),
        products=_mols_to_smiles(products + agents),
        agents=agent_side,
        has_reactant_product_mapping=has_mapping,
    )


def _reaction_from_smiles(rxn_smiles: str) -> tuple[ReactionProtocol, bool]:
    rxn_text = str(rxn_smiles).strip()
    if not rxn_text:
        raise ValueError("Reaction SMILES must not be empty")
    try:
        reaction_from_smiles_obj = getattr(rdChemReactions, "ReactionFromSmiles", None)
        if callable(reaction_from_smiles_obj):
            reaction_from_smiles = cast(Callable[[str], object | None], reaction_from_smiles_obj)
            rxn = reaction_from_smiles(rxn_text)
            sanitize_templates = True
        else:
            reaction_from_smarts = cast(Callable[..., object | None], rdChemReactions.ReactionFromSmarts)
            rxn = reaction_from_smarts(rxn_text, useSmiles=True)
            sanitize_templates = False
    except Exception as exc:
        raise ValueError(f"Invalid reaction SMILES: {rxn_smiles!r}") from exc
    if rxn is None:
        raise ValueError(f"Invalid reaction SMILES: {rxn_smiles!r}")
    return cast(ReactionProtocol, rxn), sanitize_templates


def _sanitize_reaction(rxn: ReactionProtocol) -> None:
    try:
        sanitize_as_mols = cast(Callable[[object], None], rdChemReactions.SanitizeRxnAsMols)
        sanitize_rxn = cast(Callable[[object], None], rdChemReactions.SanitizeRxn)
        sanitize_as_mols(cast(object, rxn))
        sanitize_rxn(cast(object, rxn))
        rxn.Initialize()
    except Exception as exc:
        raise ValueError("RDKit could not sanitize reaction SMILES") from exc


def _initialize_reaction(rxn: ReactionProtocol) -> None:
    try:
        rxn.Initialize()
    except Exception as exc:
        raise ValueError("RDKit could not initialize reaction SMILES") from exc


def _clean_mol(mol: Chem.Mol) -> Chem.Mol:
    cleaned = Chem.Mol(mol)
    try:
        has_mapped_hydrogens = _has_mapped_hydrogens(cleaned)
        Chem.SanitizeMol(cleaned)
        if not has_mapped_hydrogens:
            cleaned = rdMolStandardize.Cleanup(cleaned)
            cleaned = Chem.RemoveHs(cleaned, sanitize=True)
    except Exception as exc:
        smiles = _mol_to_smiles(cleaned, canonical=False, isomeric_smiles=True)
        raise ValueError(f"Invalid molecule in reaction SMILES: {smiles!r}") from exc
    return cleaned


def _mols_to_smiles(mols: list[Chem.Mol]) -> str:
    if not mols:
        return ""
    parts = [_mol_to_smiles(mol, canonical=True, isomeric_smiles=True) for mol in mols]
    if any(_has_mapped_hydrogens(mol) for mol in mols):
        return ".".join(sorted(parts))
    return Chem.CanonSmiles(".".join(parts), useChiral=True)


def _mol_to_smiles(mol: Chem.Mol, *, canonical: bool, isomeric_smiles: bool) -> str:
    mol_to_smiles = cast(Callable[..., str], Chem.MolToSmiles)
    return mol_to_smiles(mol, canonical=canonical, isomericSmiles=isomeric_smiles)


def _sides_have_shared_atom_maps(reactants: list[Chem.Mol], products: list[Chem.Mol]) -> bool:
    reactant_maps = set(_atom_maps(reactants))
    product_maps = set(_atom_maps(products))
    return bool(reactant_maps and product_maps and reactant_maps.intersection(product_maps))


def _atom_maps(mols: list[Chem.Mol]) -> list[int]:
    maps: list[int] = []
    for mol in mols:
        maps.extend(atom.GetAtomMapNum() for atom in _mol_atoms(mol) if atom.GetAtomMapNum() > 0)
    return maps


def _has_mapped_hydrogens(mol: Chem.Mol) -> bool:
    return any(
        atom.GetAtomicNum() == 1 and atom.GetAtomMapNum() > 0
        for atom in _mol_atoms(mol)
    )


def _assign_agent_maps_by_atom_order(mols: list[Chem.Mol], *, start: int) -> list[Chem.Mol]:
    next_map = start
    mapped_mols: list[Chem.Mol] = []
    for mol in mols:
        mapped = Chem.Mol(mol)
        for atom in _mol_atoms(mapped):
            next_map += 1
            atom.SetAtomMapNum(next_map)
        mapped_mols.append(mapped)
    return mapped_mols


def _without_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    copied = Chem.Mol(mol)
    for atom in _mol_atoms(copied):
        atom.SetAtomMapNum(0)
    return copied


def _mol_atoms(mol: Chem.Mol) -> list[Chem.Atom]:
    return [mol.GetAtomWithIdx(atom_idx) for atom_idx in range(mol.GetNumAtoms())]
