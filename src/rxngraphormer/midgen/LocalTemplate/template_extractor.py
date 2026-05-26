"""
This python script is modified from rdchiral template extractor
https://github.com/connorcoley/rdchiral/blob/master/rdchiral/template_extractor.py
"""

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, overload

from rdkit import Chem
from rdkit.Chem import Atom, Bond, Mol
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

from .template_extract_extention import expand_atoms_to_use, get_special_groups, get_strict_smarts_for_special_atom


@dataclass
class ExtractorSettings:
    """Configuration settings for the template extractor."""

    verbose: bool = False
    use_stereo: bool = False
    include_neighbors: bool = True
    use_symbol: bool = False
    max_unmap: int = 5
    retro: bool = False
    remote: bool = True
    least_atom_num: int = 2


@overload
def clean_map_and_sort(
    smiles_list: list[str],
    no_clean_numbers: list[int],
    return_mols: Literal[False],
) -> list[str]: ...
@overload
def clean_map_and_sort(
    smiles_list: list[str],
    no_clean_numbers: list[int],
    return_mols: Literal[True],
) -> list[Mol]: ...
def clean_map_and_sort(
    smiles_list: list[str], no_clean_numbers: list[int] = [], return_mols: bool = False
) -> list[str] | list[Mol]:
    mols = []
    for smiles in smiles_list:
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        # logic commented out in original code preserved as comment
        # [atom.SetAtomMapNum(0) for atom in mol.GetAtoms() if atom.GetAtomMapNum() not in no_clean_numbers]
        if mol:
            mols.append(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

    mols = sorted(mols, key=lambda m: m.GetNumAtoms(), reverse=True)
    if return_mols:
        return mols
    else:
        return [Chem.MolToSmiles(mol) for mol in mols]


def replace_deuterated(smi: str) -> str:
    return re.sub(r"\[2H\]", r"[H]", smi)


def clear_mapnum(mol: Chem.Mol):
    [
        mol.GetAtomWithIdx(atom_idx).ClearProp("molAtomMapNumber")
        for atom_idx in range(mol.GetNumAtoms())
        if mol.GetAtomWithIdx(atom_idx).HasProp("molAtomMapNumber")
    ]
    return mol


def get_tagged_atoms_from_mols(mols: list[Mol]) -> tuple[list[Atom], list[str]]:
    """Takes a list of RDKit molecules and returns total list of
    atoms and their tags"""
    atoms = []
    atom_tags = []
    for mol in mols:
        new_atoms, new_atom_tags = get_tagged_atoms_from_mol(mol)
        atoms += new_atoms
        atom_tags += new_atom_tags
    return atoms, atom_tags


def get_tagged_atoms_from_mol(mol: Mol) -> tuple[list[Atom], list[str]]:
    """Takes an RDKit molecule and returns list of tagged atoms and their
    corresponding numbers"""
    atoms = []
    atom_tags = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.HasProp("molAtomMapNumber"):
            atoms.append(atom)
            atom_tags.append(str(atom.GetProp("molAtomMapNumber")))
    return atoms, atom_tags


def bond_to_smarts(bond: Bond) -> str:
    """This function takes an RDKit bond and creates a label describing
    the most important attributes"""
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp("molAtomMapNumber"):
        a1_label += bond.GetBeginAtom().GetProp("molAtomMapNumber")
    if bond.GetEndAtom().HasProp("molAtomMapNumber"):
        a2_label += bond.GetEndAtom().GetProp("molAtomMapNumber")
    atoms = sorted([a1_label, a2_label])
    bond_smarts = bond.GetSmarts()  # type: ignore
    if bond_smarts == "":
        bond_smarts = "-"

    return f"{atoms[0]}{bond_smarts}{atoms[1]}"


def atom_neighbors(atom: Atom) -> list[int]:
    # Helper needed for atoms_are_different (inferred from original code usage)
    # The original code used this but didn't define it in the snippet,
    # assuming standard neighbor map num extraction.
    neighbor_maps = []
    for n in atom.GetNeighbors():
        if n.HasProp("molAtomMapNumber"):
            neighbor_maps.append(int(n.GetProp("molAtomMapNumber")))
        else:
            neighbor_maps.append(0)
    return sorted(neighbor_maps)


def atoms_are_different(atom1: Atom, atom2: Atom, settings: ExtractorSettings) -> bool:
    """Compares two RDKit atoms based on basic properties"""

    if atom1.GetAtomicNum() != atom2.GetAtomicNum():
        return True  # must be true for atom mapping

    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons():
        return True
    if settings.remote:
        if atom1.GetFormalCharge() != atom2.GetFormalCharge():
            return True
        # may be wrong information due to wrong atom mapping
        if atom1.GetTotalNumHs() != atom2.GetTotalNumHs():
            return True

    # add or break bonds
    if atom_neighbors(atom1) != atom_neighbors(atom2):
        return True

    # change bonds
    bonds1 = sorted([bond_to_smarts(bond) for bond in atom1.GetBonds()])
    bonds2 = sorted([bond_to_smarts(bond) for bond in atom2.GetBonds()])
    if bonds1 != bonds2:
        return True

    return False


def find_map_num(mol: Mol, mapnum: str) -> tuple[int, Atom]:
    return [
        (atom_id, mol.GetAtomWithIdx(atom_id))
        for atom_id in range(mol.GetNumAtoms())
        if mol.GetAtomWithIdx(atom_id).HasProp("molAtomMapNumber")
        and mol.GetAtomWithIdx(atom_id).GetProp("molAtomMapNumber") == str(mapnum)
    ][0]


def get_tetrahedral_atoms(reactants: list[Mol], products: list[Mol]) -> list[tuple[str, Atom, Atom]]:
    tetrahedral_atoms = []
    for reactant in reactants:
        for ar_id in range(reactant.GetNumAtoms()):
            ar = reactant.GetAtomWithIdx(ar_id)
            if not ar.HasProp("molAtomMapNumber"):
                continue
            atom_tag = ar.GetProp("molAtomMapNumber")
            for product in products:
                try:
                    (ip, ap) = find_map_num(product, atom_tag)
                    if (
                        ar.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
                        or ap.GetChiralTag() != ChiralType.CHI_UNSPECIFIED
                    ):
                        tetrahedral_atoms.append((atom_tag, ar, ap))
                except IndexError:
                    pass
    return tetrahedral_atoms


def set_isotope_to_equal_mapnum(mol: Mol) -> None:
    for atom_id in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(atom_id)
        if a.HasProp("molAtomMapNumber"):
            a.SetIsotope(int(a.GetProp("molAtomMapNumber")))


def get_frag_around_tetrahedral_center(mol: Mol, idx: int, settings: ExtractorSettings) -> str:
    """Builds a MolFragment using neighbors of a tetrahedral atom,
    where the molecule has already been updated to include isotopes"""
    ids_to_include = [idx]
    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
        ids_to_include.append(neighbor.GetIdx())
    symbols = [
        f"[{mol.GetAtomWithIdx(atom_id).GetIsotope()}{mol.GetAtomWithIdx(atom_id).GetSymbol()}]"
        if mol.GetAtomWithIdx(atom_id).GetIsotope() != 0
        else f"[#{mol.GetAtomWithIdx(atom_id).GetAtomicNum()}]"
        for atom_id in range(mol.GetNumAtoms())
    ]
    return Chem.MolFragmentToSmiles(
        mol,
        ids_to_include,
        isomericSmiles=settings.use_stereo,
        atomSymbols=symbols,
        allBondsExplicit=True,
        allHsExplicit=True,
    )


def check_tetrahedral_centers_equivalent(atom1: Atom, atom2: Atom, settings: ExtractorSettings) -> bool:
    """Checks to see if tetrahedral centers are equivalent in
    chirality, ignoring the ChiralTag. Owning molecules of the
    input atoms must have been Isotope-mapped"""
    atom1_frag = get_frag_around_tetrahedral_center(atom1.GetOwningMol(), atom1.GetIdx(), settings)
    atom1_neighborhood = Chem.MolFromSmiles(atom1_frag, sanitize=False)
    for matched_ids in atom2.GetOwningMol().GetSubstructMatches(atom1_neighborhood, useChirality=True):
        if atom2.GetIdx() in matched_ids:
            return True
    return False


def clear_isotope(mol: Mol):
    [mol.GetAtomWithIdx(atom_id).SetIsotope(0) for atom_id in range(mol.GetNumAtoms())]


def get_changed_atoms(
    reactants: list[Mol], products: list[Mol], settings: ExtractorSettings
) -> tuple[list[Atom], list[str], int]:
    """Looks at mapped atoms in a reaction and determines which ones changed"""

    err = 0
    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mols(products)

    if settings.verbose:
        print(f"Products contain {len(prod_atoms)} tagged atoms")
        print(f"Products contain {len(set(prod_atom_tags))} unique atom numbers")

    reac_atoms, reac_atom_tags = get_tagged_atoms_from_mols(reactants)
    if len(set(prod_atom_tags)) != len(set(reac_atom_tags)):
        if settings.verbose:
            print("warning: different atom tags appear in reactants and products")
    if len(prod_atoms) != len(reac_atoms):
        if settings.verbose:
            print("warning: total number of tagged atoms differ, stoichometry != 1?")

    # Find differences
    changed_atoms = []  # actual reactant atom species
    changed_atom_tags = []  # atom map numbers of those atoms

    # Product atoms that are different from reactant atom equivalent
    for i, prod_tag in enumerate(prod_atom_tags):
        for j, reac_tag in enumerate(reac_atom_tags):
            if reac_tag != prod_tag:
                continue
            if reac_tag not in changed_atom_tags:  # don't bother comparing if we know this atom changes
                # If atom changed, add
                if atoms_are_different(prod_atoms[i], reac_atoms[j], settings):
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break
                # If reac_tag appears multiple times, add (need for stoichometry > 1)
                if prod_atom_tags.count(reac_tag) > 1:
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break

    # Reactant atoms that do not appear in product (tagged leaving groups)
    for j, reac_tag in enumerate(reac_atom_tags):
        if reac_tag not in changed_atom_tags:
            if reac_tag not in prod_atom_tags:
                changed_atoms.append(reac_atoms[j])
                changed_atom_tags.append(reac_tag)

    [clear_isotope(reactant) for reactant in reactants]
    [clear_isotope(product) for product in products]

    if settings.verbose:
        print(f"{len(changed_atom_tags)} tagged atoms in reactants change 1-atom properties")
        for smarts in [atom.GetSmarts() for atom in changed_atoms]:
            print(f"  {smarts}")

    return changed_atoms, changed_atom_tags, err


def template_scorer(template: str, atom_dict: dict[str, dict[str, int]]) -> float:
    score = 0.0
    for s, b in enumerate(["-", ":", "=", "#"]):
        score += template.count(b) * (s + 1)
    for n in re.findall(r"\:([0-9]+)\]", template):
        if n in atom_dict:
            score += 0.1 * atom_dict[n]["charge"] + 0.01 * atom_dict[n]["Hs"]
    return score


def inv_temp(template: str) -> str:  # for more canonical representation
    symbols = re.findall(r"\[[a-zA-Z@]+\:.*?\]", template)
    nums = [int(n) for n in re.findall(r"\[[a-zA-Z@]+\:(.*?)\]", template)]
    if len(nums) not in [2, 3] or "]1" in template:
        return template
    if nums[0] < nums[1]:
        return template
    if len(nums) == 3:
        if nums[0] < nums[2]:
            return template
    bonds = [""] + [sorted(bond)[1] for bond in re.findall(r"]([-=#:])|]1([-=#:])", template)]
    return "".join([f"{a}{b}" for a, b in zip(symbols[::-1], bonds[::-1])])


def inverse_template(template: str) -> str:  # for more canonical representation
    n_atoms = sum([Chem.MolFromSmarts(smarts).GetNumAtoms() for smarts in template.split(">>")])
    labels = re.findall(r"(\[[a-zA-Z@]+\:.*?\])", template)
    if n_atoms > len(labels):  # include leaving group
        return template

    def score_bonds(bonds):
        bond_dict = {b: str(i + 1) for i, b in enumerate(["-", ":", "=", "#"])}
        return eval("".join([bond_dict[b] for b in bonds]))

    if "]1" in template:
        ring_template = True
    else:
        ring_template = False
    bonds1 = [sorted(bond)[1] for bond in re.findall(r"]([-=#:])|]1([-=#:])", template)]
    bonds2 = bonds1[::-1]
    if len(bonds1) == 0 or ")" in template or score_bonds(bonds1) <= score_bonds(bonds2):
        return template

    labels = re.findall(r"\[.*?]", template)[::-1]
    inv_template = labels[0]
    for i in range(len(bonds2)):
        if ring_template:
            if i == 0:
                inv_template += "1"
            if i + 1 == len(labels):
                inv_template += bonds2[0]
                inv_template += "1"
            else:
                inv_template += bonds2[i + 1]
                inv_template += labels[i + 1]
        else:
            inv_template += bonds2[i]
            inv_template += labels[i + 1]
    return inv_template


def canonicalize_smarts(smarts: str, settings: ExtractorSettings) -> str:  # for more canonical representation
    # This logic was mostly commented out or conditional in original, preserving structure
    if settings.use_symbol:
        # Note: The original code returned immediately if USE_ATOM_SYMBOL was true,
        # but seemingly also did logic below. Assuming early return based on original logic structure.
        return smarts

    preserved_info = {"[#0:{}]".format(a.split(":")[-1].split("]")[0]): a for a in re.findall(r"\[.*?]", smarts)}
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmarts(smarts))
        smarts_ = Chem.MolToSmarts(Chem.MolFromSmiles(smiles))
    except:  # noqa: E722
        return smarts
    if "(" not in smarts_:
        smarts = smarts_
        for k, v in preserved_info.items():
            smarts = smarts.replace(k, v)
    return smarts


def sort_template(
    transform: str, atom_dict: dict[str, dict[str, int]], settings: ExtractorSettings
) -> str:  # for more canonical representation
    transform = (
        transform.split(">>")[0][1:-1].replace(").(", ".") + ">>" + transform.split(">>")[1][1:-1].replace(").(", ".")
    )
    templates = []
    for tt in transform.split(">>"):
        ts = []
        for smarts in sorted(tt.split("."), key=lambda s: template_scorer(s, atom_dict)):
            try:
                ts.append(inverse_template(canonicalize_smarts(smarts, settings)))
            except:  # noqa: E722
                ts.append(canonicalize_smarts(smarts, settings))
        templates.append(".".join(ts))
    return ">>".join(templates)


def permutations(template: str) -> list[list[str]]:
    n_atoms = sum([Chem.MolFromSmarts(smarts).GetNumAtoms() for smarts in template.split(">>")])
    labels = re.findall(r"(\[[a-zA-Z@]+\:.*?\])", template)
    if len(labels) == 1 or "(" in template or n_atoms > len(labels):  # include leaving group
        return [labels]
    charges = re.findall(r"\;(.+?[0-9]+)\:", template)
    bonds = re.findall(r"\]([-=#:])\[", template)
    if "".join(bonds) != "".join(bonds[::-1]) or "".join(charges) != "".join(charges[::-1]):
        return [labels]
    return [labels, labels[::-1]]


def enumerate_mapping(transform: str) -> list[list[str]]:
    r_permutes = []
    p_permutes = []
    for i, templates in enumerate(transform.split(">>")):
        grow_template = None
        for template in templates.split("."):
            pert_template = permutations(template)
            if grow_template is None:
                grow_template = pert_template
            else:
                growed_template = []
                for t in grow_template:
                    for p in pert_template:
                        growed_template.append(t + p)
                grow_template = growed_template
        if i == 0:
            r_permutes = grow_template if grow_template else []
        else:
            p_permutes = grow_template if grow_template else []

    t_permutes = []
    for r in r_permutes:
        for p in p_permutes:
            t_permutes.append(r + p)
    return t_permutes


def reassign_atom_mapping(
    transform: str, atom_dict: dict[str, dict[str, int]], settings: ExtractorSettings
) -> tuple[str, dict[str, str]]:
    if not settings.retro:
        transform = ">>".join(transform.split(">>")[::-1])
    transform = sort_template(transform, atom_dict, settings)

    p_labels = enumerate_mapping(transform)
    templates = set()
    templates_sort = {}
    replacement_dicts = {}
    for all_labels in p_labels:
        # Define list of replacements which matches all_labels *IN ORDER*
        replacements = []
        replacement_dict_symbol = {}
        replacement_dict = {}
        counter = 1
        for label in all_labels:  # keep in order! this is important
            atom_map = label.split(":")[1].split("]")[0]
            if atom_map not in replacement_dict:
                replacement_dict_symbol[label] = "{}:{}]".format(
                    label.split(":")[0],
                    counter,
                )
                replacement_dict[atom_map] = str(counter)
                counter += 1
            else:
                replacement_dict_symbol[label] = "{}:{}]".format(
                    label.split(":")[0],
                    replacement_dict[atom_map],
                )
            replacements.append(replacement_dict_symbol[label])

        # Perform replacements in order
        transform_newmaps = re.sub(r"\[[a-zA-Z@]+\:.*?\]", lambda match: replacements.pop(0), transform)
        if settings.retro:
            pass
        else:
            transform_newmaps = ">>".join(transform_newmaps.split(">>")[::-1])

        templates.add(transform_newmaps)
        templates_sort[transform_newmaps] = "".join(re.findall(r"\[[a-zA-Z@]+\:.*?\]", transform_newmaps))
        replacement_dicts[transform_newmaps] = replacement_dict

    transform_newmaps = sorted(list(templates), key=lambda t: templates_sort[t])[0]
    replacement_dict = replacement_dicts[transform_newmaps]
    return transform_newmaps, replacement_dict


def get_strict_smarts_for_atom(atom: Atom, settings: ExtractorSettings, special_group: bool = False) -> str:
    """
    For an RDkit atom object, generate a SMARTS pattern that
    matches the atom as strictly as possible
    """

    symbol = atom.GetSymbol()
    if atom.GetAtomMapNum() <= 100:
        general_symbol = symbol
    else:
        if symbol in ["F", "Cl", "Br", "I"]:
            general_symbol = "X"
        else:
            general_symbol = symbol

    if settings.use_symbol or special_group:
        symbol = f"[{atom.GetSymbol()}:{atom.GetAtomMapNum()}]"
        if "H" in symbol and "Hg" not in symbol:
            symbol = symbol.replace("H", "")
        if atom.GetIsAromatic():
            symbol = symbol.lower()
    else:
        symbol = f"[{general_symbol}:{atom.GetAtomMapNum()}]"
        if atom.GetIsAromatic():
            symbol = symbol.lower()

    if atom.GetSymbol() == "H":
        symbol = "[#1]"

    if "[" not in symbol:
        symbol = "[" + symbol + "]"

    return symbol


def get_fragments_for_changed_atoms(
    mols: list[Mol],
    changed_atom_tags: list[str],
    settings: ExtractorSettings,
    category: str = "reactant",
    special_atoms: list[str] = [],
) -> tuple[str, bool, bool, list[Mol], list[str]]:
    fragments = ""
    mols_changed = []
    mols_special = []
    for mol in mols:
        # Initialize list of replacement symbols (updated during expansion)
        symbol_replacements = []

        # Build list of atoms to use
        atoms_to_use = []

        for atom in mol.GetAtoms():  # type: ignore
            # Check self (only tagged atoms)
            if ":" in atom.GetSmarts():
                atom_map = atom.GetSmarts().split(":")[1][:-1]
                if atom_map in changed_atom_tags + special_atoms:
                    atoms_to_use.append(atom.GetIdx())
                    if atom_map in changed_atom_tags:
                        symbol = get_strict_smarts_for_atom(atom, settings)
                    else:
                        symbol = get_strict_smarts_for_special_atom(atom)
                    #                     if (category == 'product' and settings.retro) or (category == 'reactant' and not settings.retro):
                    #                         symbol = symbol.replace('@', '') # remove chiral information in smarts
                    if symbol != atom.GetSmarts():
                        symbol_replacements.append((atom.GetIdx(), symbol))
                    continue

        # Are we looking for special reactive groups? (reactants only)
        # Fully define leaving groups if this is deprotection reaction (len(atoms_to_use) == 2)
        if category == "reactant":
            if settings.include_neighbors:
                groups = get_special_groups(mol)
                atoms_to_use, symbol_replacements, mol = expand_atoms_to_use(
                    mol,
                    atoms_to_use,
                    groups=groups,
                    symbol_replacements=symbol_replacements,
                )
                for atom_idx in atoms_to_use:
                    atom_map = str(mol.GetAtomWithIdx(atom_idx).GetAtomMapNum())
                    if atom_map not in changed_atom_tags:
                        special_atoms.append(atom_map)

            for atom in mol.GetAtoms():  # type: ignore
                if not atom.HasProp("molAtomMapNumber"):  # LG in reactant is also mapped in foward synthesis
                    atoms_to_use.append(atom.GetIdx())

        else:
            groups = []
        mols_special.append(mol)

        # Define new symbols based on symbol_replacements
        symbols = [atom.GetSmarts() for atom in mol.GetAtoms()]  # type: ignore
        for i, symbol in symbol_replacements:
            symbols[i] = symbol
        if not atoms_to_use:
            continue

        mol_copy = deepcopy(mol)
        [x.ClearProp("molAtomMapNumber") for x in mol_copy.GetAtoms()]  # type: ignore
        this_fragment = Chem.MolFragmentToSmiles(
            mol_copy,
            atoms_to_use,
            atomSymbols=symbols,
            allHsExplicit=True,
            isomericSmiles=settings.use_stereo,
            allBondsExplicit=True,
        )
        fragments += "(" + this_fragment + ")."
        mols_changed.append(Chem.MolToSmiles(clear_mapnum(Chem.MolFromSmiles(Chem.MolToSmiles(mol, True))), True))

    # auxiliary template information: is this an intramolecular reaction or dimerization?
    intra_only = 1 == len(mols_changed)
    dimer_only = (1 == len(set(mols_changed))) and (len(mols_changed) == 2)
    return fragments[:-1], intra_only, dimer_only, mols_special, special_atoms


def canonicalize_transform(
    transform: str, atom_dict: dict[str, dict[str, int]], settings: ExtractorSettings
) -> tuple[str, dict[str, str]]:
    """This function takes an atom-mapped SMARTS transform and
    converts it to a canonical form by, if nececssary, rearranging
    the order of reactant and product templates and reassigning
    atom maps."""

    transform_reordered = ">>".join([canonicalize_template(x) for x in transform.split(">>")])
    return reassign_atom_mapping(transform_reordered, atom_dict, settings)


def canonicalize_template(template: str) -> str:
    """This function takes one-half of a template SMARTS string
    (i.e., reactants or products) and re-orders them based on
    an equivalent string without atom mapping."""

    # Strip labels to get sort orders
    template_nolabels = re.sub(r"\:[0-9]+\]", "]", template)

    # Split into separate molecules *WITHOUT wrapper parentheses*
    template_nolabels_mols = template_nolabels[1:-1].split(").(")
    template_mols = template[1:-1].split(").(")

    # Split into fragments within those molecules
    for i in range(len(template_mols)):
        nolabel_mol_frags = template_nolabels_mols[i].split(".")
        mol_frags = template_mols[i].split(".")

        # Get sort order within molecule, defined WITHOUT labels
        sortorder = [j[0] for j in sorted(enumerate(nolabel_mol_frags), key=lambda x: x[1])]

        # Apply sorting and merge list back into overall mol fragment
        template_nolabels_mols[i] = ".".join([nolabel_mol_frags[j] for j in sortorder])
        template_mols[i] = ".".join([mol_frags[j] for j in sortorder])

    # Get sort order between molecules, defined WITHOUT labels
    sortorder = [j[0] for j in sorted(enumerate(template_nolabels_mols), key=lambda x: x[1])]

    # Apply sorting and merge list back into overall transform
    template = "(" + ").(".join([template_mols[i] for i in sortorder]) + ")"

    return template


def extend_atom_tag(reactant: Mol, max_num: int, settings: ExtractorSettings) -> tuple[bool, int]:
    untagged_neighbors = []
    is_reagent = True
    # check atom map
    for atom in reactant.GetAtoms():  # type: ignore
        if atom.GetAtomMapNum() == 0:
            pass
        else:
            is_reagent = False

    if is_reagent or settings.retro:
        return is_reagent, max_num

    # tag the untag neighbors
    for atom in reactant.GetAtoms():  # type: ignore
        if atom.GetAtomMapNum() == 0:
            continue
        for n_atom in atom.GetNeighbors():
            if n_atom.GetAtomMapNum() == 0:
                untagged_neighbors.append(n_atom.GetIdx())

    for idx in untagged_neighbors:
        max_num += 1
        atom = reactant.GetAtomWithIdx(idx)
        atom.SetAtomMapNum(max_num)
    return is_reagent, max_num


def split_reagents(reaction: dict[str, str], settings: ExtractorSettings) -> tuple[list[str], list[str], list[str]]:
    rs, ps = (
        replace_deuterated(reaction["reactants"]).split("."),
        replace_deuterated(reaction["products"]).split("."),
    )
    largest_p = [Chem.MolFromSmiles(smiles).GetNumAtoms() for smiles in ps if smiles not in rs and smiles != ""]
    if len(largest_p) == 0:
        return [], [], []
    least_atom_n = min([max(largest_p), settings.least_atom_num])
    reagents = [smiles for smiles in rs if smiles in ps]
    ps = [smiles for smiles in ps if Chem.MolFromSmiles(smiles).GetNumAtoms() >= least_atom_n]
    return (
        [r for r in rs if r not in reagents],
        [p for p in ps if p not in reagents],
        reagents,
    )


def add_atom_num(reaction: dict[str, str]) -> None:
    rs, ps = reaction["reactants"], reaction["products"]
    new_rs = []

    max_num = 100
    for r in rs.split("."):
        rmol = Chem.MolFromSmiles(r)
        if r in ps:  # reagent
            [atom.SetAtomMapNum(0) for atom in rmol.GetAtoms()]  # type: ignore
        if sum([atom.GetAtomMapNum() > 0 for atom in rmol.GetAtoms()]) > 0:  # type: ignore
            for atom in rmol.GetAtoms():  # type: ignore
                if atom.GetAtomMapNum() == 0:
                    max_num += 1
                    atom.SetAtomMapNum(max_num)
        new_rs.append(Chem.MolToSmiles(rmol))
    reaction["reactants"] = ".".join(new_rs)
    return


def extract_from_reaction(
    reaction: dict[str, str] | str,
    setting: dict[str, Any] | ExtractorSettings | None = None,
) -> tuple[str, str, dict[str, str]] | None:
    # Handle settings input flexibility (dict or dataclass)
    if setting is None:
        settings = ExtractorSettings()
    elif isinstance(setting, dict):
        settings = ExtractorSettings(**setting)
    else:
        settings = setting  # Already a dataclass instance

    if isinstance(reaction, str):
        reaction = {
            "reactants": reaction.split(">>")[0],
            "products": reaction.split(">>")[1],
        }

    #     add_atom_num(reaction)
    reactants_list, products_list, reagents_list = split_reagents(reaction, settings)

    if len(products_list) == 0:
        return None
    product_maps = [
        atom.GetAtomMapNum()
        for products in products_list
        for atom in Chem.MolFromSmiles(products).GetAtoms()  # type: ignore
    ]
    products = clean_map_and_sort(products_list, product_maps, return_mols=True)
    reactants_ = clean_map_and_sort(reactants_list, product_maps, return_mols=True)

    max_num = 100
    reactants = []
    for reactant in reactants_:
        is_reagent, max_num = extend_atom_tag(reactant, max_num, settings)
        if is_reagent:
            reagents_list.append(Chem.MolToSmiles(reactant))
        else:
            reactants.append(reactant)
    # if rdkit cant understand molecule, return
    if None in reactants:
        return None
    if None in products:
        return None
    # try to sanitize molecules
    try:
        for i in range(len(reactants)):
            reactants[i] = Chem.RemoveHs(reactants[i])  # *might* not be safe
        for i in range(len(products)):
            products[i] = Chem.RemoveHs(products[i])  # *might* not be safe
        [Chem.SanitizeMol(mol) for mol in reactants + products]  # redundant w/ RemoveHs
        [mol.UpdatePropertyCache() for mol in reactants + products]
    except Exception as e:
        if settings.verbose:
            print(e)
            print("Could not load SMILES or sanitize")
        return None

    if None in reactants + products:
        if settings.verbose:
            print("Could not parse all molecules in reaction, skipping")
        return None

    # Calculate changed atoms
    changed_atoms, changed_atom_tags, err = get_changed_atoms(reactants, products, settings)

    if err:
        if settings.verbose:
            print("Could not get changed atoms")
        return None
    if not changed_atom_tags:
        if settings.verbose:
            print("No atoms changed?")
        return None

    try:
        reactant_fragments, _, _, reactants, special_atoms = get_fragments_for_changed_atoms(
            reactants, changed_atom_tags, settings, "reactant", []
        )
        product_fragments, _, _, _, _ = get_fragments_for_changed_atoms(
            products, changed_atom_tags, settings, "product", special_atoms
        )
        reaction["reactants"] = ".".join([Chem.MolToSmiles(m, canonical=False) for m in reactants])
    except ValueError as e:
        if settings.verbose:
            print(e)
        return None

    # Put together and canonicalize (as best as possible)
    rxn_string = reactant_fragments + ">>" + product_fragments
    if settings.verbose:
        print(rxn_string)
    atom_dict = {
        str(atom.GetAtomMapNum()): {
            "charge": atom.GetFormalCharge(),
            "Hs": atom.GetNumExplicitHs(),
        }
        for atom in changed_atoms
    }

    rxn_canonical, replacement_dict = canonicalize_transform(rxn_string, atom_dict, settings)

    reactants_string = rxn_canonical.split(">>")[0]
    products_string = rxn_canonical.split(">>")[1]

    # Used for validation logic or debug but unused in final return in original script logic
    # products_smiles = ".".join([Chem.MolToSmiles(p) for p in products])
    # reactants_smiles = ".".join([Chem.MolToSmiles(r) for r in reactants])

    try:
        products_string = canonicalize_smarts(products_string, settings)
        reactants_string = canonicalize_smarts(reactants_string, settings)
    except:  # noqa: E722
        pass

    canonical_template = f"{reactants_string}>>{products_string}"
    rxn_obj = ReactionFromSmarts(canonical_template)
    if rxn_obj.Validate()[1] != 0:
        if settings.verbose:
            print("Could not validate reaction successfully")
            print(f"canonical_template: {canonical_template}")
            print(reaction["reactants"] + ">>" + reaction["products"])
        return None

    return (
        reaction["reactants"] + ">>" + reaction["products"],
        canonical_template,
        replacement_dict,
    )
