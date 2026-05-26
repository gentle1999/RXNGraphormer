################### From https://github.com/snu-micc/MechFinder ###################
import importlib.resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # Added missing import
from rdkit import Chem

# Assuming these are available in the package context
from .LocalTemplate.template_extractor import extract_from_reaction
from .utils.criterions import find_map_num
from .utils.utils import (
    clean_leaving_mapping,
    get_acidic_EWG_map_nums,
    get_EWG_connected_atom,
    get_map_numbered_matches,
    get_neighbor_props,
)

# Type aliases for clarity
RxnString = str
TemplateString = str
AtomMapIdx = int
ReplacementDict = dict[int | str, int]
PathAtom = int | float | str
PathStep = PathAtom | list[PathAtom]
PathValue = PathAtom | list[int | float]
ElectronPath = tuple[tuple[int | float, ...], ...]

out_of_scope_mechanisms: list[str] = [
    "nitro_reduction",
    "alkene_reduction",
    "hydrogenative_deprotection",
    "radical_reaction",
    "aromatic_dehalogenation",
    "Heck",
    "alkyne_reduction",
    "Stille_coupling",
    "protodesilylation",
    "Grignard_reagent_prep",
    "Suzuki_coupling",
    "Negishi_coupling",
    "one_pot_Grignard_synthesis",
    "one_pot_Weinreb_ketone_synthesis",
    "Ullmann",
    "Bouveault_aldehyde_synthesis",
    "Huisgen_cycloaddition",
    "Rosenmund_von_Braun",
    "catalytic_amination",
    "catalytic_coupling",
    "Barton_McCombie_deoxygenation",
    "radical_dehalogenation",
]


wrong_atom_mapped_reactions: list[str] = [
    "wrong_atom_mapped_esterification",
    "wrong_atom_mapped_ester_hydrolysis",
    "wrong_atom_mapped_reduction",
    "wrong_atom_mapped_(hemi)acetal_hydrolysis",
    "wrong_atom_mapped_(hemi)acetal_formation",
    "wrong_atom_mapped_hydroboration_oxidation",
    "wrong_atom_mapped_alcohol_condensation",
    "wrong_atom_mapped_amide_formation",
    "wrong_atom_mapped_carboxylic_acid_reduction",
    "wrong_atom_mapped_Williamson_ether_synthesis",
    "wrong_atom_mapped_Friedel_Crafts_acylation",
    "wrong_atom_mapped_carboxylic_acid_LAH_reduction",
]


def add_reagent(
    rxn: RxnString,
    reagents: dict[str, str | int] | None,
    replacement_dict: ReplacementDict,
) -> tuple[RxnString, ReplacementDict]:
    if reagents:
        reactants, products = rxn.split(">>")
        reagent_n = 300
        for reagent, template_no in reagents.items():
            if template_no == "_":
                ll = reagent.split(":")
                for i, part in enumerate(ll):
                    if part[0].isdigit():
                        reagent_n += 1
                        j = 1
                        while part[j].isdigit():
                            j += 1
                        ll[i] = str(reagent_n) + part[j:]
                        replacement_dict[int(part[:j])] = reagent_n
                reagent_updated = ":".join(ll)
                reactants = f"{reactants}.{reagent_updated}"
            elif reagent not in reactants.split("."):
                reagent_n += 1
                reactants = f"{reactants}.[{reagent}:{reagent_n}]"
                replacement_dict[template_no] = reagent_n
        rxn = reactants + ">>" + products
    return rxn, replacement_dict


def change_atom_map(replacement_dict: ReplacementDict, template_path: list[PathStep]) -> tuple[PathValue, ...]:
    map_path: list[PathValue] = []
    for path in template_path:
        if not isinstance(path, list):  # if it's some atom (int) or hydrogen (float) or 'xl'
            scalar_path: PathAtom = path
            scalar_path_val = replacement_dict[int(scalar_path)] if isinstance(scalar_path, (int, str)) else replacement_dict[int(scalar_path)] + 0.1
            map_path.append(scalar_path_val)
        else:
            list_path: list[PathAtom] = path
            list_path_val = [
                replacement_dict[p] if isinstance(p, (int, str)) else replacement_dict[int(p)] + 0.1
                for p in list_path
            ]
            map_path.append(list_path_val)
    return tuple(map_path)


def adjust_template_atom_map(adjust_dict: dict[int, int], mech_path: list[PathStep]) -> tuple[PathValue, ...]:
    template_path: list[PathValue] = []
    for path in mech_path:
        if not isinstance(path, list):  # if it's some atom (int) or hydrogen (float) or 'xl'
            path_atom: PathAtom = path
            scalar_path_val = (
                adjust_dict[int(path_atom)] if isinstance(path_atom, (int, str)) else adjust_dict[int(path_atom)] + 0.1
            )
            template_path.append(scalar_path_val)
        else:
            path_atoms: list[PathAtom] = path
            list_path_val = [
                adjust_dict[int(p)] if isinstance(p, (int, str)) else adjust_dict[int(p)] + 0.1 for p in path_atoms
            ]
            template_path.append(list_path_val)
    return tuple(template_path)


def swap_map_nums(
    smiles: str,
    replacement_dict: ReplacementDict,
    temp_no_1: int,
    map_num_1: int,
    temp_no_2: int,
    map_num_2: int,
) -> str:
    ll = smiles.split(":")
    for i, part in enumerate(ll):
        if part[0].isdigit():
            j = 1
            while part[j].isdigit():
                j += 1
            if part[:j] == str(map_num_1):
                ll[i] = str(map_num_2) + part[j:]
                replacement_dict[temp_no_1] = map_num_2
            elif part[:j] == str(map_num_2):
                ll[i] = str(map_num_1) + part[j:]
                replacement_dict[temp_no_2] = map_num_1
    return ":".join(ll)


def replace_dict(rxn: RxnString, replacement_dict: ReplacementDict, return_idx: bool = False) -> dict[int, int]:
    if return_idx:
        reactants, _ = rxn.split(">>")
        rmol = Chem.MolFromSmiles(reactants)
        map2idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rmol.GetAtoms()}  # type: ignore
        return {int(v): map2idx[int(k)] for k, v in replacement_dict.items()}
    else:
        return {int(v): int(k) for k, v in replacement_dict.items()}


def neutralize_charge(rxn: RxnString) -> RxnString:
    charged_atoms = {"NH3+": "NH2", "NH4+": "NH3", "O-:": "OH:"}
    for k, v in charged_atoms.items():
        rxn = rxn.replace(k, v)
    return rxn


def build_ext_dict(rxn: RxnString, replacement_dict: ReplacementDict, ext_info: dict[str, Any]) -> dict[str, int]:
    rmol = Chem.MolFromSmiles(rxn.split(">>")[0])
    temp_map_num, fg, ext_strings = (
        ext_info["target_map_num"],
        ext_info["FG"],
        ext_info["ext_strings"],
    )
    atom_map_num = replacement_dict[temp_map_num]
    target_idx = find_map_num(rmol, atom_map_num)[0]  # index of atom that is connected to leaving group

    ext_dict: dict[str, int] = dict()

    if fg in "aromatic_ring":  # case where LRT extension applies to the aromatic ring atoms beyond target_map_num
        # get idx of EWG-attached atom along with EWG indices (of double bond in EWG)
        ring_atom, EWG_db_indices = get_EWG_connected_atom(
            rmol, target_idx
        )  # ring_atom might be EWG_connected_atom_idx (int) if SNAr_ortho; or ortho, meta, para ring atoms (list) if SNAr_para
        EWG_indices = [ring_atom] + EWG_db_indices if isinstance(ring_atom, int) else ring_atom + EWG_db_indices

        for i, s in enumerate(ext_strings):
            ext_dict[s] = rmol.GetAtomWithIdx(
                EWG_indices[i]
            ).GetAtomMapNum()  # 'xl1' -> ortho atom; 'xl2' & 'xl3' -> EWG db
    elif fg == "alpha_EWG":
        EWG_map_nums = get_acidic_EWG_map_nums(rmol, target_idx)
        for i, s in enumerate(ext_strings):
            ext_dict[s] = EWG_map_nums[i]
    else:  # simple fg case, e.g., fg = 'C=O'
        fg_matches = get_map_numbered_matches(rmol, fg)
        match: tuple[int, ...] = ()
        for m in fg_matches:
            match = m  # default assignment
            if match[0] in get_neighbor_props(rmol, atom_map_num, "map_num"):
                break

        if match:
            for i, s in enumerate(ext_strings):
                ext_dict[s] = match[i]
    return ext_dict


def replace_xl(mech_pathway: list[tuple[Any, ...]], ext_dict: dict[str, int]) -> list[tuple[Any, ...]]:
    """Given final, actual atom map numbered mech_pathway with 'xl' strings to be replaced"""
    updated_mech_pathway = []
    for attack in mech_pathway:
        updated_attack = []
        for path in attack:
            if not isinstance(path, list):  # if it's some atom (int) or hydrogen (float) or 'xl'
                path_res = ext_dict[path] if isinstance(path, str) else path
            else:
                path_res = [ext_dict[p] if isinstance(p, str) else p for p in path]
            updated_attack.append(path_res)
        updated_mech_pathway.append(tuple(updated_attack))
    return updated_mech_pathway


class MechFinder:
    MT_collection: dict[Any, Any]
    LRT_collection: dict[Any, Any]
    out_of_scope_mechanisms: list[str]
    wrong_atom_mapped_reactions: list[str]
    debug: bool

    def __init__(self, collection_dir: str | None = None, debug: bool = False) -> None:
        if collection_dir:
            base_path = Path(collection_dir)
            mt_path = base_path / "MT_library.csv"
            lrt_path = base_path / "LRT_library.csv"
            self.MT_collection = pd.read_csv(mt_path).replace(np.nan, None).set_index("MT_class").to_dict("index")
            self.LRT_collection = pd.read_csv(lrt_path).replace(np.nan, None).set_index("LRT").to_dict("index")
        else:
            ref_mt = importlib.resources.files("rxngraphormer.midgen.collections") / "MT_library.csv"
            ref_lrt = importlib.resources.files("rxngraphormer.midgen.collections") / "LRT_library.csv"

            with importlib.resources.as_file(ref_mt) as mt_path:
                self.MT_collection = pd.read_csv(mt_path).replace(np.nan, None).set_index("MT_class").to_dict("index")

            with importlib.resources.as_file(ref_lrt) as lrt_path:
                self.LRT_collection = pd.read_csv(lrt_path).replace(np.nan, None).set_index("LRT").to_dict("index")

    def check_exception(self, MT_class: str) -> str | bool:
        if MT_class in ["wrong atom-mapping", "mechanism not in collection"]:
            return MT_class
        elif MT_class in self.out_of_scope_mechanisms:
            return "outside_scope_of_arrow_pushing"
        elif MT_class in self.wrong_atom_mapped_reactions:
            return "wrong_atom_mapping"
        elif MT_class == "missing_info":
            return "missing_info"
        return False

    def get_LRT(
        self, rxn: RxnString
    ) -> tuple[RxnString, TemplateString | None, ReplacementDict | None, dict[str, Any]]:
        rxn, rmaps = clean_leaving_mapping(rxn)
        # print(rxn, rmaps,extract_from_reaction(rxn))
        extract_res = extract_from_reaction(rxn)
        if extract_res is None:
            return rxn, None, None, {"MT_class": "wrong atom-mapping"}

        rxn, template, replacement_dict = extract_res

        if not template:
            return rxn, template, None, {"MT_class": "wrong atom-mapping"}
        if template not in self.LRT_collection:
            return rxn, template, None, {"MT_class": "mechanism not in collection"}
        LRT_info = self.LRT_collection[template]
        replacement_dict = replace_dict(rxn, replacement_dict)  # type: ignore
        rxn = neutralize_charge(rxn)
        return rxn, template, replacement_dict, LRT_info  # type: ignore

    def get_electron_path(self, rxn: RxnString) -> tuple[RxnString, TemplateString | None, str, str | list[Any]]:
        rxn, template, replacement_dict, LRT_info = self.get_LRT(rxn)

        MT_class = LRT_info["MT_class"]
        exception = self.check_exception(MT_class)
        if exception:
            if exception == "wrong_atom_mapping" and self.debug:
                print("#########################################################")
                print("MT: %s, LRT: %s", (MT_class, template))
                print("Reaction:", rxn)
                print("#########################################################")
            return rxn, template, MT_class, str(exception)

        criterion, reagents, adjust_dict = (
            LRT_info["Criterion"],
            LRT_info["Reagent"],
            LRT_info["remap"],
        )
        criterion_result: Any = None
        ext_info: Any = None
        ext_dict: Any = None
        if criterion:
            # eval is used to evaluate criteria strings from the library
            criterion_func, see_maps = eval(criterion)
            criterion_result = criterion_func(rxn, replacement_dict, see_maps)
            MT_class = eval(MT_class)[criterion_result]
            exception = self.check_exception(MT_class)
            if exception:
                if exception == "wrong_atom_mapping" and self.debug:
                    print("#########################################################")
                    print("MT: %s, LRT: %s", (MT_class, template))
                    print("Reaction:", rxn)
                    print("#########################################################")
                return rxn, template, MT_class, str(exception)

            if adjust_dict and not pd.isna(adjust_dict):  # using pd.isna for robustness
                adjust_dict = eval(adjust_dict)[criterion_result]
            if reagents:
                reagents = eval(reagents)[criterion_result]
                rxn, replacement_dict = add_reagent(rxn, reagents, replacement_dict)  # type: ignore
        else:
            if adjust_dict and not pd.isna(adjust_dict):
                adjust_dict = eval(adjust_dict)
            if reagents:
                reagents = eval(reagents)
                rxn, replacement_dict = add_reagent(rxn, reagents, replacement_dict)  # type: ignore

        mechanistic_pathway = eval(self.MT_collection[MT_class]["mechanistic pathway"])
        if LRT_info["LRT_extension"]:
            ext_info = (
                eval(LRT_info["LRT_extension"])[criterion_result] if criterion else eval(LRT_info["LRT_extension"])
            )
            if ext_info:
                for d in [replacement_dict, adjust_dict]:
                    if isinstance(d, dict):
                        for s in ext_info["ext_strings"]:
                            d.update({s: s})
                ext_dict = build_ext_dict(rxn, replacement_dict, ext_info)  # type: ignore

        electron_path = []
        for path in mechanistic_pathway:
            if adjust_dict and not pd.isna(adjust_dict):
                path = adjust_template_atom_map(adjust_dict, path)
            electron_path.append(change_atom_map(replacement_dict, path))  # type: ignore

        if LRT_info["LRT_extension"] and ext_info:
            electron_path = replace_xl(electron_path, ext_dict)
        return rxn, template, MT_class, electron_path
