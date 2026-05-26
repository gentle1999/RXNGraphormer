import copy
from typing import Literal, overload

from rdkit import Chem
from rdkit.Chem import Atom, Draw, Mol
from rdkit.Chem.rdchem import BondType

# from .utils import * # 假设 utils 已被正确导入或在此处定义

# 类型别名定义
MapNum = int
# source/sink 可以是原子(int), 氢原子标记(float), 或键(list)
SourceType = int | list[int | float]
SinkType = int | float | list[int | float]
PathwayType = SourceType | SinkType

bond_dict1: dict[int, BondType | None] = {
    0: None,
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
}
bond_dict2: dict[str, int] = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}


def find_map_num(mol: Mol, mapnum: int | str) -> tuple[int, Atom]:
    return [
        (a.GetIdx(), a)
        for a in mol.GetAtoms()  # type: ignore
        if a.HasProp("molAtomMapNumber") and a.GetProp("molAtomMapNumber") == str(mapnum)
    ][0]


@overload
def update_map_nums(mol: Mol, get_added: Literal[False]) -> Mol: ...
@overload
def update_map_nums(mol: Mol, get_added: Literal[True]) -> tuple[Mol, list[int]]: ...
def update_map_nums(mol: Mol, get_added: bool = False) -> Mol | tuple[Mol, list[int]]:
    """Given a molecule (mol), this function adds mapping to unmapped atoms
    without changing the rest (e.g. after adding hydrogens)
    * parameter get_added is set to True if newly added map numbers are needed"""
    all_map_nums = [
        atom.GetAtomMapNum()
        for atom in mol.GetAtoms()  # type: ignore
        if atom.HasProp("molAtomMapNumber")
    ]
    added = []
    for atom in mol.GetAtoms(): # type: ignore
        map_num = atom.GetAtomMapNum()
        if not map_num:
            map_num = max(all_map_nums) + 1
            all_map_nums.append(map_num)
            if get_added:
                added.append(map_num)
        atom.SetAtomMapNum(map_num)
    return (mol, added) if get_added else mol


def preprocess(mol: Mol, map_num: int | float) -> tuple[int, Atom]:
    if isinstance(map_num, float):  # if it's hydrogen
        atom = [
            neighbor for neighbor in find_map_num(mol, int(map_num))[-1].GetNeighbors() if neighbor.GetAtomicNum() == 1
        ][-1]
        atom_idx = atom.GetIdx()
    elif isinstance(map_num, int):  # if it's not hydrogen
        atom_idx, atom = find_map_num(mol, map_num)
    # 静态分析补充：确保在所有路径都有返回值，实际逻辑中应保证 map_num 只为 int 或 float
    return atom_idx, atom  # type: ignore


def get_h_attached_atoms(mol: Mol, pathway: PathwayType | list[PathwayType]) -> list[int]:
    h_attached_atoms = []
    # 这里处理递归结构，pathway 可能是单一的 source/sink，也可能是它们的列表
    if isinstance(pathway, list):
        for path in pathway:
            if isinstance(path, float):
                h_attached_atoms.append(int(path))
            elif isinstance(path, list):
                h_attached_atoms.extend(get_h_attached_atoms(mol, path))  # type: ignore
    elif isinstance(pathway, float):
        h_attached_atoms.append(int(pathway))

    return h_attached_atoms


def arrow_pushing(
    omol: Mol,
    source: SourceType,
    sink: SinkType,
    visualize: bool = False,
    rxn_class: str | None = None,
    step_no: int | None = None,
) -> Mol:
    mol = copy.copy(omol)
    # 此处传入 [source, sink] 作为列表
    h_attached_atoms_indices = [
        find_map_num(mol, map_num)[0]
        for map_num in get_h_attached_atoms(mol, [source, sink])  # type: ignore
    ]

    if h_attached_atoms_indices:
        mol = update_map_nums(Chem.AddHs(mol, onlyOnAtoms=h_attached_atoms_indices), get_added=False)
        #     mol = Chem.rdmolops.AddHs(mol)

        # #     Visualize the intermediate before path occurrence
    if visualize:
        mol_vis = Chem.RemoveHs(mol, sanitize=False)
        Draw.ShowMol(
            mol_vis,
            size=(1200, 550),
            title=f"{rxn_class} reaction. Step #{step_no}: {source} =>  {sink}",
        )

    # attack from lone pair
    if isinstance(source, int):
        # set formal charge of source atom
        source_atom = find_map_num(mol, source)[-1]
        source_atom.SetFormalCharge(source_atom.GetFormalCharge() + 1)

        # to atom
        if isinstance(sink, int):
            sink_atom = find_map_num(mol, sink)[-1]
            sink_atom.SetFormalCharge(sink_atom.GetFormalCharge() - 1)
            emol = Chem.EditableMol(mol)
            emol.AddBond(
                find_map_num(mol, source)[0],
                find_map_num(mol, sink)[0],
                Chem.rdchem.BondType.SINGLE,
            )
            mol = emol.GetMol()

        # to bond
        elif isinstance(sink, list):
            sink_start_idx, sink_start = find_map_num(mol, int(sink[0]))
            sink_end_idx, sink_end = find_map_num(mol, int(sink[1]))

            sink_end.SetFormalCharge(sink_end.GetFormalCharge() - 1)
            bond = mol.GetBondBetweenAtoms(sink_start_idx, sink_end_idx)
            bond_type_str = str(bond.GetBondType())
            # if there's an attack of lone pair to a bond which is AROMATIC, we consider it as degree one for mechanistic purposes
            if bond_type_str == "AROMATIC":
                bond_type_str = "SINGLE"  # old bond

            # bond_dict2[bond_type] + 1 意味着键级增加
            new_bond_type = bond_dict1[bond_dict2[bond_type_str] + 1]
            if new_bond_type:
                bond.SetBondType(new_bond_type)

        # to hydrogen
        # if sink is float, it represents hydrogen and is denoted by map num of attached atom, e.g, 5.1 shows hydrogen attached to atom having MAP NUMBER = 5 (not index)
        elif isinstance(sink, float):
            sink_hydrogen = [
                neighbor for neighbor in find_map_num(mol, int(sink))[-1].GetNeighbors() if neighbor.GetAtomicNum() == 1
            ][-1]
            sink_hydrogen.SetFormalCharge(sink_hydrogen.GetFormalCharge() - 1)
            emol = Chem.EditableMol(mol)
            emol.AddBond(
                find_map_num(mol, source)[0],
                sink_hydrogen.GetIdx(),
                Chem.rdchem.BondType.SINGLE,
            )
            mol = emol.GetMol()

    # attack from bond
    elif isinstance(source, list):
        # set formal charge of source atom
        source_start_idx = 0
        source_end_idx = 0

        for i, source_map_num in enumerate(source):
            if i == 0:
                # source item is Union[int, float], usually int for bonds
                source_start_idx, source_start = preprocess(mol, source_map_num)  # type: ignore
                source_start.SetFormalCharge(source_start.GetFormalCharge() + 1)
            elif i == 1:
                source_end_idx, source_end = preprocess(mol, source_map_num)  # type: ignore

        # set new bond
        bond = mol.GetBondBetweenAtoms(source_start_idx, source_end_idx)
        bond_type_str = str(bond.GetBondType())  # original bond

        ## only aromatic DOUBLE bonds attack
        if bond_type_str == "AROMATIC":
            bond_type_str = "DOUBLE"

        new_bond_type = bond_dict1[
            bond_dict2[bond_type_str] - 1
        ]  # breaking bond ==> new_bond_type = original_bond_type - 1

        if not new_bond_type:  # if new_bond_type = 0 (None) ==> remove bond
            emol = Chem.EditableMol(mol)
            emol.RemoveBond(source_start_idx, source_end_idx)
            mol = emol.GetMol()
        else:  # else ==> replace with new_bond_type
            bond.SetBondType(new_bond_type)

        # to atom
        if isinstance(sink, int):
            sink_atom_idx, sink_atom = find_map_num(mol, sink)
            sink_atom.SetFormalCharge(sink_atom.GetFormalCharge() - 1)

            # attack to different atom
            if sink_atom_idx != source_end_idx:
                emol = Chem.EditableMol(mol)
                emol.AddBond(source_end_idx, sink_atom_idx, Chem.rdchem.BondType.SINGLE)
                mol = emol.GetMol()

        # to hydrogen
        elif isinstance(sink, float):
            sink_atom_idx, sink_atom = preprocess(mol, sink)
            sink_atom.SetFormalCharge(sink_atom.GetFormalCharge() - 1)

            # to different hydrogen
            if sink_atom_idx != source_end_idx:
                emol = Chem.EditableMol(mol)
                emol.AddBond(source_end_idx, sink_atom_idx, Chem.rdchem.BondType.SINGLE)
                mol = emol.GetMol()

        # to bond
        elif isinstance(sink, list):
            sink_start_idx, sink_start = find_map_num(mol, int(sink[0]))
            sink_end_idx, sink_end = find_map_num(mol, int(sink[1]))
            if all(atom.GetIsAromatic() for atom in [sink_start, sink_end]):  # if new set bond is aromatic
                sink_end.SetFormalCharge(sink_end.GetFormalCharge() - 1)
                bond = mol.GetBondBetweenAtoms(sink_start_idx, sink_end_idx)
                bond.SetBondType(Chem.rdchem.BondType.AROMATIC)
            else:
                sink_end.SetFormalCharge(sink_end.GetFormalCharge() - 1)
                bond = mol.GetBondBetweenAtoms(sink_start_idx, sink_end_idx)
                bond_type_str = str(bond.GetBondType())
                new_bond_type = bond_dict1[bond_dict2[bond_type_str] + 1]
                if new_bond_type:
                    bond.SetBondType(new_bond_type)

    return mol
