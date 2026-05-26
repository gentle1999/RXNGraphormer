"""Legacy molecular feature constants."""

from rdkit import Chem, RDLogger

disable_rdkit_log = getattr(RDLogger, "DisableLog", None)
if callable(disable_rdkit_log):
    disable_rdkit_log("rdApp.*")

NUM_ATOM_TYPE = 65
NUM_DEGRESS_TYPE = 11
NUM_FORMCHRG_TYPE = 5
NUM_HYBRIDTYPE = 6
NUM_CHIRAL_TYPE = 3
NUM_AROMATIC_NUM = 2
NUM_VALENCE_TYPE = 7
NUM_Hs_TYPE = 5
NUM_RS_TPYE = 3

NUM_BOND_TYPE = 6
NUM_BOND_DIRECTION = 3
NUM_BOND_STEREO = 3
NUM_BOND_INRING = 2
NUM_BOND_ISCONJ = 2
ATOM_FEAT_DIMS = [
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
BOND_FEAT_DIME = [
    NUM_BOND_TYPE,
    NUM_BOND_DIRECTION,
    NUM_BOND_STEREO,
    NUM_BOND_INRING,
    NUM_BOND_ISCONJ,
]
ATOM_LST = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "W",
    "Ru",
    "Nb",
    "Re",
    "Te",
    "Rh",
    "Ta",
    "Tc",
    "Ba",
    "Bi",
    "Hf",
    "Mo",
    "U",
    "Sm",
    "Os",
    "Ir",
    "Ce",
    "Gd",
    "Ga",
    "Cs",
    "*",
    "unk",
]
ATOM_DICT = {symbol: i for i, symbol in enumerate(ATOM_LST)}
MAX_NEIGHBORS = 10
CHIRAL_TAG_LST = [
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
]
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG_LST)}
HYBRIDTYPE_LST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
HYBRIDTYPE_DICT = {hb: i for i, hb in enumerate(HYBRIDTYPE_LST)}
VALENCE_LST = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE_LST)}
NUM_Hs_LST = [0, 1, 3, 4, 5]
NUM_Hs_DICT = {nH: i for i, nH in enumerate(NUM_Hs_LST)}
BOND_TYPE_LST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.DATIVE,
    Chem.rdchem.BondType.UNSPECIFIED,
]
BOND_DIR_LST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]
BOND_STEREO_LST = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOZ,
]
FORMAL_CHARGE_LST = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE_LST)}
RS_TAG_LST = ["R", "S", "None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG_LST)}

__all__ = [
    "ATOM_DICT",
    "ATOM_FEAT_DIMS",
    "ATOM_LST",
    "BOND_DIR_LST",
    "BOND_FEAT_DIME",
    "BOND_STEREO_LST",
    "BOND_TYPE_LST",
    "CHIRAL_TAG_DICT",
    "CHIRAL_TAG_LST",
    "FC_DICT",
    "FORMAL_CHARGE_LST",
    "HYBRIDTYPE_DICT",
    "HYBRIDTYPE_LST",
    "MAX_NEIGHBORS",
    "NUM_AROMATIC_NUM",
    "NUM_ATOM_TYPE",
    "NUM_BOND_DIRECTION",
    "NUM_BOND_INRING",
    "NUM_BOND_ISCONJ",
    "NUM_BOND_STEREO",
    "NUM_BOND_TYPE",
    "NUM_CHIRAL_TYPE",
    "NUM_DEGRESS_TYPE",
    "NUM_FORMCHRG_TYPE",
    "NUM_HYBRIDTYPE",
    "NUM_Hs_DICT",
    "NUM_Hs_LST",
    "NUM_Hs_TYPE",
    "NUM_RS_TPYE",
    "NUM_VALENCE_TYPE",
    "RS_TAG_DICT",
    "RS_TAG_LST",
    "VALENCE_DICT",
    "VALENCE_LST",
]
