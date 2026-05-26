"""Molecular descriptor and fingerprint feature helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import Mol
from rdkit.ML.Descriptors import MoleculeDescriptors

descs = [desc_name[0] for desc_name in Descriptors._descList]
sel_descs = [
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "RingCount",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
]
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(sel_descs)
rdkit_desc_dim = len(sel_descs)
disable_rdkit_log = getattr(RDLogger, "DisableLog", None)
if callable(disable_rdkit_log):
    disable_rdkit_log("rdApp.*")

FingerprintParams = Mapping[str, int | bool | float | str]
FeatureArray: TypeAlias = npt.NDArray[np.float64] | npt.NDArray[np.int_]


def get_fp(
    mol: Mol,
    radius: int = 2,
    nBits: int = 2048,
    useChirality: bool = True,
    fp_type: str = "morgan",
) -> npt.NDArray[np.int_]:
    if fp_type.lower() == "morgan":
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=nBits,
            useChirality=useChirality,
        )
    elif fp_type.lower() == "atompair":
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            mol,
            nBits=nBits,
        )
    elif fp_type.lower() == "toptorsion":
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            mol,
            nBits=nBits,
        )
    elif fp_type.lower() == "rdfp":
        fp = Chem.RDKFingerprint(
            mol,
            fpSize=nBits,
        )
    else:
        raise ValueError("Invalid fingerprint type")

    array = np.zeros((nBits,), dtype=np.int_)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def get_rdkit_desc(mol: Mol, **kwargs: object) -> npt.NDArray[np.float64]:
    return np.array(list(desc_calc.CalcDescriptors(mol)))


def ext_feat_gen(
    mol: Mol,
    params: FingerprintParams | None = None,
    desc_type: str = "Morgan",
    multi_readout: str = "mean",
) -> FeatureArray:
    if params is None:
        params = {"radius": 2, "nBits": 2048, "useChirality": True}
    radius = int(params.get("radius", 2))
    n_bits = int(params.get("nBits", 2048))
    use_chirality = bool(params.get("useChirality", True))
    if "++" in desc_type:
        desc_type_lst = desc_type.split("++")
        desc_lst: list[npt.NDArray[np.int_]] = []
        for desc_type in desc_type_lst:
            desc_lst.append(
                get_fp(
                    mol,
                    radius=radius,
                    nBits=n_bits,
                    useChirality=use_chirality,
                    fp_type=desc_type.lower(),
                )
            )
        if multi_readout.lower() == "mean":
            return cast(npt.NDArray[np.float64], np.mean(desc_lst, axis=0))
        if multi_readout.lower() == "concat":
            return cast(npt.NDArray[np.int_], np.concatenate(desc_lst, axis=0))
        raise ValueError('Invalid multi_readout type, please choose from "mean", "concat"')

    if desc_type.lower() in ["morgan", "atompair", "toptorsion", "rdfp"]:
        return get_fp(
            mol,
            radius=radius,
            nBits=n_bits,
            useChirality=use_chirality,
            fp_type=desc_type,
        )
    if desc_type.lower() == "rdkit":
        return get_rdkit_desc(mol)
    raise ValueError(
        'Invalid descriptor type, please choose from "Morgan", "RDKit", "AtomPair", "TopTorsion", "RDFP"'
    )


__all__ = [
    "desc_calc",
    "descs",
    "ext_feat_gen",
    "get_fp",
    "get_rdkit_desc",
    "rdkit_desc_dim",
    "sel_descs",
]
