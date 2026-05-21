from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from rxngraphormer.utils import canonical_smiles
from rxngraphormer.reaction import (
    canonicalize_reaction_side,
    parse_reaction_smiles,
    reaction_has_atom_mapping,
    split_reaction_smiles,
)


@dataclass(frozen=True)
class PreprocessTableFiles:
    rct_data_file: str
    pdt_data_file: str
    mid_data_file: str = ""
    rct_name_regrex: str = ""
    pdt_name_regrex: str = ""
    mid_name_regrex: str = ""


def write_reaction_table_files(
    path: str | Path,
    *,
    output_dir: str | Path,
    output_prefix: str | None = None,
    rxn_smiles_column: str = "rxn_smiles",
    rct_smiles_column: str = "rct_smiles",
    pdt_smiles_column: str = "pdt_smiles",
    mid_smiles_column: str = "mid_smiles",
    target_column: str | None = None,
    generate_mid: bool = False,
    mapping_policy: str = "auto",
) -> PreprocessTableFiles:
    table = read_reaction_table(path)
    if table.empty:
        raise ValueError(f"Input table is empty: {path}")

    input_path = Path(path)
    prefix = output_prefix or input_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rct_name = f"{prefix}_rct.csv"
    pdt_name = f"{prefix}_pdt.csv"
    mid_name = f"{prefix}_mid.csv" if generate_mid or _has_nonempty_column(table, mid_smiles_column) else ""

    rct_rows: list[dict[str, str]] = []
    pdt_rows: list[dict[str, str]] = []
    mid_rows: list[dict[str, str]] = []
    for row in table.to_dict(orient="records"):
        reactants, products = reaction_sides_from_row(
            row,
            rxn_smiles_column=rxn_smiles_column,
            rct_smiles_column=rct_smiles_column,
            pdt_smiles_column=pdt_smiles_column,
        )
        target = _optional_value(row, target_column, default="0")
        _validate_legacy_target(target)
        rct_rows.append({"smiles": canonicalize_reaction_side(reactants), "target": target})
        pdt_rows.append({"smiles": canonicalize_reaction_side(products), "target": target})

        if mid_name:
            if _has_value(row, mid_smiles_column):
                mid_smiles = _required_value(row, mid_smiles_column)
            elif generate_mid:
                mid_smiles = generate_mid_smiles(
                    f"{reactants}>>{products}",
                    mapping_policy=mapping_policy,
                )
            else:
                raise ValueError(
                    f"Input table is missing column {mid_smiles_column!r}; "
                    "set generate_mid=true or provide mid SMILES explicitly."
                )
            mid_rows.append({"smiles": canonical_smiles(mid_smiles), "target": target})

    _write_legacy_smiles_target_csv(output_dir / rct_name, rct_rows)
    _write_legacy_smiles_target_csv(output_dir / pdt_name, pdt_rows)
    if mid_name:
        _write_legacy_smiles_target_csv(output_dir / mid_name, mid_rows)

    return PreprocessTableFiles(
        rct_data_file=rct_name,
        pdt_data_file=pdt_name,
        mid_data_file=mid_name,
        rct_name_regrex=rct_name,
        pdt_name_regrex=pdt_name,
        mid_name_regrex=mid_name,
    )


def read_reaction_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).fillna("").astype(str)
    raise ValueError("Preprocessing input table must be .csv, .parquet, or .pq")


def reaction_sides_from_row(
    row: dict[str, Any],
    *,
    rxn_smiles_column: str,
    rct_smiles_column: str,
    pdt_smiles_column: str,
) -> tuple[str, str]:
    if _has_value(row, rxn_smiles_column):
        parsed = parse_reaction_smiles(_required_value(row, rxn_smiles_column))
        return parsed.reactants, parsed.products
    return _required_value(row, rct_smiles_column), _required_value(row, pdt_smiles_column)


def generate_mid_smiles(rxn_smiles: str, *, mapping_policy: str = "auto") -> str:
    mapping_policy = mapping_policy.lower()
    if mapping_policy not in {"auto", "always", "never"}:
        raise ValueError("mapping_policy must be one of: auto, always, never")

    reactants, products = split_reaction_smiles(rxn_smiles)
    folded_rxn = f"{reactants}>>{products}"
    has_mapping = reaction_has_atom_mapping(rxn_smiles)
    if mapping_policy == "never":
        if not has_mapping:
            raise ValueError("mapping_policy='never' requires atom-mapped reaction SMILES")
        mapped_rxn = folded_rxn
    elif mapping_policy == "auto" and has_mapping:
        mapped_rxn = folded_rxn
    else:
        from rxngraphormer.midgen.midmol import gen_mech_mid_smi

        return gen_mech_mid_smi((reactants, products))[0]

    from rxngraphormer.midgen.midmol import finder, get_mid_smi_from_rxn, remove_atmmap

    updated_reaction, _lrt, _mt_class, _electron_path = finder.get_electron_path(mapped_rxn)
    mid_smi_lst = get_mid_smi_from_rxn(updated_reaction)
    rct_smi_lst: list[str] = []
    pdt_smi_lst: list[str] = []
    for smi in reactants.split("."):
        rct_smi_lst.extend(remove_atmmap(smi).split("."))
    for smi in products.split("."):
        pdt_smi_lst.extend(remove_atmmap(smi).split("."))
    pot_mech_smi_lst: list[str] = []
    for side in updated_reaction.split(">>"):
        for smi in side.split("."):
            pot_mech_smi_lst.extend(remove_atmmap(smi).split("."))
    mech_mid_smi_lst = [
        smi
        for smi in pot_mech_smi_lst + mid_smi_lst
        if smi not in rct_smi_lst and smi not in pdt_smi_lst
    ]
    return canonical_smiles(".".join(sorted(set(mech_mid_smi_lst))))


def _required_value(row: dict[str, Any], column: str) -> str:
    if column not in row:
        raise ValueError(f"Input table is missing required column: {column}")
    value = row[column]
    if value is None or str(value) == "":
        raise ValueError(f"Input table contains an empty value in required column: {column}")
    return str(value)


def _optional_value(row: dict[str, Any], column: str | None, *, default: str) -> str:
    if column is None or column == "" or not _has_value(row, column):
        return default
    return str(row[column])


def _has_value(row: dict[str, Any], column: str) -> bool:
    return column in row and row[column] is not None and str(row[column]) != ""


def _has_nonempty_column(table: pd.DataFrame, column: str) -> bool:
    return column in table.columns and table[column].astype(str).ne("").any()


def _validate_legacy_target(target: str) -> None:
    if "," in target or "\n" in target or "\r" in target:
        raise ValueError(
            "Targets used by the legacy graph builder must be scalar values "
            "without commas or newlines"
        )


def _write_legacy_smiles_target_csv(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows, columns=["smiles", "target"]).to_csv(
        path,
        header=False,
        index=False,
        lineterminator="\n",
    )
