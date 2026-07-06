from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from rxngraphormer.preprocessing.chemistry import canonical_smiles
from rxngraphormer.preprocessing.reactions import (
    canonicalize_reaction_side,
    parse_reaction_smiles,
    split_reaction_smiles,
)

LEGACY_RCT_NAME = "rct_smiles_0.csv"
LEGACY_PDT_NAME = "pdt_smiles_0.csv"
LEGACY_MID_NAME = "mid_smiles_0.csv"


@dataclass(frozen=True)
class LegacyInputFiles:
    root: str
    rct_name: str = LEGACY_RCT_NAME
    pdt_name: str = LEGACY_PDT_NAME
    mid_name: str | None = None


@dataclass(frozen=True)
class MaterializedTableFiles:
    rct_data_file: str
    pdt_data_file: str
    mid_data_file: str = ""
    rct_name_regrex: str = ""
    pdt_name_regrex: str = ""
    mid_name_regrex: str = ""


def read_reaction_table(path: str | os.PathLike, *, empty_error: str | None = None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif suffix in {".parquet", ".pq"}:
        table = pd.read_parquet(path).fillna("").astype(str)
    else:
        raise ValueError("Reaction input table must be .csv, .parquet, or .pq")
    if table.empty and empty_error is not None:
        raise ValueError(empty_error)
    return table


def read_prediction_table(path: str | os.PathLike) -> list[dict[str, object]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path).to_dict(orient="records")  # type: ignore
    raise ValueError("Prediction table must be .csv, .parquet, or .pq")


def reaction_smiles_from_rows(
    rows: list[dict[str, object]],
    *,
    rxn_smiles_column: str,
    rct_smiles_column: str,
    pdt_smiles_column: str,
) -> list[str]:
    return [
        reaction_smiles_from_row(
            row,
            rxn_smiles_column=rxn_smiles_column,
            rct_smiles_column=rct_smiles_column,
            pdt_smiles_column=pdt_smiles_column,
        )
        for row in rows
    ]


def reaction_smiles_from_row(
    row: dict[str, object],
    *,
    rxn_smiles_column: str,
    rct_smiles_column: str,
    pdt_smiles_column: str,
) -> str:
    if has_value(row, rxn_smiles_column):
        return required_value(row, rxn_smiles_column)
    return f"{required_value(row, rct_smiles_column)}>>{required_value(row, pdt_smiles_column)}"


def reaction_sides_from_row(
    row: dict[str, object],
    *,
    rxn_smiles_column: str,
    rct_smiles_column: str,
    pdt_smiles_column: str,
) -> tuple[str, str]:
    if has_value(row, rxn_smiles_column):
        parsed = parse_reaction_smiles(required_value(row, rxn_smiles_column))
        return parsed.reactants, parsed.products
    return required_value(row, rct_smiles_column), required_value(row, pdt_smiles_column)


def reaction_smiles_pair_lines(rxn_smiles: Iterable[str]) -> tuple[list[str], list[str]]:
    rct_lines: list[str] = []
    pdt_lines: list[str] = []
    for smi in rxn_smiles:
        reactant, product = split_reaction_smiles(smi)
        rct_lines.append(f"{canonicalize_reaction_side(reactant)},0")
        pdt_lines.append(f"{canonicalize_reaction_side(product)},0")
    return rct_lines, pdt_lines


def regression_table_lines(
    rows: list[dict[str, object]],
    *,
    rxn_smiles_column: str,
    rct_smiles_column: str,
    pdt_smiles_column: str,
    mid_smiles_column: str,
    target_column: str | None,
    require_mid: bool,
) -> tuple[list[str], list[str], list[str]]:
    rct_lines: list[str] = []
    pdt_lines: list[str] = []
    mid_lines: list[str] = []
    for row in rows:
        reactant, product = reaction_sides_from_row(
            row,
            rxn_smiles_column=rxn_smiles_column,
            rct_smiles_column=rct_smiles_column,
            pdt_smiles_column=pdt_smiles_column,
        )
        target = optional_value(row, target_column, default="0")
        validate_legacy_target(target)
        rct_lines.append(f"{canonicalize_reaction_side(reactant)},{target}")
        pdt_lines.append(f"{canonicalize_reaction_side(product)},{target}")
        if require_mid:
            mid_lines.append(f"{canonical_smiles(required_value(row, mid_smiles_column))},{target}")
    return rct_lines, pdt_lines, mid_lines


def write_reaction_smiles_pair_files(root: str | os.PathLike, rxn_smiles: Iterable[str]) -> LegacyInputFiles:
    rct_lines, pdt_lines = reaction_smiles_pair_lines(rxn_smiles)
    return write_legacy_input_files(root, rct_lines, pdt_lines)


def write_regression_table_files(
    root: str | os.PathLike,
    rows: list[dict[str, object]],
    *,
    rxn_smiles_column: str,
    rct_smiles_column: str,
    pdt_smiles_column: str,
    mid_smiles_column: str,
    target_column: str | None,
    require_mid: bool,
) -> LegacyInputFiles:
    rct_lines, pdt_lines, mid_lines = regression_table_lines(
        rows,
        rxn_smiles_column=rxn_smiles_column,
        rct_smiles_column=rct_smiles_column,
        pdt_smiles_column=pdt_smiles_column,
        mid_smiles_column=mid_smiles_column,
        target_column=target_column,
        require_mid=require_mid,
    )
    return write_legacy_input_files(root, rct_lines, pdt_lines, mid_lines if require_mid else None)


def write_reaction_table_files(
    path: str | os.PathLike,
    *,
    output_dir: str | os.PathLike,
    output_prefix: str | None = None,
    rxn_smiles_column: str = "rxn_smiles",
    rct_smiles_column: str = "rct_smiles",
    pdt_smiles_column: str = "pdt_smiles",
    mid_smiles_column: str = "mid_smiles",
    target_column: str | None = None,
    generate_mid: bool = False,
    mapping_policy: str = "auto",
) -> MaterializedTableFiles:
    table = read_reaction_table(path, empty_error=f"Input table is empty: {path}")

    input_path = Path(path)
    prefix = output_prefix or input_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rct_name = f"{prefix}_rct.csv"
    pdt_name = f"{prefix}_pdt.csv"
    mid_name = f"{prefix}_mid.csv" if generate_mid or has_nonempty_column(table, mid_smiles_column) else ""

    rct_rows: list[dict[str, str]] = []
    pdt_rows: list[dict[str, str]] = []
    mid_rows: list[dict[str, str]] = []
    for row in table.to_dict(orient="records"):
        reactants, products = reaction_sides_from_row(
            row,  # type: ignore
            rxn_smiles_column=rxn_smiles_column,
            rct_smiles_column=rct_smiles_column,
            pdt_smiles_column=pdt_smiles_column,
        )
        target = optional_value(row, target_column, default="0")  # type: ignore
        validate_legacy_target(target)
        rct_rows.append({"smiles": canonicalize_reaction_side(reactants), "target": target})
        pdt_rows.append({"smiles": canonicalize_reaction_side(products), "target": target})

        if mid_name:
            if has_value(row, mid_smiles_column):  # type: ignore
                mid_smiles = required_value(row, mid_smiles_column)  # type: ignore
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

    write_legacy_smiles_target_csv(output_dir / rct_name, rct_rows)
    write_legacy_smiles_target_csv(output_dir / pdt_name, pdt_rows)
    if mid_name:
        write_legacy_smiles_target_csv(output_dir / mid_name, mid_rows)

    return MaterializedTableFiles(
        rct_data_file=rct_name,
        pdt_data_file=pdt_name,
        mid_data_file=mid_name,
        rct_name_regrex=rct_name,
        pdt_name_regrex=pdt_name,
        mid_name_regrex=mid_name,
    )


def generate_mid_smiles(rxn_smiles: str, *, mapping_policy: str = "auto") -> str:
    mapping_policy = mapping_policy.lower()
    if mapping_policy not in {"auto", "always", "never"}:
        raise ValueError("mapping_policy must be one of: auto, always, never")

    parsed = parse_reaction_smiles(rxn_smiles)
    reactants = parsed.reactants
    products = parsed.products
    folded_rxn = f"{reactants}>>{products}"
    if mapping_policy == "never":
        if not parsed.has_reactant_product_mapping:
            raise ValueError("mapping_policy='never' requires atom-mapped reaction SMILES")
        mapped_rxn = folded_rxn
    elif mapping_policy == "auto" and parsed.has_reactant_product_mapping:
        mapped_rxn = folded_rxn
    else:
        from rxngraphormer.midgen import midmol

        return midmol.gen_mech_mid_smi((reactants, products))[0]

    return _mid_smiles_from_mapped_reaction(reactants, products, mapped_rxn)


def generate_mid_smiles_batch(
    rxn_smiles: Iterable[str],
    *,
    mapping_policy: str = "auto",
) -> list[str | Exception]:
    mapping_policy = mapping_policy.lower()
    if mapping_policy not in {"auto", "always", "never"}:
        raise ValueError("mapping_policy must be one of: auto, always, never")

    rxn_smiles_list = list(rxn_smiles)
    results: list[str | Exception | None] = [None] * len(rxn_smiles_list)
    mapper_indices: list[int] = []
    mapper_tasks: list[tuple[str, str]] = []
    for idx, rxn_smi in enumerate(rxn_smiles_list):
        try:
            parsed = parse_reaction_smiles(rxn_smi)
            reactants = parsed.reactants
            products = parsed.products
            folded_rxn = f"{reactants}>>{products}"
            if mapping_policy == "never":
                if not parsed.has_reactant_product_mapping:
                    raise ValueError("mapping_policy='never' requires atom-mapped reaction SMILES")
                results[idx] = _mid_smiles_from_mapped_reaction(reactants, products, folded_rxn)
            elif mapping_policy == "auto" and parsed.has_reactant_product_mapping:
                results[idx] = _mid_smiles_from_mapped_reaction(reactants, products, folded_rxn)
            else:
                mapper_indices.append(idx)
                mapper_tasks.append((reactants, products))
        except Exception as exc:
            results[idx] = exc

    if mapper_tasks:
        from rxngraphormer.midgen import midmol

        mapper_results = midmol.gen_mech_mid_smis(mapper_tasks)
        for idx, mapper_result in zip(mapper_indices, mapper_results, strict=True):
            if isinstance(mapper_result, Exception):
                results[idx] = mapper_result
            else:
                results[idx] = mapper_result[0]

    return [
        result if result is not None else RuntimeError("mid SMILES generation did not produce a result")
        for result in results
    ]


def _mid_smiles_from_mapped_reaction(reactants: str, products: str, mapped_rxn: str) -> str:
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
        smi for smi in pot_mech_smi_lst + mid_smi_lst if smi not in rct_smi_lst and smi not in pdt_smi_lst
    ]
    return canonical_smiles(".".join(sorted(set(mech_mid_smi_lst))))


def write_legacy_input_files(
    root: str | os.PathLike,
    rct_lines: list[str],
    pdt_lines: list[str],
    mid_lines: list[str] | None = None,
) -> LegacyInputFiles:
    root = Path(root)
    write_lines(root / LEGACY_RCT_NAME, rct_lines)
    write_lines(root / LEGACY_PDT_NAME, pdt_lines)
    if mid_lines is None:
        return LegacyInputFiles(root=os.fspath(root), mid_name=None)
    write_lines(root / LEGACY_MID_NAME, mid_lines)
    return LegacyInputFiles(root=os.fspath(root), mid_name=LEGACY_MID_NAME)


def write_lines(path: str | os.PathLike, lines: list[str]) -> None:
    with open(path, "w") as handle:
        handle.writelines("\n".join(lines))


def write_legacy_smiles_target_csv(path: str | os.PathLike, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows, columns=cast(Any, ["smiles", "target"])).to_csv(
        path,
        header=False,
        index=False,
        lineterminator="\n",
    )


def required_value(row: dict[str, Any], column: str) -> str:
    if column not in row:
        raise ValueError(f"Input table is missing required column: {column}")
    value = row[column]
    if value is None or str(value) == "":
        raise ValueError(f"Input table contains an empty value in required column: {column}")
    return str(value)


def optional_value(row: dict[str, Any], column: str | None, *, default: str = "0") -> str:
    if column is None or column == "" or not has_value(row, column):
        return default
    return str(row[column])


def has_value(row: dict[str, Any], column: str) -> bool:
    return column in row and row[column] is not None and str(row[column]) != ""


def has_nonempty_column(table: pd.DataFrame, column: str) -> bool:
    return bool(column in table.columns and table[column].astype(str).ne("").any())


def validate_legacy_target(target: str) -> None:
    if "," in target or "\n" in target or "\r" in target:
        raise ValueError("Targets used by the legacy graph builder must be scalar values without commas or newlines")
