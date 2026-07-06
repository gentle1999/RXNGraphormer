from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an initial substrate-combination sample and k-fold splits for dataset/full_df.xlsx. "
            "The initial sample chooses exactly one ene/diene combination for each diene, covers every ene, "
            "and minimizes the selected product count. K-fold assignment is done on ene and diene axes by default; "
            "initial-sample combinations are moved from each fold holdout into train."
        )
    )
    parser.add_argument("--input", type=Path, default=ROOT / "dataset/full_df.xlsx")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dataset/full_df_initial_kfold")
    parser.add_argument("--ene-col", default="ene_smiles")
    parser.add_argument("--diene-col", default="diene_smiles")
    parser.add_argument("--product-col", default="prod_smiles")
    parser.add_argument(
        "--cost-mode",
        choices=("product_nunique", "row_count"),
        default="product_nunique",
        help="Cost minimized by the initial sample.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--fold-mode",
        choices=("axis-separate", "axis-union", "axis-intersection", "ene", "diene", "combo"),
        default="axis-separate",
        help=(
            "Fold geometry. axis-separate writes ene_fold_0..N and diene_fold_0..N as separate splits; "
            "axis-union holds out combos whose ene or diene belongs to the fold; axis-intersection holds out the "
            "fold block; ene/diene hold out one axis only; combo assigns each combo once."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-combos-per-ene",
        type=int,
        default=None,
        help="Optional upper bound for how many initial diene assignments one ene can receive.",
    )
    parser.add_argument("--milp-time-limit", type=float, default=60.0)
    parser.add_argument(
        "--fold-cost-weight",
        type=float,
        default=1.0,
        help="Greedy fold assignment weight for balancing product-count cost.",
    )
    parser.add_argument(
        "--fold-row-weight",
        type=float,
        default=1.0,
        help="Greedy fold assignment weight for balancing row counts.",
    )
    parser.add_argument(
        "--fold-combo-weight",
        type=float,
        default=0.2,
        help="Greedy fold assignment weight for balancing combo counts.",
    )
    parser.add_argument(
        "--fold-substrate-weight",
        type=float,
        default=1.0,
        help="Greedy fold assignment weight for balancing ene/diene appearances.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def read_dataset(path: Path, ene_col: str, diene_col: str, product_col: str) -> pd.DataFrame:
    df = pd.read_excel(path).reset_index(drop=True)
    require_columns(df, [ene_col, diene_col, product_col])
    if "source_row_id" in df.columns:
        raise ValueError("Input already has a source_row_id column; choose a clean input or rename it first.")
    df.insert(0, "source_row_id", np.arange(len(df), dtype=int))
    return df


def build_combo_table(
    df: pd.DataFrame,
    *,
    ene_col: str,
    diene_col: str,
    product_col: str,
    cost_mode: str,
) -> pd.DataFrame:
    grouped = (
        df.groupby([ene_col, diene_col], sort=True)
        .agg(
            row_count=("source_row_id", "size"),
            product_nunique=(product_col, "nunique"),
        )
        .reset_index()
    )
    grouped["cost"] = grouped["product_nunique" if cost_mode == "product_nunique" else "row_count"].astype(float)
    grouped.insert(0, "combo_id", np.arange(len(grouped), dtype=int))
    return grouped


def solve_initial_sample(
    combo: pd.DataFrame,
    *,
    ene_col: str,
    diene_col: str,
    seed: int,
    max_combos_per_ene: int | None,
    time_limit: float,
) -> pd.DataFrame:
    enes = sorted(combo[ene_col].unique())
    dienes = sorted(combo[diene_col].unique())
    ene_index = {ene: idx for idx, ene in enumerate(enes)}
    diene_index = {diene: idx for idx, diene in enumerate(dienes)}

    n_variables = len(combo)
    n_constraints = len(dienes) + len(enes)
    matrix = lil_matrix((n_constraints, n_variables), dtype=float)
    for col, row in enumerate(combo.itertuples(index=False)):
        ene = getattr(row, ene_col)
        diene = getattr(row, diene_col)
        matrix[diene_index[diene], col] = 1.0
        matrix[len(dienes) + ene_index[ene], col] = 1.0

    lower = np.concatenate([np.ones(len(dienes)), np.ones(len(enes))])
    ene_upper = np.full(len(enes), np.inf)
    if max_combos_per_ene is not None:
        if max_combos_per_ene < 1:
            raise ValueError("--max-combos-per-ene must be >= 1")
        ene_upper[:] = max_combos_per_ene
    upper = np.concatenate([np.ones(len(dienes)), ene_upper])

    rng = np.random.default_rng(seed)
    # Product counts are integer, so this epsilon can only break ties.
    objective = combo["cost"].to_numpy(dtype=float) + rng.random(n_variables) * 1e-6
    result = milp(
        c=objective,
        integrality=np.ones(n_variables),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    if not result.success:
        raise RuntimeError(f"Initial sample MILP failed: {result.message}")

    selected_mask = np.asarray(result.x) > 0.5
    selected = combo.loc[selected_mask].copy()
    selected["initial_selected"] = True

    if len(selected) != len(dienes):
        raise RuntimeError(f"Expected one selected combo per diene ({len(dienes)}), got {len(selected)}")
    if selected[diene_col].nunique() != len(dienes):
        raise RuntimeError("Initial sample does not cover every diene exactly once")
    if selected[ene_col].nunique() != len(enes):
        raise RuntimeError("Initial sample does not cover every ene")
    return selected


def assign_combo_folds(
    combo: pd.DataFrame,
    *,
    ene_col: str,
    diene_col: str,
    folds: int,
    seed: int,
    cost_weight: float,
    row_weight: float,
    combo_weight: float,
    substrate_weight: float,
) -> pd.DataFrame:
    if folds < 2:
        raise ValueError("--folds must be >= 2")

    assigned = combo.copy()
    rng = np.random.default_rng(seed)
    assigned["_shuffle"] = rng.random(len(assigned))
    assigned = assigned.sort_values(
        ["cost", "row_count", "_shuffle"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    enes = sorted(assigned[ene_col].unique())
    dienes = sorted(assigned[diene_col].unique())
    ene_index = {ene: idx for idx, ene in enumerate(enes)}
    diene_index = {diene: idx for idx, diene in enumerate(dienes)}

    fold_costs = np.zeros(folds, dtype=float)
    fold_rows = np.zeros(folds, dtype=float)
    fold_combos = np.zeros(folds, dtype=float)
    ene_counts = np.zeros((len(enes), folds), dtype=float)
    diene_counts = np.zeros((len(dienes), folds), dtype=float)

    target_cost = max(float(assigned["cost"].sum()) / folds, 1.0)
    target_rows = max(float(assigned["row_count"].sum()) / folds, 1.0)
    target_combos = max(len(assigned) / folds, 1.0)
    ene_targets = assigned.groupby(ene_col).size().reindex(enes).to_numpy(dtype=float) / folds
    diene_targets = assigned.groupby(diene_col).size().reindex(dienes).to_numpy(dtype=float) / folds
    ene_targets = np.maximum(ene_targets, 1.0)
    diene_targets = np.maximum(diene_targets, 1.0)

    fold_ids: list[int] = []
    for row in assigned.itertuples(index=False):
        ene = getattr(row, ene_col)
        diene = getattr(row, diene_col)
        ene_idx = ene_index[ene]
        diene_idx = diene_index[diene]
        cost = float(row.cost)
        row_count = float(row.row_count)

        candidate_scores: list[tuple[float, float, float, float, int]] = []
        for fold in range(folds):
            next_costs = fold_costs.copy()
            next_rows = fold_rows.copy()
            next_combos = fold_combos.copy()
            next_ene_counts = ene_counts[ene_idx].copy()
            next_diene_counts = diene_counts[diene_idx].copy()

            next_costs[fold] += cost
            next_rows[fold] += row_count
            next_combos[fold] += 1.0
            next_ene_counts[fold] += 1.0
            next_diene_counts[fold] += 1.0

            cost_score = float(np.sum(((next_costs - target_cost) / target_cost) ** 2))
            row_score = float(np.sum(((next_rows - target_rows) / target_rows) ** 2))
            combo_score = float(np.sum(((next_combos - target_combos) / target_combos) ** 2))
            ene_score = float(np.sum(((next_ene_counts - ene_targets[ene_idx]) / ene_targets[ene_idx]) ** 2))
            diene_score = float(
                np.sum(((next_diene_counts - diene_targets[diene_idx]) / diene_targets[diene_idx]) ** 2)
            )
            score = (
                cost_weight * cost_score
                + row_weight * row_score
                + combo_weight * combo_score
                + substrate_weight * (ene_score + diene_score)
            )
            candidate_scores.append((score, fold_costs[fold], fold_rows[fold], fold_combos[fold], fold))

        fold = min(candidate_scores)[-1]
        fold_ids.append(fold)
        fold_costs[fold] += cost
        fold_rows[fold] += row_count
        fold_combos[fold] += 1
        ene_counts[ene_idx, fold] += 1
        diene_counts[diene_idx, fold] += 1

    assigned["fold"] = fold_ids
    return assigned.drop(columns=["_shuffle"])


def assign_axis_folds(
    combo: pd.DataFrame,
    *,
    ene_col: str,
    diene_col: str,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    if folds < 2:
        raise ValueError("--folds must be >= 2")

    rng = np.random.default_rng(seed)
    enes = np.array(sorted(combo[ene_col].unique()), dtype=object)
    dienes = np.array(sorted(combo[diene_col].unique()), dtype=object)
    rng.shuffle(enes)
    rng.shuffle(dienes)

    ene_fold = {value: int(idx % folds) for idx, value in enumerate(enes)}
    diene_fold = {value: int(idx % folds) for idx, value in enumerate(dienes)}

    folded = combo[["combo_id", ene_col, diene_col]].copy()
    folded["ene_fold"] = folded[ene_col].map(ene_fold).astype(int)
    folded["diene_fold"] = folded[diene_col].map(diene_fold).astype(int)
    return folded[["combo_id", "ene_fold", "diene_fold"]]


def fold_candidate_mask(rows: pd.DataFrame, *, fold: int, fold_mode: str) -> pd.Series:
    if fold_mode == "combo":
        return rows["fold"] == fold
    if fold_mode == "axis-union":
        return (rows["ene_fold"] == fold) | (rows["diene_fold"] == fold)
    if fold_mode == "axis-intersection":
        return (rows["ene_fold"] == fold) & (rows["diene_fold"] == fold)
    if fold_mode == "ene":
        return rows["ene_fold"] == fold
    if fold_mode == "diene":
        return rows["diene_fold"] == fold
    raise ValueError(f"Unsupported fold mode: {fold_mode}")


def split_specs(folds: int, fold_mode: str) -> list[dict[str, object]]:
    if fold_mode == "axis-separate":
        specs: list[dict[str, object]] = []
        for fold in range(folds):
            specs.append({"id": f"ene_{fold}", "dir": f"ene_fold_{fold}", "axis": "ene", "fold": fold})
        for fold in range(folds):
            specs.append({"id": f"diene_{fold}", "dir": f"diene_fold_{fold}", "axis": "diene", "fold": fold})
        return specs
    return [{"id": str(fold), "dir": f"fold_{fold}", "axis": None, "fold": fold} for fold in range(folds)]


def split_candidate_mask(rows: pd.DataFrame, *, spec: dict[str, object], fold_mode: str) -> pd.Series:
    fold = int(spec["fold"])
    if fold_mode == "axis-separate":
        axis = spec["axis"]
        if axis == "ene":
            return rows["ene_fold"] == fold
        if axis == "diene":
            return rows["diene_fold"] == fold
        raise ValueError(f"Unsupported axis split: {axis}")
    return fold_candidate_mask(rows, fold=fold, fold_mode=fold_mode)


def fold_reason_counts(rows: pd.DataFrame, *, fold: int, fold_mode: str) -> dict[str, int]:
    if fold_mode == "combo":
        return {"combo": int((rows["fold"] == fold).sum())}

    ene_hit = rows["ene_fold"] == fold
    diene_hit = rows["diene_fold"] == fold
    return {
        "ene_only": int((ene_hit & ~diene_hit).sum()),
        "diene_only": int((~ene_hit & diene_hit).sum()),
        "both": int((ene_hit & diene_hit).sum()),
    }


def split_reason_counts(rows: pd.DataFrame, *, spec: dict[str, object], fold_mode: str) -> dict[str, int]:
    fold = int(spec["fold"])
    if fold_mode == "axis-separate":
        axis = str(spec["axis"])
        return {axis: int(len(rows))}
    return fold_reason_counts(rows, fold=fold, fold_mode=fold_mode)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_outputs(
    df: pd.DataFrame,
    combo: pd.DataFrame,
    initial: pd.DataFrame,
    folded: pd.DataFrame,
    *,
    out_dir: Path,
    ene_col: str,
    diene_col: str,
    folds: int,
    source_path: Path,
    cost_mode: str,
    fold_mode: str,
    seed: int,
    max_combos_per_ene: int | None,
    milp_time_limit: float,
    fold_weights: dict[str, float],
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = out_dir / "folds"
    folds_dir.mkdir(exist_ok=True)

    initial_ids = set(initial["combo_id"].tolist())
    folded_assignment_columns = ["combo_id"] + [
        column for column in ("fold", "ene_fold", "diene_fold") if column in folded.columns
    ]
    folded_assignments = folded[folded_assignment_columns]
    combo_splits = combo.merge(folded_assignments, on="combo_id", how="left")
    combo_splits["initial_selected"] = combo_splits["combo_id"].isin(initial_ids)
    combo_splits["split_role"] = np.where(combo_splits["initial_selected"], "initial", "cv")
    for column in ("fold", "ene_fold", "diene_fold"):
        if column in combo_splits.columns:
            if combo_splits[column].isna().any():
                raise RuntimeError(f"Some combos did not receive {column} assignments")
            combo_splits[column] = combo_splits[column].astype(int)

    combo_path = out_dir / "combo_splits.csv"
    initial_combo_path = out_dir / "initial_combos.csv"
    row_assignment_path = out_dir / "row_assignments.csv"
    initial_rows_path = out_dir / "initial_rows.csv"
    remaining_rows_path = out_dir / "remaining_rows.csv"

    combo_splits.to_csv(combo_path, index=False)
    combo_splits.loc[combo_splits["initial_selected"]].to_csv(initial_combo_path, index=False)

    row_assignments = df.merge(
        combo_splits[
            [
                "combo_id",
                ene_col,
                diene_col,
                "row_count",
                "product_nunique",
                "cost",
                "initial_selected",
                "split_role",
            ]
            + [column for column in ("fold", "ene_fold", "diene_fold") if column in combo_splits.columns]
        ],
        on=[ene_col, diene_col],
        how="left",
        validate="many_to_one",
    )
    if row_assignments["combo_id"].isna().any():
        raise RuntimeError("Some rows did not receive combo assignments")

    row_assignments.to_csv(row_assignment_path, index=False)
    row_assignments.loc[row_assignments["initial_selected"]].to_csv(initial_rows_path, index=False)
    row_assignments.loc[~row_assignments["initial_selected"]].to_csv(remaining_rows_path, index=False)

    specs = split_specs(folds, fold_mode)
    fold_records: dict[str, object] = {}
    for spec in specs:
        split_id = str(spec["id"])
        fold = int(spec["fold"])
        fold_dir = folds_dir / str(spec["dir"])
        fold_dir.mkdir(exist_ok=True)
        candidate_mask = split_candidate_mask(row_assignments, spec=spec, fold_mode=fold_mode)
        holdout_mask = (~row_assignments["initial_selected"]) & candidate_mask
        train_mask = row_assignments["initial_selected"] | ((~row_assignments["initial_selected"]) & (~candidate_mask))

        train_path = fold_dir / "train.csv"
        valid_path = fold_dir / "valid.csv"
        test_path = fold_dir / "test.csv"
        row_assignments.loc[train_mask].to_csv(train_path, index=False)
        row_assignments.loc[holdout_mask].to_csv(valid_path, index=False)
        row_assignments.loc[holdout_mask].to_csv(test_path, index=False)

        holdout = row_assignments.loc[holdout_mask]
        train = row_assignments.loc[train_mask]
        moved_initial = row_assignments.loc[row_assignments["initial_selected"] & candidate_mask]
        combo_candidate_mask = split_candidate_mask(combo_splits, spec=spec, fold_mode=fold_mode)
        combo_holdout_mask = (~combo_splits["initial_selected"]) & combo_candidate_mask
        fold_records[split_id] = {
            "split_id": split_id,
            "split_axis": spec["axis"],
            "axis_fold": fold,
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "train_combos": int(train["combo_id"].nunique()),
            "holdout_combos": int(holdout["combo_id"].nunique()),
            "initial_moved_from_holdout_rows": int(len(moved_initial)),
            "initial_moved_from_holdout_combos": int(moved_initial["combo_id"].nunique()),
            "holdout_product_nunique_sum": int(combo_splits.loc[combo_holdout_mask, "product_nunique"].sum()),
            "holdout_row_count_sum": int(combo_splits.loc[combo_holdout_mask, "row_count"].sum()),
            "holdout_unique_enes": int(holdout[ene_col].nunique()),
            "holdout_unique_dienes": int(holdout[diene_col].nunique()),
            "candidate_combo_reason_counts": split_reason_counts(
                combo_splits.loc[combo_candidate_mask],
                spec=spec,
                fold_mode=fold_mode,
            ),
            "train_path": rel(train_path),
            "valid_path": rel(valid_path),
            "test_path": rel(test_path),
        }

    manifest: dict[str, object] = {
        "dataset": rel(source_path),
        "seed": seed,
        "folds": folds,
        "split_count": len(specs),
        "ene_col": ene_col,
        "diene_col": diene_col,
        "cost_mode": cost_mode,
        "fold_mode": fold_mode,
        "max_combos_per_ene": max_combos_per_ene,
        "milp_time_limit": milp_time_limit,
        "fold_weights": fold_weights,
        "rows": int(len(df)),
        "combos": int(len(combo)),
        "unique_enes": int(combo[ene_col].nunique()),
        "unique_dienes": int(combo[diene_col].nunique()),
        "initial": {
            "constraints": [
                "exactly one selected ene/diene combo per diene",
                "every ene appears in at least one selected combo",
                "objective minimizes the selected combo cost",
            ],
            "combos": int(len(initial)),
            "rows": int(row_assignments["initial_selected"].sum()),
            "product_nunique_sum": int(initial["product_nunique"].sum()),
            "row_count_sum": int(initial["row_count"].sum()),
            "unique_enes": int(initial[ene_col].nunique()),
            "unique_dienes": int(initial[diene_col].nunique()),
            "path": rel(initial_combo_path),
            "rows_path": rel(initial_rows_path),
        },
        "cv": {
            "combos": int((combo_splits["split_role"] == "cv").sum()),
            "rows": int((~row_assignments["initial_selected"]).sum()),
            "combo_splits_path": rel(combo_path),
            "row_assignments_path": rel(row_assignment_path),
            "remaining_rows_path": rel(remaining_rows_path),
            "folds": fold_records,
            "notes": [
                "Fold assignment is performed according to fold_mode.",
                "axis-separate writes 2 * folds split directories: ene_fold_* and diene_fold_*.",
                "For axis fold modes, ene_fold and diene_fold are assigned separately before fold files are written.",
                "Fold holdout files exclude the initial sample even when an initial combo belongs to that fold holdout geometry.",
                "Each fold train.csv contains all initial combos plus all non-holdout combos.",
                "valid.csv and test.csv are identical holdout files; create a second-level validation split if needed.",
            ],
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    df = read_dataset(args.input, args.ene_col, args.diene_col, args.product_col)
    combo = build_combo_table(
        df,
        ene_col=args.ene_col,
        diene_col=args.diene_col,
        product_col=args.product_col,
        cost_mode=args.cost_mode,
    )
    initial = solve_initial_sample(
        combo,
        ene_col=args.ene_col,
        diene_col=args.diene_col,
        seed=args.seed,
        max_combos_per_ene=args.max_combos_per_ene,
        time_limit=args.milp_time_limit,
    )
    if args.fold_mode == "combo":
        folded = assign_combo_folds(
            combo,
            ene_col=args.ene_col,
            diene_col=args.diene_col,
            folds=args.folds,
            seed=args.seed,
            cost_weight=args.fold_cost_weight,
            row_weight=args.fold_row_weight,
            combo_weight=args.fold_combo_weight,
            substrate_weight=args.fold_substrate_weight,
        )
    else:
        folded = assign_axis_folds(
            combo,
            ene_col=args.ene_col,
            diene_col=args.diene_col,
            folds=args.folds,
            seed=args.seed,
        )
    manifest = write_outputs(
        df,
        combo,
        initial,
        folded,
        out_dir=args.out_dir,
        ene_col=args.ene_col,
        diene_col=args.diene_col,
        folds=args.folds,
        source_path=args.input,
        cost_mode=args.cost_mode,
        fold_mode=args.fold_mode,
        seed=args.seed,
        max_combos_per_ene=args.max_combos_per_ene,
        milp_time_limit=args.milp_time_limit,
        fold_weights={
            "cost": args.fold_cost_weight,
            "row": args.fold_row_weight,
            "combo": args.fold_combo_weight,
            "substrate": args.fold_substrate_weight,
        },
    )
    print(json.dumps({"manifest": rel(args.out_dir / "manifest.json"), "initial": manifest["initial"]}, indent=2))


if __name__ == "__main__":
    main()
