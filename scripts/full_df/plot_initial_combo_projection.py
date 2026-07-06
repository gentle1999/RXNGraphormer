from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the logical ene/diene combination grid with matplotlib/seaborn. "
            "Each heatmap cell is one (ene_smiles, diene_smiles) combination."
        )
    )
    parser.add_argument("--combo-splits", type=Path, default=ROOT / "dataset/full_df_initial_kfold/combo_splits.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/full_df_initial_sampling")
    parser.add_argument("--ene-order", choices=("smiles", "first"), default="smiles")
    parser.add_argument("--diene-order", choices=("smiles", "first"), default="smiles")
    parser.add_argument(
        "--fold-mode",
        choices=("axis-separate", "axis-union", "axis-intersection", "ene", "diene", "combo"),
        default="axis-separate",
        help="Fold geometry used for the overlay plot.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def ordered_values(series: pd.Series, mode: str) -> list[str]:
    values = series.astype(str)
    if mode == "first":
        return list(dict.fromkeys(values))
    return sorted(values.unique())


def add_logical_coordinates(combo: pd.DataFrame, *, ene_order: str, diene_order: str) -> pd.DataFrame:
    enes = ordered_values(combo["ene_smiles"], ene_order)
    dienes = ordered_values(combo["diene_smiles"], diene_order)
    ene_index = {smiles: idx for idx, smiles in enumerate(enes)}
    diene_index = {smiles: idx for idx, smiles in enumerate(dienes)}

    out = combo.copy()
    out["ene_index"] = out["ene_smiles"].astype(str).map(ene_index).astype(int)
    out["diene_index"] = out["diene_smiles"].astype(str).map(diene_index).astype(int)
    out["logical_x"] = out["ene_index"]
    out["logical_y"] = out["diene_index"]
    return out


def write_axis_lookup(rows: pd.DataFrame, out_path: Path) -> None:
    ene_lookup = (
        rows[["ene_index", "ene_smiles"]]
        .drop_duplicates()
        .sort_values("ene_index")
        .rename(columns={"ene_index": "index", "ene_smiles": "smiles"})
    )
    ene_lookup.insert(0, "axis", "ene")
    diene_lookup = (
        rows[["diene_index", "diene_smiles"]]
        .drop_duplicates()
        .sort_values("diene_index")
        .rename(columns={"diene_index": "index", "diene_smiles": "smiles"})
    )
    diene_lookup.insert(0, "axis", "diene")
    pd.concat([ene_lookup, diene_lookup], ignore_index=True).to_csv(out_path, index=False)


def grid_matrix(rows: pd.DataFrame, value_col: str, *, fill_value: float = 0.0) -> np.ndarray:
    n_enes = int(rows["ene_index"].max()) + 1
    n_dienes = int(rows["diene_index"].max()) + 1
    matrix = np.full((n_dienes, n_enes), fill_value, dtype=float)
    for row in rows.itertuples(index=False):
        matrix[int(row.diene_index), int(row.ene_index)] = float(getattr(row, value_col))
    return matrix


def initial_mask(rows: pd.DataFrame) -> np.ndarray:
    n_enes = int(rows["ene_index"].max()) + 1
    n_dienes = int(rows["diene_index"].max()) + 1
    mask = np.zeros((n_dienes, n_enes), dtype=bool)
    for row in rows.loc[rows["initial_selected"]].itertuples(index=False):
        mask[int(row.diene_index), int(row.ene_index)] = True
    return mask


def fold_candidate_mask(rows: pd.DataFrame, *, fold: int, fold_mode: str) -> pd.Series:
    if fold_mode == "combo":
        if "fold" not in rows.columns:
            raise ValueError("combo fold mode requires a fold column")
        return rows["fold"] == fold

    missing = [column for column in ("ene_fold", "diene_fold") if column not in rows.columns]
    if missing:
        raise ValueError(f"{fold_mode} fold mode requires columns: {missing}")

    ene_hit = rows["ene_fold"] == fold
    diene_hit = rows["diene_fold"] == fold
    if fold_mode == "axis-union":
        return ene_hit | diene_hit
    if fold_mode == "axis-intersection":
        return ene_hit & diene_hit
    if fold_mode == "ene":
        return ene_hit
    if fold_mode == "diene":
        return diene_hit
    raise ValueError(f"Unsupported fold mode: {fold_mode}")


def split_specs(rows: pd.DataFrame, fold_mode: str) -> list[dict[str, object]]:
    if fold_mode == "combo":
        return [
            {"id": str(value), "label": f"Fold {value}", "axis": None, "fold": int(value)}
            for value in sorted(int(value) for value in rows["fold"].unique())
        ]
    fold_values = sorted(set(rows["ene_fold"].astype(int)).union(set(rows["diene_fold"].astype(int))))
    if fold_mode == "axis-separate":
        specs: list[dict[str, object]] = []
        specs.extend(
            {"id": f"ene_{value}", "label": f"Ene fold {value}", "axis": "ene", "fold": int(value)}
            for value in fold_values
        )
        specs.extend(
            {"id": f"diene_{value}", "label": f"Diene fold {value}", "axis": "diene", "fold": int(value)}
            for value in fold_values
        )
        return specs
    return [{"id": str(value), "label": f"Fold {value}", "axis": None, "fold": int(value)} for value in fold_values]


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


def configure_grid_axes(ax: plt.Axes, *, n_enes: int, n_dienes: int) -> None:
    ax.set_xlabel("ene_index")
    ax.set_ylabel("diene_index")
    ax.set_xticks(np.arange(0.5, n_enes + 0.5, 5))
    ax.set_xticklabels([str(value) for value in range(0, n_enes, 5)], rotation=0)
    ax.set_yticks(np.arange(0.5, n_dienes + 0.5, 5))
    ax.set_yticklabels([str(value) for value in range(0, n_dienes, 5)], rotation=0)
    ax.tick_params(axis="both", labelsize=8, length=0)


def overlay_initial_rectangles(ax: plt.Axes, mask: np.ndarray, *, alpha: float = 0.9, linewidth: float = 0.35) -> None:
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs, strict=True):
        ax.add_patch(
            Rectangle(
                (x, y),
                1,
                1,
                facecolor="#dc2626",
                edgecolor="#7f1d1d",
                linewidth=linewidth,
                alpha=alpha,
            )
        )


def plot_initial_grid(rows: pd.DataFrame, *, png_path: Path, pdf_path: Path, dpi: int) -> None:
    product_matrix = grid_matrix(rows, "product_nunique")
    init_mask = initial_mask(rows)
    selected = rows.loc[rows["initial_selected"]]
    n_dienes, n_enes = product_matrix.shape

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(10.8, 14.2), constrained_layout=True)
    sns.heatmap(
        product_matrix,
        ax=ax,
        cmap="Greys",
        cbar_kws={"label": "product_nunique"},
        square=True,
        linewidths=0.08,
        linecolor="#ffffff",
    )
    overlay_initial_rectangles(ax, init_mask, alpha=0.92, linewidth=0.45)
    configure_grid_axes(ax, n_enes=n_enes, n_dienes=n_dienes)
    ax.set_title(
        "Initial sample on logical ene x diene grid\n"
        f"{len(selected)} initial combos, {selected['ene_index'].nunique()}/{n_enes} enes, "
        f"{selected['diene_index'].nunique()}/{n_dienes} dienes, "
        f"product sum={int(selected['product_nunique'].sum())}",
        fontsize=12,
        pad=14,
    )
    ax.legend(
        handles=[Patch(facecolor="#dc2626", edgecolor="#7f1d1d", label="initial sample")],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def fold_state_matrix(rows: pd.DataFrame, *, spec: dict[str, object], fold_mode: str) -> np.ndarray:
    n_enes = int(rows["ene_index"].max()) + 1
    n_dienes = int(rows["diene_index"].max()) + 1
    matrix = np.zeros((n_dienes, n_enes), dtype=int)
    candidate = split_candidate_mask(rows, spec=spec, fold_mode=fold_mode)
    holdout = rows.loc[candidate & (~rows["initial_selected"])]
    initial = rows.loc[rows["initial_selected"]]
    for row in holdout.itertuples(index=False):
        matrix[int(row.diene_index), int(row.ene_index)] = 1
    for row in initial.itertuples(index=False):
        matrix[int(row.diene_index), int(row.ene_index)] = 2
    return matrix


def plot_fold_overlay(
    rows: pd.DataFrame, *, fold_mode: str, png_path: Path, pdf_path: Path, dpi: int
) -> dict[str, object]:
    specs = split_specs(rows, fold_mode)
    n_enes = int(rows["ene_index"].max()) + 1
    n_dienes = int(rows["diene_index"].max()) + 1
    cmap = ListedColormap(["#e5e7eb", "#2563eb", "#dc2626"])

    sns.set_theme(style="white", context="paper")
    cols = min(5, len(specs))
    panel_rows = int(np.ceil(len(specs) / cols))
    fig, axes = plt.subplots(panel_rows, cols, figsize=(cols * 3.6, panel_rows * 5.15), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(-1)

    fold_records: dict[str, object] = {}
    for ax, spec in zip(axes_array, specs, strict=False):
        split_id = str(spec["id"])
        matrix = fold_state_matrix(rows, spec=spec, fold_mode=fold_mode)
        candidate = split_candidate_mask(rows, spec=spec, fold_mode=fold_mode)
        holdout = rows.loc[candidate & (~rows["initial_selected"])]
        moved_initial = rows.loc[candidate & rows["initial_selected"]]
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=2,
            cbar=False,
            square=True,
            linewidths=0.02,
            linecolor="#ffffff",
        )
        configure_grid_axes(ax, n_enes=n_enes, n_dienes=n_dienes)
        ax.set_title(
            f"{spec['label']}\n"
            f"test combos={holdout['combo_id'].nunique()}, moved initial={moved_initial['combo_id'].nunique()}",
            fontsize=9,
            pad=8,
        )
        fold_records[split_id] = {
            "split_axis": spec["axis"],
            "axis_fold": int(spec["fold"]),
            "holdout_combos": int(holdout["combo_id"].nunique()),
            "holdout_rows": int(holdout["row_count"].sum()),
            "moved_initial_combos": int(moved_initial["combo_id"].nunique()),
            "moved_initial_rows": int(moved_initial["row_count"].sum()),
            "holdout_unique_enes": int(holdout["ene_index"].nunique()),
            "holdout_unique_dienes": int(holdout["diene_index"].nunique()),
        }

    for ax in axes_array[len(specs) :]:
        ax.axis("off")

    fig.suptitle(
        f"{len(specs)} split {fold_mode} distribution overlaid on initial sample\n"
        "blue = fold test/holdout, red = fixed initial sample moved to train, gray = train/other-fold combos",
        fontsize=14,
    )
    fig.legend(
        handles=[
            Patch(facecolor="#2563eb", label="fold test/holdout"),
            Patch(facecolor="#dc2626", label="initial sample"),
            Patch(facecolor="#e5e7eb", label="train or other fold"),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return fold_records


def main() -> None:
    args = parse_args()
    combo = pd.read_csv(args.combo_splits)
    required = {"combo_id", "ene_smiles", "diene_smiles", "product_nunique", "row_count", "initial_selected"}
    missing = sorted(required - set(combo.columns))
    if missing:
        raise ValueError(f"Missing required columns in {args.combo_splits}: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    projected = add_logical_coordinates(combo, ene_order=args.ene_order, diene_order=args.diene_order)
    projection_csv = args.out_dir / "initial_combo_projection.csv"
    axis_lookup_csv = args.out_dir / "initial_combo_axis_lookup.csv"
    initial_png = args.out_dir / "initial_combo_projection.png"
    initial_pdf = args.out_dir / "initial_combo_projection.pdf"
    fold_overlay_png = args.out_dir / "fold_overlay_projection.png"
    fold_overlay_pdf = args.out_dir / "fold_overlay_projection.pdf"
    manifest_path = args.out_dir / "projection_manifest.json"

    projected.to_csv(projection_csv, index=False)
    write_axis_lookup(projected, axis_lookup_csv)
    plot_initial_grid(projected, png_path=initial_png, pdf_path=initial_pdf, dpi=args.dpi)
    fold_records = plot_fold_overlay(
        projected,
        fold_mode=args.fold_mode,
        png_path=fold_overlay_png,
        pdf_path=fold_overlay_pdf,
        dpi=args.dpi,
    )

    selected = projected[projected["initial_selected"]]
    manifest = {
        "projection_type": "logical_ene_diene_grid",
        "plot_backend": "matplotlib/seaborn",
        "combo_splits": str(args.combo_splits),
        "projection_csv": str(projection_csv),
        "axis_lookup_csv": str(axis_lookup_csv),
        "initial_png": str(initial_png),
        "initial_pdf": str(initial_pdf),
        "fold_overlay_png": str(fold_overlay_png),
        "fold_overlay_pdf": str(fold_overlay_pdf),
        "ene_order": args.ene_order,
        "diene_order": args.diene_order,
        "fold_mode": args.fold_mode,
        "combos": int(len(projected)),
        "unique_enes": int(projected["ene_smiles"].nunique()),
        "unique_dienes": int(projected["diene_smiles"].nunique()),
        "initial_combos": int(len(selected)),
        "initial_unique_enes": int(selected["ene_smiles"].nunique()),
        "initial_unique_dienes": int(selected["diene_smiles"].nunique()),
        "initial_product_nunique_sum": int(selected["product_nunique"].sum()),
        "folds": fold_records,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
