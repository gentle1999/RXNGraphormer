from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/full_df/create_initial_combo_kfold_splits.py"


def load_sampling_module():
    spec = importlib.util.spec_from_file_location("create_initial_combo_kfold_splits", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_combo_rows() -> pd.DataFrame:
    costs = {
        ("E0", "D0"): 1,
        ("E0", "D1"): 1,
        ("E0", "D2"): 10,
        ("E0", "D3"): 10,
        ("E1", "D0"): 10,
        ("E1", "D1"): 10,
        ("E1", "D2"): 1,
        ("E1", "D3"): 10,
        ("E2", "D0"): 10,
        ("E2", "D1"): 10,
        ("E2", "D2"): 10,
        ("E2", "D3"): 1,
    }
    rows = []
    for (ene, diene), cost in costs.items():
        for idx in range(cost):
            rows.append(
                {
                    "source_row_id": len(rows),
                    "ene_smiles": ene,
                    "diene_smiles": diene,
                    "prod_smiles": f"P{idx}",
                }
            )
    return pd.DataFrame(rows)


def test_initial_sample_covers_each_substrate_and_minimizes_product_count():
    sampling = load_sampling_module()
    combo = sampling.build_combo_table(
        synthetic_combo_rows(),
        ene_col="ene_smiles",
        diene_col="diene_smiles",
        product_col="prod_smiles",
        cost_mode="product_nunique",
    )

    selected = sampling.solve_initial_sample(
        combo,
        ene_col="ene_smiles",
        diene_col="diene_smiles",
        seed=0,
        max_combos_per_ene=None,
        time_limit=10.0,
    )

    assert len(selected) == 4
    assert selected["diene_smiles"].nunique() == 4
    assert selected["ene_smiles"].nunique() == 3
    assert int(selected["product_nunique"].sum()) == 4


def test_fold_outputs_move_initial_combos_from_holdout_to_train(tmp_path: Path):
    sampling = load_sampling_module()
    df = synthetic_combo_rows()
    combo = sampling.build_combo_table(
        df,
        ene_col="ene_smiles",
        diene_col="diene_smiles",
        product_col="prod_smiles",
        cost_mode="product_nunique",
    )
    initial = combo.loc[combo["combo_id"].isin([0, 1])].copy()
    initial["initial_selected"] = True
    folded = combo[["combo_id"]].copy()
    folded["fold"] = folded["combo_id"] % 2

    sampling.write_outputs(
        df,
        combo,
        initial,
        folded,
        out_dir=tmp_path,
        ene_col="ene_smiles",
        diene_col="diene_smiles",
        folds=2,
        source_path=Path("synthetic.xlsx"),
        cost_mode="product_nunique",
        fold_mode="combo",
        seed=0,
        max_combos_per_ene=None,
        milp_time_limit=10.0,
        fold_weights={"cost": 1.0, "row": 1.0, "combo": 0.2, "substrate": 1.0},
    )

    combo_splits = pd.read_csv(tmp_path / "combo_splits.csv")
    initial_ids = set(combo_splits.loc[combo_splits["initial_selected"], "combo_id"])
    assert initial_ids == {0, 1}

    for fold in range(2):
        train = pd.read_csv(tmp_path / "folds" / f"fold_{fold}" / "train.csv")
        test = pd.read_csv(tmp_path / "folds" / f"fold_{fold}" / "test.csv")
        assigned_to_fold = set(combo_splits.loc[combo_splits["fold"] == fold, "combo_id"])
        assert set(test["combo_id"]) == assigned_to_fold - initial_ids
        assert initial_ids.issubset(set(train["combo_id"]))
        assert initial_ids.isdisjoint(set(test["combo_id"]))


def test_axis_fold_outputs_use_ene_or_diene_fold_geometry(tmp_path: Path):
    sampling = load_sampling_module()
    df = synthetic_combo_rows()
    combo = sampling.build_combo_table(
        df,
        ene_col="ene_smiles",
        diene_col="diene_smiles",
        product_col="prod_smiles",
        cost_mode="product_nunique",
    )
    initial = combo.loc[combo["combo_id"].isin([0, 1])].copy()
    initial["initial_selected"] = True
    folded = combo[["combo_id", "ene_smiles", "diene_smiles"]].copy()
    folded["ene_fold"] = folded["ene_smiles"].map({"E0": 0, "E1": 1, "E2": 0}).astype(int)
    folded["diene_fold"] = folded["diene_smiles"].map({"D0": 0, "D1": 1, "D2": 0, "D3": 1}).astype(int)
    folded = folded[["combo_id", "ene_fold", "diene_fold"]]

    sampling.write_outputs(
        df,
        combo,
        initial,
        folded,
        out_dir=tmp_path,
        ene_col="ene_smiles",
        diene_col="diene_smiles",
        folds=2,
        source_path=Path("synthetic.xlsx"),
        cost_mode="product_nunique",
        fold_mode="axis-union",
        seed=0,
        max_combos_per_ene=None,
        milp_time_limit=10.0,
        fold_weights={"cost": 1.0, "row": 1.0, "combo": 0.2, "substrate": 1.0},
    )

    combo_splits = pd.read_csv(tmp_path / "combo_splits.csv")
    initial_ids = set(combo_splits.loc[combo_splits["initial_selected"], "combo_id"])
    for fold in range(2):
        test = pd.read_csv(tmp_path / "folds" / f"fold_{fold}" / "test.csv")
        expected = set(
            combo_splits.loc[
                ((combo_splits["ene_fold"] == fold) | (combo_splits["diene_fold"] == fold))
                & (~combo_splits["initial_selected"]),
                "combo_id",
            ]
        )
        assert set(test["combo_id"]) == expected
        assert initial_ids.isdisjoint(set(test["combo_id"]))


def test_axis_separate_outputs_independent_ene_and_diene_splits(tmp_path: Path):
    sampling = load_sampling_module()
    df = synthetic_combo_rows()
    combo = sampling.build_combo_table(
        df,
        ene_col="ene_smiles",
        diene_col="diene_smiles",
        product_col="prod_smiles",
        cost_mode="product_nunique",
    )
    initial = combo.loc[combo["combo_id"].isin([0, 1])].copy()
    initial["initial_selected"] = True
    folded = combo[["combo_id", "ene_smiles", "diene_smiles"]].copy()
    folded["ene_fold"] = folded["ene_smiles"].map({"E0": 0, "E1": 1, "E2": 0}).astype(int)
    folded["diene_fold"] = folded["diene_smiles"].map({"D0": 0, "D1": 1, "D2": 0, "D3": 1}).astype(int)
    folded = folded[["combo_id", "ene_fold", "diene_fold"]]

    manifest = sampling.write_outputs(
        df,
        combo,
        initial,
        folded,
        out_dir=tmp_path,
        ene_col="ene_smiles",
        diene_col="diene_smiles",
        folds=2,
        source_path=Path("synthetic.xlsx"),
        cost_mode="product_nunique",
        fold_mode="axis-separate",
        seed=0,
        max_combos_per_ene=None,
        milp_time_limit=10.0,
        fold_weights={"cost": 1.0, "row": 1.0, "combo": 0.2, "substrate": 1.0},
    )

    combo_splits = pd.read_csv(tmp_path / "combo_splits.csv")
    initial_ids = set(combo_splits.loc[combo_splits["initial_selected"], "combo_id"])
    assert manifest["split_count"] == 4
    assert set(manifest["cv"]["folds"]) == {"ene_0", "ene_1", "diene_0", "diene_1"}
    assert (tmp_path / "folds" / "ene_fold_0" / "test.csv").exists()
    assert (tmp_path / "folds" / "diene_fold_0" / "test.csv").exists()

    ene_test = pd.read_csv(tmp_path / "folds" / "ene_fold_0" / "test.csv")
    diene_test = pd.read_csv(tmp_path / "folds" / "diene_fold_0" / "test.csv")
    expected_ene = set(combo_splits.loc[combo_splits["ene_fold"] == 0, "combo_id"]) - initial_ids
    expected_diene = set(combo_splits.loc[combo_splits["diene_fold"] == 0, "combo_id"]) - initial_ids
    assert set(ene_test["combo_id"]) == expected_ene
    assert set(diene_test["combo_id"]) == expected_diene
