import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

from rxngraphormer import cli
from rxngraphormer.compatibility import (
    ClassificationCheckpointCompatibilitySettings,
    RegressionPretrainCompatibilitySettings,
)
from rxngraphormer.evaluation import (
    ClassificationEvaluation,
    ClassificationMetrics,
    RegressionEvaluation,
    RegressionMetrics,
    classification_metrics,
)
from rxngraphormer.evaluation.classification_workflow import (
    ClassificationEvaluationSettings,
    ClassificationSplitResult,
)
from rxngraphormer.evaluation.regression_workflow import (
    RegressionEvaluationSettings,
    RegressionSplitResult,
    evaluate_regression_prediction,
)
from rxngraphormer.inference import (
    RegressionPrediction,
    RXNGraphormerPredictor,
)
from rxngraphormer.training import RXNGraphormerLitModule


class TrainCliRoutingTest(unittest.TestCase):
    def test_train_cli_routes_regression_to_lightning_by_default(self):
        calls = {}
        config = type("Config", (), {"task": "regression"})()

        def fake_fit_config(received_config, settings):
            calls["fit"] = (received_config, settings)

        argv = [
            "rxngraphormer-train",
            "--config",
            "config.json",
            "--accelerator",
            "cpu",
            "--max_epochs",
            "3",
            "--max_steps",
            "12",
            "--val_check_interval",
            "4",
            "--split_manifest",
            "split.json",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.cli.train.load_config", return_value=config) as load_config_mock,
            mock.patch("rxngraphormer.lightning.fit_config", side_effect=fake_fit_config),
        ):
            cli.train_main()

        load_config_mock.assert_called_once_with("config.json")
        self.assertIs(calls["fit"][0], config)
        self.assertEqual(calls["fit"][1].accelerator, "cpu")
        self.assertEqual(calls["fit"][1].max_epochs, 3)
        self.assertEqual(calls["fit"][1].max_steps, 12)
        self.assertEqual(calls["fit"][1].val_check_interval, 4)
        self.assertEqual(calls["fit"][1].split_manifest, "split.json")

    def test_train_cli_can_keep_legacy_regression_path(self):
        config = type("Config", (), {"task": "regression"})()

        argv = ["rxngraphormer-train", "--config_json", "config.json", "--legacy_regression"]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.cli.train.load_config", return_value=config),
            mock.patch("rxngraphormer.cli.train._run_legacy_train") as legacy_train,
            mock.patch("rxngraphormer.lightning.fit_config") as fit_config_mock,
        ):
            cli.train_main()

        legacy_train.assert_called_once_with(config, local_rank=-1)
        fit_config_mock.assert_not_called()

    def test_train_cli_routes_classification_to_lightning_by_default(self):
        calls = {}
        config = type("Config", (), {"task": "classification"})()

        def fake_fit_config(received_config, settings):
            calls["fit"] = (received_config, settings)

        argv = ["rxngraphormer-train", "--config", "config.json", "--local_rank", "2", "--accelerator", "cpu"]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.cli.train.load_config", return_value=config),
            mock.patch("rxngraphormer.cli.train._run_legacy_train") as legacy_train,
            mock.patch("rxngraphormer.lightning.fit_config", side_effect=fake_fit_config),
        ):
            cli.train_main()

        self.assertIs(calls["fit"][0], config)
        self.assertEqual(calls["fit"][1].accelerator, "cpu")
        legacy_train.assert_not_called()

    def test_train_cli_can_keep_legacy_classification_path(self):
        config = type("Config", (), {"task": "classification"})()

        argv = ["rxngraphormer-train", "--config", "config.json", "--local_rank", "2", "--legacy_classification"]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.cli.train.load_config", return_value=config),
            mock.patch("rxngraphormer.cli.train._run_legacy_train") as legacy_train,
            mock.patch("rxngraphormer.lightning.fit_config") as fit_config_mock,
        ):
            cli.train_main()

        legacy_train.assert_called_once_with(config, local_rank=2)
        fit_config_mock.assert_not_called()

    def test_pyproject_registers_all_supported_cli_entrypoints_and_root_wrappers_are_removed(self):
        repo_root = Path(__file__).resolve().parents[1]
        pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        scripts: dict[str, str] = {}
        in_scripts = False
        for raw_line in pyproject_text.splitlines():
            line = raw_line.strip()
            if line == "[project.scripts]":
                in_scripts = True
                continue
            if in_scripts and line.startswith("["):
                break
            if in_scripts and line and not line.startswith("#"):
                key, value = line.split("=", 1)
                scripts[key.strip()] = value.strip().strip('"')

        self.assertEqual(
            scripts,
            {
                "rxngraphormer": "rxngraphormer.cli:main",
                "rxngraphormer-train": "rxngraphormer.cli:train_main",
                "rxngraphormer-train-legacy": "rxngraphormer.cli:train_legacy_main",
                "rxngraphormer-train-lit": "rxngraphormer.cli:train_lit_main",
                "rxngraphormer-eval": "rxngraphormer.cli:eval_main",
                "rxngraphormer-eval-legacy": "rxngraphormer.cli:eval_legacy_main",
                "rxngraphormer-compat": "rxngraphormer.cli:compat_main",
                "rxngraphormer-predict": "rxngraphormer.cli:predict_main",
                "rxngraphormer-predict-sequence": "rxngraphormer.cli:predict_sequence_main",
                "rxngraphormer-preprocess": "rxngraphormer.cli:preprocess_main",
            },
        )
        self.assertFalse((repo_root / "train_model.py").exists())
        self.assertFalse((repo_root / "eval_model.py").exists())
        self.assertFalse((repo_root / "data_preprocess.py").exists())
        self.assertFalse((repo_root / "main.py").exists())

    def test_top_level_cli_dispatches_to_registered_subcommand(self):
        with (
            mock.patch("sys.argv", ["rxngraphormer", "train", "--config", "config.json"]),
            mock.patch("rxngraphormer.cli.train_main") as train_main,
        ):
            cli.main()

        train_main.assert_called_once_with()

    def test_new_engineering_boundaries_reexport_core_apis(self):
        from rxngraphormer.compatibility import CheckpointAdapter
        from rxngraphormer.compatibility import load_legacy_torch as boundary_load_legacy_torch
        from rxngraphormer.compatibility.torch_compat import load_legacy_torch as impl_load_legacy_torch
        from rxngraphormer.data import RXNDataset as BoundaryRXNDataset
        from rxngraphormer.data import add_dense_empty_node_edge as boundary_add_dense_empty_node_edge
        from rxngraphormer.data import add_empty_node_and_edge as boundary_add_empty_node_and_edge
        from rxngraphormer.data import ext_feat_gen as boundary_ext_feat_gen
        from rxngraphormer.data import get_rxn_pfm_info as boundary_get_rxn_pfm_info
        from rxngraphormer.data import pad_feat as boundary_pad_feat
        from rxngraphormer.data import update_batch_idx as boundary_update_batch_idx
        from rxngraphormer.data.batch import add_dense_empty_node_edge as impl_add_dense_empty_node_edge
        from rxngraphormer.data.batch import add_empty_node_and_edge as impl_add_empty_node_and_edge
        from rxngraphormer.data.batch import pad_feat as impl_pad_feat
        from rxngraphormer.data.batch import update_batch_idx as impl_update_batch_idx
        from rxngraphormer.data.datasets import RXNDataset as DatasetImplRXNDataset
        from rxngraphormer.data.features import ext_feat_gen as impl_ext_feat_gen
        from rxngraphormer.data.legacy import RXNDataset as LegacyRXNDataset
        from rxngraphormer.data.reaction_dataset import RXNDataset as ConcreteRXNDataset
        from rxngraphormer.data.reaction_graph import get_rxn_pfm_info as impl_get_rxn_pfm_info
        from rxngraphormer.evaluation import ClassificationEvaluationSettings as EvaluationClassificationSettings
        from rxngraphormer.evaluation import RegressionEvaluationSettings as EvaluationRegressionSettings
        from rxngraphormer.evaluation import classification_metrics as boundary_classification_metrics
        from rxngraphormer.evaluation.classification_workflow import (
            ClassificationEvaluationSettings as BoundaryClassificationEvaluationSettings,
        )
        from rxngraphormer.evaluation.regression_workflow import (
            RegressionEvaluationSettings as BoundaryRegressionEvaluationSettings,
        )
        from rxngraphormer.inference import RXNEMB as BoundaryRXNEMB
        from rxngraphormer.inference import RXNClassifier as BoundaryRXNClassifier
        from rxngraphormer.inference import RXNGraphormerPredictor as BoundaryPredictor
        from rxngraphormer.inference.embeddings import RXNEMB as ImplRXNEMB
        from rxngraphormer.inference.embeddings import RXNClassifier as ImplRXNClassifier
        from rxngraphormer.inference.inputs import reaction_smiles_pair_lines as inference_reaction_smiles_pair_lines
        from rxngraphormer.models import RXNGClassifier, build_classification_model
        from rxngraphormer.models import get_sin_encodings as boundary_get_sin_encodings
        from rxngraphormer.models import index_scatter as boundary_index_scatter
        from rxngraphormer.models import index_select_ND as boundary_index_select_ND
        from rxngraphormer.models import scaled_dot_product_attention as boundary_scaled_dot_product_attention
        from rxngraphormer.models.ops import get_sin_encodings as impl_get_sin_encodings
        from rxngraphormer.models.ops import index_scatter as impl_index_scatter
        from rxngraphormer.models.ops import index_select_ND as impl_index_select_ND
        from rxngraphormer.models.ops import scaled_dot_product_attention as impl_scaled_dot_product_attention
        from rxngraphormer.preprocessing import canonical_smiles as boundary_canonical_smiles
        from rxngraphormer.preprocessing import canonicalize_reaction_side as boundary_canonicalize_reaction_side
        from rxngraphormer.preprocessing import gen_mid_mols as boundary_gen_mid_mols
        from rxngraphormer.preprocessing import gen_truth_false_rxn_smi as boundary_gen_truth_false_rxn_smi
        from rxngraphormer.preprocessing import generate_mid_smiles as boundary_generate_mid_smiles
        from rxngraphormer.preprocessing import get_random_shuffle_smiles as boundary_get_random_shuffle_smiles
        from rxngraphormer.preprocessing import inchi_to_smiles as boundary_inchi_to_smiles
        from rxngraphormer.preprocessing import mod_mol as boundary_mod_mol
        from rxngraphormer.preprocessing import parse_reaction_smiles as boundary_parse_reaction_smiles
        from rxngraphormer.preprocessing import preprocess_from_config as boundary_preprocess_from_config
        from rxngraphormer.preprocessing import reaction_sides_from_row as boundary_reaction_sides_from_row
        from rxngraphormer.preprocessing import reaction_smiles_pair_lines as boundary_reaction_smiles_pair_lines
        from rxngraphormer.preprocessing import read_reaction_table as boundary_read_reaction_table
        from rxngraphormer.preprocessing import split_reaction_smiles as boundary_split_reaction_smiles
        from rxngraphormer.preprocessing import write_reaction_table_files as boundary_write_reaction_table_files
        from rxngraphormer.preprocessing.chemistry import canonical_smiles as impl_canonical_smiles
        from rxngraphormer.preprocessing.chemistry import gen_mid_mols as impl_gen_mid_mols
        from rxngraphormer.preprocessing.chemistry import gen_truth_false_rxn_smi as impl_gen_truth_false_rxn_smi
        from rxngraphormer.preprocessing.chemistry import get_random_shuffle_smiles as impl_get_random_shuffle_smiles
        from rxngraphormer.preprocessing.chemistry import inchi_to_smiles as impl_inchi_to_smiles
        from rxngraphormer.preprocessing.chemistry import mod_mol as impl_mod_mol
        from rxngraphormer.preprocessing.materialization import generate_mid_smiles as impl_generate_mid_smiles
        from rxngraphormer.preprocessing.materialization import reaction_sides_from_row as impl_reaction_sides_from_row
        from rxngraphormer.preprocessing.materialization import (
            reaction_smiles_pair_lines as impl_reaction_smiles_pair_lines,
        )
        from rxngraphormer.preprocessing.materialization import read_reaction_table as impl_read_reaction_table
        from rxngraphormer.preprocessing.materialization import (
            write_reaction_table_files as impl_write_reaction_table_files,
        )
        from rxngraphormer.preprocessing.reactions import canonicalize_reaction_side as impl_canonicalize_reaction_side
        from rxngraphormer.preprocessing.reactions import parse_reaction_smiles as impl_parse_reaction_smiles
        from rxngraphormer.preprocessing.reactions import split_reaction_smiles as impl_split_reaction_smiles
        from rxngraphormer.preprocessing.workflow import preprocess_from_config as impl_preprocess_from_config
        from rxngraphormer.training import NoamLR as BoundaryNoamLR
        from rxngraphormer.training import RXNGraphormerLitModule as BoundaryLitModule
        from rxngraphormer.training import SPLITClassifierTrainer as BoundarySplitClassifierTrainer
        from rxngraphormer.training import get_linear_scheduler_with_warmup as BoundaryLinearWarmup
        from rxngraphormer.training.legacy import SPLITClassifierTrainer as ImplSplitClassifierTrainer
        from rxngraphormer.training.schedulers import NoamLR as ImplNoamLR
        from rxngraphormer.training.schedulers import get_linear_scheduler_with_warmup as ImplLinearWarmup

        self.assertEqual(CheckpointAdapter.__name__, "CheckpointAdapter")
        self.assertIs(boundary_load_legacy_torch, impl_load_legacy_torch)
        self.assertIs(boundary_classification_metrics, classification_metrics)
        self.assertIs(EvaluationClassificationSettings, BoundaryClassificationEvaluationSettings)
        self.assertIs(EvaluationRegressionSettings, BoundaryRegressionEvaluationSettings)
        self.assertIs(BoundaryPredictor, RXNGraphormerPredictor)
        self.assertIs(BoundaryRXNClassifier, ImplRXNClassifier)
        self.assertIs(BoundaryRXNEMB, ImplRXNEMB)
        self.assertEqual(RXNGClassifier.__name__, "RXNGClassifier")
        self.assertTrue(callable(build_classification_model))
        self.assertIs(BoundarySplitClassifierTrainer, ImplSplitClassifierTrainer)
        self.assertIs(BoundaryLitModule, RXNGraphormerLitModule)
        self.assertIs(BoundaryNoamLR, ImplNoamLR)
        self.assertIs(BoundaryLinearWarmup, ImplLinearWarmup)
        self.assertIs(BoundaryRXNDataset, DatasetImplRXNDataset)
        self.assertIs(LegacyRXNDataset, DatasetImplRXNDataset)
        self.assertIs(ConcreteRXNDataset, DatasetImplRXNDataset)
        self.assertIs(boundary_get_rxn_pfm_info, impl_get_rxn_pfm_info)
        self.assertIs(boundary_pad_feat, impl_pad_feat)
        self.assertIs(boundary_update_batch_idx, impl_update_batch_idx)
        self.assertIs(boundary_add_dense_empty_node_edge, impl_add_dense_empty_node_edge)
        self.assertIs(boundary_add_empty_node_and_edge, impl_add_empty_node_and_edge)
        self.assertIs(boundary_ext_feat_gen, impl_ext_feat_gen)
        self.assertIs(boundary_read_reaction_table, impl_read_reaction_table)
        self.assertIs(boundary_reaction_sides_from_row, impl_reaction_sides_from_row)
        self.assertIs(boundary_write_reaction_table_files, impl_write_reaction_table_files)
        self.assertIs(boundary_preprocess_from_config, impl_preprocess_from_config)
        self.assertIs(boundary_generate_mid_smiles, impl_generate_mid_smiles)
        self.assertIs(boundary_canonical_smiles, impl_canonical_smiles)
        self.assertIs(boundary_gen_mid_mols, impl_gen_mid_mols)
        self.assertIs(boundary_gen_truth_false_rxn_smi, impl_gen_truth_false_rxn_smi)
        self.assertIs(boundary_get_random_shuffle_smiles, impl_get_random_shuffle_smiles)
        self.assertIs(boundary_inchi_to_smiles, impl_inchi_to_smiles)
        self.assertIs(boundary_mod_mol, impl_mod_mol)
        self.assertIs(boundary_parse_reaction_smiles, impl_parse_reaction_smiles)
        self.assertIs(boundary_split_reaction_smiles, impl_split_reaction_smiles)
        self.assertIs(boundary_canonicalize_reaction_side, impl_canonicalize_reaction_side)
        self.assertIs(boundary_reaction_smiles_pair_lines, impl_reaction_smiles_pair_lines)
        self.assertIs(inference_reaction_smiles_pair_lines, impl_reaction_smiles_pair_lines)
        self.assertIs(boundary_get_sin_encodings, impl_get_sin_encodings)
        self.assertIs(boundary_index_scatter, impl_index_scatter)
        self.assertIs(boundary_index_select_ND, impl_index_select_ND)
        self.assertIs(boundary_scaled_dot_product_attention, impl_scaled_dot_product_attention)

    def test_new_code_does_not_import_flat_legacy_paths(self):
        repo_root = Path(__file__).resolve().parents[1]
        legacy_modules = {
            "rxngraphormer.checkpointing",
            "rxngraphormer.compat",
            "rxngraphormer.classification_workflow",
            "rxngraphormer.dataloader",
            "rxngraphormer.datamodule",
            "rxngraphormer.datasets",
            "rxngraphormer.eval",
            "rxngraphormer.evaluator",
            "rxngraphormer.ext_feat",
            "rxngraphormer.layer",
            "rxngraphormer.model",
            "rxngraphormer.model_factory",
            "rxngraphormer.predictor",
            "rxngraphormer.preprocess",
            "rxngraphormer.preprocess.data",
            "rxngraphormer.preprocess.ext_feat",
            "rxngraphormer.preprocess.compat",
            "rxngraphormer.preprocess.table",
            "rxngraphormer.reaction",
            "rxngraphormer.regression_workflow",
            "rxngraphormer.rxn_emb",
            "rxngraphormer.scheduler",
            "rxngraphormer.train",
            "rxngraphormer.utils",
            "rxngraphormer.lightning_module",
            "rxngraphormer.runtime_callbacks",
        }
        violations: list[str] = []

        source_files = [
            *repo_root.glob("src/rxngraphormer/**/*.py"),
            *repo_root.glob("scripts/**/*.py"),
            *repo_root.glob("tests/**/*.py"),
        ]
        for path in sorted(source_files):
            rel_path = path.relative_to(repo_root)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = alias.name
                        if any(imported == module or imported.startswith(f"{module}.") for module in legacy_modules):
                            violations.append(f"{rel_path}:{node.lineno}: import {imported}")
                elif isinstance(node, ast.ImportFrom):
                    if not node.module:
                        continue
                    imported = _absolute_import_module(rel_path, node.module, node.level)
                    if any(imported == module or imported.startswith(f"{module}.") for module in legacy_modules):
                        violations.append(f"{rel_path}:{node.lineno}: from {imported} import ...")

        self.assertEqual(violations, [])
        missing_files = [
            Path("src", *module.split(".")).with_suffix(".py")
            for module in sorted(legacy_modules)
            if module.startswith("rxngraphormer.")
        ]
        self.assertEqual([str(path) for path in missing_files if (repo_root / path).exists()], [])

    def test_compat_cli_runs_classification_checkpoint_check(self):
        fake_report = type(
            "Report",
            (),
            {
                "ok": True,
                "summary": lambda self: {"ok": True, "bare_loaded_keys": 3, "lightning_loaded_keys": 3},
            },
        )()
        argv = [
            "rxngraphormer-compat",
            "classification-checkpoint",
            "--model_path",
            "model_dir",
            "--ckpt_file",
            "best.pt",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.compatibility.check_classification_checkpoint_compatibility", return_value=fake_report) as check,
        ):
            cli.compat_main()

        settings = check.call_args.args[0]
        self.assertIsInstance(settings, ClassificationCheckpointCompatibilitySettings)
        self.assertEqual(settings.model_path, "model_dir")
        self.assertEqual(settings.ckpt_file, "best.pt")

    def test_compat_cli_runs_regression_pretrain_check(self):
        fake_report = type(
            "Report",
            (),
            {
                "ok": True,
                "summary": lambda self: {
                    "ok": True,
                    "max_rct_encoder_abs_diff": 0.0,
                    "max_pdt_encoder_abs_diff": 0.0,
                },
            },
        )()
        argv = [
            "rxngraphormer-compat",
            "regression-pretrain",
            "--config",
            "config.json",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.compatibility.check_regression_pretrain_compatibility", return_value=fake_report) as check,
        ):
            cli.compat_main()

        settings = check.call_args.args[0]
        self.assertIsInstance(settings, RegressionPretrainCompatibilitySettings)
        self.assertEqual(settings.config_path, "config.json")

    def test_predict_sequence_cli_writes_sequence_predictions_and_passes_checkpoint(self):
        calls = {}

        def fake_reaction_prediction(model_path, rxn_smiles_lst, task_type, params, ckpt_file, device):
            calls["prediction"] = (model_path, rxn_smiles_lst, task_type, params, ckpt_file, device)
            return pd.DataFrame([["CCO", "CCN"], ["CCC", "CCCl"]], index=["Top-1", "Top-2"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "sequence_predictions.csv"
            argv = [
                "rxngraphormer-predict-sequence",
                "--model_path",
                "sequence_model",
                "--task",
                "forward-synthesis",
                "--ckpt_file",
                "best.safetensors",
                "--input_smiles",
                "CCO",
                "CCN",
                "--batch_size",
                "8",
                "--beam_size",
                "4",
                "--n_best",
                "2",
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
            with (
                mock.patch("sys.argv", argv),
                mock.patch("rxngraphormer.evaluation.legacy_eval.reaction_prediction", side_effect=fake_reaction_prediction),
            ):
                cli.predict_sequence_main()

            rows = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(calls["prediction"][0], "sequence_model")
        self.assertEqual(calls["prediction"][1], ["CCO", "CCN"])
        self.assertEqual(calls["prediction"][2], "forward-synthesis")
        self.assertEqual(calls["prediction"][3]["batch_size"], 8)
        self.assertEqual(calls["prediction"][3]["beam_size"], 4)
        self.assertEqual(calls["prediction"][3]["n_best"], 2)
        self.assertEqual(calls["prediction"][4], "best.safetensors")
        self.assertEqual(calls["prediction"][5], "cpu")
        self.assertEqual(rows[0], ",0,1")


class RegressionEvaluationWorkflowTest(unittest.TestCase):
    def test_prediction_evaluation_uses_unified_metrics(self):
        prediction = RegressionPrediction(
            preds=torch.tensor([[0.1], [0.2]]),
            targets=torch.tensor([[0.0], [0.3]]),
        )

        evaluation = evaluate_regression_prediction(prediction, scale=100.0, yield_constrain=True)

        self.assertEqual(evaluation.metrics.count, 2)
        self.assertAlmostEqual(evaluation.metrics.mae, 10.0, places=5)
        self.assertTrue(torch.equal(evaluation.preds, torch.tensor([[10.0], [20.0]])))

    def test_eval_cli_regression_uses_workflow_and_writes_reports(self):
        calls = {}
        config = type(
            "Config",
            (),
            {
                "task": "regression",
                "trained_model_path": "model_dir",
                "ckpt_file": "best.pt",
                "scale": 100.0,
                "yield_constrain": True,
            },
        )()
        fake_result = RegressionSplitResult(
            split="valid",
            evaluation=RegressionEvaluation(
                metrics=RegressionMetrics(count=2, mae=1.5, rmse=2.0, r2=0.8),
                preds=torch.tensor([[1.0], [2.0]]),
                targets=torch.tensor([[1.5], [2.5]]),
            ),
            files={"rct": "rct.csv", "pdt": "pdt.csv", "mid": "mid.csv"},
            model_path="model_dir",
            config_path="model_dir/config.json",
            checkpoint_path="model_dir/model/best.pt",
        )

        def fake_evaluate(settings):
            calls["settings"] = settings
            return fake_result

        argv = [
            "rxngraphormer-eval",
            "--config_json",
            "eval.json",
            "--split",
            "valid",
            "--batch_size",
            "64",
            "--max_batches",
            "2",
            "--output_json",
            "report.json",
            "--output_csv",
            "report.csv",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.cli.eval.load_config", return_value=config),
            mock.patch("rxngraphormer.evaluation.regression_workflow.evaluate_regression_split", side_effect=fake_evaluate),
            mock.patch("rxngraphormer.evaluation.regression_workflow.write_regression_json") as write_json,
            mock.patch("rxngraphormer.evaluation.regression_workflow.write_regression_csv") as write_csv,
        ):
            cli.eval_main()

        settings = calls["settings"]
        self.assertIsInstance(settings, RegressionEvaluationSettings)
        self.assertEqual(settings.model_path, "model_dir")
        self.assertEqual(settings.ckpt_file, "best.pt")
        self.assertEqual(settings.split, "valid")
        self.assertEqual(settings.batch_size, 64)
        self.assertEqual(settings.max_batches, 2)
        self.assertEqual(settings.scale, 100.0)
        self.assertTrue(settings.yield_constrain)
        write_json.assert_called_once_with(fake_result, "report.json")
        write_csv.assert_called_once_with(fake_result, "report.csv")

    def test_eval_cli_classification_uses_workflow_and_writes_reports(self):
        calls = {}
        config = type(
            "Config",
            (),
            {
                "task": "classification",
                "trained_model_path": "model_dir",
                "ckpt_file": "best.pt",
            },
        )()
        fake_result = ClassificationSplitResult(
            split="valid",
            evaluation=ClassificationEvaluation(
                metrics=ClassificationMetrics(count=2, accuracy=0.5, loss=1.25, mean_confidence=0.8),
                logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                probabilities=torch.tensor([[0.7, 0.3], [0.4, 0.6]]),
                preds=torch.tensor([0, 1]),
                targets=torch.tensor([0, 0]),
                confidence=torch.tensor([0.7, 0.6]),
            ),
            files={"rct": "rct.csv", "pdt": "pdt.csv"},
            model_path="model_dir",
            config_path="model_dir/config.json",
            checkpoint_path="model_dir/model/best.pt",
        )

        def fake_evaluate(settings):
            calls["settings"] = settings
            return fake_result

        argv = [
            "rxngraphormer-eval",
            "--config_json",
            "eval.json",
            "--split",
            "valid",
            "--batch_size",
            "64",
            "--max_batches",
            "2",
            "--output_json",
            "report.json",
            "--output_csv",
            "report.csv",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.cli.eval.load_config", return_value=config),
            mock.patch("rxngraphormer.evaluation.classification_workflow.evaluate_classification_split", side_effect=fake_evaluate),
            mock.patch("rxngraphormer.evaluation.classification_workflow.write_classification_json") as write_json,
            mock.patch("rxngraphormer.evaluation.classification_workflow.write_classification_csv") as write_csv,
        ):
            cli.eval_main()

        settings = calls["settings"]
        self.assertIsInstance(settings, ClassificationEvaluationSettings)
        self.assertEqual(settings.model_path, "model_dir")
        self.assertEqual(settings.ckpt_file, "best.pt")
        self.assertEqual(settings.split, "valid")
        self.assertEqual(settings.batch_size, 64)
        self.assertEqual(settings.max_batches, 2)
        write_json.assert_called_once_with(fake_result, "report.json")
        write_csv.assert_called_once_with(fake_result, "report.csv")

    def test_legacy_eval_function_delegates_to_regression_workflow(self):
        from rxngraphormer.evaluation import legacy_eval as eval_module

        fake_result = RegressionSplitResult(
            split="test",
            evaluation=RegressionEvaluation(
                metrics=RegressionMetrics(count=2, mae=1.25, rmse=1.5, r2=0.7),
                preds=torch.tensor([[1.0], [2.0]]),
                targets=torch.tensor([[1.5], [2.5]]),
            ),
            files={"rct": "rct.csv", "pdt": "pdt.csv", "mid": "mid.csv"},
            model_path="model_dir",
            config_path="model_dir/config.json",
            checkpoint_path="model_dir/model/best.pt",
        )
        with mock.patch("rxngraphormer.evaluation.legacy_eval.evaluate_regression_split", return_value=fake_result) as evaluate:
            r2, mae, preds, targets = eval_module.eval_regression_performance(
                "model_dir",
                ckpt_file="best.pt",
                scale=100.0,
                yield_constrain=True,
                max_batches=3,
            )

        settings = evaluate.call_args.args[0]
        self.assertEqual(settings.model_path, "model_dir")
        self.assertEqual(settings.ckpt_file, "best.pt")
        self.assertEqual(settings.split, "test")
        self.assertEqual(settings.max_batches, 3)
        self.assertEqual(r2, 0.7)
        self.assertEqual(mae, 1.25)
        self.assertTrue(torch.equal(preds, fake_result.preds))
        self.assertTrue(torch.equal(targets, fake_result.targets))


def _absolute_import_module(rel_path: Path, module: str, level: int) -> str:
    if level == 0:
        return module

    package_parts = list(rel_path.with_suffix("").parts[:-1])
    if not package_parts:
        return module
    base_parts = package_parts[: len(package_parts) - level + 1]
    return ".".join([*base_parts, module])
