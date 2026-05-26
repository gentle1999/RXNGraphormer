import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from rxngraphormer import cli
from rxngraphormer.data import (
    RXNG2SDataset,
    get_rxn_pfm_info,
)
from rxngraphormer.evaluation import (
    sequence_exact_match,
)
from rxngraphormer.inference import (
    RXNEMB,
    ClassificationPrediction,
    EmbeddingPrediction,
    RegressionPrediction,
    RXNGraphormerPredictor,
    export_embeddings_csv,
    export_predictions_csv,
)
from rxngraphormer.inference.inputs import (
    reaction_smiles_from_rows,
    regression_table_lines,
    write_reaction_smiles_pair_files,
)


class SequenceMetricTest(unittest.TestCase):
    def test_sequence_exact_match_checks_exact_token_match(self):
        pred = torch.tensor([[2, 3, 4], [2, 3, 5], [0, 1, 2]])
        target = torch.tensor([[2, 3, 4], [2, 3, 4], [0, 1, 2]])

        acc = sequence_exact_match(pred, target)

        self.assertTrue(torch.equal(acc, torch.tensor([1.0, 0.0, 1.0])))


class SequenceDatasetTest(unittest.TestCase):
    def test_rxng2s_dataset_reports_empty_generated_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "src.txt").write_text("notasmiles\n")
            Path(tmp_dir, "tgt.txt").write_text("C\n")
            Path(tmp_dir, "vocab.txt").write_text("_PAD\n_EOS\nC\n")

            with self.assertRaisesRegex(ValueError, "No valid sequence graph data"):
                RXNG2SDataset(
                    tmp_dir,
                    src_file="src.txt",
                    tgt_file="tgt.txt",
                    vocab_file="vocab.txt",
                    trunck=1,
                    multi_process=False,
                    oh=False,
                )


class ReactionParsingTest(unittest.TestCase):
    def test_invalid_reaction_smiles_returns_none_instead_of_unpack_error(self):
        self.assertIsNone(get_rxn_pfm_info(("notasmiles,1.0", "regression", False, "morgan", None, "mean")))


class TemporaryInputApiTest(unittest.TestCase):
    def test_gen_rxn_emb_uses_cleaned_isolated_temp_directory(self):
        api = object.__new__(RXNEMB)
        seen_roots = []

        def fake_from_dataset(root, **kwargs):
            seen_roots.append(Path(root))
            self.assertTrue((Path(root) / "rct_smiles_0.csv").exists())
            self.assertTrue((Path(root) / "pdt_smiles_0.csv").exists())
            return torch.ones(2, 3)

        api.gen_rxn_emb_from_dataset = fake_from_dataset
        with mock.patch("rxngraphormer.preprocessing.chemistry.canonical_smiles", side_effect=lambda smi: smi):
            out = api.gen_rxn_emb(["C>>O", "CC>>CO"], batch_size=2)

        self.assertTrue(torch.equal(out, torch.ones(2, 3)))
        self.assertEqual(len(seen_roots), 1)
        self.assertFalse(seen_roots[0].exists())

    def test_predictor_dataset_classification_returns_cpu_tensors(self):
        predictor = object.__new__(RXNGraphormerPredictor)
        predictor.device = torch.device("cpu")
        predictor.task = "classification"

        class FakeModel:
            def __call__(self, data):
                return torch.tensor([[0.1, 0.9], [0.8, 0.2]])

        predictor.model = FakeModel()

        class FakeDataset:
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return idx

        class FakeGraph:
            def to(self, device):
                return self

        class FakeDataLoader:
            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                yield FakeGraph(), FakeGraph()

        with (
            mock.patch("rxngraphormer.inference.predictor.MultiRXNDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.inference.predictor.PairDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.inference.predictor.torch.utils.data.DataLoader", FakeDataLoader),
        ):
            result = predictor.predict_from_dataset(
                "root",
                rct_name_regrex="rct.csv",
                pdt_name_regrex="pdt.csv",
                return_probabilities=True,
            )

        self.assertTrue(torch.equal(result.preds, torch.tensor([1, 0])))
        self.assertTrue(torch.allclose(result.confidence, torch.tensor([0.9, 0.8])))
        self.assertTrue(torch.allclose(result.probabilities, torch.tensor([[0.1, 0.9], [0.8, 0.2]])))

    def test_predictor_dataset_regression_returns_predictions_and_targets(self):
        predictor = object.__new__(RXNGraphormerPredictor)
        predictor.device = torch.device("cpu")
        predictor.task = "regression"
        predictor.config = type("Config", (), {"model": type("Model", (), {"use_mid_inf": False})()})()

        class FakeModel:
            def __call__(self, data):
                return torch.tensor([[1.5], [2.5]])

        predictor.model = FakeModel()

        class FakeDataset:
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return idx

        class FakeGraph:
            def __init__(self, y):
                self.y = y

            def to(self, device):
                return self

        class FakeDataLoader:
            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                yield FakeGraph(torch.tensor([0.1, 0.2])), FakeGraph(torch.tensor([0.3, 0.4]))

        with (
            mock.patch("rxngraphormer.inference.predictor.MultiRXNDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.inference.predictor.PairDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.inference.predictor.torch.utils.data.DataLoader", FakeDataLoader),
        ):
            result = predictor.predict_regression_from_dataset(
                "root",
                rct_name_regrex="rct.csv",
                pdt_name_regrex="pdt.csv",
                return_targets=True,
            )

        self.assertTrue(torch.allclose(result.preds, torch.tensor([[1.5], [2.5]])))
        self.assertTrue(torch.allclose(result.targets, torch.tensor([[0.1], [0.2]])))

    def test_predictor_from_checkpoint_uses_explicit_config_and_checkpoint(self):
        calls = {}

        class FakeModel:
            def parameters(self):
                return []

            def to(self, device):
                calls["to"] = device
                return self

            def eval(self):
                calls["eval"] = True
                return self

        class FakeAdapter:
            def load_into_model(self, model, path, *, map_location, mode):
                calls["load"] = (model, path, map_location, mode)

        fake_config = type("Config", (), {"model": type("Model", (), {"use_mid_inf": False})()})()
        with (
            mock.patch("rxngraphormer.inference.predictor.load_config", return_value=fake_config) as load_config_mock,
            mock.patch("rxngraphormer.inference.predictor.build_regression_model", return_value=FakeModel()) as build_mock,
            mock.patch("rxngraphormer.inference.predictor.CheckpointAdapter", return_value=FakeAdapter()),
        ):
            predictor = RXNGraphormerPredictor.from_checkpoint(
                "runs/model.ckpt",
                config_path="runs/config.json",
                device="cpu",
            )

        load_config_mock.assert_called_once_with("runs/config.json")
        build_mock.assert_called_once_with(fake_config)
        self.assertEqual(calls["load"][1], Path("runs/model.ckpt"))
        self.assertEqual(calls["load"][2], "cpu")
        self.assertEqual(calls["load"][3], "strict")
        self.assertIs(predictor.config, fake_config)

    def test_predictor_table_classification_uses_rxn_smiles_column(self):
        predictor = object.__new__(RXNGraphormerPredictor)
        predictor.task = "classification"
        seen = {}

        def fake_predict_reactions(rxn_smiles, **kwargs):
            seen["rxn_smiles"] = list(rxn_smiles)
            seen["kwargs"] = kwargs
            return ClassificationPrediction(preds=torch.tensor([1, 0]), confidence=torch.tensor([0.9, 0.8]))

        predictor.predict_reactions = fake_predict_reactions
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, "input.csv")
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["rxn_smiles"])
                writer.writeheader()
                writer.writerow({"rxn_smiles": "C>>O"})
                writer.writerow({"rxn_smiles": "CC>>CO"})

            result = predictor.predict_table(path, batch_size=7, return_probabilities=True, return_uncertainty=True)

        self.assertTrue(torch.equal(result.preds, torch.tensor([1, 0])))
        self.assertEqual(seen["rxn_smiles"], ["C>>O", "CC>>CO"])
        self.assertEqual(seen["kwargs"]["batch_size"], 7)
        self.assertTrue(seen["kwargs"]["return_probabilities"])
        self.assertTrue(seen["kwargs"]["return_uncertainty"])

    def test_predictor_table_classification_falls_back_to_split_smiles_when_rxn_empty(self):
        predictor = object.__new__(RXNGraphormerPredictor)
        predictor.task = "classification"
        seen = {}

        def fake_predict_reactions(rxn_smiles, **kwargs):
            seen["rxn_smiles"] = list(rxn_smiles)
            return ClassificationPrediction(preds=torch.tensor([1, 0]), confidence=torch.tensor([0.9, 0.8]))

        predictor.predict_reactions = fake_predict_reactions
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, "input.csv")
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["rxn_smiles", "rct_smiles", "pdt_smiles"])
                writer.writeheader()
                writer.writerow({"rxn_smiles": "CCO>O>CC=O", "rct_smiles": "", "pdt_smiles": ""})
                writer.writerow({"rxn_smiles": "", "rct_smiles": "CCN", "pdt_smiles": "CC=N"})

            predictor.predict_table(path)

        self.assertEqual(seen["rxn_smiles"], ["CCO>O>CC=O", "CCN>>CC=N"])

    def test_predictor_table_regression_writes_split_smiles_and_targets(self):
        predictor = object.__new__(RXNGraphormerPredictor)
        predictor.task = "regression"
        predictor.config = type("Config", (), {"model": type("Model", (), {"use_mid_inf": True})()})()
        seen = {}

        def fake_predict_from_dataset(root, **kwargs):
            root = Path(root)
            seen["rct"] = (root / "rct_smiles_0.csv").read_text().splitlines()
            seen["pdt"] = (root / "pdt_smiles_0.csv").read_text().splitlines()
            seen["mid"] = (root / "mid_smiles_0.csv").read_text().splitlines()
            seen["kwargs"] = kwargs
            return RegressionPrediction(preds=torch.tensor([[1.5], [2.5]]))

        predictor.predict_regression_from_dataset = fake_predict_from_dataset
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, "input.csv")
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["rct_smiles", "pdt_smiles", "mid_smiles", "target"])
                writer.writeheader()
                writer.writerow({"rct_smiles": "A", "pdt_smiles": "B", "mid_smiles": "M1", "target": "1.2"})
                writer.writerow({"rct_smiles": "C", "pdt_smiles": "D", "mid_smiles": "M2", "target": "3.4"})

            with (
                mock.patch("rxngraphormer.preprocessing.materialization.canonicalize_reaction_side", side_effect=lambda smi: smi),
                mock.patch("rxngraphormer.preprocessing.materialization.canonical_smiles", side_effect=lambda smi: smi),
            ):
                result = predictor.predict_table(path, target_column="target", return_targets=True)

        self.assertTrue(torch.equal(result.preds, torch.tensor([[1.5], [2.5]])))
        self.assertEqual(seen["rct"], ["A,1.2", "C,3.4"])
        self.assertEqual(seen["pdt"], ["B,1.2", "D,3.4"])
        self.assertEqual(seen["mid"], ["M1,1.2", "M2,3.4"])

    def test_predictor_table_regression_accepts_standard_rxn_smiles_with_agents(self):
        predictor = object.__new__(RXNGraphormerPredictor)
        predictor.task = "regression"
        predictor.config = type("Config", (), {"model": type("Model", (), {"use_mid_inf": False})()})()
        seen = {}

        def fake_predict_from_dataset(root, **kwargs):
            root = Path(root)
            seen["rct"] = (root / "rct_smiles_0.csv").read_text().splitlines()
            seen["pdt"] = (root / "pdt_smiles_0.csv").read_text().splitlines()
            seen["kwargs"] = kwargs
            return RegressionPrediction(preds=torch.tensor([[1.5]]))

        predictor.predict_regression_from_dataset = fake_predict_from_dataset
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, "input.csv")
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["rxn_smiles", "target"])
                writer.writeheader()
                writer.writerow({"rxn_smiles": "CCO>O>CC=O", "target": "1.2"})

            result = predictor.predict_table(path, target_column="target", return_targets=True)

        self.assertTrue(torch.equal(result.preds, torch.tensor([[1.5]])))
        self.assertEqual(seen["rct"], ["CCO.O,1.2"])
        self.assertEqual(seen["pdt"], ["CC=O.O,1.2"])
        self.assertTrue(seen["kwargs"]["return_targets"])
        self.assertFalse(seen["kwargs"]["use_mid_inf"])

    def test_predictor_embed_from_dataset_returns_cpu_embeddings(self):
        predictor = object.__new__(RXNGraphormerPredictor)
        predictor.device = torch.device("cpu")
        predictor.task = "classification"

        class FakeEncoder:
            def __init__(self, base):
                self.base = base

            def __call__(self, data):
                memory = torch.tensor(
                    [
                        [[self.base + 1.0, self.base + 2.0], [self.base + 3.0, self.base + 4.0]],
                        [[self.base + 5.0, self.base + 6.0], [99.0, 99.0]],
                    ]
                )
                return memory, None, torch.tensor([2, 1])

        class FakeDecoder:
            batch_norm = False
            layers = torch.nn.ModuleList()

        class FakeModel:
            trans_readout = "mean"
            split_merge_method = "all"
            rct_encoder = FakeEncoder(0.0)
            pdt_encoder = FakeEncoder(10.0)
            decoder = FakeDecoder()

        predictor.model = FakeModel()

        class FakeDataset:
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return idx

        class FakeGraph:
            def to(self, device):
                return self

        class FakeDataLoader:
            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                yield FakeGraph(), FakeGraph()

        with (
            mock.patch("rxngraphormer.inference.predictor.MultiRXNDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.inference.predictor.PairDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.inference.predictor.torch.utils.data.DataLoader", FakeDataLoader),
        ):
            result = predictor.embed_from_dataset(
                "root",
                rct_name_regrex="rct.csv",
                pdt_name_regrex="pdt.csv",
            )

        expected = torch.tensor(
            [
                [3.0, 4.0, 13.0, 14.0, 10.0, 10.0],
                [3.0, 4.0, 13.0, 14.0, 10.0, 10.0],
            ]
        )
        self.assertTrue(torch.allclose(result.embeddings, expected))

    def test_export_predictions_csv_writes_classification_probabilities(self):
        result = ClassificationPrediction(
            preds=torch.tensor([1, 0]),
            confidence=torch.tensor([0.9, 0.8]),
            probabilities=torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
            uncertainty=torch.tensor([0.1, 0.2]),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, "predictions.csv")
            export_predictions_csv(result, path)

            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["prediction"], "1")
        self.assertEqual(rows[0]["confidence"], "0.8999999761581421")
        self.assertEqual(rows[0]["prob_0"], "0.10000000149011612")
        self.assertEqual(rows[0]["prob_1"], "0.8999999761581421")
        self.assertEqual(rows[0]["uncertainty"], "0.10000000149011612")

    def test_export_predictions_csv_writes_regression_targets(self):
        result = RegressionPrediction(
            preds=torch.tensor([[1.5], [2.5]]),
            targets=torch.tensor([[1.0], [2.0]]),
            uncertainty=torch.tensor([[float("nan")], [float("nan")]]),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, "regression.csv")
            export_predictions_csv(result, path)

            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["prediction"], "1.5")
        self.assertEqual(rows[0]["target"], "1.0")
        self.assertEqual(rows[0]["uncertainty"], "nan")

    def test_export_embeddings_csv_writes_embedding_columns(self):
        result = EmbeddingPrediction(embeddings=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir, "embeddings.csv")
            export_embeddings_csv(result, path)

            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["embedding_0"], "1.0")
        self.assertEqual(rows[0]["embedding_1"], "2.0")

    def test_inference_input_adapter_normalizes_table_rows(self):
        rows = [
            {"rxn_smiles": "CCO>O>CC=O", "rct_smiles": "", "pdt_smiles": ""},
            {"rxn_smiles": "", "rct_smiles": "CCN", "pdt_smiles": "CC=N"},
        ]

        result = reaction_smiles_from_rows(
            rows,
            rxn_smiles_column="rxn_smiles",
            rct_smiles_column="rct_smiles",
            pdt_smiles_column="pdt_smiles",
        )

        self.assertEqual(result, ["CCO>O>CC=O", "CCN>>CC=N"])

    def test_inference_input_adapter_writes_legacy_pair_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = write_reaction_smiles_pair_files(tmp_dir, ["CCO>O>CC=O", "CCN>>CC=N"])

            rct_lines = Path(files.root, files.rct_name).read_text().splitlines()
            pdt_lines = Path(files.root, files.pdt_name).read_text().splitlines()

        self.assertEqual(rct_lines, ["CCO.O,0", "CCN,0"])
        self.assertEqual(pdt_lines, ["CC=O.O,0", "CC=N,0"])

    def test_inference_input_adapter_builds_regression_legacy_lines(self):
        rows = [
            {"rct_smiles": "A", "pdt_smiles": "B", "mid_smiles": "M1", "target": "1.2"},
            {"rct_smiles": "C", "pdt_smiles": "D", "mid_smiles": "M2", "target": "3.4"},
        ]
        with (
            mock.patch("rxngraphormer.preprocessing.materialization.canonicalize_reaction_side", side_effect=lambda smi: smi),
            mock.patch("rxngraphormer.preprocessing.materialization.canonical_smiles", side_effect=lambda smi: smi),
        ):
            rct_lines, pdt_lines, mid_lines = regression_table_lines(
                rows,
                rxn_smiles_column="rxn_smiles",
                rct_smiles_column="rct_smiles",
                pdt_smiles_column="pdt_smiles",
                mid_smiles_column="mid_smiles",
                target_column="target",
                require_mid=True,
            )

        self.assertEqual(rct_lines, ["A,1.2", "C,3.4"])
        self.assertEqual(pdt_lines, ["B,1.2", "D,3.4"])
        self.assertEqual(mid_lines, ["M1,1.2", "M2,3.4"])


    def test_predict_cli_wires_classification_predictor_and_export(self):
        calls = {}

        class FakePredictor:
            def __init__(self, model_path, task, ckpt_file, device):
                calls["init"] = (model_path, task, ckpt_file, device)

            def predict_from_dataset(self, root, **kwargs):
                calls["predict"] = (root, kwargs)
                return ClassificationPrediction(preds=torch.tensor([1]), confidence=torch.tensor([0.9]))

        def fake_export(result, output):
            calls["export"] = (result, output)

        argv = [
            "rxngraphormer-predict",
            "--model_path",
            "model_dir",
            "--root",
            "data_dir",
            "--rct_name_regrex",
            "rct.csv",
            "--pdt_name_regrex",
            "pdt.csv",
            "--output",
            "pred.csv",
            "--return_probabilities",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.inference.RXNGraphormerPredictor", FakePredictor),
            mock.patch("rxngraphormer.inference.export_predictions_csv", fake_export),
        ):
            cli.predict_main()

        self.assertEqual(calls["init"], ("model_dir", "classification", "valid_checkpoint.safetensors", None))
        self.assertEqual(calls["predict"][0], "data_dir")
        self.assertTrue(calls["predict"][1]["return_probabilities"])
        self.assertEqual(calls["export"][1], "pred.csv")

    def test_predict_cli_regression_keeps_use_mid_inf_auto(self):
        calls = {}

        class FakePredictor:
            def __init__(self, model_path, task, ckpt_file, device):
                calls["init"] = (model_path, task, ckpt_file, device)

            def predict_regression_from_dataset(self, root, **kwargs):
                calls["predict"] = (root, kwargs)
                return RegressionPrediction(preds=torch.tensor([[1.0]]))

        def fake_export(result, output):
            calls["export"] = (result, output)

        argv = [
            "rxngraphormer-predict",
            "--model_path",
            "model_dir",
            "--task",
            "regression",
            "--root",
            "data_dir",
            "--rct_name_regrex",
            "rct.csv",
            "--pdt_name_regrex",
            "pdt.csv",
            "--mid_name_regrex",
            "mid.csv",
            "--output",
            "pred.csv",
            "--return_targets",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.inference.RXNGraphormerPredictor", FakePredictor),
            mock.patch("rxngraphormer.inference.export_predictions_csv", fake_export),
        ):
            cli.predict_main()

        self.assertEqual(calls["init"], ("model_dir", "regression", "valid_checkpoint.safetensors", None))
        self.assertEqual(calls["predict"][0], "data_dir")
        self.assertIsNone(calls["predict"][1]["use_mid_inf"])
        self.assertTrue(calls["predict"][1]["return_targets"])
        self.assertEqual(calls["export"][1], "pred.csv")

    def test_predict_cli_wires_table_predictor(self):
        calls = {}

        class FakePredictor:
            def __init__(self, model_path, task, ckpt_file, device):
                calls["init"] = (model_path, task, ckpt_file, device)

            def predict_table(self, path, **kwargs):
                calls["predict"] = (path, kwargs)
                return ClassificationPrediction(preds=torch.tensor([1]), confidence=torch.tensor([0.9]))

        def fake_export(result, output):
            calls["export"] = (result, output)

        argv = [
            "rxngraphormer-predict",
            "--model_path",
            "model_dir",
            "--input_table",
            "input.csv",
            "--output",
            "pred.csv",
            "--return_uncertainty",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.inference.RXNGraphormerPredictor", FakePredictor),
            mock.patch("rxngraphormer.inference.export_predictions_csv", fake_export),
        ):
            cli.predict_main()

        self.assertEqual(calls["init"], ("model_dir", "classification", "valid_checkpoint.safetensors", None))
        self.assertEqual(calls["predict"][0], "input.csv")
        self.assertTrue(calls["predict"][1]["return_uncertainty"])
        self.assertEqual(calls["export"][1], "pred.csv")

    def test_predict_cli_exports_embeddings_when_requested(self):
        calls = {}

        class FakePredictor:
            def __init__(self, model_path, task, ckpt_file, device):
                calls["init"] = (model_path, task, ckpt_file, device)

            def predict_from_dataset(self, root, **kwargs):
                calls["predict"] = (root, kwargs)
                return ClassificationPrediction(preds=torch.tensor([1]), confidence=torch.tensor([0.9]))

            def embed_from_dataset(self, root, **kwargs):
                calls["embed"] = (root, kwargs)
                return EmbeddingPrediction(embeddings=torch.tensor([[1.0, 2.0]]))

        def fake_export_predictions(result, output):
            calls["prediction_export"] = (result, output)

        def fake_export_embeddings(result, output):
            calls["embedding_export"] = (result, output)

        argv = [
            "rxngraphormer-predict",
            "--model_path",
            "model_dir",
            "--root",
            "data_dir",
            "--rct_name_regrex",
            "rct.csv",
            "--pdt_name_regrex",
            "pdt.csv",
            "--output",
            "pred.csv",
            "--embeddings_output",
            "emb.csv",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.inference.RXNGraphormerPredictor", FakePredictor),
            mock.patch("rxngraphormer.inference.export_predictions_csv", fake_export_predictions),
            mock.patch("rxngraphormer.inference.export_embeddings_csv", fake_export_embeddings),
        ):
            cli.predict_main()

        self.assertEqual(calls["embed"][0], "data_dir")
        self.assertEqual(calls["embedding_export"][1], "emb.csv")
