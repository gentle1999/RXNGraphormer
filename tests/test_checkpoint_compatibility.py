import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from rxngraphormer.compatibility import (
    ClassificationCheckpointCompatibilitySettings,
    RegressionPretrainCompatibilitySettings,
    check_classification_checkpoint_compatibility,
    check_regression_pretrain_compatibility,
)
from rxngraphormer.compatibility.checkpointing import CheckpointAdapter, resolve_model_checkpoint
from rxngraphormer.serialization import save_checkpoint_payload


class CheckpointAdapterTest(unittest.TestCase):
    def test_resolve_model_checkpoint_prefers_safetensors_and_falls_back_to_torch_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            checkpoint_dir = model_dir / "model"
            checkpoint_dir.mkdir()
            safetensors_path = checkpoint_dir / "valid_checkpoint.safetensors"
            torch_path = checkpoint_dir / "valid_checkpoint.pt"
            save_checkpoint_payload({"model_state_dict": {"w": torch.tensor([1.0])}}, safetensors_path)
            torch.save({"model_state_dict": {"w": torch.tensor([2.0])}}, torch_path)

            resolved = resolve_model_checkpoint(model_dir)

        self.assertEqual(resolved, safetensors_path)

        for suffix in (".pt", ".pth", ".ckpt"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp_dir:
                model_dir = Path(tmp_dir)
                checkpoint_dir = model_dir / "model"
                checkpoint_dir.mkdir()
                legacy_path = checkpoint_dir / f"valid_checkpoint{suffix}"
                torch.save({"model_state_dict": {"w": torch.tensor([1.0])}}, legacy_path)

                resolved = resolve_model_checkpoint(model_dir)

            self.assertEqual(resolved, legacy_path)

    def test_extract_state_dict_accepts_bare_and_wrapped_checkpoint_payloads(self):
        adapter = CheckpointAdapter()
        state_dict = {"weight": torch.tensor([1.0])}

        for payload in (
            state_dict,
            {"model_state_dict": state_dict},
            {"state_dict": state_dict},
            {"model": state_dict},
        ):
            with self.subTest(keys=list(payload)):
                extracted = adapter.extract_state_dict(payload)

            self.assertTrue(torch.equal(extracted["weight"], state_dict["weight"]))

    def test_canonicalize_removes_prefixes_adds_prefix_and_maps_legacy_encoder_keys(self):
        adapter = CheckpointAdapter(add_prefix="model.")
        state_dict = {
            "module.model.rct_encoder.x_embedding.weight": torch.tensor([1.0]),
            "module.head.bias": torch.tensor([2.0]),
        }

        canonical, report = adapter.canonicalize(state_dict)

        self.assertEqual(
            sorted(canonical),
            [
                "model.head.bias",
                "model.rct_encoder.rxn_graph_encoder.x_embedding.weight",
            ],
        )
        self.assertEqual(report.removed_prefixes, ["model.", "module."])
        self.assertEqual(report.added_prefix, "model.")
        self.assertEqual(
            report.renamed_keys["module.model.rct_encoder.x_embedding.weight"],
            "model.rct_encoder.rxn_graph_encoder.x_embedding.weight",
        )

    def test_shape_compatible_mode_loads_only_matching_keys_and_reports_mismatches(self):
        model = torch.nn.Linear(2, 1)
        original_bias = model.bias.detach().clone()
        checkpoint = {
            "model_state_dict": {
                "weight": torch.full_like(model.weight, 3.0),
                "bias": torch.ones(2),
                "extra": torch.tensor([5.0]),
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "valid_checkpoint.safetensors"
            save_checkpoint_payload(checkpoint, path)

            report = CheckpointAdapter().load_into_model(model, path, mode="shape-compatible")

        self.assertTrue(torch.equal(model.weight, torch.full_like(model.weight, 3.0)))
        self.assertTrue(torch.equal(model.bias, original_bias))
        self.assertEqual(report.loaded_key_count, 1)
        self.assertIn("bias", report.shape_mismatches)
        self.assertEqual(report.missing_keys, ["bias"])
        self.assertEqual(report.unexpected_keys, ["extra"])
        self.assertFalse(report.ok)

    def test_diagnostic_mode_reports_shape_mismatch_without_raising(self):
        model = torch.nn.Linear(2, 1)
        checkpoint = {"model_state_dict": {"weight": torch.ones(3, 3), "bias": torch.ones(1)}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "valid_checkpoint.pt"
            torch.save(checkpoint, path)

            report = CheckpointAdapter().load_into_model(model, path, mode="diagnostic")

        self.assertIn("weight", report.shape_mismatches)
        self.assertEqual(report.loaded_key_count, 1)
        self.assertEqual(report.missing_keys, ["weight"])

    def test_strict_mode_raises_on_shape_mismatch(self):
        model = torch.nn.Linear(2, 1)
        checkpoint = {"model_state_dict": {"weight": torch.ones(3, 3), "bias": torch.ones(1)}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "valid_checkpoint.safetensors"
            save_checkpoint_payload(checkpoint, path)

            with self.assertRaisesRegex(RuntimeError, "Checkpoint shape mismatch"):
                CheckpointAdapter().load_into_model(model, path, mode="strict")

    def test_add_prefix_loads_bare_state_dict_into_lightning_style_wrapper(self):
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 1)

        wrapper = Wrapper()
        checkpoint = {
            "model_state_dict": {
                "weight": torch.full_like(wrapper.model.weight, 4.0),
                "bias": torch.full_like(wrapper.model.bias, 5.0),
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "valid_checkpoint.safetensors"
            save_checkpoint_payload(checkpoint, path)

            report = CheckpointAdapter(add_prefix="model.").load_into_model(wrapper, path, mode="strict")

        self.assertEqual(report.loaded_key_count, 2)
        self.assertEqual(report.added_prefix, "model.")
        self.assertEqual(report.missing_keys, [])
        self.assertEqual(report.unexpected_keys, [])
        self.assertTrue(torch.equal(wrapper.model.weight, torch.full_like(wrapper.model.weight, 4.0)))
        self.assertTrue(torch.equal(wrapper.model.bias, torch.full_like(wrapper.model.bias, 5.0)))


class CheckpointCompatibilitySmokeTest(unittest.TestCase):
    def test_classification_checkpoint_compatibility_compares_bare_and_lightning_logits(self):
        config = type(
            "Config",
            (),
            {
                "task": "classification",
                "training": type("Training", (), {"loss": "ce"})(),
            },
        )()

        class FakeClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))

            def logits(self, data):
                return data[0].x @ self.weight

            def forward(self, data):
                return torch.softmax(self.logits(data), dim=-1)

        class FakeBatch:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def to(self, device):
                self.x = self.x.to(device)
                self.y = self.y.to(device)
                return self

        class FakeAdapter:
            def load_into_model(self, model, path, *, map_location, mode):
                return type(
                    "Report",
                    (),
                    {
                        "ok": True,
                        "loaded_key_count": len(model.state_dict()),
                        "source_key_count": len(model.state_dict()),
                        "missing_keys": [],
                        "unexpected_keys": [],
                        "shape_mismatches": {},
                    },
                )()

        batch = (
            FakeBatch(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
            FakeBatch(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
        )
        with (
            mock.patch("rxngraphormer.compatibility.classification.resolve_config_file", return_value=Path("parameters.json")),
            mock.patch("rxngraphormer.compatibility.classification.load_config", return_value=config),
            mock.patch("rxngraphormer.compatibility.classification.build_classification_model", side_effect=lambda cfg: FakeClassifier()),
            mock.patch("rxngraphormer.compatibility.classification.CheckpointAdapter", return_value=FakeAdapter()),
        ):
            report = check_classification_checkpoint_compatibility(
                ClassificationCheckpointCompatibilitySettings(model_path="model_dir"),
                batch=batch,
            )

        self.assertTrue(report.ok)
        self.assertEqual(report.batch_size, 2)
        self.assertEqual(report.max_logits_abs_diff, 0.0)
        self.assertEqual(report.max_probabilities_abs_diff, 0.0)

    def test_regression_pretrain_compatibility_compares_encoder_state_dicts(self):
        config = type(
            "Config",
            (),
            {
                "model": type("Model", (), {"pretrained_model_path": "pretrained"})(),
            },
        )()

        class FakeEncoder(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[value, value + 1.0]]))

        class FakePretrained(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rct_encoder = FakeEncoder(1.0)
                self.pdt_encoder = FakeEncoder(3.0)

        class FakeRegression(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rct_encoder = FakeEncoder(1.0)
                self.pdt_encoder = FakeEncoder(3.0)

        with (
            mock.patch("rxngraphormer.compatibility.regression.load_config", return_value=config),
            mock.patch("rxngraphormer.compatibility.regression.build_pretrained_classification_model", return_value=FakePretrained()),
            mock.patch("rxngraphormer.compatibility.regression.build_regression_model_from_config", return_value=FakeRegression()),
        ):
            report = check_regression_pretrain_compatibility(
                RegressionPretrainCompatibilitySettings(config_path="config.json")
            )

        self.assertTrue(report.ok)
        self.assertEqual(report.rct_encoder_keys, 1)
        self.assertEqual(report.pdt_encoder_keys, 1)
        self.assertEqual(report.max_rct_encoder_abs_diff, 0.0)
        self.assertEqual(report.max_pdt_encoder_abs_diff, 0.0)
