import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch_geometric.data import Data, InMemoryDataset

from rxngraphormer.config import load_config
from rxngraphormer.data.graph_data import ReactionGraphData
from rxngraphormer.inference import (
    RXNClassifier,
)
from rxngraphormer.models import (
    RXNG2Sequencer,
    sequence_dependencies,
)
from rxngraphormer.serialization import (
    convert_serialized_file,
    existing_serialized_path,
    load_checkpoint_payload,
    load_processed_graph_data,
    prefer_safetensors_path,
    save_checkpoint_payload,
    save_processed_graph_data,
)


class SequenceModelCompatibilityTest(unittest.TestCase):
    def _fake_onmt_symbols(self):
        class FakeEmbeddings(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class FakeTransformerDecoder(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class FakePositionwiseFeedForward(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        class FakePositionalEncoding(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        def fake_sequence_mask(lengths):
            max_len = int(lengths.max().item())
            return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

        return {
            "BeamSearch": object,
            "Embeddings": FakeEmbeddings,
            "GNMTGlobalScorer": object,
            "GreedySearch": object,
            "PositionalEncoding": FakePositionalEncoding,
            "PositionwiseFeedForward": FakePositionwiseFeedForward,
            "TransformerDecoder": FakeTransformerDecoder,
            "sequence_mask": fake_sequence_mask,
        }

    def _sequence_config(self, att_encoder_type):
        config = load_config("config/uspto_50k_parameters.json")
        config.model.att_encoder_type = att_encoder_type
        config.model.emb_dim = 8
        config.model.gnn_num_layer = 2
        config.model.gnum_layer = 2
        config.model.trans_num_layer = 1
        config.model.tnum_layer = 1
        config.model.num_heads = 2
        config.model.encoder_filter_size = 16
        config.model.filter_size = 16
        config.model.decoder_num_layers = 1
        return config

    def test_sequence_model_accepts_legacy_attention_encoder_values(self):
        vocab = {"_PAD": 0, "_SOS": 1, "_EOS": 2, "C": 3}

        with (
            mock.patch("rxngraphormer.models.sequence._require_onmt", return_value=self._fake_onmt_symbols()),
            mock.patch("rxngraphormer.models.attention_xl._require_onmt", return_value=self._fake_onmt_symbols()),
        ):
            self.assertIsInstance(RXNG2Sequencer(self._sequence_config("attxl"), vocab), RXNG2Sequencer)
            self.assertIsInstance(RXNG2Sequencer(self._sequence_config("attn"), vocab), RXNG2Sequencer)

    def test_sequence_model_rejects_unknown_attention_encoder_at_init(self):
        vocab = {"_PAD": 0, "_SOS": 1, "_EOS": 2, "C": 3}

        with (
            mock.patch("rxngraphormer.models.sequence._require_onmt", return_value=self._fake_onmt_symbols()),
            mock.patch("rxngraphormer.models.attention_xl._require_onmt", return_value=self._fake_onmt_symbols()),
        ):
            with self.assertRaisesRegex(NotImplementedError, "Attention encoder type invalid"):
                RXNG2Sequencer(self._sequence_config("invalid"), vocab)

    def test_sequence_dependency_error_points_to_sequence_extra(self):
        original_symbols = sequence_dependencies._ONMT_SYMBOLS
        sequence_dependencies._ONMT_SYMBOLS = None

        def fake_import(name, *args, **kwargs):
            if name.startswith("onmt"):
                raise ImportError("missing onmt")
            return original_import(name, *args, **kwargs)

        original_import = __import__
        try:
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaisesRegex(ImportError, "uv sync --extra sequence"):
                    sequence_dependencies.require_onmt()
        finally:
            sequence_dependencies._ONMT_SYMBOLS = original_symbols

    def test_rxn_pred_uses_cleaned_isolated_temp_directory(self):
        api = object.__new__(RXNClassifier)
        seen_roots = []

        def fake_from_dataset(root, **kwargs):
            seen_roots.append(Path(root))
            self.assertTrue((Path(root) / "rct_smiles_0.csv").exists())
            self.assertTrue((Path(root) / "pdt_smiles_0.csv").exists())
            return torch.tensor([1, 0]), torch.tensor([0.9, 0.8])

        api.rxn_pred_from_dataset = fake_from_dataset
        with mock.patch("rxngraphormer.preprocessing.chemistry.canonical_smiles", side_effect=lambda smi: smi):
            preds, confs = api.rxn_pred(["C>>O", "CC>>CO"], batch_size=2)

        self.assertTrue(torch.equal(preds, torch.tensor([1, 0])))
        self.assertTrue(torch.allclose(confs, torch.tensor([0.9, 0.8])))
        self.assertEqual(len(seen_roots), 1)
        self.assertFalse(seen_roots[0].exists())

    def test_processed_graph_safetensors_roundtrip_keeps_tensor_fields(self):
        data = ReactionGraphData(
            x=torch.tensor([[1], [2], [3]], dtype=torch.long),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            mol_index=torch.tensor([0, 0, 1], dtype=torch.long),
            y=torch.tensor([[1.5]], dtype=torch.float),
        )
        slices = {
            "x": torch.tensor([0, 1, 3]),
            "edge_index": torch.tensor([0, 1, 2]),
            "mol_index": torch.tensor([0, 1, 3]),
            "y": torch.tensor([0, 1, 2]),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "processed.safetensors"
            save_processed_graph_data(data, slices, path)
            loaded, loaded_slices = load_processed_graph_data(path)

        self.assertIsInstance(loaded, ReactionGraphData)
        self.assertTrue(torch.equal(loaded.mol_index, data.mol_index))
        self.assertTrue(torch.equal(loaded.x, data.x))
        self.assertIsNotNone(loaded_slices)
        self.assertTrue(torch.equal(loaded_slices["mol_index"], slices["mol_index"]))

    def test_checkpoint_safetensors_roundtrip_keeps_model_state_dict(self):
        payload = {
            "epoch": 3,
            "model_state_dict": {
                "encoder.weight": torch.arange(4, dtype=torch.float).reshape(2, 2),
                "decoder.bias": torch.tensor([1.0, 2.0]),
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "valid_checkpoint.safetensors"
            save_checkpoint_payload(payload, path)
            loaded = load_checkpoint_payload(path)

        self.assertEqual(loaded["epoch"], 3)
        self.assertTrue(torch.equal(loaded["model_state_dict"]["encoder.weight"], payload["model_state_dict"]["encoder.weight"]))
        self.assertTrue(torch.equal(loaded["model_state_dict"]["decoder.bias"], payload["model_state_dict"]["decoder.bias"]))

    def test_existing_serialized_path_falls_back_to_legacy_pt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pt_path = Path(tmp_dir) / "valid_checkpoint.pt"
            safetensors_path = Path(tmp_dir) / "valid_checkpoint.safetensors"
            torch.save({"model_state_dict": {"w": torch.tensor([1.0])}}, pt_path)

            resolved = existing_serialized_path(safetensors_path)
            loaded = load_checkpoint_payload(safetensors_path)

        self.assertEqual(resolved, pt_path)
        self.assertTrue(torch.equal(loaded["model_state_dict"]["w"], torch.tensor([1.0])))

    def test_convert_serialized_file_roundtrips_checkpoint_and_processed_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            checkpoint_input = tmp_root / "valid_checkpoint.pt"
            checkpoint_payload = {
                "epoch": 1,
                "model_state_dict": {"w": torch.tensor([1.0, 2.0])},
                "optimizer_state_dict": {"state": {}},
            }
            torch.save(checkpoint_payload, checkpoint_input)
            checkpoint_output = convert_serialized_file(checkpoint_input, kind="checkpoint")
            self.assertEqual(checkpoint_output, prefer_safetensors_path(checkpoint_input))
            converted_checkpoint = load_checkpoint_payload(checkpoint_output)
            self.assertEqual(converted_checkpoint["epoch"], 1)
            self.assertTrue(torch.equal(converted_checkpoint["model_state_dict"]["w"], checkpoint_payload["model_state_dict"]["w"]))
            self.assertNotIn("optimizer_state_dict", converted_checkpoint)

            processed_input = tmp_root / "processed.pt"
            data = ReactionGraphData(
                x=torch.tensor([[1], [2]], dtype=torch.long),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                mol_index=torch.tensor([0, 1], dtype=torch.long),
                y=torch.tensor([[3.5]], dtype=torch.float32),
            )
            slices = {
                "x": torch.tensor([0, 1, 2]),
                "edge_index": torch.tensor([0, 1]),
                "mol_index": torch.tensor([0, 1, 2]),
                "y": torch.tensor([0, 1]),
            }
            torch.save((data, slices), processed_input)
            processed_output = convert_serialized_file(processed_input, kind="processed-data")
            self.assertEqual(processed_output, prefer_safetensors_path(processed_input))
            loaded_data, loaded_slices = load_processed_graph_data(processed_output)
            self.assertTrue(torch.equal(loaded_data.mol_index, data.mol_index))
            self.assertIsNotNone(loaded_slices)
            self.assertTrue(torch.equal(loaded_slices["mol_index"], slices["mol_index"]))

    def test_legacy_list_mol_index_converts_to_safetensors_tensor_with_slices(self):
        data = Data(
            x=torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.long),
            mol_index=[[0, 0, 1], [0, 1]],
            y=torch.tensor([[1.0], [2.0]], dtype=torch.float),
        )
        slices = {
            "x": torch.tensor([0, 3, 5]),
            "mol_index": torch.tensor([0, 1, 2]),
            "y": torch.tensor([0, 1, 2]),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "legacy-list.safetensors"
            save_processed_graph_data(data, slices, path)
            loaded, loaded_slices = load_processed_graph_data(path)

        self.assertIsInstance(loaded, ReactionGraphData)
        self.assertTrue(torch.equal(loaded.mol_index, torch.tensor([0, 0, 1, 0, 1])))
        self.assertIsNotNone(loaded_slices)
        self.assertTrue(torch.equal(loaded_slices["mol_index"], torch.tensor([0, 3, 5])))

    def test_mol_index_collates_as_tensor_when_source_is_tensor(self):
        data, slices = InMemoryDataset.collate(
            [
                ReactionGraphData(x=torch.ones(1, 1), mol_index=torch.tensor([0])),
                ReactionGraphData(x=torch.ones(2, 1), mol_index=torch.tensor([0, 1])),
            ]
        )

        self.assertTrue(torch.equal(data.mol_index, torch.tensor([0, 0, 1])))
        self.assertTrue(torch.equal(slices["mol_index"], torch.tensor([0, 1, 3])))
