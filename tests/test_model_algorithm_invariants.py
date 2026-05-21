import unittest
import csv
from pathlib import Path
import tempfile
from unittest import mock
from types import SimpleNamespace
import types

import numpy as np
import torch
from torch_geometric.data import Data

from rxngraphormer import rxn_emb
from rxngraphormer import cli
from rxngraphormer import model as model_module
from rxngraphormer.config import DataConfig, ModelConfig, load_config, load_config_dict, resolve_config_file
from rxngraphormer.evaluator import regression_metrics
from rxngraphormer.lightning_module import RXNGraphormerLitModule
from rxngraphormer.lightning.workflow import (
    LightningFitSettings,
    build_dataloader_settings,
    fit_config,
)
from rxngraphormer.regression_workflow import (
    RegressionEvaluationSettings,
    RegressionSplitResult,
    evaluate_regression_prediction,
)
from rxngraphormer.data import (
    ATOM_DICT,
    ATOM_FEAT_DIMS,
    _collated_data_count,
    _matched_raw_files,
    _raw_file_sort_key,
    _separate_collated_data,
    RXNG2SDataset,
    calc_batch_graph_distance,
    calc_graph_distance,
    get_rxn_pfm_info,
)
from rxngraphormer.layer import (
    MultiHeadAttention,
    gat_attention_softmax,
    get_mess_around_edge,
    self_loop_bond_attr,
    sum_mess_around_edge,
)
from rxngraphormer.model import (
    ClassifierLayer,
    RXNG2Sequencer,
    TransformerEncoder,
    _decoder_src_placeholder,
    masked_sequence_mean,
)
from rxngraphormer.predictor import (
    ClassificationPrediction,
    EmbeddingPrediction,
    RegressionPrediction,
    RXNGraphormerPredictor,
    export_embeddings_csv,
    export_predictions_csv,
)
from rxngraphormer.preprocess.cli import preprocess_from_config
from rxngraphormer.preprocess.table import (
    generate_mid_smiles,
    reaction_has_atom_mapping,
    write_reaction_table_files,
)
from rxngraphormer.reaction import canonicalize_reaction_side, parse_reaction_smiles, split_reaction_smiles
from rxngraphormer.runtime_callbacks import EpochRuntimeMonitor
from rxngraphormer.utils import (
    add_dense_empty_node_edge,
    add_empty_node_and_edge,
    get_seq_acc,
    pad_feat,
    scaled_dot_product_attention,
    update_batch_idx,
)
from rxngraphormer.evaluator import RegressionEvaluation, RegressionMetrics


class ModelAlgorithmInvariantsTest(unittest.TestCase):
    def test_masked_sequence_mean_ignores_padding(self):
        x = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
                [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            ]
        )
        lengths = torch.tensor([2, 3])

        out = masked_sequence_mean(x, lengths)

        self.assertTrue(torch.allclose(out, torch.tensor([[2.0, 3.0], [7.0, 8.0]])))

    def test_transformer_encoder_masks_padding_tokens(self):
        torch.manual_seed(7)
        encoder = TransformerEncoder(
            num_layer=2,
            hidden_size=8,
            intermediate_size=8,
            num_heads=2,
            hidden_dropout_prob=0.0,
        ).eval()
        short = torch.randn(1, 2, 8)
        long = torch.randn(1, 5, 8)

        with torch.no_grad():
            out_alone = encoder(short, torch.tensor([2]))
            padded_short = torch.cat([short, torch.zeros(1, 3, 8)], dim=1)
            batch = torch.cat([padded_short, long], dim=0)
            out_batch = encoder(batch, torch.tensor([2, 5]))[:1, :2]

        self.assertTrue(torch.allclose(out_alone, out_batch, atol=1e-6))

    def test_scaled_dot_product_attention_zeroes_fully_masked_queries(self):
        query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        value = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]])
        mask = torch.tensor([[[1, 1], [0, 0]]])

        out = scaled_dot_product_attention(query, key, value, mask=mask)

        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.allclose(out[0, 1], torch.zeros(2)))

    def test_scaled_dot_product_attention_preserves_valid_mask_softmax(self):
        query = torch.tensor([[[1.0, 0.0]]])
        key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        value = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]])
        mask = torch.tensor([[[1, 0]]])

        out = scaled_dot_product_attention(query, key, value, mask=mask)

        self.assertTrue(torch.allclose(out, value[:, :1]))

    def test_multi_head_attention_rejects_non_divisible_heads(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            MultiHeadAttention(embed_dim=10, num_heads=3)

    def test_decoder_src_placeholder_uses_memory_lengths_device_and_dtype(self):
        memory_lengths = torch.tensor([2, 5, 3], dtype=torch.long)

        out = _decoder_src_placeholder(memory_lengths)

        self.assertEqual(out.device, memory_lengths.device)
        self.assertEqual(out.dtype, memory_lengths.dtype)
        self.assertTrue(torch.equal(out, torch.zeros(5, dtype=torch.long)))

    def test_calc_graph_distance_handles_disconnected_retrosynthesis_graph(self):
        atom_feat = np.zeros((3, 1), dtype=np.int64)
        edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64).T

        dist = calc_graph_distance(atom_feat, edge_index, task="retrosynthesis")

        self.assertEqual(dist.tolist(), [[0, 1, 9], [1, 0, 9], [9, 9, 0]])

    def test_calc_batch_graph_distance_avoids_cross_graph_edges_and_guards_size(self):
        batch = torch.tensor([0, 0, 1])
        edge_index = torch.tensor([[0, 1], [1, 0]])

        dist = calc_batch_graph_distance(batch, edge_index, task="forward_prediction")

        self.assertEqual(
            dist.tolist(),
            [
                [[0, 1], [1, 0]],
                [[0, 11], [11, 11]],
            ],
        )
        with self.assertRaisesRegex(ValueError, "max_nodes=2, limit=0"):
            calc_batch_graph_distance(batch, edge_index, task="retrosynthesis", max_nodes_per_graph=0)

    def test_classifier_layer_forward_keeps_probability_api_and_logits_are_available(self):
        layer = ClassifierLayer(hidden_size=2, output_size=2, layer_num=1, batch_norm=False)
        with torch.no_grad():
            layer.projection.weight.copy_(torch.eye(2))

        x = torch.tensor([[2.0, -1.0]])

        self.assertTrue(torch.allclose(layer.logits(x), torch.tensor([[2.0, -1.0]])))
        self.assertTrue(torch.allclose(layer(x), torch.softmax(torch.tensor([[2.0, -1.0]]), dim=-1)))

    def test_pad_feat_preserves_per_batch_order_without_python_loop(self):
        feat = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
            ]
        )
        batch = torch.tensor([0, 0, 1, 2, 2])

        out = pad_feat(feat, batch, num_features=2)

        expected = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [0.0, 0.0]],
                [[4.0, 40.0], [5.0, 50.0]],
            ]
        )
        self.assertTrue(torch.equal(out, expected))

    def test_pad_feat_rejects_empty_batch(self):
        with self.assertRaisesRegex(ValueError, "at least one batch entry"):
            pad_feat(torch.empty(0, 2), torch.empty(0, dtype=torch.long), num_features=2)

    def test_update_batch_idx_offsets_molecule_ids_per_reaction(self):
        mol_index = [[0, 0, 1], torch.tensor([0, 1]), [0, 0, 0]]

        batch_mol_index, batch = update_batch_idx(mol_index, device=torch.device("cpu"))

        self.assertTrue(torch.equal(batch_mol_index, torch.tensor([0, 0, 1, 2, 3, 4, 4, 4])))
        self.assertTrue(torch.equal(batch, torch.tensor([0, 0, 1, 1, 2])))

    def test_update_batch_idx_rejects_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least one molecule index block"):
            update_batch_idx([], device=torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            update_batch_idx([[0], []], device=torch.device("cpu"))

    def test_update_batch_idx_accepts_single_flat_sample(self):
        batch_mol_index, batch = update_batch_idx(torch.tensor([0, 0, 1]), device=torch.device("cpu"))

        self.assertTrue(torch.equal(batch_mol_index, torch.tensor([0, 0, 1])))
        self.assertTrue(torch.equal(batch, torch.tensor([0, 0])))

    def test_raw_file_sort_key_accepts_numbered_and_plain_names(self):
        files = ["data/rct.csv", "data/rct_10.csv", "data/rct_2.csv", "data/abc.csv"]

        sorted_files = sorted(files, key=_raw_file_sort_key)

        self.assertEqual(sorted_files, ["data/rct_2.csv", "data/rct_10.csv", "data/abc.csv", "data/rct.csv"])

    def test_matched_raw_files_filters_directories_and_reports_empty_matches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "rct_2.csv").write_text("A,0\n")
            Path(tmp_dir, "rct_10.csv").write_text("B,0\n")
            Path(tmp_dir, "rct_dir.csv").mkdir()

            files = _matched_raw_files(tmp_dir, "rct*")

            self.assertEqual([Path(path).name for path in files], ["rct_2.csv", "rct_10.csv"])
            with self.assertRaisesRegex(FileNotFoundError, "No raw data files matched pattern"):
                _matched_raw_files(tmp_dir, "missing*")

    def test_single_item_collated_block_is_indexable(self):
        data = Data(x=torch.tensor([[1], [2]]), y=torch.tensor([0.5]))

        self.assertEqual(_collated_data_count(None), 1)
        item = _separate_collated_data(data, None, 0)

        self.assertTrue(torch.equal(item.x, data.x))
        with self.assertRaisesRegex(IndexError, "idx=0"):
            _separate_collated_data(data, None, 1)

    def test_sum_mess_around_edge_matches_padded_legacy_sum(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 2]])
        mess = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )

        legacy = get_mess_around_edge(edge_index, mess).sum(dim=1)
        optimized = sum_mess_around_edge(edge_index, mess, num_nodes=3)

        self.assertTrue(torch.equal(optimized, legacy))

    def test_get_mess_around_edge_preserves_dtype_for_isolated_nodes(self):
        edge_index = torch.tensor([[0], [2]])
        mess = torch.tensor([[1, 2]], dtype=torch.long)

        out = get_mess_around_edge(edge_index, mess)

        self.assertEqual(out.dtype, mess.dtype)
        self.assertEqual(out.device, mess.device)
        self.assertTrue(torch.equal(out[1, 0], torch.zeros(2, dtype=torch.long)))

    def test_sum_mess_around_edge_handles_edgeless_graph(self):
        out = sum_mess_around_edge(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 4)),
            num_nodes=3,
        )

        self.assertTrue(torch.equal(out, torch.zeros(3, 4)))

    def test_self_loop_bond_attr_preserves_edge_attr_dtype_and_pattern(self):
        edge_attr = torch.ones((2, 5), dtype=torch.long)

        out = self_loop_bond_attr(num_nodes=3, num_features=5, edge_attr=edge_attr)

        self.assertEqual(out.dtype, edge_attr.dtype)
        self.assertEqual(out.device, edge_attr.device)
        self.assertTrue(torch.equal(out[:, 0], torch.full((3,), 5, dtype=torch.long)))
        self.assertTrue(torch.equal(out[:, 1:], torch.zeros((3, 4), dtype=torch.long)))

    def test_gru_cell_batched_call_matches_per_node_loop(self):
        torch.manual_seed(13)
        rnn = torch.nn.GRUCell(4, 4)
        x = torch.randn(5, 4)
        hidden = torch.zeros_like(x)

        loop_out = torch.stack([rnn(x[i], hidden[i]) for i in range(x.size(0))])
        batch_out = rnn(x, hidden)

        self.assertTrue(torch.allclose(batch_out, loop_out))

    def test_gat_attention_softmax_normalizes_by_target_node(self):
        alpha = torch.tensor([[0.0], [1.0], [2.0]])
        edge_index = torch.tensor([[0, 2, 2], [1, 1, 0]])

        attn = gat_attention_softmax(alpha, edge_index)

        expected_target_one = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
        self.assertTrue(torch.allclose(attn[:2, 0], expected_target_one))
        self.assertTrue(torch.allclose(attn[2, 0], torch.tensor(1.0)))

    def test_add_dense_empty_node_edge_preserves_dtype_and_offsets_edges(self):
        data = Data(
            x=torch.tensor([[1, 2, 3, 4, 0, 1, 2, 3, 0], [3, 4, 5, 0, 1, 0, 2, 3, 0]], dtype=torch.long),
            edge_attr=torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.long),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            mol_index=[[0, 0]],
        )

        out = add_dense_empty_node_edge(data)

        self.assertEqual(out.x.dtype, torch.long)
        self.assertEqual(out.edge_attr.dtype, torch.long)
        self.assertEqual(out.edge_index.dtype, torch.long)
        self.assertTrue(torch.equal(out.x[0], torch.tensor([ATOM_DICT["*"], 0, 0, 0, 0, 0, 0, 0, 0])))
        self.assertTrue(torch.equal(out.edge_attr[0], torch.zeros(5, dtype=torch.long)))
        self.assertTrue(torch.equal(out.edge_index, torch.tensor([[0, 1], [0, 2]])))
        self.assertEqual(out.mol_index, [[0], [0, 0]])

    def test_add_empty_node_and_edge_preserves_sparse_feature_dtype(self):
        onehot_dim = sum(ATOM_FEAT_DIMS)
        original_edge_oh_attr = torch.tensor([[0, 0, 1]], dtype=torch.long)
        expected_edge_oh_attr = original_edge_oh_attr.clone()
        data = Data(
            x_oh=torch.ones((1, onehot_dim), dtype=torch.long),
            edge_oh_attr=original_edge_oh_attr,
            a_graphs=torch.tensor([[1, 999999999]], dtype=torch.long),
            b_graphs=torch.tensor([[2, 999999999]], dtype=torch.long),
        )
        edge_ref = data.edge_oh_attr

        out = add_empty_node_and_edge(data)

        self.assertEqual(out.x_oh.dtype, torch.long)
        self.assertEqual(out.edge_oh_attr.dtype, torch.long)
        self.assertTrue(torch.equal(edge_ref, expected_edge_oh_attr))
        self.assertTrue(torch.equal(out.edge_oh_attr[0], torch.zeros(3, dtype=torch.long)))
        self.assertTrue(torch.equal(out.edge_oh_attr[1, :2], torch.tensor([1, 1])))


class _CountingOptimizer:
    def __init__(self):
        self.steps = 0
        self.zeroes = 0

    def step(self):
        self.steps += 1

    def zero_grad(self):
        self.zeroes += 1


class _CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class _Scaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def _run_accumulation_smoke(num_batches, accum_steps):
    weight = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = _CountingOptimizer()
    scheduler = _CountingScheduler()
    scaler = _Scaler()
    optimizer.zero_grad()
    pending_steps = 0
    for _ in range(num_batches):
        loss = weight * 2.0
        scaler.scale(loss / accum_steps).backward()
        pending_steps += 1
        if pending_steps == accum_steps:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            pending_steps = 0
    if pending_steps > 0:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
    return optimizer.steps, scheduler.steps


class SequenceGradientAccumulationTest(unittest.TestCase):
    def test_no_extra_optimizer_step_when_batches_divide_accumulation(self):
        self.assertEqual(_run_accumulation_smoke(num_batches=4, accum_steps=2), (2, 2))

    def test_tail_microbatch_gets_one_optimizer_step(self):
        self.assertEqual(_run_accumulation_smoke(num_batches=3, accum_steps=2), (2, 2))


class LightningRuntimeTest(unittest.TestCase):
    def test_lightning_workflow_builds_dataloader_settings_from_config_and_overrides(self):
        config = type(
            "Config",
            (),
            {
                "data": type(
                    "Data",
                    (),
                    {
                        "batch_size": 32,
                        "num_workers": 2,
                        "pin_memory": False,
                        "persistent_workers": False,
                        "prefetch_factor": 3,
                    },
                )(),
            },
        )()

        settings = build_dataloader_settings(
            config,
            LightningFitSettings(
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
            ),
        )

        self.assertEqual(settings.batch_size, 32)
        self.assertEqual(settings.num_workers, 8)
        self.assertTrue(settings.pin_memory)
        self.assertTrue(settings.persistent_workers)
        self.assertEqual(settings.prefetch_factor, 4)

    def test_lightning_workflow_wires_fit_artifacts_and_preserves_cli_settings(self):
        config = type(
            "Config",
            (),
            {
                "task": "regression",
                "runtime": type(
                    "Runtime",
                    (),
                    {
                        "deterministic": False,
                        "compile_model": False,
                        "compile_mode": None,
                        "early_stopping_patience": 0,
                    },
                )(),
                "data": type(
                    "Data",
                    (),
                    {
                        "batch_size": 16,
                        "num_workers": 1,
                        "pin_memory": False,
                        "persistent_workers": False,
                        "prefetch_factor": None,
                    },
                )(),
                "model": type("Model", (), {"save_dir": "default_save"})(),
            },
        )()
        fake_datamodule = mock.Mock(name="datamodule")
        fake_lit_module = mock.Mock(name="lit_module")
        fake_trainer = mock.Mock(name="trainer")

        settings = LightningFitSettings(
            default_root_dir="runs/lit",
            accelerator="cpu",
            devices="1",
            precision="32-true",
            max_epochs=7,
            split_manifest="split.json",
            resume_from_checkpoint="resume.ckpt",
            compile_model=True,
            compile_mode="reduce-overhead",
            early_stopping_patience=5,
            write_manifest=False,
        )
        with (
            mock.patch("rxngraphormer.lightning.workflow.RXNGraphormerDataModule", return_value=fake_datamodule) as datamodule_cls,
            mock.patch("rxngraphormer.lightning.workflow.RXNGraphormerLitModule.from_config", return_value=fake_lit_module) as from_config,
            mock.patch("rxngraphormer.lightning.workflow.build_trainer", return_value=fake_trainer) as trainer_builder,
        ):
            artifacts = fit_config(config, settings)

        datamodule_cls.assert_called_once()
        self.assertEqual(datamodule_cls.call_args.kwargs["split_manifest"], "split.json")
        from_config.assert_called_once_with(config)
        trainer_builder.assert_called_once_with(
            config,
            default_root_dir="runs/lit",
            accelerator="cpu",
            devices="1",
            precision="32-true",
            max_epochs=7,
        )
        fake_trainer.fit.assert_called_once_with(fake_lit_module, datamodule=fake_datamodule, ckpt_path="resume.ckpt")
        self.assertIs(artifacts.trainer, fake_trainer)
        self.assertTrue(config.runtime.compile_model)
        self.assertEqual(config.runtime.compile_mode, "reduce-overhead")
        self.assertEqual(config.runtime.early_stopping_patience, 5)

    def test_lightning_workflow_writes_manifest_and_optional_eval_report(self):
        config = type(
            "Config",
            (),
            {
                "task": "regression",
                "runtime": type(
                    "Runtime",
                    (),
                    {
                        "deterministic": False,
                        "compile_model": False,
                        "compile_mode": None,
                        "early_stopping_patience": 0,
                    },
                )(),
                "data": type(
                    "Data",
                    (),
                    {
                        "batch_size": 16,
                        "num_workers": 0,
                        "pin_memory": False,
                        "persistent_workers": False,
                        "prefetch_factor": None,
                    },
                )(),
                "model": type("Model", (), {"save_dir": "default_save"})(),
            },
        )()
        fake_datamodule = mock.Mock(name="datamodule")
        fake_lit_module = mock.Mock(name="lit_module")
        fake_trainer = mock.Mock(name="trainer")
        fake_trainer.logger.log_dir = "runs/lit/version_0"
        fake_trainer.checkpoint_callback.best_model_path = "runs/lit/version_0/checkpoints/best.ckpt"
        fake_trainer.checkpoint_callback.last_model_path = ""
        fake_trainer.callback_metrics = {"val_mae": torch.tensor(0.5)}

        settings = LightningFitSettings(
            default_root_dir="runs/lit",
            config_path="config.json",
            eval_after_fit=True,
            eval_splits=("valid", "test"),
            eval_batch_size=64,
            eval_scale=100.0,
            eval_yield_constrain=True,
        )
        with (
            mock.patch("rxngraphormer.lightning.workflow.RXNGraphormerDataModule", return_value=fake_datamodule),
            mock.patch("rxngraphormer.lightning.workflow.RXNGraphormerLitModule.from_config", return_value=fake_lit_module),
            mock.patch("rxngraphormer.lightning.workflow.build_trainer", return_value=fake_trainer),
            mock.patch("rxngraphormer.lightning.workflow.run_post_fit_regression_eval", return_value={"json": "eval.json"}) as post_eval,
            mock.patch("rxngraphormer.lightning.workflow.write_fit_manifest", return_value={"output_manifest": "manifest.json"}) as write_manifest,
        ):
            artifacts = fit_config(config, settings)

        post_eval.assert_called_once()
        self.assertEqual(post_eval.call_args.kwargs["checkpoint_path"], "runs/lit/version_0/checkpoints/best.ckpt")
        self.assertEqual(post_eval.call_args.kwargs["splits"], ("valid", "test"))
        self.assertEqual(post_eval.call_args.kwargs["scale"], 100.0)
        write_manifest.assert_called_once()
        manifest = write_manifest.call_args.args[0]
        self.assertEqual(manifest["config_path"], "config.json")
        self.assertEqual(manifest["checkpoint"]["best_model_path"], "runs/lit/version_0/checkpoints/best.ckpt")
        self.assertEqual(manifest["eval_reports"], {"regression": {"json": "eval.json"}})
        self.assertEqual(artifacts.manifest_paths, {"output_manifest": "manifest.json"})

    def test_compile_forward_does_not_register_extra_state_dict_keys(self):
        config = type(
            "Config",
            (),
            {
                "training": type("Training", (), {"loss": "l1"})(),
                "runtime": type("Runtime", (), {"compile_model": True, "compile_mode": "default"})(),
            },
        )()
        model = torch.nn.Linear(2, 1)
        module = RXNGraphormerLitModule(model, config)
        compiled = mock.Mock(side_effect=lambda x: model(x) + 1.0)

        with mock.patch("rxngraphormer.lightning.module.torch.compile", return_value=compiled) as compile_mock:
            module.setup("fit")
            out = module(torch.ones(1, 2))

        compile_mock.assert_called_once()
        self.assertTrue(torch.allclose(out, model(torch.ones(1, 2)) + 1.0))
        self.assertEqual(sorted(module.state_dict().keys()), ["model.bias", "model.weight"])

    def test_epoch_runtime_monitor_logs_wall_time(self):
        callback = EpochRuntimeMonitor(log_epoch_time=True, log_gpu_memory=False)
        module = mock.Mock()

        callback.on_train_epoch_start(trainer=None, pl_module=module)
        callback.on_train_epoch_end(trainer=None, pl_module=module)

        logged = module.log_dict.call_args.args[0]
        self.assertIn("epoch_seconds", logged)
        self.assertGreaterEqual(logged["epoch_seconds"], 0.0)

    def test_finetune_optimizer_uses_scaled_pretrained_encoder_lr(self):
        config = type(
            "Config",
            (),
            {
                "training": type("Training", (), {"loss": "l1"})(),
                "runtime": type("Runtime", (), {"compile_model": False})(),
                "optimizer": type(
                    "Optimizer",
                    (),
                    {"optimizer": "AdamW", "learning_rate": 0.4, "weight_decay": 0.0},
                )(),
                "scheduler": type("Scheduler", (), {"type": "none"})(),
                "model": type(
                    "Model",
                    (),
                    {
                        "pretrained_model_path": "model_path/pretrained_classification_model",
                        "pretrained_model_freeze": False,
                        "pretrained_lr_scaled_coef": 0.1,
                        "use_mid_inf": True,
                    },
                )(),
            },
        )()

        class FakeFineTuneModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rct_encoder = torch.nn.Linear(2, 2)
                self.pdt_encoder = torch.nn.Linear(2, 2)
                self.decoder = torch.nn.Linear(2, 1)
                self.mid_encoder = torch.nn.Linear(2, 2)
                self.mid_iteract = torch.nn.Linear(2, 2)
                self.mid_decoder = torch.nn.Linear(2, 1)

        module = RXNGraphormerLitModule(FakeFineTuneModel(), config)
        optimizer = module.configure_optimizers()

        self.assertTrue(np.allclose([group["lr"] for group in optimizer.param_groups], [0.04, 0.04, 0.4, 0.4, 0.4, 0.4]))

    def test_regression_metrics_accepts_bfloat16_predictions(self):
        result = regression_metrics(
            torch.tensor([[1.0], [2.0]], dtype=torch.bfloat16),
            torch.tensor([[1.5], [1.5]], dtype=torch.bfloat16),
        )

        self.assertEqual(result.metrics.count, 2)
        self.assertEqual(result.preds.dtype, torch.float32)


class SequenceMetricTest(unittest.TestCase):
    def test_get_seq_acc_checks_exact_token_match(self):
        pred = torch.tensor([[2, 3, 4], [2, 3, 5], [0, 1, 2]])
        target = torch.tensor([[2, 3, 4], [2, 3, 4], [0, 1, 2]])

        acc = get_seq_acc(pred, target)

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
        api = object.__new__(rxn_emb.RXNEMB)
        seen_roots = []

        def fake_from_dataset(root, **kwargs):
            seen_roots.append(Path(root))
            self.assertTrue((Path(root) / "rct_smiles_0.csv").exists())
            self.assertTrue((Path(root) / "pdt_smiles_0.csv").exists())
            return torch.ones(2, 3)

        api.gen_rxn_emb_from_dataset = fake_from_dataset
        with mock.patch.object(rxn_emb, "canonical_smiles", side_effect=lambda smi: smi):
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
            mock.patch("rxngraphormer.predictor.MultiRXNDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.predictor.PairDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.predictor.torch.utils.data.DataLoader", FakeDataLoader),
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
            mock.patch("rxngraphormer.predictor.MultiRXNDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.predictor.PairDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.predictor.torch.utils.data.DataLoader", FakeDataLoader),
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
            mock.patch("rxngraphormer.predictor.load_config", return_value=fake_config) as load_config_mock,
            mock.patch("rxngraphormer.predictor.build_regression_model", return_value=FakeModel()) as build_mock,
            mock.patch("rxngraphormer.predictor.CheckpointAdapter", return_value=FakeAdapter()),
        ):
            predictor = RXNGraphormerPredictor.from_checkpoint(
                "runs/model.ckpt",
                config_path="runs/config.json",
                device="cpu",
            )

        load_config_mock.assert_called_once_with("runs/config.json")
        build_mock.assert_called_once_with(fake_config)
        self.assertEqual(calls["load"][1], "runs/model.ckpt")
        self.assertEqual(calls["load"][2], torch.device("cpu"))
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
                mock.patch("rxngraphormer.predictor.canonicalize_reaction_side", side_effect=lambda smi: smi),
                mock.patch("rxngraphormer.predictor.canonical_smiles", side_effect=lambda smi: smi),
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
            mock.patch("rxngraphormer.predictor.MultiRXNDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.predictor.PairDataset", return_value=FakeDataset()),
            mock.patch("rxngraphormer.predictor.torch.utils.data.DataLoader", FakeDataLoader),
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
            mock.patch("rxngraphormer.predictor.RXNGraphormerPredictor", FakePredictor),
            mock.patch("rxngraphormer.predictor.export_predictions_csv", fake_export),
        ):
            cli.predict_main()

        self.assertEqual(calls["init"], ("model_dir", "classification", "valid_checkpoint.pt", None))
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
            mock.patch("rxngraphormer.predictor.RXNGraphormerPredictor", FakePredictor),
            mock.patch("rxngraphormer.predictor.export_predictions_csv", fake_export),
        ):
            cli.predict_main()

        self.assertEqual(calls["init"], ("model_dir", "regression", "valid_checkpoint.pt", None))
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
            mock.patch("rxngraphormer.predictor.RXNGraphormerPredictor", FakePredictor),
            mock.patch("rxngraphormer.predictor.export_predictions_csv", fake_export),
        ):
            cli.predict_main()

        self.assertEqual(calls["init"], ("model_dir", "classification", "valid_checkpoint.pt", None))
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
            mock.patch("rxngraphormer.predictor.RXNGraphormerPredictor", FakePredictor),
            mock.patch("rxngraphormer.predictor.export_predictions_csv", fake_export_predictions),
            mock.patch("rxngraphormer.predictor.export_embeddings_csv", fake_export_embeddings),
        ):
            cli.predict_main()

        self.assertEqual(calls["embed"][0], "data_dir")
        self.assertEqual(calls["embedding_export"][1], "emb.csv")


class ConfigCompatibilityTest(unittest.TestCase):
    def test_legacy_data_file_fields_populate_name_regex_aliases(self):
        config = load_config("config_toml/bh_scratch_reproduce.toml")

        self.assertEqual(config.data.rct_name_regrex, config.data.rct_data_file)
        self.assertEqual(config.data.pdt_name_regrex, config.data.pdt_data_file)
        self.assertEqual(config.data.mid_name_regrex, config.data.mid_data_file)
        self.assertEqual(config.data.task, config.task)

    def test_input_table_populates_legacy_data_files_from_prefix(self):
        data = DataConfig(input_table="inputs/reactions.csv")

        self.assertEqual(data.rct_data_file, "reactions_rct.csv")
        self.assertEqual(data.pdt_data_file, "reactions_pdt.csv")
        self.assertEqual(data.mid_data_file, "reactions_mid.csv")
        self.assertEqual(data.rct_name_regrex, "reactions_rct.csv")
        self.assertEqual(data.pdt_name_regrex, "reactions_pdt.csv")
        self.assertEqual(data.mid_name_regrex, "reactions_mid.csv")

    def test_sequence_attention_encoder_legacy_values_load(self):
        config = load_config("config/uspto_50k_parameters.json")

        self.assertEqual(config.task, "sequence_generation")
        self.assertEqual(config.model.att_encoder_type, "attxl")

    def test_sequence_attention_encoder_aliases_normalize_to_model_values(self):
        self.assertEqual(ModelConfig(att_encoder_type="rxngraphormer").att_encoder_type, "attxl")
        self.assertEqual(ModelConfig(att_encoder_type="onmt").att_encoder_type, "attn")

    def test_jsonc_config_supports_line_and_block_comments(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "parameters.jsonc"
            config_path.write_text(
                """
                {
                  // line comment
                  "task": "regression",
                  "model": {
                    "gnn_type": "gcn",
                    "save_dir": "runs/http://example.test/model" // URL-like string stays intact
                  },
                  /*
                    block comment
                  */
                  "data": {
                    "task": "regression",
                    "rct_data_file": "rct.json",
                    "pdt_data_file": "pdt.json"
                  }
                }
                """,
                encoding="utf-8",
            )

            self.assertEqual(resolve_config_file(tmp_dir), config_path)
            raw = load_config_dict(config_path)
            config = load_config(config_path)

        self.assertEqual(raw["model"]["save_dir"], "runs/http://example.test/model")
        self.assertEqual(config.task, "regression")
        self.assertEqual(config.data.rct_name_regrex, "rct.json")
        self.assertEqual(config.data.pdt_name_regrex, "pdt.json")


class PreprocessTableTest(unittest.TestCase):
    def test_reaction_has_atom_mapping_detects_shared_map_numbers(self):
        self.assertTrue(reaction_has_atom_mapping("[CH3:1][OH:2]>>[CH3:1][OH:2]"))
        self.assertFalse(reaction_has_atom_mapping("CCO>>CC=O"))

    def test_reaction_smiles_parser_copies_agents_to_both_legacy_sides(self):
        reactants, products = split_reaction_smiles("CCO>O>CC=O")

        self.assertEqual(reactants, "CCO.O")
        self.assertEqual(products, "CC=O.O")

    def test_reaction_smiles_parser_maps_agents_when_reaction_is_mapped(self):
        parsed = parse_reaction_smiles("[CH3:1][OH:2]>O>[CH3:1][OH:2]")

        self.assertTrue(parsed.has_reactant_product_mapping)
        self.assertIn("[OH2:3]", parsed.reactants)
        self.assertIn("[OH2:3]", parsed.products)
        self.assertEqual(parsed.reactants, parsed.products)

    def test_reaction_smiles_parser_preserves_explicit_mapped_hydrogens(self):
        parsed = parse_reaction_smiles(
            "[C:1](=[C:2]([H:5])[H:6])([H:3])[H:4]"
            ">>"
            "[C:1]([H:3])([H:4])([H:5])[H:6]"
        )

        self.assertIn("[H:3]", parsed.reactants)
        self.assertIn("[H:3]", parsed.products)
        self.assertNotIn("[*:1]", parsed.reactants)
        self.assertNotIn("[*:2]", parsed.reactants)

    def test_generate_mid_smiles_auto_skips_mapper_for_mapped_reaction(self):
        finder_mock = mock.Mock()
        finder_mock.get_electron_path.return_value = ("[CH3:1]>>[CH3:1]", None, None, None)
        fake_midmol = types.SimpleNamespace(
            finder=finder_mock,
            get_mid_smi_from_rxn=mock.Mock(return_value=["M"]),
            remove_atmmap=mock.Mock(side_effect=lambda smi: smi),
            gen_mech_mid_smi=mock.Mock(return_value=("SHOULD_NOT_USE", "", "")),
        )
        with (
            mock.patch.dict("sys.modules", {"rxngraphormer.midgen.midmol": fake_midmol}),
            mock.patch("rxngraphormer.preprocess.table.canonicalize_reaction_side", side_effect=lambda smi: smi),
            mock.patch("rxngraphormer.preprocess.table.canonical_smiles", side_effect=lambda smi: smi),
        ):
            out = generate_mid_smiles("[CH3:1]>>[CH3:1]", mapping_policy="auto")

        self.assertEqual(out, "M")
        finder_mock.get_electron_path.assert_called_once()
        fake_midmol.gen_mech_mid_smi.assert_not_called()

    def test_midmol_mapper_initialization_is_lazy(self):
        import importlib
        import sys

        import rxngraphormer.midgen as midgen_pkg

        module_name = "rxngraphormer.midgen.midmol"
        old_module = sys.modules.pop(module_name, None)
        had_attr = hasattr(midgen_pkg, "midmol")
        old_attr = getattr(midgen_pkg, "midmol", None)
        localmapper_mock = mock.Mock(return_value=mock.Mock())
        rxn_mapper_mock = mock.Mock()
        fake_localmapper = types.SimpleNamespace(localmapper=localmapper_mock)
        fake_rxnmapper = types.SimpleNamespace(RXNMapper=rxn_mapper_mock)
        try:
            with mock.patch.dict(
                "sys.modules",
                {"localmapper": fake_localmapper, "rxnmapper": fake_rxnmapper},
            ):
                midmol = importlib.import_module(module_name)

            self.assertIsNone(midmol.mapper)
            self.assertIsNone(midmol.rxn_mapper)
            localmapper_mock.assert_not_called()
            rxn_mapper_mock.assert_not_called()
        finally:
            sys.modules.pop(module_name, None)
            if old_module is not None:
                sys.modules[module_name] = old_module
            if had_attr:
                setattr(midgen_pkg, "midmol", old_attr)
            elif hasattr(midgen_pkg, "midmol"):
                delattr(midgen_pkg, "midmol")

    def test_generate_mid_smiles_never_requires_atom_mapping(self):
        with self.assertRaisesRegex(ValueError, "requires atom-mapped reaction SMILES"):
            generate_mid_smiles("CCO>>CC=O", mapping_policy="never")

    def test_write_reaction_table_files_uses_rxn_smiles_and_mid_smiles_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir, "reactions.csv")
            pd = __import__("pandas")
            pd.DataFrame(
                [
                    {"rxn_smiles": "CCO>>CC=O", "target": 1.2, "mid_smiles": "CC[O]"},
                    {"rxn_smiles": "CCN>>CC=N", "target": 3.4, "mid_smiles": "CC[N]"},
                ]
            ).to_csv(input_path, index=False)

            files = write_reaction_table_files(
                input_path,
                output_dir=tmp_dir,
                target_column="target",
                generate_mid=False,
            )

            self.assertEqual(files.rct_data_file, "reactions_rct.csv")
            self.assertTrue((Path(tmp_dir) / "reactions_rct.csv").exists())
            self.assertTrue((Path(tmp_dir) / "reactions_mid.csv").exists())
            self.assertEqual((Path(tmp_dir) / "reactions_rct.csv").read_text().splitlines()[0], "CCO,1.2")

    def test_write_reaction_table_files_materializes_agents_on_both_sides(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir, "reactions.csv")
            pd = __import__("pandas")
            pd.DataFrame(
                [
                    {"rxn_smiles": "CCO>O>CC=O", "target": 1.2},
                ]
            ).to_csv(input_path, index=False)

            write_reaction_table_files(
                input_path,
                output_dir=tmp_dir,
                target_column="target",
                generate_mid=False,
            )

            self.assertEqual((Path(tmp_dir) / "reactions_rct.csv").read_text().strip(), "CCO.O,1.2")
            self.assertEqual((Path(tmp_dir) / "reactions_pdt.csv").read_text().strip(), "CC=O.O,1.2")

    def test_write_reaction_table_files_preserves_explicit_mapped_hydrogens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir, "reactions.csv")
            pd = __import__("pandas")
            pd.DataFrame(
                [
                    {
                        "rxn_smiles": "[C:1](=[C:2]([H:5])[H:6])([H:3])[H:4]>>[C:1]([H:3])([H:4])([H:5])[H:6]",
                        "target": 1.2,
                    },
                ]
            ).to_csv(input_path, index=False)

            write_reaction_table_files(
                input_path,
                output_dir=tmp_dir,
                target_column="target",
                generate_mid=False,
            )

            rct_line = (Path(tmp_dir) / "reactions_rct.csv").read_text().strip()
            self.assertIn("[H:3]", rct_line)
            self.assertNotIn("[*:1]", rct_line)

    def test_preprocess_cli_materializes_input_table_before_regression_processing(self):
        calls = {}

        fake_config = SimpleNamespace(
            task="regression",
            data=SimpleNamespace(
                input_table="inputs/reactions.csv",
                data_path="dataset",
                output_prefix="",
                rxn_smiles_column="rxn_smiles",
                rct_smiles_column="rct_smiles",
                pdt_smiles_column="pdt_smiles",
                mid_smiles_column="mid_smiles",
                target_column="target",
                generate_mid=False,
                mapping_policy="auto",
                rct_data_file="",
                pdt_data_file="",
                mid_data_file="",
                rct_name_regrex="",
                pdt_name_regrex="",
                mid_name_regrex="",
                data_trunck=0,
                file_num_trunck=0,
                multi_process=False,
                task="regression",
            ),
            model=SimpleNamespace(use_mid_inf=False),
        )

        class FakeTableFiles:
            rct_data_file = "reactions_rct.csv"
            pdt_data_file = "reactions_pdt.csv"
            mid_data_file = ""
            rct_name_regrex = "reactions_rct.csv"
            pdt_name_regrex = "reactions_pdt.csv"
            mid_name_regrex = ""

        def fake_write(*args, **kwargs):
            calls["write"] = (args, kwargs)
            return FakeTableFiles()

        def fake_rxn_dataset(*args, **kwargs):
            calls.setdefault("datasets", []).append((args, kwargs))
            return object()

        with (
            mock.patch("rxngraphormer.preprocess.cli.load_config", return_value=fake_config),
            mock.patch("rxngraphormer.preprocess.cli.write_reaction_table_files", side_effect=fake_write),
            mock.patch("rxngraphormer.preprocess.cli.RXNDataset", side_effect=fake_rxn_dataset),
        ):
            preprocess_from_config("config.json")

        self.assertIn("write", calls)
        self.assertEqual(len(calls["datasets"]), 2)


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
            "--split_manifest",
            "split.json",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.cli.load_config", return_value=config) as load_config_mock,
            mock.patch("rxngraphormer.lightning.fit_config", side_effect=fake_fit_config),
        ):
            cli.train_main()

        load_config_mock.assert_called_once_with("config.json")
        self.assertIs(calls["fit"][0], config)
        self.assertEqual(calls["fit"][1].accelerator, "cpu")
        self.assertEqual(calls["fit"][1].max_epochs, 3)
        self.assertEqual(calls["fit"][1].split_manifest, "split.json")

    def test_train_cli_can_keep_legacy_regression_path(self):
        config = type("Config", (), {"task": "regression"})()

        argv = ["rxngraphormer-train", "--config_json", "config.json", "--legacy_regression"]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.cli.load_config", return_value=config),
            mock.patch("rxngraphormer.cli._run_legacy_train") as legacy_train,
            mock.patch("rxngraphormer.lightning.fit_config") as fit_config_mock,
        ):
            cli.train_main()

        legacy_train.assert_called_once_with(config, local_rank=-1)
        fit_config_mock.assert_not_called()

    def test_train_cli_keeps_non_regression_on_legacy_path(self):
        config = type("Config", (), {"task": "classification"})()

        argv = ["rxngraphormer-train", "--config", "config.json", "--local_rank", "2"]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("rxngraphormer.cli.load_config", return_value=config),
            mock.patch("rxngraphormer.cli._run_legacy_train") as legacy_train,
            mock.patch("rxngraphormer.lightning.fit_config") as fit_config_mock,
        ):
            cli.train_main()

        legacy_train.assert_called_once_with(config, local_rank=2)
        fit_config_mock.assert_not_called()


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
            mock.patch("rxngraphormer.cli.load_config", return_value=config),
            mock.patch("rxngraphormer.regression_workflow.evaluate_regression_split", side_effect=fake_evaluate),
            mock.patch("rxngraphormer.regression_workflow.write_regression_json") as write_json,
            mock.patch("rxngraphormer.regression_workflow.write_regression_csv") as write_csv,
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

    def test_legacy_eval_function_delegates_to_regression_workflow(self):
        from rxngraphormer import eval as eval_module

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
        with mock.patch("rxngraphormer.eval.evaluate_regression_split", return_value=fake_result) as evaluate:
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

        with mock.patch.object(model_module, "_require_onmt", return_value=self._fake_onmt_symbols()):
            self.assertIsInstance(RXNG2Sequencer(self._sequence_config("attxl"), vocab), RXNG2Sequencer)
            self.assertIsInstance(RXNG2Sequencer(self._sequence_config("attn"), vocab), RXNG2Sequencer)

    def test_sequence_model_rejects_unknown_attention_encoder_at_init(self):
        vocab = {"_PAD": 0, "_SOS": 1, "_EOS": 2, "C": 3}

        with mock.patch.object(model_module, "_require_onmt", return_value=self._fake_onmt_symbols()):
            with self.assertRaisesRegex(NotImplementedError, "Attention encoder type invalid"):
                RXNG2Sequencer(self._sequence_config("invalid"), vocab)

    def test_sequence_dependency_error_points_to_sequence_extra(self):
        original_symbols = model_module._ONMT_SYMBOLS
        model_module._ONMT_SYMBOLS = None

        def fake_import(name, *args, **kwargs):
            if name.startswith("onmt"):
                raise ImportError("missing onmt")
            return original_import(name, *args, **kwargs)

        original_import = __import__
        try:
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaisesRegex(ImportError, "uv sync --extra sequence"):
                    model_module._require_onmt()
        finally:
            model_module._ONMT_SYMBOLS = original_symbols

    def test_rxn_pred_uses_cleaned_isolated_temp_directory(self):
        api = object.__new__(rxn_emb.RXNClassifier)
        seen_roots = []

        def fake_from_dataset(root, **kwargs):
            seen_roots.append(Path(root))
            self.assertTrue((Path(root) / "rct_smiles_0.csv").exists())
            self.assertTrue((Path(root) / "pdt_smiles_0.csv").exists())
            return torch.tensor([1, 0]), torch.tensor([0.9, 0.8])

        api.rxn_pred_from_dataset = fake_from_dataset
        with mock.patch.object(rxn_emb, "canonical_smiles", side_effect=lambda smi: smi):
            preds, confs = api.rxn_pred(["C>>O", "CC>>CO"], batch_size=2)

        self.assertTrue(torch.equal(preds, torch.tensor([1, 0])))
        self.assertTrue(torch.allclose(confs, torch.tensor([0.9, 0.8])))
        self.assertEqual(len(seen_roots), 1)
        self.assertFalse(seen_roots[0].exists())


if __name__ == "__main__":
    unittest.main()
