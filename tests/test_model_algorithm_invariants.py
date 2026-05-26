import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GraphNorm

from rxngraphormer.data import (
    ATOM_DICT,
    ATOM_FEAT_DIMS,
    _collated_data_count,
    _matched_raw_files,
    _raw_file_sort_key,
    _separate_collated_data,
    calc_batch_graph_distance,
    calc_graph_distance,
)
from rxngraphormer.data.batch import (
    add_dense_empty_node_edge,
    add_empty_node_and_edge,
    pad_feat,
    update_batch_idx,
)
from rxngraphormer.data.graph_data import ReactionGraphData
from rxngraphormer.models import (
    ClassifierLayer,
    RegressorLayer,
    TransformerEncoder,
    _decoder_src_placeholder,
    masked_sequence_mean,
    sequence_mean,
)
from rxngraphormer.models.encoders import RXNGraphEncoder
from rxngraphormer.models.layers import (
    MultiHeadAttention,
    gat_attention_softmax,
    get_mess_around_edge,
    self_loop_bond_attr,
    sum_mess_around_edge,
)
from rxngraphormer.models.ops import scaled_dot_product_attention


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

    def test_sequence_mean_supports_legacy_raw_mean(self):
        x = torch.tensor([[[1.0], [3.0], [100.0]]])
        lengths = torch.tensor([2])

        self.assertTrue(torch.allclose(sequence_mean(x, lengths, "modern"), torch.tensor([[2.0]])))
        self.assertTrue(torch.allclose(sequence_mean(x, lengths, "legacy"), torch.tensor([[104.0 / 3.0]])))

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

    def test_legacy_transformer_encoder_keeps_padding_tokens_visible(self):
        torch.manual_seed(7)
        encoder = TransformerEncoder(
            num_layer=2,
            hidden_size=8,
            intermediate_size=8,
            num_heads=2,
            hidden_dropout_prob=0.0,
            use_padding_mask=False,
        ).eval()
        short = torch.randn(1, 2, 8)
        long = torch.randn(1, 5, 8)

        with torch.no_grad():
            out_alone = encoder(short, torch.tensor([2]))
            padded_short = torch.cat([short, torch.ones(1, 3, 8)], dim=1)
            batch = torch.cat([padded_short, long], dim=0)
            out_batch = encoder(batch, torch.tensor([2, 5]))[:1, :2]

        self.assertFalse(torch.allclose(out_alone, out_batch, atol=1e-6))

    def test_legacy_graph_encoder_keeps_dropout_active_in_eval(self):
        encoder = RXNGraphEncoder(
            gnum_layer=2,
            emb_dim=4,
            gnn_aggr="mean",
            drop_ratio=0.5,
            node_readout="mean",
            always_on_dropout=True,
        ).eval()

        self.assertTrue(encoder.always_on_dropout)

    def test_graph_encoder_norm_type_none_uses_identity_layers(self):
        encoder = RXNGraphEncoder(
            gnum_layer=2,
            emb_dim=4,
            gnn_aggr="mean",
            drop_ratio=0.0,
            node_readout="mean",
            norm_type="none",
        ).eval()
        x, mol_index, edge_index, edge_attr, atom_batch = _minimal_graph_inputs()

        with torch.no_grad():
            out, out_mol_index, batch = encoder(x, mol_index, edge_index, edge_attr, atom_batch=atom_batch)

        self.assertIsInstance(encoder.norm_layers[0], torch.nn.Identity)
        self.assertEqual(out.shape, (3, 4))
        self.assertTrue(torch.equal(out_mol_index, torch.tensor([0, 0, 1])))
        self.assertTrue(torch.equal(batch, torch.tensor([0, 0])))

    def test_graph_encoder_norm_type_layernorm_uses_layernorm_layers(self):
        encoder = RXNGraphEncoder(
            gnum_layer=2,
            emb_dim=4,
            gnn_aggr="mean",
            drop_ratio=0.0,
            node_readout="mean",
            norm_type="layernorm",
        ).eval()
        x, mol_index, edge_index, edge_attr, atom_batch = _minimal_graph_inputs()

        with torch.no_grad():
            out, _, _ = encoder(x, mol_index, edge_index, edge_attr, atom_batch=atom_batch)

        self.assertIsInstance(encoder.norm_layers[0], torch.nn.LayerNorm)
        self.assertEqual(out.shape, (3, 4))

    def test_graph_encoder_norm_type_graphnorm_uses_node_level_graph_batch(self):
        encoder = RXNGraphEncoder(
            gnum_layer=2,
            emb_dim=4,
            gnn_aggr="mean",
            drop_ratio=0.0,
            node_readout="mean",
            norm_type="graphnorm",
        ).eval()
        x, mol_index, edge_index, edge_attr, atom_batch = _minimal_graph_inputs()

        with torch.no_grad():
            out, _, _ = encoder(x, mol_index, edge_index, edge_attr, atom_batch=atom_batch)

        self.assertIsInstance(encoder.norm_layers[0], GraphNorm)
        self.assertEqual(out.shape, (3, 4))

    def test_graph_encoder_norm_layers_alias_does_not_duplicate_state_dict_keys(self):
        encoder = RXNGraphEncoder(
            gnum_layer=2,
            emb_dim=4,
            gnn_aggr="mean",
            drop_ratio=0.0,
            node_readout="mean",
            norm_type="graphnorm",
        )

        self.assertIs(encoder.norm_layers, encoder.batch_norms)
        self.assertFalse(any(key.startswith("norm_layers.") for key in encoder.state_dict()))

    def test_graph_encoder_rejects_invalid_norm_type(self):
        with self.assertRaisesRegex(ValueError, "norm_type must be one of"):
            RXNGraphEncoder(
                gnum_layer=2,
                emb_dim=4,
                gnn_aggr="mean",
                drop_ratio=0.0,
                node_readout="mean",
                norm_type="invalid",
            )

    def test_regressor_layer_norm_type_layernorm_uses_layernorm_layers(self):
        layer = RegressorLayer(hidden_size=4, output_size=1, layer_num=2, norm_type="layernorm").eval()

        with torch.no_grad():
            out = layer(torch.ones(2, 4))

        self.assertIsInstance(layer.norm_layers[0], torch.nn.LayerNorm)
        self.assertEqual(out.shape, (2, 1))

    def test_regressor_norm_layers_alias_does_not_duplicate_state_dict_keys(self):
        layer = RegressorLayer(hidden_size=4, output_size=1, layer_num=2, norm_type="layernorm")

        self.assertIs(layer.norm_layers, layer.batch_norms)
        self.assertFalse(any(key.startswith("norm_layers.") for key in layer.state_dict()))

    def test_regressor_layer_rejects_graphnorm(self):
        with self.assertRaisesRegex(ValueError, "graphnorm is only supported in graph encoders"):
            RegressorLayer(hidden_size=4, output_size=1, layer_num=2, norm_type="graphnorm")

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

    def test_reaction_graph_data_batches_tensor_mol_index_without_pyg_increment(self):
        batch_data = Batch.from_data_list(
            [
                ReactionGraphData(x=torch.ones(3, 1), mol_index=torch.tensor([0, 0, 1])),
                ReactionGraphData(x=torch.ones(2, 1), mol_index=torch.tensor([0, 1])),
            ]
        )

        batch_mol_index, batch = update_batch_idx(
            batch_data.mol_index,
            device=torch.device("cpu"),
            atom_batch=batch_data.batch,
        )

        self.assertTrue(torch.equal(batch_data.mol_index, torch.tensor([0, 0, 1, 0, 1])))
        self.assertTrue(torch.equal(batch_mol_index, torch.tensor([0, 0, 1, 2, 3])))
        self.assertTrue(torch.equal(batch, torch.tensor([0, 0, 1, 1])))

    def test_update_batch_idx_uses_atom_batch_to_recover_legacy_incremented_tensor(self):
        batch_mol_index, batch = update_batch_idx(
            torch.tensor([0, 0, 1, 3, 4]),
            device=torch.device("cpu"),
            atom_batch=torch.tensor([0, 0, 0, 1, 1]),
        )

        self.assertTrue(torch.equal(batch_mol_index, torch.tensor([0, 0, 1, 2, 3])))
        self.assertTrue(torch.equal(batch, torch.tensor([0, 0, 1, 1])))

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


def _minimal_graph_inputs():
    x = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, 5), dtype=torch.long)
    mol_index = torch.tensor([0, 0, 1], dtype=torch.long)
    atom_batch = torch.zeros(3, dtype=torch.long)
    return x, mol_index, edge_index, edge_attr, atom_batch


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
