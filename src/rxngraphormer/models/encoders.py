from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GlobalAttention

from ..config import NormType
from ..data.batch import pad_feat, update_batch_idx
from ..data.constants import (
    NUM_AROMATIC_NUM,
    NUM_ATOM_TYPE,
    NUM_CHIRAL_TYPE,
    NUM_DEGRESS_TYPE,
    NUM_FORMCHRG_TYPE,
    NUM_HYBRIDTYPE,
    NUM_RS_TPYE,
    NUM_VALENCE_TYPE,
    NUM_Hs_TYPE,
)
from ..data.graph import GraphTask, calc_batch_graph_distance
from .attention_xl import AttnEncoderXL
from .layers import GATConv, GCNConv, GINConv
from .norms import make_graph_norm, normalize_norm_type
from .transformer import TransformerEncoder


class RXNGraphEncoder(nn.Module):
    def __init__(
        self,
        gnum_layer,
        emb_dim,
        gnn_aggr="add",
        bond_feat_red="mean",
        gnn_type="gcn",
        JK="last",
        drop_ratio=0.0,
        node_readout="sum",
        always_on_dropout=False,
        norm_type: NormType | str | None = "batchnorm",
    ):
        super().__init__()
        self.gnum_layer = gnum_layer
        self.emb_dim = emb_dim
        self.gnn_aggr = gnn_aggr
        self.gnn_type = gnn_type
        self.JK = JK
        self.drop_ratio = drop_ratio
        self.node_readout = node_readout
        self.always_on_dropout = always_on_dropout
        self.norm_type = normalize_norm_type(norm_type)
        assert self.gnum_layer >= 2, "Number of RXNGraphEncoder layers must be greater than 1."

        self.x_embedding1 = torch.nn.Embedding(NUM_ATOM_TYPE, self.emb_dim)  ## atom type
        self.x_embedding2 = torch.nn.Embedding(NUM_DEGRESS_TYPE, self.emb_dim)  ## atom degree
        self.x_embedding3 = torch.nn.Embedding(NUM_FORMCHRG_TYPE, self.emb_dim)  ## formal charge
        self.x_embedding4 = torch.nn.Embedding(NUM_HYBRIDTYPE, self.emb_dim)  ## hybrid type
        self.x_embedding5 = torch.nn.Embedding(NUM_CHIRAL_TYPE, self.emb_dim)  ## chiral type
        self.x_embedding6 = torch.nn.Embedding(NUM_AROMATIC_NUM, self.emb_dim)  ## aromatic or not
        self.x_embedding7 = torch.nn.Embedding(NUM_VALENCE_TYPE, self.emb_dim)  ## valence
        self.x_embedding8 = torch.nn.Embedding(NUM_Hs_TYPE, self.emb_dim)  ## number of Hs
        self.x_embedding9 = torch.nn.Embedding(NUM_RS_TPYE, self.emb_dim)  ## R or S

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding8.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding9.weight.data)

        self.x_emedding_lst = [
            self.x_embedding1,
            self.x_embedding2,
            self.x_embedding3,
            self.x_embedding4,
            self.x_embedding5,
            self.x_embedding6,
            self.x_embedding7,
            self.x_embedding8,
            self.x_embedding9,
        ]

        ## List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(self.gnum_layer):
            if self.gnn_type.lower() == "gcn":
                self.gnns.append(GCNConv(self.emb_dim, aggr=self.gnn_aggr, bond_feat_red=bond_feat_red))
            elif self.gnn_type.lower() == "gin":
                self.gnns.append(GINConv(self.emb_dim, self.emb_dim, aggr=self.gnn_aggr, bond_feat_red=bond_feat_red))
            elif self.gnn_type.lower() == "gat":
                self.gnns.append(GATConv(self.emb_dim, aggr=self.gnn_aggr, bond_feat_red=bond_feat_red))
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type.lower()}")

        ## Keep the historical attribute name for strict BatchNorm checkpoint compatibility.
        self.batch_norm = self.norm_type == "batchnorm"
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.gnum_layer):
            self.batch_norms.append(make_graph_norm(self.norm_type, self.emb_dim))

    @property
    def norm_layers(self) -> torch.nn.ModuleList:
        return self.batch_norms

    def forward(self, x, mol_index, edge_index, edge_attr, atom_batch=None):
        graph_norm_batch = _node_graph_batch(mol_index, x.shape[0], x.device, atom_batch=atom_batch)
        if atom_batch is not None and torch.is_tensor(mol_index) and mol_index.numel() != atom_batch.numel():
            atom_batch = None
        mol_index, batch = update_batch_idx(mol_index, device=x.device, atom_batch=atom_batch)
        mol_index = mol_index.to(x.device)
        batch = batch.to(x.device)
        x_emb_lst = []
        for i in range(x.shape[1]):
            _x_emb = self.x_emedding_lst[i](x[:, i])
            x_emb_lst.append(_x_emb)
        if self.node_readout == "sum":
            x_emb = torch.stack(x_emb_lst).sum(dim=0)
        elif self.node_readout == "mean":
            x_emb = torch.stack(x_emb_lst).mean(dim=0)
        else:
            raise NotImplementedError(f"Unknown node_readout: {self.node_readout}")
        h_list = [x_emb]
        for layer in range(self.gnum_layer):
            h = self.gnns[layer](h_list[layer], edge_index=edge_index, edge_attr=edge_attr)
            if self.norm_type == "graphnorm":
                h = self.batch_norms[layer](h, graph_norm_batch)
            else:
                h = self.batch_norms[layer](h)
            dropout_training = self.training or self.always_on_dropout
            if layer == self.gnum_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=dropout_training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=dropout_training)
            h_list.append(h)
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "mean":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.mean(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "last+first":
            node_representation = h_list[-1] + h_list[0]
        else:
            raise NotImplementedError

        return node_representation, mol_index, batch


def _node_graph_batch(
    mol_index: object,
    node_count: int,
    device: torch.device,
    *,
    atom_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    if atom_batch is not None and atom_batch.numel() == node_count:
        return atom_batch.to(device=device, dtype=torch.long).view(-1)
    if torch.is_tensor(mol_index):
        return torch.zeros(node_count, dtype=torch.long, device=device)
    mol_index_blocks = _mol_index_blocks(mol_index)
    if len(mol_index_blocks) <= 1:
        return torch.zeros(node_count, dtype=torch.long, device=device)
    block_sizes = [int(torch.as_tensor(block).numel()) for block in mol_index_blocks]
    if sum(block_sizes) != node_count:
        return torch.zeros(node_count, dtype=torch.long, device=device)
    return torch.repeat_interleave(
        torch.arange(len(block_sizes), dtype=torch.long, device=device),
        torch.tensor(block_sizes, dtype=torch.long, device=device),
    )


def _mol_index_blocks(mol_index: object) -> list[object]:
    if isinstance(mol_index, np.ndarray):
        return cast(list[object], mol_index.astype(object).tolist())
    if isinstance(mol_index, Sequence) and not isinstance(mol_index, (str, bytes)):
        sequence = cast(Sequence[object], mol_index)
        if len(sequence) == 0:
            return []
        first = sequence[0]
        if torch.is_tensor(first) or isinstance(first, (list, tuple, np.ndarray)):
            return [item for item in sequence]
    return [mol_index]


class RXNGEncoder(torch.nn.Module):
    def __init__(
        self,
        gnum_layer,
        tnum_layer,
        emb_dim,
        JK="last",
        drop_ratio=0.0,
        attn_drop_ratio=0.0,
        num_heads=4,
        gnn_type="gcn",
        bond_feat_red="mean",
        gnn_aggr="add",
        node_readout="sum",
        graph_pooling="attention",
        encoder_filter_size=2048,
        rel_pos_buckets=11,
        enc_pos_encoding=None,
        rel_pos="emb_only",
        task="retrosynthesis",
        add_empty_node=False,
        forward_compat_mode="modern",
        norm_type: NormType | str | None = "batchnorm",
    ):
        super().__init__()
        self.forward_compat_mode = forward_compat_mode
        self.use_padding_mask = self.forward_compat_mode != "legacy"
        self.rxn_graph_encoder = RXNGraphEncoder(
            gnum_layer=gnum_layer,
            emb_dim=emb_dim,
            gnn_aggr=gnn_aggr,
            bond_feat_red=bond_feat_red,
            gnn_type=gnn_type,
            JK=JK,
            drop_ratio=drop_ratio,
            node_readout=node_readout,
            always_on_dropout=self.forward_compat_mode == "legacy",
            norm_type=norm_type,
        )

        # self.gnum_layer = gnum_layer
        self.tnum_layer = tnum_layer
        self.drop_ratio = drop_ratio
        self.attn_drop_ratio = attn_drop_ratio
        # self.JK = JK
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        # self.node_readout = node_readout
        self.graph_pooling = graph_pooling
        self.encoder_filter_size = encoder_filter_size  ## attention_xl
        self.rel_pos_buckets = rel_pos_buckets
        self.enc_pos_encoding = enc_pos_encoding
        self.rel_pos = rel_pos
        if task not in {"forward_prediction", "retrosynthesis"}:
            raise ValueError(f"Unknown graph task: {task}")
        self.task: GraphTask = cast(GraphTask, task)
        # self.gnn_type = gnn_type
        self.add_empty_node = add_empty_node

        if self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.emb_dim, 1))

        elif self.graph_pooling == "attentionxl":
            self.pool = AttnEncoderXL(
                self.tnum_layer,
                d_model=self.emb_dim,
                heads=self.num_heads,
                d_ff=self.encoder_filter_size,
                dropout=self.drop_ratio,
                attention_dropout=self.attn_drop_ratio,
                rel_pos_buckets=self.rel_pos_buckets,
                enc_pos_encoding=self.enc_pos_encoding,
                rel_pos=self.rel_pos,
            )

        self.t_encoder = TransformerEncoder(
            num_layer=self.tnum_layer,
            hidden_size=self.emb_dim,
            intermediate_size=self.emb_dim,
            num_heads=num_heads,
            hidden_dropout_prob=self.drop_ratio,
            use_padding_mask=self.use_padding_mask,
        )

    def forward(self, data):
        x = data.x
        mol_index = data.mol_index
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_representation, mol_index, batch = self.rxn_graph_encoder(
            x,
            mol_index,
            edge_index,
            edge_attr,
            atom_batch=getattr(data, "batch", None),
        )

        if self.graph_pooling == "attention":
            memory_lengths = torch.bincount(batch).long().to(device=node_representation.device)
            rxn_representation = self.pool(node_representation, mol_index)  ## node_representation is equal to hatom
            padded_feat = pad_feat(rxn_representation, batch, self.emb_dim)
            rxn_transf_emb = self.t_encoder(padded_feat, memory_lengths)
            padded_memory_bank = rxn_transf_emb.transpose(1, 0)

        elif self.graph_pooling == "attentionxl":  ## TODO name it
            memory_lengths = torch.bincount(data.batch).long().to(device=node_representation.device)
            assert sum(memory_lengths) == node_representation.size(0), (
                f"Memory lengths calculation error, encoder output: {node_representation.size(0)}, memory_lengths: {memory_lengths}"
            )
            ## add an empty node in original paper
            memory_bank = torch.split(
                node_representation, memory_lengths.cpu().tolist(), dim=0
            )  # [n_atoms, h] => 1+b tup of (t, h)
            padded_memory_bank = []
            max_length = int(memory_lengths.max().item())
            for length, h in zip(memory_lengths, memory_bank):
                m = nn.ZeroPad2d((0, 0, 0, int(max_length - int(length))))
                padded_memory_bank.append(m(h))

            padded_memory_bank = torch.stack(padded_memory_bank, dim=1)  # list of b (max_t, h) => [max_t, b, h]
            distances = calc_batch_graph_distance(batch=data.batch, edge_index=data.edge_index, task=self.task)
            padded_memory_bank = self.pool(padded_memory_bank, memory_lengths, distances)

        else:
            raise NotImplementedError

        return padded_memory_bank, batch, memory_lengths
