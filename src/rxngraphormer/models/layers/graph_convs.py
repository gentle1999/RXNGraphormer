# pyright: reportIncompatibleMethodOverride=false
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, scatter

from ...data.constants import (
    NUM_BOND_DIRECTION,
    NUM_BOND_INRING,
    NUM_BOND_ISCONJ,
    NUM_BOND_STEREO,
    NUM_BOND_TYPE,
)
from .message import gat_attention_softmax, self_loop_bond_attr, sum_mess_around_edge


class GCNConv(MessagePassing):
    # adapted from https://github.com/junxia97/Mole-BERT
    def __init__(self, emb_dim, aggr="add", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)

        self.edge_embedding_lst = [
            self.edge_embedding1,
            self.edge_embedding2,
            self.edge_embedding3,
            self.edge_embedding4,
            self.edge_embedding5,
        ]

        self.aggr = aggr
        self.bond_feat_red = bond_feat_red

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )
        row, col = edge_index
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce="sum")
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = self_loop_bond_attr(
            x.size(0), len(self.edge_embedding_lst), edge_attr
        )
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:, i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
        else:
            raise ValueError(
                "Invalid bond feature reduction method. Please choose from 'mean' or 'sum'"
            )

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        return self.propagate(
            edge_index=edge_index,
            aggr=self.aggr,
            x=x,
            edge_attr=edge_embeddings,
            norm=norm,
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GCNEdgeConv(MessagePassing):
    # adapted from https://github.com/junxia97/Mole-BERT
    def __init__(self, emb_dim, aggr="add", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)

        self.edge_embedding_lst = [
            self.edge_embedding1,
            self.edge_embedding2,
            self.edge_embedding3,
            self.edge_embedding4,
            self.edge_embedding5,
        ]
        self.bridge_layer = torch.nn.Linear(emb_dim * 3, emb_dim)
        self.aggr = aggr
        self.bond_feat_red = bond_feat_red

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )
        row, col = edge_index
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce="sum")
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = self_loop_bond_attr(
            x.size(0), len(self.edge_embedding_lst), edge_attr
        )
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:, i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
        else:
            raise ValueError(
                "Invalid bond feature reduction method. Please choose from 'mean' or 'sum'"
            )

        mess1 = x.index_select(
            index=edge_index[0], dim=0
        )  # mess1 = h.index_select(index=edge_index[0], dim=0)
        mess2 = edge_embeddings.clone()  # mess2 = edge_attr
        mess = torch.cat(
            [mess1, mess2], dim=-1
        )  # mess = torch.cat([mess1,mess2],dim=-1)
        nei_mess = sum_mess_around_edge(edge_index, mess, num_nodes=x.size(0))
        node_emb = torch.cat([x, nei_mess], dim=1)
        x = self.bridge_layer(node_emb)
        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        return self.propagate(
            edge_index=edge_index,
            aggr=self.aggr,
            x=x,
            edge_attr=edge_embeddings,
            norm=norm,
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class SimpGCNConv(MessagePassing):
    # adapted from https://github.com/junxia97/Mole-BERT
    def __init__(self, emb_dim, aggr="add", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Embedding(
            max(
                [
                    NUM_BOND_TYPE,
                    NUM_BOND_DIRECTION,
                    NUM_BOND_STEREO,
                    NUM_BOND_INRING,
                    NUM_BOND_ISCONJ,
                ]
            ),
            emb_dim,
        )
        self.edge_dim_size = len(
            [
                NUM_BOND_TYPE,
                NUM_BOND_DIRECTION,
                NUM_BOND_STEREO,
                NUM_BOND_INRING,
                NUM_BOND_ISCONJ,
            ]
        )
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr
        self.bond_feat_red = bond_feat_red

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )
        row, col = edge_index
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce="sum")
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = self_loop_bond_attr(x.size(0), self.edge_dim_size, edge_attr)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        if self.bond_feat_red == "mean":
            edge_embeddings = self.edge_embedding(edge_attr).mean(dim=1)
        elif self.bond_feat_red == "sum":
            edge_embeddings = self.edge_embedding(edge_attr).sum(dim=1)
        else:
            raise ValueError(
                "Invalid bond feature reduction method. Please choose from 'mean' or 'sum'"
            )

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        return self.propagate(
            edge_index=edge_index,
            aggr=self.aggr,
            x=x,
            edge_attr=edge_embeddings,
            norm=norm,
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class RGCNConv(MessagePassing):
    # adapted from https://github.com/junxia97/Mole-BERT
    def __init__(self, emb_dim, aggr="add", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.rnn = nn.GRUCell(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)

        self.edge_embedding_lst = [
            self.edge_embedding1,
            self.edge_embedding2,
            self.edge_embedding3,
            self.edge_embedding4,
            self.edge_embedding5,
        ]

        self.aggr = aggr
        self.bond_feat_red = bond_feat_red

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )
        row, col = edge_index
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce="sum")
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = self_loop_bond_attr(
            x.size(0), len(self.edge_embedding_lst), edge_attr
        )
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:, i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
        else:
            raise ValueError(
                "Invalid bond feature reduction method. Please choose from 'mean' or 'sum'"
            )

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        x = self.propagate(
            edge_index=edge_index,
            aggr=self.aggr,
            x=x,
            edge_attr=edge_embeddings,
            norm=norm,
        )

        hidden_state = torch.zeros_like(x)
        return self.rnn(x, hidden_state)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GINConv(MessagePassing):
    """
    Adapted from https://github.com/junxia97/Mole-BERT
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, out_dim, aggr="add", bond_feat_red="mean"):
        self.aggr = aggr
        super().__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, out_dim),
        )
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)

        self.edge_embedding_lst = [
            self.edge_embedding1,
            self.edge_embedding2,
            self.edge_embedding3,
            self.edge_embedding4,
            self.edge_embedding5,
        ]
        self.bond_feat_red = bond_feat_red

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = self_loop_bond_attr(
            x.size(0), len(self.edge_embedding_lst), edge_attr
        )
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:, i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
            # edge_embeddings = (self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2]) + self.edge_embedding4(edge_attr[:,3]) + self.edge_embedding5(edge_attr[:,4]))/5
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
            # edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2]) + self.edge_embedding4(edge_attr[:,3]) + self.edge_embedding5(edge_attr[:,4])
        else:
            raise ValueError(
                "Invalid bond feature reduction method. Please choose from 'mean' or 'sum'"
            )

        return self.propagate(
            edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings
        )

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class SimpGINConv(MessagePassing):
    """
    Adapted from https://github.com/junxia97/Mole-BERT
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, out_dim, aggr="add", bond_feat_red="mean"):
        self.aggr = aggr
        super().__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, out_dim),
        )
        self.edge_embedding = torch.nn.Embedding(
            max(
                [
                    NUM_BOND_TYPE,
                    NUM_BOND_DIRECTION,
                    NUM_BOND_STEREO,
                    NUM_BOND_INRING,
                    NUM_BOND_ISCONJ,
                ]
            ),
            emb_dim,
        )
        self.edge_dim_size = len(
            [
                NUM_BOND_TYPE,
                NUM_BOND_DIRECTION,
                NUM_BOND_STEREO,
                NUM_BOND_INRING,
                NUM_BOND_ISCONJ,
            ]
        )
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.bond_feat_red = bond_feat_red

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = self_loop_bond_attr(x.size(0), self.edge_dim_size, edge_attr)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        if self.bond_feat_red == "mean":
            edge_embeddings = self.edge_embedding(edge_attr).mean(dim=1)

        elif self.bond_feat_red == "sum":
            edge_embeddings = self.edge_embedding(edge_attr).sum(dim=1)
        else:
            raise ValueError(
                "Invalid bond feature reduction method. Please choose from 'mean' or 'sum'"
            )

        return self.propagate(
            edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings
        )

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GATConv(MessagePassing):
    # TODO
    def __init__(
        self, emb_dim, heads=2, negative_slope=0.2, aggr="add", bond_feat_red="mean"
    ):
        super().__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, heads * emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, heads * emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, heads * emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)

        self.edge_embedding_lst = [
            self.edge_embedding1,
            self.edge_embedding2,
            self.edge_embedding3,
            self.edge_embedding4,
            self.edge_embedding5,
        ]

        self.reset_parameters()
        self.bond_feat_red = bond_feat_red

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = self_loop_bond_attr(
            x.size(0), len(self.edge_embedding_lst), edge_attr
        )
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:, i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
            # edge_embeddings = (self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2]) + self.edge_embedding4(edge_attr[:,3]) + self.edge_embedding5(edge_attr[:,4]))/5
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
            # edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2]) + self.edge_embedding4(edge_attr[:,3]) + self.edge_embedding5(edge_attr[:,4])
        else:
            raise ValueError(
                "Invalid bond feature reduction method. Please choose from 'mean' or 'sum'"
            )

        # x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        x = self.weight_linear(x).view(-1, self.heads * self.emb_dim)
        return self.propagate(
            edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings
        )

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j + edge_attr
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = gat_attention_softmax(alpha, edge_index)
        return (x_j * alpha.view(-1, self.heads, 1)).mean(dim=1)

    def update(self, aggr_out):
        # aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    # TODO
    def __init__(self, emb_dim, aggr="mean", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)

        self.aggr = aggr

        self.edge_embedding_lst = [
            self.edge_embedding1,
            self.edge_embedding2,
            self.edge_embedding3,
            self.edge_embedding4,
            self.edge_embedding5,
        ]
        self.bond_feat_red = bond_feat_red

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = self_loop_bond_attr(
            x.size(0), len(self.edge_embedding_lst), edge_attr
        )
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:, i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)

        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)

        else:
            raise ValueError(
                "Invalid bond feature reduction method. Please choose from 'mean' or 'sum'"
            )

        x = self.linear(x)

        return self.propagate(
            edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings
        )

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)
