from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import scatter, softmax

from ...data.constants import NUM_BOND_TYPE


def get_mess_around_edge(edge_index: torch.Tensor, mess: torch.Tensor) -> torch.Tensor:
    # create node to edge map
    num_nodes = int(torch.max(edge_index).item()) + 1
    node_to_edges: list[list[int]] = [[] for _ in range(num_nodes)]
    for edge_idx, edge in enumerate(edge_index.t()):
        src = int(edge[0].item())
        dst = int(edge[1].item())
        node_to_edges[src].append(edge_idx)
        node_to_edges[dst].append(edge_idx)

    # abstract node-edge features
    node_edge_features: list[torch.Tensor] = []
    for edges in node_to_edges:
        if edges:
            edge_features = mess[edges]
            node_edge_features.append(edge_features)
        else:
            node_edge_features.append(mess.new_zeros((1, mess.size(1))))

    node_edge_features_padded = pad_sequence(
        node_edge_features, batch_first=True, padding_value=0
    )
    return node_edge_features_padded


def sum_mess_around_edge(
    edge_index: torch.Tensor,
    mess: torch.Tensor,
    num_nodes: int | None = None,
) -> torch.Tensor:
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
    if edge_index.numel() == 0:
        return mess.new_zeros((num_nodes, mess.size(-1)))

    incident_nodes = torch.cat((edge_index[0], edge_index[1]), dim=0)
    incident_mess = torch.cat((mess, mess), dim=0)
    return scatter(
        incident_mess, incident_nodes, dim=0, dim_size=num_nodes, reduce="sum"
    )


def self_loop_bond_attr(num_nodes: int, num_features: int, edge_attr: torch.Tensor) -> torch.Tensor:
    self_loop_attr = edge_attr.new_zeros((num_nodes, num_features))
    self_loop_attr[:, 0] = NUM_BOND_TYPE - 1
    return self_loop_attr


def gat_attention_softmax(alpha: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    return softmax(alpha, edge_index[1])
