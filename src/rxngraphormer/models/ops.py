"""Small tensor operations shared by model components."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def get_sin_encodings(rel_pos_buckets: int, model_dim: int) -> torch.Tensor:
    pe = torch.zeros(rel_pos_buckets + 1, model_dim)
    position = torch.arange(0, rel_pos_buckets).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, model_dim, 2, dtype=torch.float) * -(math.log(10000.0) / model_dim)
    )
    pe[:-1, 0::2] = torch.sin(position.float() * div_term)
    pe[:-1, 1::2] = torch.cos(position.float() * div_term)

    return pe


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    key_mask: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_k)
    valid_mask: torch.Tensor | None = None
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        valid_mask = mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1)
    if valid_mask is not None:
        weights = weights.masked_fill(~valid_mask, 0.0)
    return torch.bmm(weights, value)


def index_scatter(sub_data: torch.Tensor, all_data: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0, index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)

    return all_data * mask.unsqueeze(-1) + buf


def index_select_ND(source: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))

    return target.view(final_size)


__all__ = [
    "get_sin_encodings",
    "index_scatter",
    "index_select_ND",
    "scaled_dot_product_attention",
]
