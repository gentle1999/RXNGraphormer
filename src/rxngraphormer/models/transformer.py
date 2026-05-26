from __future__ import annotations

import torch
from torch import nn

from .layers import FeedForward, MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size,intermediate_size,num_heads,hidden_dropout_prob):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size,num_heads)
        self.feed_forward = FeedForward(hidden_size,intermediate_size,hidden_dropout_prob)

    def forward(self, x, mask=None):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layer,hidden_size,intermediate_size,num_heads,hidden_dropout_prob, use_padding_mask=True):
        super().__init__()
        self.use_padding_mask = use_padding_mask
        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_size=hidden_size,intermediate_size=intermediate_size,
                                    num_heads=num_heads,hidden_dropout_prob=hidden_dropout_prob) for _ in range(num_layer)])

    def forward(self, x, lengths=None):
        valid_mask = None
        attention_mask = None
        if lengths is not None and self.use_padding_mask:
            max_len = x.size(1)
            valid_mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.to(x.device).unsqueeze(1)
            attention_mask = valid_mask.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
            if valid_mask is not None:
                x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


def masked_sequence_mean(x, lengths):
    max_len = x.size(1)
    mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.to(x.device).unsqueeze(1)
    denom = lengths.to(device=x.device, dtype=x.dtype).clamp_min(1).unsqueeze(1)
    return (x * mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1) / denom
