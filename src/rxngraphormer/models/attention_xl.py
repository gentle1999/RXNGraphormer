from __future__ import annotations

import math

import torch
from torch import nn

from .ops import get_sin_encodings
from .sequence_dependencies import require_onmt as _require_onmt


class MultiHeadedRelAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout, rel_pos_buckets, u, v, rel_pos="emb_only"):
        """
        rel_pos : "emb_only" or "enc_only"
        """
        super().__init__()
        assert model_dim % head_count == 0, "model_dim must be divisible by head_count (ERROR from MultiHeadedRelAttention)"
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, model_dim)
        self.linear_values = nn.Linear(model_dim, model_dim)
        self.linear_query = nn.Linear(model_dim, model_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.rel_pos_buckets = rel_pos_buckets
        self.rel_pos = rel_pos
        if self.rel_pos == "enc_only":
            self.relative_pe = nn.Embedding.from_pretrained(
                embeddings=get_sin_encodings(rel_pos_buckets, model_dim),
                freeze=True,
                padding_idx=rel_pos_buckets
            )
            # self.W_kR = nn.Parameter(
            #     torch.Tensor(self.head_count, self.dim_per_head, self.dim_per_head), requires_grad=True)
            # self.b_kR = nn.Parameter(
            #     torch.Tensor(self.head_count, self.dim_per_head), requires_grad=True)

        elif self.rel_pos == "emb_only":
            self.relative_pe = nn.Embedding(
                rel_pos_buckets + 1,
                model_dim,
                padding_idx=rel_pos_buckets
            )
            # self.W_kR = nn.Parameter(
            #     torch.Tensor(self.head_count, self.dim_per_head, self.dim_per_head), requires_grad=True)
            # self.b_kR = nn.Parameter(
            #     torch.Tensor(self.head_count, self.dim_per_head), requires_grad=True)

        else:
            self.relative_pe = None
            self.W_kR = None
            self.b_kR = None

        self.u = u
        self.v = v

    def forward(self, inputs, mask, distances):
        """
        Compute the context vector and the attention vectors.

        Args:
           inputs (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
           distances: graph distance matrix (BUCKETED), ``(batch, key_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """

        batch_size = inputs.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query. Seems that we don't need layer_cache here
        query = self.linear_query(inputs)
        key = self.linear_keys(inputs)
        value = self.linear_values(inputs)

        key = shape(key)                # (b, t_k, h) -> (b, head, t_k, h/head)
        value = shape(value)
        query = shape(query)            # (b, t_q, h) -> (b, head, t_q, h/head)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)

        if self.relative_pe is None:
            scores = torch.matmul(
                query, key.transpose(2, 3))                 # (b, head, t_q, t_k)

        else:
            # a + c
            u = self.u.reshape(1, head_count, 1, dim_per_head)
            a_c = torch.matmul(query + u, key.transpose(2, 3))

            rel_emb = self.relative_pe(distances)           # (b, t_q, t_k) -> (b, t_q, t_k, h)
            rel_emb = rel_emb.reshape(                      # (b, t_q, t_k, h) -> (b, t_q, t_k, head, h/head)
                batch_size, query_len, key_len, head_count, dim_per_head)

            # W_kR = self.W_kR.reshape(1, 1, 1, head_count, dim_per_head, dim_per_head)
            # rel_emb = torch.matmul(rel_emb, W_kR)           # (b, t_q, t_k, head, 1, h/head)
            # rel_emb = rel_emb.squeeze(-2)                   # (b, t_q, t_k, head, h/head)
            #
            # b_kR = self.b_kR.reshape(1, 1, 1, head_count, dim_per_head)
            # rel_emb = rel_emb + b_kR                        # (b, t_q, t_k, head, h/head)

            # b + d
            query = query.unsqueeze(-2)                     # (b, head, t_q, h/head) -> (b, head, t_q, 1, h/head)
            rel_emb_t = rel_emb.permute(0, 3, 1, 4, 2)      # (b, t_q, t_k, head, h/head) -> (b, head, t_q, h/head, t_k)

            v = self.v.reshape(1, head_count, 1, 1, dim_per_head)
            b_d = torch.matmul(query + v, rel_emb_t
                               ).squeeze(-2)                # (b, head, t_q, 1, t_k) -> (b, head, t_q, t_k)

            scores = a_c + b_d

        scores = scores.float()

        mask = mask.unsqueeze(1)                            # (B, 1, 1, T_values)
        scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)   # -> (b, head, t_q, h/head)
        context = unshape(context_original)                 # -> (b, t_q, h)

        output = self.final_linear(context)
        attns = attn.view(batch_size, head_count, query_len, key_len)

        return output, attns

class SALayerXL(nn.Module):
    """
    A single layer of the self-attention encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout: dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout, rel_pos_buckets, u, v, rel_pos="emb_only"):
        super().__init__()
        onmt = _require_onmt()

        self.self_attn = MultiHeadedRelAttention(
            heads, d_model, dropout=attention_dropout,
            rel_pos_buckets=rel_pos_buckets,
            u=u,
            v=v,
            rel_pos=rel_pos
        )
        self.feed_forward = onmt["PositionwiseFeedForward"](d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask, distances):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``
            distances (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, mask=mask, distances=distances)
        out = self.dropout(context) + inputs

        return self.feed_forward(out)

class AttnEncoderXL(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, attention_dropout, rel_pos_buckets,
                 enc_pos_encoding: str | None = "transformer", rel_pos="emb_only", encoder_emb_scale="sqrt"):
        super().__init__()
        onmt = _require_onmt()
        #self.args = args
        """
        self.num_layers = args.attn_enc_num_layers
        self.d_model = args.attn_enc_hidden_size
        self.heads = args.attn_enc_heads
        self.d_ff = args.attn_enc_filter_size
        self.attention_dropout = args.attn_dropout
        self.rel_pos_buckets = args.rel_pos_buckets
        """
        self.num_layers = num_layers                    # attn_enc_num_layers
        self.d_model = d_model                          # attn_enc_hidden_size
        self.heads = heads                              # attn_enc_heads
        self.d_ff = d_ff                                # attn_enc_filter_size
        self.dropout = dropout                          # dropout
        self.attention_dropout = attention_dropout      # attn_dropout
        self.rel_pos_buckets = rel_pos_buckets          # rel_pos_buckets
        self.rel_pos = rel_pos
        self.enc_pos_encoding = enc_pos_encoding        # encoder_positional_encoding
        self.encoder_pe = None
        self.encoder_emb_scale = encoder_emb_scale
        if self.enc_pos_encoding == "transformer":
            self.encoder_pe = onmt["PositionalEncoding"](
                dropout=self.dropout,
                dim=self.d_model,
                max_len=1024        # temporary hard-code. Seems that onmt fix the denominator as 10000.0
            )
        else:
            self.dropout = nn.Dropout(p=dropout)

        if self.rel_pos in ["enc_only", "emb_only"]:
            self.u = nn.Parameter(torch.randn(self.d_model), requires_grad=True)
            self.v = nn.Parameter(torch.randn(self.d_model), requires_grad=True)
        else:
            self.u = None
            self.v = None

        self.attention_layers = nn.ModuleList(
            [SALayerXL(
                self.d_model, self.heads, self.d_ff, dropout, self.attention_dropout,
                self.rel_pos_buckets, self.u, self.v, self.rel_pos)
             for i in range(self.num_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

    def forward(self, src, lengths, distances):
        """adapt from onmt TransformerEncoder
            src: (t, b, h)
            lengths: (b,)
            distances: (b, t, t)
        """

        if self.encoder_pe is not None:
            emb = self.encoder_pe(src)
            out = emb.transpose(0, 1).contiguous()
        else:
            out = src.transpose(0, 1).contiguous()
            if self.encoder_emb_scale == "sqrt":
                out = out * math.sqrt(self.d_model)
            out = self.dropout(out)

        mask = ~_require_onmt()["sequence_mask"](lengths).unsqueeze(1)

        for layer in self.attention_layers:
            out = layer(out, mask, distances)
        out = self.layer_norm(out)

        return out.transpose(0, 1).contiguous()
