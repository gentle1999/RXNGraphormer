from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import NormType
from ..config_utils import norm_from_bool
from .encoders import RXNGEncoder
from .heads import ClassifierLayer, RegressorLayer
from .sequence import RXNG2Sequencer
from .transformer import masked_sequence_mean


def sequence_mean(x: torch.Tensor, lengths: torch.Tensor, forward_compat_mode: str) -> torch.Tensor:
    if forward_compat_mode == "legacy":
        return x.mean(dim=1)
    return masked_sequence_mean(x, lengths)


class RXNGRegressor(torch.nn.Module):
    def __init__(
        self,
        gnum_layer,
        tnum_layer,
        onum_layer,
        emb_dim,
        JK="last",
        output_size=1,
        drop_ratio=0.0,
        num_heads=4,
        gnn_type="gcn",
        bond_feat_red="mean",
        gnn_aggr="add",
        node_readout="sum",
        trans_readout="mean",
        graph_pooling="attention",
        attn_drop_ratio=0.0,
        encoder_filter_size=2048,
        rel_pos_buckets=11,
        rel_pos="emb_only",
        forward_compat_mode="modern",
        pretrained_encoder=None,
        pretrained_rct_encoder=None,
        pretrained_pdt_encoder=None,
        output_norm=False,
        split_process=False,
        use_mid_inf=False,
        interaction=False,
        interaction_layer_num=3,
        pretrained_mid_encoder=None,
        mid_iteract_method="attention",
        split_merge_method="all",
        output_act_func="relu",
        rct_batch_norm=True,
        pdt_batch_norm=True,
        mid_batch_norm=True,
        mid_layer_num=1,
        rct_norm: NormType | str | None = None,
        pdt_norm: NormType | str | None = None,
        mid_norm: NormType | str | None = None,
        head_norm: NormType | str | None = None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.trans_readout = trans_readout
        self.split_process = split_process
        self.use_mid_inf = use_mid_inf
        self.interaction = interaction
        self.interaction_layer_num = interaction_layer_num
        self.split_merge_method = split_merge_method
        self.mid_iteract_method = mid_iteract_method
        self.output_act_func = output_act_func
        self.rct_batch_norm = rct_batch_norm
        self.pdt_batch_norm = pdt_batch_norm
        self.mid_batch_norm = mid_batch_norm
        self.mid_layer_num = mid_layer_num
        self.rct_norm = rct_norm or norm_from_bool(rct_batch_norm)
        self.pdt_norm = pdt_norm or norm_from_bool(pdt_batch_norm)
        self.mid_norm = mid_norm or norm_from_bool(mid_batch_norm)
        self.head_norm = head_norm or norm_from_bool(output_norm)
        self.forward_compat_mode = forward_compat_mode
        if self.forward_compat_mode not in {"modern", "legacy"}:
            raise ValueError("forward_compat_mode must be 'modern' or 'legacy'")
        assert self.split_merge_method in ["only_diff", "all", "rct_pdt"], (
            "split_merge_method must be one of ['only_diff','all','rct_pdt']"
        )
        assert self.mid_iteract_method in ["fc", "1dconv", "attention"], (
            "mid_merge_method must be one of ['fc','1dconv','attention']"
        )
        if not self.split_process:
            ## not be used
            if pretrained_encoder is None:
                self.encoder = RXNGEncoder(
                    gnum_layer,
                    tnum_layer,
                    emb_dim,
                    JK=JK,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    num_heads=num_heads,
                    gnn_type=gnn_type,
                    bond_feat_red=bond_feat_red,
                    gnn_aggr=gnn_aggr,
                    node_readout=node_readout,
                    graph_pooling=graph_pooling,
                    encoder_filter_size=encoder_filter_size,
                    rel_pos_buckets=rel_pos_buckets,
                    enc_pos_encoding=None,
                    rel_pos=rel_pos,
                    task="retrosynthesis",
                    forward_compat_mode=self.forward_compat_mode,
                    norm_type=self.rct_norm,
                )
            else:
                self.encoder = pretrained_encoder

            self.decoder = RegressorLayer(
                hidden_size=self.emb_dim,
                output_size=output_size,
                layer_num=onum_layer,
                batch_norm=output_norm,
                act_func=self.output_act_func,
                norm_type=self.head_norm,
            )
        else:
            if pretrained_rct_encoder is None:
                self.rct_encoder = RXNGEncoder(
                    gnum_layer,
                    tnum_layer,
                    emb_dim,
                    JK=JK,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    num_heads=num_heads,
                    gnn_type=gnn_type,
                    bond_feat_red=bond_feat_red,
                    gnn_aggr=gnn_aggr,
                    node_readout=node_readout,
                    graph_pooling=graph_pooling,
                    encoder_filter_size=encoder_filter_size,
                    rel_pos_buckets=rel_pos_buckets,
                    enc_pos_encoding=None,
                    rel_pos=rel_pos,
                    task="retrosynthesis",
                    forward_compat_mode=self.forward_compat_mode,
                    norm_type=self.rct_norm,
                )
            else:
                self.rct_encoder = pretrained_rct_encoder

            if pretrained_pdt_encoder is None:
                self.pdt_encoder = RXNGEncoder(
                    gnum_layer,
                    tnum_layer,
                    emb_dim,
                    JK=JK,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    num_heads=num_heads,
                    gnn_type=gnn_type,
                    bond_feat_red=bond_feat_red,
                    gnn_aggr=gnn_aggr,
                    node_readout=node_readout,
                    graph_pooling=graph_pooling,
                    encoder_filter_size=encoder_filter_size,
                    rel_pos_buckets=rel_pos_buckets,
                    enc_pos_encoding=None,
                    rel_pos=rel_pos,
                    task="retrosynthesis",
                    forward_compat_mode=self.forward_compat_mode,
                    norm_type=self.pdt_norm,
                )
            else:
                self.pdt_encoder = pretrained_pdt_encoder

            if self.use_mid_inf:
                if self.split_merge_method == "only_diff":
                    mid_emb_dim = 1 * emb_dim
                elif self.split_merge_method == "all":
                    mid_emb_dim = 3 * emb_dim
                elif self.split_merge_method == "rct_pdt":
                    mid_emb_dim = 2 * emb_dim
                else:
                    raise ValueError(f"Unknown split_merge_method: {self.split_merge_method}")

                if pretrained_mid_encoder is None:
                    self.mid_encoder = RXNGEncoder(
                        gnum_layer,
                        tnum_layer,
                        emb_dim=mid_emb_dim,
                        JK=JK,
                        drop_ratio=drop_ratio,
                        attn_drop_ratio=attn_drop_ratio,
                        num_heads=num_heads,
                        gnn_type=gnn_type,
                        bond_feat_red=bond_feat_red,
                        gnn_aggr=gnn_aggr,
                        node_readout=node_readout,
                        graph_pooling=graph_pooling,
                        encoder_filter_size=encoder_filter_size,
                        rel_pos_buckets=rel_pos_buckets,
                        enc_pos_encoding=None,
                        rel_pos=rel_pos,
                        task="retrosynthesis",
                        forward_compat_mode=self.forward_compat_mode,
                        norm_type=self.mid_norm,
                    )
                else:
                    self.mid_encoder = pretrained_mid_encoder

                if self.mid_iteract_method == "fc":
                    layers = [nn.Linear(2 * mid_emb_dim, mid_emb_dim), nn.ReLU()]
                    for i in range(self.mid_layer_num - 1):
                        layers.append(nn.Linear(mid_emb_dim, mid_emb_dim))
                        layers.append(nn.ReLU())
                    self.mid_iteract = nn.Sequential(*layers)
                elif self.mid_iteract_method == "1dconv":
                    self.mid_iteract = nn.Conv1d(2, 1, kernel_size=1)
                elif self.mid_iteract_method == "attention":
                    self.mid_iteract = nn.MultiheadAttention(mid_emb_dim, num_heads=1)
                else:
                    raise ValueError(f"Unknown mid_iteract_method: {self.mid_iteract_method}")

                self.mid_decoder = RegressorLayer(
                    hidden_size=mid_emb_dim,
                    output_size=output_size,
                    layer_num=onum_layer,
                    batch_norm=output_norm,
                    act_func=self.output_act_func,
                    norm_type=self.head_norm,
                )  ## TODO
                # raise NotImplemented(f"Not implemented pretrain ability!")
            if self.split_merge_method == "only_diff":
                dec_emb_dim = 1 * emb_dim
            elif self.split_merge_method == "all":
                dec_emb_dim = 3 * emb_dim
            elif self.split_merge_method == "rct_pdt":
                dec_emb_dim = 2 * emb_dim
            else:
                raise ValueError(f"Unknown split_merge_method: {self.split_merge_method}")

            self.decoder = RegressorLayer(
                hidden_size=dec_emb_dim,
                output_size=output_size,
                layer_num=onum_layer,
                batch_norm=output_norm,
                act_func=self.output_act_func,
                norm_type=self.head_norm,
            )

    def forward(self, data):
        output: torch.Tensor
        if not self.split_process:
            padded_memory_bank, batch, memory_lengths = self.encoder(data)
            rxn_transf_emb = padded_memory_bank.transpose(0, 1)
            if self.trans_readout == "mean":
                rxn_transf_emb_merg = sequence_mean(rxn_transf_emb, memory_lengths, self.forward_compat_mode)
            else:
                raise NotImplementedError(f"Unsupported trans_readout: {self.trans_readout}")

            output = self.decoder(rxn_transf_emb_merg)

        else:
            if not self.use_mid_inf:
                rct_data, pdt_data = data
                rct_padded_memory_bank, rct_batch, rct_memory_lengths = self.rct_encoder(rct_data)
                pdt_padded_memory_bank, pdt_batch, pdt_memory_lengths = self.pdt_encoder(pdt_data)
                rct_rxn_transf_emb = rct_padded_memory_bank.transpose(0, 1)
                pdt_rxn_transf_emb = pdt_padded_memory_bank.transpose(0, 1)
                if self.trans_readout == "mean":
                    rct_rxn_transf_emb_merg = sequence_mean(
                        rct_rxn_transf_emb, rct_memory_lengths, self.forward_compat_mode
                    )
                    pdt_rxn_transf_emb_merg = sequence_mean(
                        pdt_rxn_transf_emb, pdt_memory_lengths, self.forward_compat_mode
                    )
                else:
                    raise NotImplementedError(f"Unsupported trans_readout: {self.trans_readout}")

                diff_emb = torch.abs(rct_rxn_transf_emb_merg - pdt_rxn_transf_emb_merg)
                cat_emb = torch.cat([rct_rxn_transf_emb_merg, pdt_rxn_transf_emb_merg, diff_emb], dim=-1)
                rct_pdt_cat_emb = torch.cat([rct_rxn_transf_emb_merg, pdt_rxn_transf_emb_merg], dim=-1)
                if self.split_merge_method == "only_diff":
                    output = self.decoder(diff_emb)
                elif self.split_merge_method == "all":
                    output = self.decoder(cat_emb)
                elif self.split_merge_method == "rct_pdt":
                    output = self.decoder(rct_pdt_cat_emb)
                else:
                    raise ValueError(f"Unknown split_merge_method: {self.split_merge_method}")

            else:
                rct_data, pdt_data, mid_data = data
                rct_padded_memory_bank, rct_batch, rct_memory_lengths = self.rct_encoder(rct_data)
                pdt_padded_memory_bank, pdt_batch, pdt_memory_lengths = self.pdt_encoder(pdt_data)
                mid_padded_memory_bank, mid_batch, mid_memory_lengths = self.mid_encoder(mid_data)
                rct_rxn_transf_emb = rct_padded_memory_bank.transpose(0, 1)
                pdt_rxn_transf_emb = pdt_padded_memory_bank.transpose(0, 1)
                mid_rxn_transf_emb = mid_padded_memory_bank.transpose(0, 1)
                if self.trans_readout == "mean":
                    rct_rxn_transf_emb_merg = sequence_mean(
                        rct_rxn_transf_emb, rct_memory_lengths, self.forward_compat_mode
                    )
                    pdt_rxn_transf_emb_merg = sequence_mean(
                        pdt_rxn_transf_emb, pdt_memory_lengths, self.forward_compat_mode
                    )
                    mid_rxn_transf_emb_merg = sequence_mean(
                        mid_rxn_transf_emb, mid_memory_lengths, self.forward_compat_mode
                    )
                else:
                    raise NotImplementedError(f"Unsupported trans_readout: {self.trans_readout}")

                diff_emb = torch.abs(
                    rct_rxn_transf_emb_merg - pdt_rxn_transf_emb_merg
                )  ## shape: (batch_size, emb_dim) eg. 32, 256
                cat_emb = torch.cat(
                    [rct_rxn_transf_emb_merg, pdt_rxn_transf_emb_merg, diff_emb], dim=-1
                )  ## shape: (batch_size, 3 * emb_dim) eg. 32, 3 * 256
                rct_pdt_cat_emb = torch.cat(
                    [rct_rxn_transf_emb_merg, pdt_rxn_transf_emb_merg], dim=-1
                )  ## shape: (batch_size, 2 * emb_dim) eg. 32, 2 * 256
                if self.split_merge_method == "only_diff":
                    stack_emb = torch.stack(
                        [diff_emb, mid_rxn_transf_emb_merg], dim=1
                    )  ## shape: (batch_size, 2, emb_dim) eg. 32, 2, 256
                    # output = self.decoder(diff_emb)
                elif self.split_merge_method == "all":
                    stack_emb = torch.stack(
                        [cat_emb, mid_rxn_transf_emb_merg], dim=1
                    )  ## shape: (batch_size, 2, 3 * emb_dim) eg. 32, 2, 3 * 256
                    # output = self.decoder(cat_emb)
                elif self.split_merge_method == "rct_pdt":
                    stack_emb = torch.stack(
                        [rct_pdt_cat_emb, mid_rxn_transf_emb_merg], dim=1
                    )  ## shape: (batch_size, 2, 2 * emb_dim) eg. 32, 2, 2 * 256
                    # output = self.decoder(rct_pdt_cat_emb)
                else:
                    raise ValueError(f"Unknown split_merge_method: {self.split_merge_method}")

                if self.mid_iteract_method == "fc":
                    stack_emb_flat = stack_emb.view(
                        stack_emb.shape[0], -1
                    )  ## batch_size, 2, mid_emb_dim -> batch_size, 2 * mid_emb_dim
                    stack_emb_output = self.mid_iteract(
                        stack_emb_flat
                    )  ## fc = nn.Linear(2 * mid_emb_dim , mid_emb_dim) , (batch_size, 2 * mid_emb_dim) -> (batch_size, mid_emb_dim)
                elif self.mid_iteract_method == "1dconv":
                    stack_emb_output = self.mid_iteract(
                        stack_emb
                    ).squeeze(
                        1
                    )  ##   nn.Conv1d(2, 1, kernel_size=1), (batch_size, 2, mid_emb_dim) -> (batch_size, 1, mid_emb_dim) -> (batch_size, mid_emb_dim)
                elif self.mid_iteract_method == "attention":
                    stack_emb_perm = stack_emb.permute(
                        1, 0, 2
                    )  ##  (batch_size, 2, mid_emb_dim) -> (2, batch_size, mid_emb_dim)
                    stack_emb_perm_att, _ = self.mid_iteract(
                        stack_emb_perm, stack_emb_perm, stack_emb_perm
                    )  ##   nn.MultiheadAttention(mid_emb_dim, num_heads=1)
                    stack_emb_perm_att = stack_emb_perm_att.permute(1, 0, 2)
                    stack_emb_output = stack_emb_perm_att.mean(
                        dim=1
                    )  ##  (batch_size, 2, mid_emb_dim) -> (batch_size, mid_emb_dim)
                else:
                    raise ValueError(f"Unknown mid_iteract_method: {self.mid_iteract_method}")

                output = self.mid_decoder(stack_emb_output)

        return output


class RXNGClassifier(torch.nn.Module):
    def __init__(
        self,
        gnum_layer,
        tnum_layer,
        onum_layer,
        emb_dim=256,
        JK="last",
        output_size=2,
        drop_ratio=0.0,
        num_heads=4,
        gnn_type="gcn",
        bond_feat_red="mean",
        gnn_aggr="add",
        node_readout="sum",
        trans_readout="mean",
        graph_pooling="attention",
        attn_drop_ratio=0.0,
        encoder_filter_size=2048,
        rel_pos_buckets=11,
        rel_pos="emb_only",
        forward_compat_mode="modern",
        split_process=False,
        split_merge_method="all",
        output_act_func="relu",
        rct_norm: NormType | str | None = "batchnorm",
        pdt_norm: NormType | str | None = "batchnorm",
        head_norm: NormType | str | None = "batchnorm",
        use_mid_inf=False,
        mid_iteract_method="attention",
        mid_batch_norm=True,
        mid_layer_num=1,
        mid_norm: NormType | str | None = None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.trans_readout = trans_readout
        self.split_process = split_process
        self.use_mid_inf = use_mid_inf
        self.split_merge_method = split_merge_method.lower()
        assert self.split_merge_method in ["only_diff", "all", "rct_pdt"]
        self.mid_iteract_method = mid_iteract_method
        assert self.mid_iteract_method in ["fc", "1dconv", "attention"], (
            "mid_merge_method must be one of ['fc','1dconv','attention']"
        )
        self.output_act_func = output_act_func
        self.rct_norm = rct_norm
        self.pdt_norm = pdt_norm
        self.mid_batch_norm = mid_batch_norm
        self.mid_layer_num = mid_layer_num
        self.mid_norm = mid_norm or norm_from_bool(mid_batch_norm)
        self.head_norm = head_norm
        self.forward_compat_mode = forward_compat_mode
        if self.forward_compat_mode not in {"modern", "legacy"}:
            raise ValueError("forward_compat_mode must be 'modern' or 'legacy'")
        if not self.split_process:
            ## This option will be removed in the future
            self.encoder = RXNGEncoder(
                gnum_layer,
                tnum_layer,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                num_heads=num_heads,
                gnn_type=gnn_type,
                bond_feat_red=bond_feat_red,
                gnn_aggr=gnn_aggr,
                node_readout=node_readout,
                graph_pooling=graph_pooling,
                encoder_filter_size=encoder_filter_size,
                rel_pos_buckets=rel_pos_buckets,
                enc_pos_encoding=None,
                rel_pos=rel_pos,
                task="retrosynthesis",
                forward_compat_mode=self.forward_compat_mode,
                norm_type=self.rct_norm,
            )
            self.decoder = ClassifierLayer(
                hidden_size=self.emb_dim,
                output_size=output_size,
                layer_num=onum_layer,
                act_func=self.output_act_func,
                norm_type=self.head_norm,
            )
        else:
            self.rct_encoder = RXNGEncoder(
                gnum_layer,
                tnum_layer,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                num_heads=num_heads,
                gnn_type=gnn_type,
                bond_feat_red=bond_feat_red,
                gnn_aggr=gnn_aggr,
                node_readout=node_readout,
                graph_pooling=graph_pooling,
                encoder_filter_size=encoder_filter_size,
                rel_pos_buckets=rel_pos_buckets,
                enc_pos_encoding=None,
                rel_pos=rel_pos,
                task="retrosynthesis",
                forward_compat_mode=self.forward_compat_mode,
                norm_type=self.rct_norm,
            )
            self.pdt_encoder = RXNGEncoder(
                gnum_layer,
                tnum_layer,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                num_heads=num_heads,
                gnn_type=gnn_type,
                bond_feat_red=bond_feat_red,
                gnn_aggr=gnn_aggr,
                node_readout=node_readout,
                graph_pooling=graph_pooling,
                encoder_filter_size=encoder_filter_size,
                rel_pos_buckets=rel_pos_buckets,
                enc_pos_encoding=None,
                rel_pos=rel_pos,
                task="retrosynthesis",
                forward_compat_mode=self.forward_compat_mode,
                norm_type=self.pdt_norm,
            )

            if self.split_merge_method == "only_diff":
                dec_emb_dim = self.emb_dim
            elif self.split_merge_method == "all":
                dec_emb_dim = self.emb_dim * 3
            elif self.split_merge_method == "rct_pdt":
                dec_emb_dim = self.emb_dim * 2
            else:
                raise ValueError(f"Unknown split_merge_method: {self.split_merge_method}")

            if self.use_mid_inf:
                self.mid_encoder = RXNGEncoder(
                    gnum_layer,
                    tnum_layer,
                    dec_emb_dim,
                    JK=JK,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    num_heads=num_heads,
                    gnn_type=gnn_type,
                    bond_feat_red=bond_feat_red,
                    gnn_aggr=gnn_aggr,
                    node_readout=node_readout,
                    graph_pooling=graph_pooling,
                    encoder_filter_size=encoder_filter_size,
                    rel_pos_buckets=rel_pos_buckets,
                    enc_pos_encoding=None,
                    rel_pos=rel_pos,
                    task="retrosynthesis",
                    forward_compat_mode=self.forward_compat_mode,
                    norm_type=self.mid_norm,
                )
                if self.mid_iteract_method == "fc":
                    layers = [nn.Linear(2 * dec_emb_dim, dec_emb_dim), nn.ReLU()]
                    for _i in range(self.mid_layer_num - 1):
                        layers.append(nn.Linear(dec_emb_dim, dec_emb_dim))
                        layers.append(nn.ReLU())
                    self.mid_iteract = nn.Sequential(*layers)
                elif self.mid_iteract_method == "1dconv":
                    self.mid_iteract = nn.Conv1d(2, 1, kernel_size=1)
                elif self.mid_iteract_method == "attention":
                    self.mid_iteract = nn.MultiheadAttention(dec_emb_dim, num_heads=1)
                else:
                    raise ValueError(f"Unknown mid_iteract_method: {self.mid_iteract_method}")
                self.decoder = ClassifierLayer(
                    hidden_size=dec_emb_dim,
                    output_size=output_size,
                    layer_num=onum_layer,
                    act_func=self.output_act_func,
                    norm_type=self.head_norm,
                )
                return

            if self.split_merge_method == "all":
                self.decoder = ClassifierLayer(
                    hidden_size=self.emb_dim * 3,
                    output_size=output_size,
                    layer_num=onum_layer,
                    act_func=self.output_act_func,
                    norm_type=self.head_norm,
                )  # *3 -> rct, pdt, diff
            elif self.split_merge_method == "only_diff":
                self.decoder = ClassifierLayer(
                    hidden_size=self.emb_dim * 1,
                    output_size=output_size,
                    layer_num=onum_layer,
                    act_func=self.output_act_func,
                    norm_type=self.head_norm,
                )  # *1 -> diff
            elif self.split_merge_method == "rct_pdt":
                self.decoder = ClassifierLayer(
                    hidden_size=self.emb_dim * 2,
                    output_size=output_size,
                    layer_num=onum_layer,
                    act_func=self.output_act_func,
                    norm_type=self.head_norm,
                )  # *2 -> rct, pdt

    def logits(self, data):
        output: torch.Tensor
        if not self.split_process:
            padded_memory_bank, batch, memory_lengths = self.encoder(data)
            rxn_transf_emb = padded_memory_bank.transpose(0, 1)
            if self.trans_readout == "mean":
                rxn_transf_emb_merg = sequence_mean(rxn_transf_emb, memory_lengths, self.forward_compat_mode)
            else:
                raise NotImplementedError(f"Unsupported trans_readout: {self.trans_readout}")
            output = self.decoder.logits(rxn_transf_emb_merg)
        else:
            if self.use_mid_inf:
                rct_data, pdt_data, mid_data = data
            else:
                rct_data, pdt_data = data
                mid_data = None
            rct_padded_memory_bank, rct_batch, rct_memory_lengths = self.rct_encoder(rct_data)
            pdt_padded_memory_bank, pdt_batch, pdt_memory_lengths = self.pdt_encoder(pdt_data)
            rct_rxn_transf_emb = rct_padded_memory_bank.transpose(0, 1)
            pdt_rxn_transf_emb = pdt_padded_memory_bank.transpose(0, 1)
            if self.trans_readout == "mean":
                rct_rxn_transf_emb_merg = sequence_mean(
                    rct_rxn_transf_emb, rct_memory_lengths, self.forward_compat_mode
                )
                pdt_rxn_transf_emb_merg = sequence_mean(
                    pdt_rxn_transf_emb, pdt_memory_lengths, self.forward_compat_mode
                )
            else:
                raise NotImplementedError(f"Unsupported trans_readout: {self.trans_readout}")

            diff_emb = torch.abs(rct_rxn_transf_emb_merg - pdt_rxn_transf_emb_merg)
            if self.split_merge_method == "all":
                cat_emb = torch.cat([rct_rxn_transf_emb_merg, pdt_rxn_transf_emb_merg, diff_emb], dim=-1)
                base_emb = cat_emb
            elif self.split_merge_method == "only_diff":
                base_emb = diff_emb
            elif self.split_merge_method == "rct_pdt":
                rct_pdt_cat_emb = torch.cat([rct_rxn_transf_emb_merg, pdt_rxn_transf_emb_merg], dim=-1)
                base_emb = rct_pdt_cat_emb
            else:
                raise ValueError(f"Unknown split_merge_method: {self.split_merge_method}")

            if not self.use_mid_inf:
                output = self.decoder.logits(base_emb)
            else:
                assert mid_data is not None
                mid_padded_memory_bank, mid_batch, mid_memory_lengths = self.mid_encoder(mid_data)
                mid_rxn_transf_emb = mid_padded_memory_bank.transpose(0, 1)
                if self.trans_readout == "mean":
                    mid_rxn_transf_emb_merg = sequence_mean(
                        mid_rxn_transf_emb, mid_memory_lengths, self.forward_compat_mode
                    )
                else:
                    raise NotImplementedError(f"Unsupported trans_readout: {self.trans_readout}")
                stack_emb = torch.stack([base_emb, mid_rxn_transf_emb_merg], dim=1)
                if self.mid_iteract_method == "fc":
                    stack_emb_output = self.mid_iteract(stack_emb.view(stack_emb.shape[0], -1))
                elif self.mid_iteract_method == "1dconv":
                    stack_emb_output = self.mid_iteract(stack_emb).squeeze(1)
                elif self.mid_iteract_method == "attention":
                    stack_emb_perm = stack_emb.permute(1, 0, 2)
                    stack_emb_perm_att, _ = self.mid_iteract(stack_emb_perm, stack_emb_perm, stack_emb_perm)
                    stack_emb_output = stack_emb_perm_att.permute(1, 0, 2).mean(dim=1)
                else:
                    raise ValueError(f"Unknown mid_iteract_method: {self.mid_iteract_method}")
                output = self.decoder.logits(stack_emb_output)
        return output

    def forward(self, data):
        return F.softmax(self.logits(data), dim=-1)


class RXNGraphormer:
    def __init__(
        self,
        task_type,
        config,
        vocab,
        pretrained_ensemble={
            "pretrained_encoder": None,
            "pretrained_rct_encoder": None,
            "pretrained_pdt_encoder": None,
            "pretrained_mid_encoder": None,
        },
    ):
        assert task_type in ["regression", "classification", "sequence_generation"]
        self.task_type = task_type
        self.config = config
        self.vocab = vocab
        self.pretrained_ensemble = pretrained_ensemble

    def get_model(
        self,
        task_type=None,
        config=None,
        vocab=None,
        pretrained_ensemble={
            "pretrained_encoder": None,
            "pretrained_rct_encoder": None,
            "pretrained_pdt_encoder": None,
            "pretrained_mid_encoder": None,
        },
    ):
        if task_type is None:
            task_type = self.task_type
            config = self.config
            vocab = self.vocab
            pretrained_ensemble = self.pretrained_ensemble
        else:
            assert task_type is not None and config is not None

        if task_type == "regression":
            model = RXNGRegressor(
                gnum_layer=config.gnum_layer,
                tnum_layer=config.tnum_layer,
                onum_layer=config.onum_layer,
                emb_dim=config.emb_dim,
                JK=config.JK,
                output_size=config.output_size,
                drop_ratio=config.drop_ratio,
                num_heads=config.num_heads,
                gnn_type=config.gnn_type,
                bond_feat_red=config.bond_feat_red,
                gnn_aggr=config.gnn_aggr,
                node_readout=config.node_readout,
                trans_readout=config.trans_readout,
                graph_pooling=config.graph_pooling,
                attn_drop_ratio=config.attn_drop_ratio,
                encoder_filter_size=config.encoder_filter_size,
                rel_pos_buckets=config.rel_pos_buckets,
                rel_pos=config.rel_pos,
                forward_compat_mode=config.forward_compat_mode,
                pretrained_encoder=pretrained_ensemble["pretrained_encoder"],
                pretrained_rct_encoder=pretrained_ensemble["pretrained_rct_encoder"],
                pretrained_pdt_encoder=pretrained_ensemble["pretrained_pdt_encoder"],
                output_norm=config.output_norm,
                split_process=config.split_process,
                use_mid_inf=config.use_mid_inf,
                interaction=config.interaction,
                interaction_layer_num=config.interaction_layer_num,
                pretrained_mid_encoder=pretrained_ensemble["pretrained_mid_encoder"],
                mid_iteract_method=config.mid_iteract_method,
                split_merge_method=config.split_merge_method,
                output_act_func=config.output_act_func,
                rct_batch_norm=config.rct_batch_norm,
                pdt_batch_norm=config.pdt_batch_norm,
                mid_batch_norm=config.mid_batch_norm,
                mid_layer_num=config.mid_layer_num,
                rct_norm=config.rct_norm,
                pdt_norm=config.pdt_norm,
                mid_norm=config.mid_norm,
                head_norm=config.head_norm,
            )

        elif task_type == "classification":
            model = RXNGClassifier(
                gnum_layer=config.gnum_layer,
                tnum_layer=config.tnum_layer,
                onum_layer=config.onum_layer,
                emb_dim=config.emb_dim,
                JK=config.JK,
                output_size=config.output_size,
                drop_ratio=config.drop_ratio,
                num_heads=config.num_heads,
                gnn_type=config.gnn_type,
                bond_feat_red=config.bond_feat_red,
                gnn_aggr=config.gnn_aggr,
                node_readout=config.node_readout,
                trans_readout=config.trans_readout,
                graph_pooling=config.graph_pooling,
                attn_drop_ratio=config.attn_drop_ratio,
                encoder_filter_size=config.encoder_filter_size,
                rel_pos_buckets=config.rel_pos_buckets,
                rel_pos=config.rel_pos,
                forward_compat_mode=config.forward_compat_mode,
                split_process=config.split_process,
                split_merge_method=config.split_merge_method,
                output_act_func=config.output_act_func,
                rct_norm=config.rct_norm,
                pdt_norm=config.pdt_norm,
                head_norm=config.head_norm,
                use_mid_inf=config.use_mid_inf,
                mid_iteract_method=config.mid_iteract_method,
                mid_batch_norm=config.mid_batch_norm,
                mid_layer_num=config.mid_layer_num,
                mid_norm=config.mid_norm,
            )

        elif task_type == "sequence_generation":
            model = RXNG2Sequencer(config, vocab)
        else:
            raise NotImplementedError
        return model
