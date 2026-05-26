from __future__ import annotations

from typing import Any, cast

import torch
from torch.nn.init import xavier_uniform_

from ..config import ForwardCompatMode
from ..config_utils import ConfigValue, align_config, as_bool
from .tasks import RXNG2Sequencer, RXNGClassifier, RXNGraphormer, RXNGRegressor


def regression_config_kwargs(config: Any, *, output_size: int = 1) -> dict[str, ConfigValue]:
    return {
        "emb_dim": config.model.emb_dim,
        "gnn_type": config.model.gnn_type,
        "gnn_aggr": config.model.gnn_aggr,
        "gnum_layer": config.model.gnn_num_layer,
        "node_readout": config.model.node_readout,
        "num_heads": config.model.num_heads,
        "JK": config.model.gnn_jk,
        "graph_pooling": config.model.graph_pooling,
        "tnum_layer": config.model.trans_num_layer,
        "trans_readout": config.model.trans_readout,
        "onum_layer": config.model.output_num_layer,
        "drop_ratio": config.model.drop_ratio,
        "bond_feat_red": config.model.bond_feat_red,
        "attn_drop_ratio": config.model.attn_drop_ratio,
        "encoder_filter_size": config.model.encoder_filter_size,
        "rel_pos_buckets": config.model.rel_pos_buckets,
        "rel_pos": config.model.rel_pos,
        "forward_compat_mode": getattr(config.model, "forward_compat_mode", "modern"),
        "output_size": output_size,
        "output_norm": as_bool(config.model.output_norm),
        "split_process": True,
        "split_merge_method": config.model.split_merge_method,
        "output_act_func": config.model.output_act_func,
        "rct_batch_norm": as_bool(config.model.rct_batch_norm),
        "pdt_batch_norm": as_bool(config.model.pdt_batch_norm),
        "rct_norm": getattr(config.model, "rct_norm", None),
        "pdt_norm": getattr(config.model, "pdt_norm", None),
        "head_norm": getattr(config.model, "head_norm", None),
        "use_mid_inf": as_bool(config.model.use_mid_inf),
        "mid_iteract_method": config.model.mid_iteract_method,
        "mid_batch_norm": as_bool(config.model.mid_batch_norm),
        "mid_norm": getattr(config.model, "mid_norm", None),
        "mid_layer_num": config.model.mid_layer_num,
    }


def classification_config_kwargs(config: Any, *, output_size: int = 2) -> dict[str, ConfigValue]:
    return {
        "emb_dim": config.model.emb_dim,
        "gnn_type": config.model.gnn_type,
        "gnn_aggr": config.model.gnn_aggr,
        "gnum_layer": config.model.gnn_num_layer,
        "node_readout": config.model.node_readout,
        "num_heads": config.model.num_heads,
        "JK": config.model.gnn_jk,
        "graph_pooling": config.model.graph_pooling,
        "tnum_layer": config.model.trans_num_layer,
        "trans_readout": config.model.trans_readout,
        "onum_layer": config.model.output_num_layer,
        "drop_ratio": config.model.drop_ratio,
        "bond_feat_red": config.model.bond_feat_red,
        "attn_drop_ratio": config.model.attn_drop_ratio,
        "encoder_filter_size": config.model.encoder_filter_size,
        "rel_pos_buckets": config.model.rel_pos_buckets,
        "rel_pos": config.model.rel_pos,
        "forward_compat_mode": getattr(config.model, "forward_compat_mode", "modern"),
        "output_size": output_size,
        "split_process": True,
        "split_merge_method": config.model.split_merge_method,
        "output_act_func": config.model.output_act_func,
        "rct_norm": getattr(config.model, "rct_norm", None),
        "pdt_norm": getattr(config.model, "pdt_norm", None),
        "head_norm": getattr(config.model, "head_norm", None),
    }


def build_regression_model(
    config: Any,
    *,
    pretrained_ensemble: dict[str, torch.nn.Module | None] | None = None,
) -> RXNGRegressor:
    return cast(
        RXNGRegressor,
        RXNGraphormer(
            "regression",
            align_config(regression_config_kwargs(config), "regressor"),
            "",
            pretrained_ensemble
            or {
                "pretrained_encoder": None,
                "pretrained_rct_encoder": None,
                "pretrained_pdt_encoder": None,
                "pretrained_mid_encoder": None,
            },
        ).get_model(),
    )


def build_classification_model(config: Any, *, initialize: bool = False) -> RXNGClassifier:
    model = cast(
        RXNGClassifier,
        RXNGraphormer(
            "classification",
            align_config(classification_config_kwargs(config), "classifier"),
            "",
        ).get_model(),
    )
    if initialize:
        _xavier_initialize_trainable(model)
    return model


def build_pretrained_classification_model(
    pretrained_model_path: str,
    *,
    forward_compat_mode: ForwardCompatMode | None = None,
) -> RXNGClassifier:
    from rxngraphormer.config import load_train_config, resolve_config_file

    from ..compatibility.checkpointing import CheckpointAdapter, resolve_model_checkpoint

    pretrained_config = load_train_config(resolve_config_file(pretrained_model_path))
    if forward_compat_mode is not None:
        pretrained_config.model.forward_compat_mode = forward_compat_mode
    model: RXNGClassifier = build_classification_model(pretrained_config)
    CheckpointAdapter().load_into_model(
        model,
        resolve_model_checkpoint(pretrained_model_path),
        map_location="cpu",
        mode="strict",
    )
    return model


def build_regression_model_from_config(config: Any) -> RXNGRegressor:
    pretrained_path = getattr(config.model, "pretrained_model_path", "")
    pretrained_ensemble: dict[str, torch.nn.Module | None] | None = None
    unfreeze_pretrained = False
    if pretrained_path:
        pretrained_model: RXNGClassifier = build_pretrained_classification_model(
            cast(str, pretrained_path),
            forward_compat_mode=getattr(config.model, "forward_compat_mode", "modern"),
        )
        rct_encoder = pretrained_model.rct_encoder
        pdt_encoder = pretrained_model.pdt_encoder
        unfreeze_pretrained = not as_bool(getattr(config.model, "pretrained_model_freeze", False))
        for param in rct_encoder.parameters():
            param.requires_grad = False
        for param in pdt_encoder.parameters():
            param.requires_grad = False
        pretrained_ensemble = {
            "pretrained_encoder": None,
            "pretrained_rct_encoder": rct_encoder,
            "pretrained_pdt_encoder": pdt_encoder,
            "pretrained_mid_encoder": None,
        }
    model = build_regression_model(config, pretrained_ensemble=pretrained_ensemble)
    _xavier_initialize_trainable(model)
    if pretrained_path and unfreeze_pretrained:
        for param in model.rct_encoder.parameters():
            param.requires_grad = True
        for param in model.pdt_encoder.parameters():
            param.requires_grad = True
    return model


def _xavier_initialize_trainable(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.dim() > 1 and param.requires_grad:
            xavier_uniform_(param)


def build_sequence_model(config: Any, vocab: dict[str, int]) -> RXNG2Sequencer:
    return cast(RXNG2Sequencer, RXNGraphormer("sequence_generation", config, vocab).get_model())
