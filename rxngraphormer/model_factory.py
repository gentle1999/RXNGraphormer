from __future__ import annotations

from typing import Any

from torch.nn.init import xavier_uniform_

from .model import RXNGraphormer
from .utils import align_config, as_bool


def regression_config_kwargs(config: Any, *, output_size: int = 1) -> dict[str, Any]:
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
        "output_size": output_size,
        "output_norm": as_bool(config.model.output_norm),
        "split_process": True,
        "split_merge_method": config.model.split_merge_method,
        "output_act_func": config.model.output_act_func,
        "rct_batch_norm": as_bool(config.model.rct_batch_norm),
        "pdt_batch_norm": as_bool(config.model.pdt_batch_norm),
        "use_mid_inf": as_bool(config.model.use_mid_inf),
        "mid_iteract_method": config.model.mid_iteract_method,
        "mid_batch_norm": as_bool(config.model.mid_batch_norm),
        "mid_layer_num": config.model.mid_layer_num,
    }


def classification_config_kwargs(config: Any, *, output_size: int = 2) -> dict[str, Any]:
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
        "output_size": output_size,
        "split_process": True,
        "split_merge_method": config.model.split_merge_method,
        "output_act_func": config.model.output_act_func,
    }


def build_regression_model(config: Any, *, pretrained_ensemble: dict[str, Any] | None = None):
    return RXNGraphormer(
        "regression",
        align_config(regression_config_kwargs(config), "regressor"),
        "",
        pretrained_ensemble or {
            "pretrained_encoder": None,
            "pretrained_rct_encoder": None,
            "pretrained_pdt_encoder": None,
            "pretrained_mid_encoder": None,
        },
    ).get_model()


def build_classification_model(config: Any):
    return RXNGraphormer(
        "classification",
        align_config(classification_config_kwargs(config), "classifier"),
        "",
    ).get_model()


def build_pretrained_classification_model(pretrained_model_path: str):
    from .checkpointing import CheckpointAdapter
    from rxngraphormer.config import load_config, resolve_config_file

    pretrained_config = load_config(resolve_config_file(pretrained_model_path))
    model = build_classification_model(pretrained_config)
    CheckpointAdapter().load_into_model(
        model,
        f"{pretrained_model_path}/model/valid_checkpoint.pt",
        map_location="cpu",
        mode="strict",
    )
    return model


def build_regression_model_from_config(config: Any):
    pretrained_path = getattr(config.model, "pretrained_model_path", "")
    pretrained_ensemble = None
    unfreeze_pretrained = False
    if pretrained_path:
        pretrained_model = build_pretrained_classification_model(pretrained_path)
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


def _xavier_initialize_trainable(model):
    for param in model.parameters():
        if param.dim() > 1 and param.requires_grad:
            xavier_uniform_(param)


def build_sequence_model(config: Any, vocab: dict[str, int]):
    return RXNGraphormer("sequence_generation", config, vocab).get_model()
