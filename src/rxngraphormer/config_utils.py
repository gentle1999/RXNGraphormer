"""Configuration compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from .config import Box

AlignedConfigType = Literal["classifier", "regressor", "sequence_generation"]
ConfigValue = bool | int | float | str | None
ConfigMapping = Mapping[str, ConfigValue]
MutableConfigDict = dict[str, ConfigValue]


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return bool(value)


def norm_from_bool(value: object) -> str:
    return "batchnorm" if as_bool(value) else "none"


def align_config(input_dict: ConfigMapping, type_: AlignedConfigType = "classifier") -> object:
    class_dict: MutableConfigDict = {
        "emb_dim": 256,
        "JK": "last",
        "output_size": 2,
        "drop_ratio": 0.0,
        "num_heads": 4,
        "gnn_type": "gcn",
        "bond_feat_red": "mean",
        "gnn_aggr": "add",
        "node_readout": "sum",
        "trans_readout": "mean",
        "graph_pooling": "attention",
        "attn_drop_ratio": 0.0,
        "encoder_filter_size": 2048,
        "rel_pos_buckets": 11,
        "rel_pos": "emb_only",
        "forward_compat_mode": "modern",
        "encoder_norm": "batchnorm",
        "rct_norm": "batchnorm",
        "pdt_norm": "batchnorm",
        "mid_norm": "batchnorm",
        "head_norm": "batchnorm",
        "split_process": False,
        "split_merge_method": "all",
        "output_act_func": "relu",
    }
    regress_dict: MutableConfigDict = {
        "JK": "last",
        "output_size": 1,
        "drop_ratio": 0.0,
        "num_heads": 4,
        "gnn_type": "gcn",
        "bond_feat_red": "mean",
        "gnn_aggr": "add",
        "node_readout": "sum",
        "trans_readout": "mean",
        "graph_pooling": "attention",
        "attn_drop_ratio": 0.0,
        "encoder_filter_size": 2048,
        "rel_pos_buckets": 11,
        "rel_pos": "emb_only",
        "forward_compat_mode": "modern",
        "encoder_norm": "batchnorm",
        "rct_norm": "batchnorm",
        "pdt_norm": "batchnorm",
        "mid_norm": "batchnorm",
        "head_norm": "none",
        "output_norm": False,
        "split_process": False,
        "use_mid_inf": False,
        "interaction": False,
        "interaction_layer_num": 3,
        "mid_iteract_method": "attention",
        "split_merge_method": "all",
        "output_act_func": "relu",
        "rct_batch_norm": True,
        "pdt_batch_norm": True,
        "mid_batch_norm": True,
        "mid_layer_num": 1,
    }

    if type_ == "classifier":
        class_input = dict(input_dict)
        class_dict.update(class_input)
        _align_norm_fields(class_dict, class_input, head_default="batchnorm")
        return Box(class_dict)
    if type_ == "regressor":
        regress_input = dict(input_dict)
        regress_dict.update(regress_input)
        _align_norm_fields(regress_dict, regress_input, head_default=norm_from_bool(regress_dict["output_norm"]))
        return Box(regress_dict)
    return Box(cast(Mapping[str, object], input_dict))


def _align_norm_fields(
    target: MutableConfigDict,
    source: Mapping[str, ConfigValue],
    *,
    head_default: str,
) -> None:
    encoder_norm_value = source.get("encoder_norm", target.get("encoder_norm", "batchnorm"))
    encoder_norm = "batchnorm" if encoder_norm_value is None else str(encoder_norm_value)
    target["encoder_norm"] = encoder_norm
    for norm_key, legacy_key in (
        ("rct_norm", "rct_batch_norm"),
        ("pdt_norm", "pdt_batch_norm"),
        ("mid_norm", "mid_batch_norm"),
    ):
        if source.get(norm_key) is not None:
            continue
        if legacy_key in source:
            target[norm_key] = norm_from_bool(source[legacy_key])
        else:
            target[norm_key] = encoder_norm
    if source.get("head_norm") is None:
        target["head_norm"] = head_default


__all__ = ["align_config", "as_bool", "norm_from_bool"]
