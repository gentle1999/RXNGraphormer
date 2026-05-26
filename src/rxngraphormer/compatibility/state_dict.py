from __future__ import annotations

from collections.abc import Mapping

import torch


def update_state_dict_keys(
    old_state_dict: Mapping[str, torch.Tensor],
    prefix: str = "module.",
    compat: bool = True,
) -> dict[str, torch.Tensor]:
    new_state_dict: dict[str, torch.Tensor] = {}
    for key, value in old_state_dict.items():
        if key.startswith(prefix):
            new_state_dict[key[len(prefix) :]] = value
        else:
            new_state_dict[key] = value

    if not compat:
        return new_state_dict

    compat_state_dict: dict[str, torch.Tensor] = {}
    legacy_encoder_prefixes = (
        "rct_encoder.x_embedding",
        "rct_encoder.gnns",
        "rct_encoder.batch_norms",
        "pdt_encoder.x_embedding",
        "pdt_encoder.gnns",
        "pdt_encoder.batch_norms",
    )
    for key, value in new_state_dict.items():
        if key.startswith(legacy_encoder_prefixes):
            name_blocks = key.split(".")
            name_blocks.insert(1, "rxn_graph_encoder")
            compat_state_dict[".".join(name_blocks)] = value
        else:
            compat_state_dict[key] = value
    return compat_state_dict


__all__ = ["update_state_dict_keys"]
