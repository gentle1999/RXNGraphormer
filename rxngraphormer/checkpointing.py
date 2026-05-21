from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from .compat import load_legacy_torch


@dataclass
class CheckpointLoadReport:
    path: str
    mode: str
    source_key_count: int
    loaded_key_count: int
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    shape_mismatches: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = field(default_factory=dict)
    removed_prefixes: list[str] = field(default_factory=list)
    added_prefix: str | None = None
    renamed_keys: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys and not self.shape_mismatches


class CheckpointAdapter:
    """Load legacy and Lightning checkpoints into canonical RXNG modules.

    The canonical state dict for model weights is the bare
    ``RXNGRegressor.state_dict()``/``RXNGClassifier.state_dict()`` namespace.
    Lightning checkpoints are treated as training-resume artifacts and adapted
    by removing the wrapper ``model.`` prefix when loading into bare modules.
    """

    CHECKPOINT_KEYS = ("state_dict", "model_state_dict", "model")
    DEFAULT_REMOVE_PREFIXES = ("module.", "model.")

    def __init__(
        self,
        *,
        remove_prefixes: tuple[str, ...] = DEFAULT_REMOVE_PREFIXES,
        add_prefix: str | None = None,
        rename_map: Mapping[str, str] | None = None,
        legacy_rxn_graph_encoder: bool = True,
    ) -> None:
        self.remove_prefixes = remove_prefixes
        self.add_prefix = add_prefix
        self.rename_map = dict(rename_map or {})
        self.legacy_rxn_graph_encoder = legacy_rxn_graph_encoder

    def load_state_dict_file(self, path: str | Path, map_location: Any = "cpu") -> dict[str, torch.Tensor]:
        path = Path(path)
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:  # pragma: no cover - depends on optional install
                raise ImportError("safetensors is required to load .safetensors checkpoints") from exc
            checkpoint = load_file(str(path), device=str(map_location) if isinstance(map_location, str) else "cpu")
        else:
            checkpoint = load_legacy_torch(str(path), map_location=map_location)
        return self.extract_state_dict(checkpoint)

    def extract_state_dict(self, checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, Mapping):
            for key in self.CHECKPOINT_KEYS:
                value = checkpoint.get(key)
                if self._looks_like_state_dict(value):
                    return dict(value)
            if self._looks_like_state_dict(checkpoint):
                return dict(checkpoint)
        raise ValueError("Checkpoint does not contain a recognizable model state_dict")

    def canonicalize(self, state_dict: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], CheckpointLoadReport]:
        renamed_keys: dict[str, str] = {}
        removed_prefixes: list[str] = []
        canonical: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in self.remove_prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        removed_prefixes.append(prefix)
                        changed = True

            new_key = self.rename_map.get(new_key, new_key)
            if self.legacy_rxn_graph_encoder:
                new_key = self._map_legacy_encoder_key(new_key)
            if self.add_prefix and not new_key.startswith(self.add_prefix):
                new_key = f"{self.add_prefix}{new_key}"
            if new_key != key:
                renamed_keys[key] = new_key
            canonical[new_key] = value

        report = CheckpointLoadReport(
            path="",
            mode="canonicalize",
            source_key_count=len(state_dict),
            loaded_key_count=len(canonical),
            removed_prefixes=sorted(set(removed_prefixes)),
            added_prefix=self.add_prefix,
            renamed_keys=renamed_keys,
        )
        return canonical, report

    def load_into_model(
        self,
        model: torch.nn.Module,
        path: str | Path,
        *,
        map_location: Any = "cpu",
        mode: str = "strict",
    ) -> CheckpointLoadReport:
        if mode not in {"strict", "shape-compatible", "diagnostic"}:
            raise ValueError("mode must be one of: strict, shape-compatible, diagnostic")

        source = self.load_state_dict_file(path, map_location=map_location)
        canonical, report = self.canonicalize(source)
        report.path = str(path)
        report.mode = mode

        model_state = model.state_dict()
        shape_mismatches = {
            key: (tuple(value.shape), tuple(model_state[key].shape))
            for key, value in canonical.items()
            if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
        }
        report.shape_mismatches = shape_mismatches

        if mode == "shape-compatible":
            loadable = {
                key: value
                for key, value in canonical.items()
                if key in model_state and key not in shape_mismatches
            }
            result = model.load_state_dict(loadable, strict=False)
        elif mode == "diagnostic":
            loadable = {
                key: value
                for key, value in canonical.items()
                if key in model_state and key not in shape_mismatches
            }
            result = model.load_state_dict(loadable, strict=False)
        else:
            if shape_mismatches:
                mismatch_text = ", ".join(
                    f"{key}: ckpt{src_shape} != model{dst_shape}"
                    for key, (src_shape, dst_shape) in shape_mismatches.items()
                )
                raise RuntimeError(f"Checkpoint shape mismatch: {mismatch_text}")
            result = model.load_state_dict(canonical, strict=True)

        report.loaded_key_count = sum(
            1 for key in canonical if key in model_state and key not in shape_mismatches
        )
        report.missing_keys = list(result.missing_keys)
        report.unexpected_keys = list(result.unexpected_keys) or [
            key for key in canonical if key not in model_state
        ]
        return report

    def save_canonical_state_dict(self, model: torch.nn.Module, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        if path.suffix == ".safetensors":
            try:
                from safetensors.torch import save_file
            except ImportError as exc:  # pragma: no cover
                raise ImportError("safetensors is required to save .safetensors checkpoints") from exc
            save_file(state_dict, str(path))
        else:
            torch.save(state_dict, path)

    @staticmethod
    def _looks_like_state_dict(value: Any) -> bool:
        return isinstance(value, Mapping) and bool(value) and all(
            isinstance(key, str) for key in value.keys()
        )

    @staticmethod
    def _map_legacy_encoder_key(key: str) -> str:
        prefixes = (
            "rct_encoder.x_embedding",
            "rct_encoder.gnns",
            "rct_encoder.batch_norms",
            "pdt_encoder.x_embedding",
            "pdt_encoder.gnns",
            "pdt_encoder.batch_norms",
        )
        if key.startswith(prefixes):
            parts = key.split(".")
            parts.insert(1, "rxn_graph_encoder")
            return ".".join(parts)
        return key


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    map_location: Any = "cpu",
    mode: str = "strict",
    add_prefix: str | None = None,
) -> CheckpointLoadReport:
    adapter = CheckpointAdapter(add_prefix=add_prefix)
    return adapter.load_into_model(model, path, map_location=map_location, mode=mode)
