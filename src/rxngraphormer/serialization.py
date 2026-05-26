"""Serialization helpers for weights and processed graph datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypeGuard, cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData

from .compatibility.torch_compat import MapLocation, load_legacy_torch

SAFETENSORS_SUFFIX = ".safetensors"
TORCH_SUFFIXES = {".pt", ".pth", ".ckpt"}
PROCESSED_DATA_FORMAT = "rxngraphormer-pyg-processed-v1"
CHECKPOINT_FORMAT = "rxngraphormer-checkpoint-v1"
DEFAULT_CHECKPOINT_FILE = "valid_checkpoint.safetensors"
LEGACY_CHECKPOINT_FILE = "valid_checkpoint.pt"
METADATA_KEY = "rxngraphormer.metadata"
JSONMetadataValue: TypeAlias = str | int | float | bool | None | list["JSONMetadataValue"] | dict[str, "JSONMetadataValue"]
PathInput: TypeAlias = str | PathLike[Any]


class _SafeTensorReader(Protocol):
    def __enter__(self) -> _SafeTensorReader:
        ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        ...

    def metadata(self) -> dict[str, str] | None:
        ...

    def keys(self) -> list[str]:
        ...

    def get_tensor(self, key: str) -> torch.Tensor:
        ...


def _safe_open_reader(path: PathInput, map_location: MapLocation) -> _SafeTensorReader:
    return cast(
        _SafeTensorReader,
        safe_open(str(path), framework="pt", device=_safetensors_device(map_location)),
    )


def _save_safetensors_file(
    tensors: dict[str, torch.Tensor],
    filename: PathInput,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    save_file(tensors, filename, metadata)


def has_safetensors() -> bool:
    return True


def prefer_safetensors_path(path: PathInput) -> Path:
    path = Path(path)
    if path.suffix in TORCH_SUFFIXES:
        return path.with_suffix(SAFETENSORS_SUFFIX)
    return path


def fallback_torch_path(path: PathInput) -> Path:
    path = Path(path)
    if path.suffix == SAFETENSORS_SUFFIX:
        return path.with_suffix(".pt")
    return path


def serialized_path_candidates(path: PathInput) -> list[Path]:
    path = Path(path)
    candidates = [path]
    if path.suffix == SAFETENSORS_SUFFIX:
        candidates.extend(path.with_suffix(suffix) for suffix in (".pt", ".pth", ".ckpt"))
    elif path.suffix in TORCH_SUFFIXES:
        candidates.append(path.with_suffix(SAFETENSORS_SUFFIX))
    return _unique_paths(candidates)


def existing_serialized_path(path: PathInput) -> Path:
    for candidate in serialized_path_candidates(path):
        if candidate.exists():
            return candidate
    return Path(path)


def resolve_checkpoint_path(
    model_path: PathInput,
    *,
    ckpt_file: str = DEFAULT_CHECKPOINT_FILE,
    checkpoint_path: PathInput | None = None,
) -> Path:
    if checkpoint_path is not None:
        return existing_serialized_path(checkpoint_path)
    return existing_serialized_path(Path(model_path) / "model" / ckpt_file)


def processed_safetensors_name(name: str) -> str:
    return str(Path(name).with_suffix(SAFETENSORS_SUFFIX))


def existing_processed_file_name(processed_dir: PathInput, preferred_name: str) -> str:
    preferred = Path(processed_safetensors_name(preferred_name))
    resolved = existing_serialized_path(Path(processed_dir) / preferred.name)
    return resolved.name if resolved.exists() else preferred.name


def existing_processed_path(path: PathInput) -> Path:
    return existing_serialized_path(path)


def processed_file_exists(path: PathInput) -> bool:
    return existing_serialized_path(path).exists()


def processed_safetensors_path(path: PathInput) -> Path:
    return prefer_safetensors_path(path)


def save_checkpoint_payload(payload: Mapping[str, object], path: PathInput) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != SAFETENSORS_SUFFIX:
        torch.save(dict(payload), path)
        return

    tensors, metadata = _checkpoint_tensors_and_metadata(payload)
    _save_safetensors_file(tensors, str(path), metadata=_safetensors_metadata(metadata))


def load_checkpoint_payload(path: PathInput, map_location: MapLocation = "cpu") -> object:
    path = existing_serialized_path(path)
    if path.suffix != SAFETENSORS_SUFFIX:
        return load_legacy_torch(str(path), map_location=map_location)

    tensors, metadata = _load_safetensors_with_metadata(path, map_location)
    if metadata is None:
        return tensors

    if metadata.get("format") != CHECKPOINT_FORMAT:
        return tensors

    payload: dict[str, object] = dict(_metadata_mapping(metadata.get("non_tensor_keys", {})))
    for key in _metadata_string_list(metadata.get("tensor_keys", [])):
        payload[key] = tensors[_join_safetensors_key("tensor", key)]
    for group, keys in _metadata_tensor_groups(metadata.get("tensor_groups", {})).items():
        payload[group] = {
            key: tensors[_join_safetensors_key(f"group.{group}", key)]
            for key in keys
        }
    return payload


def save_state_dict(state_dict: Mapping[str, torch.Tensor], path: PathInput) -> None:
    save_checkpoint_payload({"model_state_dict": state_dict}, path)


def load_state_dict(path: PathInput, map_location: MapLocation = "cpu") -> dict[str, torch.Tensor]:
    payload = load_checkpoint_payload(path, map_location=map_location)
    if _looks_like_tensor_mapping(payload):
        return dict(payload)
    if isinstance(payload, Mapping):
        payload_mapping = cast(Mapping[str, object], payload)
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload_mapping.get(key)
            if _looks_like_tensor_mapping(value):
                return dict(value)
    raise ValueError("Checkpoint does not contain a recognizable model state_dict")


def export_model_state_dict(model: torch.nn.Module, path: PathInput) -> None:
    save_state_dict(model.state_dict(), path)


def save_inference_checkpoint(model: torch.nn.Module, path: PathInput, **metadata: object) -> None:
    payload: dict[str, object] = {"model_state_dict": model.state_dict()}
    payload.update(metadata)
    save_checkpoint_payload(payload, path)


def save_training_checkpoint_pair(payload: Mapping[str, object], path: PathInput) -> None:
    """Save a full torch resume checkpoint plus adjacent safetensors weights."""

    path = Path(path)
    torch_path = path if path.suffix in TORCH_SUFFIXES else path.with_suffix(".pt")
    torch_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), torch_path)

    state_dict = payload.get("model_state_dict")
    if _looks_like_tensor_mapping(state_dict):
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
        }
        save_checkpoint_payload(
            {"model_state_dict": state_dict, **metadata},
            torch_path.with_suffix(SAFETENSORS_SUFFIX),
        )


def save_processed_graph_data(data: BaseData, slices: Mapping[str, torch.Tensor] | None, path: PathInput) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != SAFETENSORS_SUFFIX:
        torch.save((data, slices), path)
        return

    tensors: dict[str, torch.Tensor] = {}
    data_fields: dict[str, dict[str, str]] = {}
    slice_fields: list[str] = []
    normalized_slices = dict(slices) if slices is not None else None
    metadata: dict[str, object] = {
        "format": PROCESSED_DATA_FORMAT,
        "data_fields": data_fields,
        "slice_fields": slice_fields,
    }
    for key in sorted(data.keys()):
        value = data[key]
        slice_override: torch.Tensor | None = None
        if key == "mol_index" and not torch.is_tensor(value):
            value, slice_override = _legacy_mol_index_tensor_and_slices(value)
        elif not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if slice_override is not None:
            if normalized_slices is None:
                normalized_slices = {}
            normalized_slices[key] = slice_override
        tensors[_join_safetensors_key("data", key)] = value.detach().cpu().contiguous()
        data_fields[key] = {"kind": "tensor"}

    metadata["none_slices"] = normalized_slices is None
    if normalized_slices is not None:
        for key in sorted(normalized_slices):
            value = normalized_slices[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            tensors[_join_safetensors_key("slices", key)] = value.detach().cpu().contiguous()
            slice_fields.append(key)

    _save_safetensors_file(tensors, str(path), metadata=_safetensors_metadata(metadata))


def load_processed_graph_data(
    path: str | Path,
    *,
    map_location: MapLocation = "cpu",
) -> tuple[Data, dict[str, torch.Tensor] | None]:
    path = existing_serialized_path(path)
    if path.suffix != SAFETENSORS_SUFFIX:
        data, slices = cast(tuple[Data, dict[str, torch.Tensor] | None], load_legacy_torch(str(path), map_location=map_location))
        return _processed_data_container(data), slices

    tensors, metadata = _load_safetensors_with_metadata(path, map_location)
    if metadata is None:
        raise ValueError("Processed graph safetensors file is missing RXNGraphormer metadata")
    if metadata.get("format") != PROCESSED_DATA_FORMAT:
        raise ValueError(f"Unsupported processed graph safetensors format: {metadata.get('format')!r}")

    from .data.graph_data import ReactionGraphData

    data = ReactionGraphData()
    for key, field_info_obj in _metadata_mapping(metadata.get("data_fields", {})).items():
        field_info = _metadata_mapping(field_info_obj)
        if field_info.get("kind") != "tensor":
            raise ValueError(f"Unsupported processed graph field kind: {field_info.get('kind')!r}")
        data[key] = tensors[_join_safetensors_key("data", key)]

    slices = None
    if not metadata.get("none_slices", False):
        slices = {
            key: tensors[_join_safetensors_key("slices", key)]
            for key in _metadata_string_list(metadata.get("slice_fields", []))
        }
    return data, slices


def _processed_data_container(data: Data) -> Data:
    from .data.graph_data import ReactionGraphData

    if isinstance(data, ReactionGraphData):
        return data
    if "mol_index" not in data:
        return data
    graph_data = ReactionGraphData()
    for key in data.keys():
        graph_data[key] = data[key]
    return graph_data


def _legacy_mol_index_tensor_and_slices(value: object) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return torch.as_tensor(value, dtype=torch.long), None
    items = list(value)
    if not items:
        return torch.empty(0, dtype=torch.long), torch.tensor([0], dtype=torch.long)
    first = items[0]
    if not torch.is_tensor(first) and (
        not isinstance(first, Sequence) or isinstance(first, (str, bytes))
    ):
        return torch.as_tensor(items, dtype=torch.long), None

    blocks = [torch.as_tensor(item, dtype=torch.long).view(-1) for item in items]
    lengths = torch.tensor([block.numel() for block in blocks], dtype=torch.long)
    flat = torch.cat(blocks) if blocks else torch.empty(0, dtype=torch.long)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, dim=0)])
    return flat, offsets


def convert_serialized_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    kind: str,
    map_location: MapLocation = "cpu",
) -> Path:
    input_path = Path(input_path)
    output_path = prefer_safetensors_path(input_path) if output_path is None else Path(output_path)
    if output_path.exists() and output_path.resolve() == input_path.resolve():
        raise ValueError("input_path and output_path must be different")

    if kind == "checkpoint":
        loaded_payload = load_checkpoint_payload(input_path, map_location=map_location)
        if _looks_like_tensor_mapping(loaded_payload):
            payload: Mapping[str, object] = {"model_state_dict": loaded_payload}
        elif isinstance(loaded_payload, Mapping):
            payload = cast(Mapping[str, object], loaded_payload)
        else:
            raise ValueError("Checkpoint conversion expected a mapping or state_dict payload")
        save_checkpoint_payload(_checkpoint_export_payload(payload), output_path)
    elif kind == "processed-data":
        data, slices = load_processed_graph_data(input_path, map_location=map_location)
        save_processed_graph_data(data, slices, output_path)
    else:
        raise ValueError("kind must be 'checkpoint' or 'processed-data'")
    return output_path


def _load_safetensors_with_metadata(
    path: str | Path,
    map_location: MapLocation,
) -> tuple[dict[str, torch.Tensor], dict[str, object] | None]:
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, object] | None = None
    with _safe_open_reader(path, map_location) as handle:
        raw_metadata = handle.metadata() or {}
        if METADATA_KEY in raw_metadata:
            metadata = cast(dict[str, object], json.loads(raw_metadata[METADATA_KEY]))
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors, metadata


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _safetensors_device(map_location: MapLocation) -> str:
    if isinstance(map_location, torch.device):
        return str(map_location) if map_location.type != "cuda" else "cpu"
    if isinstance(map_location, str) and not map_location.startswith("cuda"):
        return map_location
    return "cpu"


def _escape_key(key: str) -> str:
    return key.replace("%", "%25").replace(".", "%2E")


def _join_safetensors_key(prefix: str, key: str) -> str:
    return f"{prefix}.{_escape_key(key)}"


def _safetensors_metadata(payload: Mapping[str, object]) -> dict[str, str]:
    return {METADATA_KEY: json.dumps(payload, sort_keys=True, separators=(",", ":"))}


def _jsonable_value(value: object) -> JSONMetadataValue:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, tuple):
        tuple_value = cast(tuple[object, ...], value)
        return [_jsonable_value(item) for item in tuple_value]
    if isinstance(value, list):
        list_value = cast(list[object], value)
        return [_jsonable_value(item) for item in list_value]
    if isinstance(value, dict):
        dict_value = cast(dict[object, object], value)
        return {str(key): _jsonable_value(item) for key, item in dict_value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot encode non-tensor value of type {type(value).__name__} in safetensors metadata")


def _looks_like_tensor_mapping(value: object) -> TypeGuard[Mapping[str, torch.Tensor]]:
    if not isinstance(value, Mapping) or not value:
        return False
    mapping_value = cast(Mapping[object, object], value)
    return all(
        isinstance(key, str) and torch.is_tensor(item)
        for key, item in mapping_value.items()
    )


def _checkpoint_tensors_and_metadata(payload: Mapping[str, object]) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    tensors: dict[str, torch.Tensor] = {}
    tensor_groups: dict[str, list[str]] = {}
    tensor_keys: list[str] = []
    non_tensor_keys: dict[str, object] = {}
    metadata: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "tensor_groups": tensor_groups,
        "tensor_keys": tensor_keys,
        "non_tensor_keys": non_tensor_keys,
    }

    for key, value in payload.items():
        key = str(key)
        if torch.is_tensor(value):
            tensor_value = cast(torch.Tensor, value)
            tensor_keys.append(key)
            tensors[_join_safetensors_key("tensor", key)] = tensor_value.detach().cpu().contiguous()
        elif _looks_like_tensor_mapping(value):
            tensor_groups[key] = []
            for tensor_key, tensor_value in value.items():
                tensor_groups[key].append(tensor_key)
                tensors[_join_safetensors_key(f"group.{key}", tensor_key)] = tensor_value.detach().cpu().contiguous()
        else:
            non_tensor_keys[key] = _jsonable_value(value)
    return tensors, metadata


def _checkpoint_export_payload(payload: Mapping[str, object]) -> dict[str, object]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if _looks_like_tensor_mapping(value):
            exported: dict[str, object] = {"model_state_dict": value}
            for meta_key, meta_value in payload.items():
                if meta_key in {"model_state_dict", "state_dict", "model", "optimizer_state_dict", "scheduler_state_dict"}:
                    continue
                if torch.is_tensor(meta_value) or isinstance(meta_value, (str, int, float, bool, type(None))):
                    exported[str(meta_key)] = meta_value
            return exported
    return dict(payload)


def _metadata_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return {
            str(key): item
            for key, item in cast(Mapping[object, object], value).items()
        }
    return {}


def _metadata_string_list(value: object) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        list_value = cast(list[object], value)
        return [cast(str, item) for item in list_value]
    return []


def _metadata_tensor_groups(value: object) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for key, item in _metadata_mapping(value).items():
        groups[key] = _metadata_string_list(item)
    return groups


__all__ = [
    "CHECKPOINT_FORMAT",
    "DEFAULT_CHECKPOINT_FILE",
    "LEGACY_CHECKPOINT_FILE",
    "METADATA_KEY",
    "PROCESSED_DATA_FORMAT",
    "SAFETENSORS_SUFFIX",
    "TORCH_SUFFIXES",
    "convert_serialized_file",
    "existing_processed_file_name",
    "existing_processed_path",
    "existing_serialized_path",
    "export_model_state_dict",
    "fallback_torch_path",
    "has_safetensors",
    "load_checkpoint_payload",
    "load_processed_graph_data",
    "load_state_dict",
    "prefer_safetensors_path",
    "processed_file_exists",
    "processed_safetensors_name",
    "processed_safetensors_path",
    "resolve_checkpoint_path",
    "save_checkpoint_payload",
    "save_inference_checkpoint",
    "save_processed_graph_data",
    "save_state_dict",
    "save_training_checkpoint_pair",
    "serialized_path_candidates",
]
