"""Centralized device handling for model runtime code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

import torch

T = TypeVar("T")
M = TypeVar("M", bound=torch.nn.Module)


class SupportsTo(Protocol):
    def to(self, *args: object, **kwargs: object) -> object:
        ...


def resolve_device(device: torch.device | str | int | None = None) -> torch.device:
    """Resolve user/config device values into a safe torch device.

    ``None`` and ``"auto"`` prefer CUDA when available. CUDA requests fall back
    to CPU when this process has no CUDA runtime, matching the historical
    single-process trainer behavior.
    """

    if device is None or str(device).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        requested = device
    elif isinstance(device, int):
        requested = torch.device(f"cuda:{device}")
    else:
        text = str(device).strip().lower()
        if text in {"gpu", "cuda"}:
            requested = torch.device("cuda")
        elif text.isdigit():
            requested = torch.device(f"cuda:{text}")
        else:
            requested = torch.device(text)

    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def module_device(module: torch.nn.Module, default: torch.device | str | int | None = None) -> torch.device:
    """Return the first parameter/buffer device for a module."""

    for tensor in module.parameters(recurse=True):
        return tensor.device
    for tensor in module.buffers(recurse=True):
        return tensor.device
    return resolve_device(default)


def move_to_device(value: T, device: torch.device | str | int | None, *, non_blocking: bool = False) -> T:
    """Recursively move tensors, PyG Data objects, and containers to a device."""

    resolved = resolve_device(device)
    if torch.is_tensor(value):
        tensor_value = cast(torch.Tensor, value)
        return cast(T, tensor_value.to(resolved, non_blocking=non_blocking))
    if isinstance(value, Mapping):
        mapping_value = cast(Mapping[object, object], value)
        return cast(
            T,
            {
                key: move_to_device(item, resolved, non_blocking=non_blocking)
                for key, item in mapping_value.items()
            },
        )
    if isinstance(value, tuple):
        tuple_value = cast(tuple[object, ...], value)
        return cast(T, tuple(move_to_device(item, resolved, non_blocking=non_blocking) for item in tuple_value))
    if isinstance(value, list):
        list_value = cast(list[object], value)
        return cast(T, [move_to_device(item, resolved, non_blocking=non_blocking) for item in list_value])
    to_method = getattr(value, "to", None)
    if callable(to_method):
        try:
            return cast(T, to_method(resolved, non_blocking=non_blocking))
        except TypeError:
            return cast(T, to_method(resolved))
    return value


@dataclass(frozen=True)
class DeviceManager:
    """Small runtime object that owns the selected device and move policy."""

    device: torch.device
    non_blocking: bool = False

    @classmethod
    def from_value(
        cls,
        device: torch.device | str | int | None = None,
        *,
        non_blocking: bool = False,
    ) -> DeviceManager:
        return cls(resolve_device(device), non_blocking=non_blocking)

    def move(self, value: T) -> T:
        return move_to_device(value, self.device, non_blocking=self.non_blocking)

    def move_module(self, module: M) -> M:
        return cast(M, module.to(self.device))

    @property
    def map_location(self) -> str:
        return "cpu" if self.device.type == "cuda" else str(self.device)
