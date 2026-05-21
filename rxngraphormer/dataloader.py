from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .utils import as_bool


@dataclass
class DataLoaderSettings:
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None


def dataloader_settings_from_config(config: Any, *, batch_size: int | None = None) -> DataLoaderSettings:
    data = config.data
    return DataLoaderSettings(
        batch_size=int(batch_size if batch_size is not None else data.batch_size),
        num_workers=int(getattr(data, "num_workers", 0)),
        pin_memory=as_bool(getattr(data, "pin_memory", False)),
        persistent_workers=as_bool(getattr(data, "persistent_workers", False)),
        prefetch_factor=getattr(data, "prefetch_factor", None),
    )


def dataloader_kwargs(
    settings: DataLoaderSettings,
    *,
    shuffle: bool = False,
    collate_fn: Callable | None = None,
    sampler: Any = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "pin_memory": settings.pin_memory,
        "persistent_workers": settings.persistent_workers and settings.num_workers > 0,
    }
    if sampler is None:
        kwargs["shuffle"] = shuffle
    else:
        kwargs["sampler"] = sampler
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if settings.num_workers > 0 and settings.prefetch_factor is not None:
        kwargs["prefetch_factor"] = settings.prefetch_factor
    return kwargs
