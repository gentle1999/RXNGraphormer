from __future__ import annotations

import time
from typing import Any

import lightning.pytorch as pl
import torch


class EpochRuntimeMonitor(pl.Callback):
    """Log epoch wall time and CUDA peak memory for Lightning training."""

    def __init__(self, *, log_epoch_time: bool = True, log_gpu_memory: bool = True) -> None:
        self.log_epoch_time = log_epoch_time
        self.log_gpu_memory = log_gpu_memory
        self._train_epoch_start: float | None = None

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        if self.log_epoch_time:
            self._train_epoch_start = time.perf_counter()
        if self.log_gpu_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        metrics: dict[str, float] = {}
        if self.log_epoch_time and self._train_epoch_start is not None:
            metrics["epoch_seconds"] = time.perf_counter() - self._train_epoch_start
        if self.log_gpu_memory and torch.cuda.is_available():
            metrics["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if metrics:
            pl_module.log_dict(metrics, prog_bar=False, logger=True, sync_dist=False)
