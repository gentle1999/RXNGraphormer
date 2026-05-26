from __future__ import annotations

from typing import Any, Literal, cast

import torch

from ..config_utils import as_bool

Float32MatmulPrecision = Literal["highest", "high", "medium"]


def configure_torch_runtime(runtime: Any) -> None:
    """Apply process-local torch performance switches from runtime config."""

    precision = getattr(runtime, "float32_matmul_precision", "highest")
    if precision in {"highest", "high", "medium"}:
        torch.set_float32_matmul_precision(cast(Float32MatmulPrecision, precision))

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = as_bool(getattr(runtime, "benchmark", False))
