from __future__ import annotations

import logging
import math
import os
import random
import sys
from datetime import datetime
from os import PathLike
from typing import cast

import numpy as np
import torch
from rdkit import RDLogger
from torch.optim import Optimizer


def _disable_rdkit_log(pattern: str) -> None:
    disable_log = cast(object, getattr(RDLogger, "DisableLog"))
    if callable(disable_log):
        disable_log(pattern)


def setup_logger(save_dir: str | PathLike[str]) -> logging.Logger:
    _disable_rdkit_log("rdApp.*")
    _disable_rdkit_log("rdApp.warning")
    os.makedirs(save_dir, exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{save_dir}/{dt}.log")
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def param_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def param_norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(float(param.norm().item()) ** 2 for param in model.parameters()))


def grad_norm(model: torch.nn.Module) -> float:
    return math.sqrt(
        sum(float(param.grad.norm().item()) ** 2 for param in model.parameters() if param.grad is not None)
    )


def get_lr(optimizer: Optimizer) -> str:
    return ",".join(str(round(float(param_group["lr"]), 8)) for param_group in optimizer.param_groups)


def set_seed(seed: int) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


__all__ = [
    "get_lr",
    "grad_norm",
    "param_count",
    "param_norm",
    "set_seed",
    "setup_logger",
]
