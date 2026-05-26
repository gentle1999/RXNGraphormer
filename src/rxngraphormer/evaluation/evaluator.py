from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..models.protocols import SupportsLogits
from ..runtime import DeviceManager


@dataclass
class RegressionMetrics:
    count: int
    mae: float
    rmse: float
    r2: float


@dataclass
class RegressionEvaluation:
    metrics: RegressionMetrics
    preds: torch.Tensor
    targets: torch.Tensor


@dataclass
class ClassificationMetrics:
    count: int
    accuracy: float
    loss: float
    mean_confidence: float


@dataclass
class ClassificationEvaluation:
    metrics: ClassificationMetrics
    logits: torch.Tensor
    probabilities: torch.Tensor
    preds: torch.Tensor
    targets: torch.Tensor
    confidence: torch.Tensor


def _manager_from_device(device: DeviceManager | torch.device | str | int | None) -> DeviceManager:
    if isinstance(device, DeviceManager):
        return device
    return DeviceManager.from_value(device)


def regression_batch_input(
    batch_data,
    device: DeviceManager | torch.device | str | int | None,
    *,
    move: bool = True,
):
    if len(batch_data) == 2:
        rct_data, pdt_data = _manager_from_device(device).move(batch_data) if move else batch_data
        return [rct_data, pdt_data], rct_data.y.unsqueeze(1)
    if len(batch_data) == 3:
        rct_data, pdt_data, mid_data = _manager_from_device(device).move(batch_data) if move else batch_data
        return [rct_data, pdt_data, mid_data], rct_data.y.unsqueeze(1)
    raise ValueError("Regression batches must contain pair or triple graph data")


def classification_batch_input(
    batch_data,
    device: DeviceManager | torch.device | str | int | None,
    *,
    move: bool = True,
):
    if len(batch_data) != 2:
        raise ValueError("Classification batches must contain reactant/product graph pairs")
    rct_data, pdt_data = _manager_from_device(device).move(batch_data) if move else batch_data
    return [rct_data, pdt_data], rct_data.y.view(-1).long()


def regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale: float = 1.0,
    yield_constrain: bool = False,
) -> RegressionEvaluation:
    preds = preds.detach().cpu().float()
    targets = targets.detach().cpu().float()
    if yield_constrain:
        preds = torch.clamp(preds, 0.0, 1.0)
    preds = preds * scale
    targets = targets * scale
    return RegressionEvaluation(
        metrics=RegressionMetrics(
            count=int(targets.shape[0]),
            mae=float(mean_absolute_error(targets, preds)),
            rmse=float(mean_squared_error(targets, preds) ** 0.5),
            r2=float(r2_score(targets, preds)),
        ),
        preds=preds,
        targets=targets,
    )


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> ClassificationEvaluation:
    logits = logits.detach().cpu().float()
    targets = targets.detach().cpu().view(-1).long()
    probabilities = F.softmax(logits, dim=-1)
    confidence, preds = probabilities.max(dim=-1)
    return ClassificationEvaluation(
        metrics=ClassificationMetrics(
            count=int(targets.shape[0]),
            accuracy=float((preds == targets).float().mean().item()),
            loss=float(F.cross_entropy(logits, targets).item()),
            mean_confidence=float(confidence.mean().item()),
        ),
        logits=logits,
        probabilities=probabilities,
        preds=preds,
        targets=targets,
        confidence=confidence,
    )


def evaluate_regression(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: DeviceManager | torch.device | str | int | None,
    *,
    scale: float = 1.0,
    yield_constrain: bool = False,
    max_batches: int | None = None,
) -> RegressionEvaluation:
    model.eval()
    manager = _manager_from_device(device)
    pred_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            model_input, target = regression_batch_input(batch_data, manager)
            out = model(model_input)
            pred_chunks.append(out.detach().cpu().float())
            target_chunks.append(target.detach().cpu().float())
    if not pred_chunks:
        raise ValueError("Cannot evaluate an empty dataloader")
    return regression_metrics(
        torch.cat(pred_chunks, dim=0),
        torch.cat(target_chunks, dim=0),
        scale=scale,
        yield_constrain=yield_constrain,
    )


def evaluate_classification(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: DeviceManager | torch.device | str | int | None,
    *,
    max_batches: int | None = None,
) -> ClassificationEvaluation:
    model.eval()
    manager = _manager_from_device(device)
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            model_input, target = classification_batch_input(batch_data, manager)
            if isinstance(model, SupportsLogits):
                logits = model.logits(model_input)
            else:
                probabilities = model(model_input)
                probabilities = cast(torch.Tensor, probabilities)
                logits = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
            logits_chunks.append(logits.detach().cpu().float())
            target_chunks.append(target.detach().cpu().long())
    if not logits_chunks:
        raise ValueError("Cannot evaluate an empty dataloader")
    return classification_metrics(
        torch.cat(logits_chunks, dim=0),
        torch.cat(target_chunks, dim=0),
    )
