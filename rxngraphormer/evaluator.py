from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def regression_batch_input(batch_data, device: torch.device | str):
    if len(batch_data) == 2:
        rct_data, pdt_data = batch_data
        rct_data = rct_data.to(device)
        pdt_data = pdt_data.to(device)
        return [rct_data, pdt_data], rct_data.y.unsqueeze(1)
    if len(batch_data) == 3:
        rct_data, pdt_data, mid_data = batch_data
        rct_data = rct_data.to(device)
        pdt_data = pdt_data.to(device)
        mid_data = mid_data.to(device)
        return [rct_data, pdt_data, mid_data], rct_data.y.unsqueeze(1)
    raise ValueError("Regression batches must contain pair or triple graph data")


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


def evaluate_regression(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device | str,
    *,
    scale: float = 1.0,
    yield_constrain: bool = False,
    max_batches: int | None = None,
) -> RegressionEvaluation:
    model.eval()
    pred_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            model_input, target = regression_batch_input(batch_data, device)
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
