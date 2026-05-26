from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch_geometric.data.data import BaseData

from rxngraphormer.config import load_train_config, resolve_config_file
from rxngraphormer.serialization import DEFAULT_CHECKPOINT_FILE

from ..compatibility.checkpointing import resolve_model_checkpoint
from ..config_utils import as_bool
from ..data.collate import pair_collate_fn, triple_collate_fn
from ..data.pairing import PairDataset, SizedDataset, TripleDataset
from ..data.reaction_dataset import RXNDataset
from ..data.splits import get_idx_split
from ..inference import RegressionPrediction, RXNGraphormerPredictor
from .evaluator import RegressionEvaluation, evaluate_regression, regression_metrics


@dataclass
class RegressionEvaluationSettings:
    model_path: str | os.PathLike
    ckpt_file: str = DEFAULT_CHECKPOINT_FILE
    config_path: str | os.PathLike | None = None
    checkpoint_path: str | os.PathLike | None = None
    split: str = "test"
    specific_val: bool = False
    batch_size: int | None = None
    scale: float = 1.0
    yield_constrain: bool = False
    max_batches: int | None = None
    device: torch.device | str | None = None
    use_mid_inf: bool | None = None


@dataclass
class RegressionSplitResult:
    split: str
    evaluation: RegressionEvaluation
    files: dict[str, str]
    model_path: str
    config_path: str
    checkpoint_path: str

    @property
    def preds(self) -> torch.Tensor:
        return self.evaluation.preds

    @property
    def targets(self) -> torch.Tensor:
        return self.evaluation.targets

    def summary(self) -> dict[str, Any]:
        preds = self.evaluation.preds
        targets = self.evaluation.targets
        metrics = self.evaluation.metrics
        return {
            "split": self.split,
            "files": self.files,
            "count": metrics.count,
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "r2": metrics.r2,
            "target_min": float(targets.min()),
            "target_max": float(targets.max()),
            "pred_min": float(preds.min()),
            "pred_max": float(preds.max()),
            "model_path": self.model_path,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
        }


def resolve_regression_checkpoint(
    model_path: str | os.PathLike,
    *,
    ckpt_file: str = DEFAULT_CHECKPOINT_FILE,
    checkpoint_path: str | os.PathLike | None = None,
) -> Path:
    return resolve_model_checkpoint(
        model_path,
        ckpt_file=ckpt_file,
        checkpoint_path=checkpoint_path,
    )


def load_regression_predictor(settings: RegressionEvaluationSettings) -> RXNGraphormerPredictor:
    config_path = settings.config_path or resolve_config_file(settings.model_path)
    checkpoint_path = resolve_regression_checkpoint(
        settings.model_path,
        ckpt_file=settings.ckpt_file,
        checkpoint_path=settings.checkpoint_path,
    )
    return RXNGraphormerPredictor.from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        task="regression",
        device=settings.device,
    )


def split_name(name: str) -> str:
    aliases = {"val": "valid", "validation": "valid"}
    normalized = aliases.get(str(name).lower(), str(name).lower())
    if normalized not in {"train", "valid", "test"}:
        raise ValueError("split must be one of: train, valid, test")
    return normalized


def regression_split_files(config: Any, split: str, *, specific_val: bool) -> dict[str, str]:
    split = split_name(split)
    if specific_val or _uses_explicit_regression_split_files(config):
        if split == "train":
            return {
                "rct": config.data.train_rct_data_file,
                "pdt": config.data.train_pdt_data_file,
                "mid": config.data.train_mid_data_file,
            }
        if split == "valid":
            return {
                "rct": config.data.val_rct_data_file,
                "pdt": config.data.val_pdt_data_file,
                "mid": config.data.val_mid_data_file,
            }
        return {
            "rct": config.data.test_rct_data_file,
            "pdt": config.data.test_pdt_data_file,
            "mid": config.data.test_mid_data_file,
        }
    return {
        "rct": config.data.rct_data_file,
        "pdt": config.data.pdt_data_file,
        "mid": config.data.mid_data_file,
    }


def _uses_explicit_regression_split_files(config: Any) -> bool:
    data = config.data
    return (
        not bool(getattr(data, "rct_data_file", ""))
        and bool(getattr(data, "train_rct_data_file", ""))
        and bool(getattr(data, "val_rct_data_file", ""))
        and bool(getattr(data, "test_rct_data_file", ""))
    )


def build_regression_dataset(
    config: Any,
    *,
    split: str = "test",
    specific_val: bool = False,
    use_mid_inf: bool | None = None,
):
    split = split_name(split)
    explicit_files = specific_val or _uses_explicit_regression_split_files(config)
    files = regression_split_files(config, split, specific_val=specific_val)
    use_mid = as_bool(config.model.use_mid_inf) if use_mid_inf is None else bool(use_mid_inf)
    data_path = config.data.data_path
    trunck = config.data.data_trunck
    required = ("rct", "pdt", "mid") if use_mid else ("rct", "pdt")
    missing = [key for key in required if not files[key]]
    if missing:
        raise ValueError(
            f"Regression evaluation split {split!r} is missing dataset file fields: "
            + ", ".join(missing)
        )

    rct = RXNDataset(root=data_path, name=files["rct"], trunck=trunck)
    pdt = RXNDataset(root=data_path, name=files["pdt"], trunck=trunck)
    mid = RXNDataset(root=data_path, name=files["mid"], trunck=trunck) if use_mid else None

    if not explicit_files:
        split_ids = get_idx_split(
            len(rct),
            int(config.data.train_ratio * len(rct)),
            int(config.data.valid_ratio * len(rct)),
            config.data.seed,
        )[split]
        rct_dataset: SizedDataset[BaseData] = cast(SizedDataset[BaseData], rct[split_ids])
        pdt_dataset: SizedDataset[BaseData] = cast(SizedDataset[BaseData], pdt[split_ids])
        mid_dataset = None if mid is None else cast(SizedDataset[BaseData], mid[split_ids])
    else:
        rct_dataset = cast(SizedDataset[BaseData], rct)
        pdt_dataset = cast(SizedDataset[BaseData], pdt)
        mid_dataset = None if mid is None else cast(SizedDataset[BaseData], mid)

    if mid_dataset is None:
        pair_dataset: PairDataset[object, object] = PairDataset(rct_dataset, pdt_dataset)
        return pair_dataset, files, use_mid
    triple_dataset: TripleDataset[object, object, object] = TripleDataset(rct_dataset, pdt_dataset, mid_dataset)
    return triple_dataset, files, use_mid


def build_regression_dataloader(
    config: Any,
    *,
    split: str = "test",
    specific_val: bool = False,
    batch_size: int | None = None,
    use_mid_inf: bool | None = None,
):
    dataset, files, use_mid = build_regression_dataset(
        config,
        split=split,
        specific_val=specific_val,
        use_mid_inf=use_mid_inf,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size or config.data.batch_size,
        shuffle=False,
        collate_fn=triple_collate_fn if use_mid else pair_collate_fn,
    )
    return dataloader, files


def _evaluate_regression_split_loaded(
    config: Any,
    predictor: RXNGraphormerPredictor,
    settings: RegressionEvaluationSettings,
) -> RegressionSplitResult:
    config_path = settings.config_path or resolve_config_file(settings.model_path)
    dataloader, files = build_regression_dataloader(
        config,
        split=settings.split,
        specific_val=settings.specific_val,
        batch_size=settings.batch_size,
        use_mid_inf=settings.use_mid_inf,
    )
    evaluation = evaluate_regression(
        predictor.model,
        dataloader,
        predictor.device,
        scale=settings.scale,
        yield_constrain=settings.yield_constrain,
        max_batches=settings.max_batches,
    )
    checkpoint_path = resolve_regression_checkpoint(
        settings.model_path,
        ckpt_file=settings.ckpt_file,
        checkpoint_path=settings.checkpoint_path,
    )
    return RegressionSplitResult(
        split=split_name(settings.split),
        evaluation=evaluation,
        files=files,
        model_path=os.fspath(settings.model_path),
        config_path=os.fspath(config_path),
        checkpoint_path=os.fspath(checkpoint_path),
    )


def evaluate_regression_split(settings: RegressionEvaluationSettings) -> RegressionSplitResult:
    config_path = settings.config_path or resolve_config_file(settings.model_path)
    config = load_train_config(config_path)
    predictor = load_regression_predictor(settings)
    return _evaluate_regression_split_loaded(config, predictor, settings)


def evaluate_regression_splits(
    settings: RegressionEvaluationSettings,
    splits: Iterable[str] = ("train", "valid", "test"),
) -> dict[str, RegressionSplitResult]:
    config_path = settings.config_path or resolve_config_file(settings.model_path)
    config = load_train_config(config_path)
    predictor = load_regression_predictor(settings)
    results: dict[str, RegressionSplitResult] = {}
    for split in splits:
        split_key = split_name(split)
        results[split_key] = _evaluate_regression_split_loaded(
            config,
            predictor,
            RegressionEvaluationSettings(
                model_path=settings.model_path,
                ckpt_file=settings.ckpt_file,
                config_path=config_path,
                checkpoint_path=settings.checkpoint_path,
                split=split_key,
                specific_val=settings.specific_val,
                batch_size=settings.batch_size,
                scale=settings.scale,
                yield_constrain=settings.yield_constrain,
                max_batches=settings.max_batches,
                device=settings.device,
                use_mid_inf=settings.use_mid_inf,
            )
        )
    return results


def evaluate_regression_prediction(
    prediction: RegressionPrediction,
    *,
    scale: float = 1.0,
    yield_constrain: bool = False,
) -> RegressionEvaluation:
    if prediction.targets is None:
        raise ValueError("RegressionPrediction.targets is required for evaluation")
    return regression_metrics(
        prediction.preds,
        prediction.targets,
        scale=scale,
        yield_constrain=yield_constrain,
    )


def write_regression_json(results: RegressionSplitResult | dict[str, RegressionSplitResult], path: str | os.PathLike) -> None:
    if isinstance(results, RegressionSplitResult):
        payload: Any = results.summary()
    else:
        payload = {split: result.summary() for split, result in results.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_regression_csv(results: RegressionSplitResult | dict[str, RegressionSplitResult], path: str | os.PathLike) -> None:
    items = [results] if isinstance(results, RegressionSplitResult) else list(results.values())
    fields = [
        "split",
        "count",
        "mae",
        "rmse",
        "r2",
        "target_min",
        "target_max",
        "pred_min",
        "pred_max",
        "model_path",
        "config_path",
        "checkpoint_path",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in items:
            summary = result.summary()
            writer.writerow({field: summary[field] for field in fields})
