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
from ..data.collate import pair_collate_fn
from ..data.multi_reaction_dataset import MultiRXNDataset
from ..data.pairing import PairDataset, SizedDataset
from ..data.reaction_dataset import RXNDataset
from ..data.splits import get_idx_split
from ..inference import RXNGraphormerPredictor
from .evaluator import ClassificationEvaluation, evaluate_classification


@dataclass
class ClassificationEvaluationSettings:
    model_path: str | os.PathLike
    ckpt_file: str = DEFAULT_CHECKPOINT_FILE
    config_path: str | os.PathLike | None = None
    checkpoint_path: str | os.PathLike | None = None
    split: str = "valid"
    specific_val: bool = False
    batch_size: int | None = None
    max_batches: int | None = None
    device: torch.device | str | None = None


@dataclass
class ClassificationSplitResult:
    split: str
    evaluation: ClassificationEvaluation
    files: dict[str, str]
    model_path: str
    config_path: str
    checkpoint_path: str

    def summary(self) -> dict[str, Any]:
        metrics = self.evaluation.metrics
        return {
            "split": self.split,
            "files": self.files,
            "count": metrics.count,
            "accuracy": metrics.accuracy,
            "loss": metrics.loss,
            "mean_confidence": metrics.mean_confidence,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
        }


def resolve_classification_checkpoint(
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


def load_classification_predictor(settings: ClassificationEvaluationSettings) -> RXNGraphormerPredictor:
    config_path = settings.config_path or resolve_config_file(settings.model_path)
    checkpoint_path = resolve_classification_checkpoint(
        settings.model_path,
        ckpt_file=settings.ckpt_file,
        checkpoint_path=settings.checkpoint_path,
    )
    return RXNGraphormerPredictor.from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        task="classification",
        device=settings.device,
    )


def split_name(name: str) -> str:
    aliases = {"val": "valid", "validation": "valid"}
    normalized = aliases.get(str(name).lower(), str(name).lower())
    if normalized not in {"train", "valid", "test"}:
        raise ValueError("split must be one of: train, valid, test")
    return normalized


def classification_split_files(config: Any, split: str, *, specific_val: bool) -> dict[str, str]:
    split = split_name(split)
    if specific_val or _uses_explicit_classification_split_files(config):
        if split == "train":
            return {"rct": config.data.train_rct_data_file, "pdt": config.data.train_pdt_data_file}
        if split == "valid":
            return {"rct": config.data.val_rct_data_file, "pdt": config.data.val_pdt_data_file}
        return {"rct": config.data.test_rct_data_file, "pdt": config.data.test_pdt_data_file}
    return {
        "rct": config.data.rct_name_regrex or config.data.rct_data_file,
        "pdt": config.data.pdt_name_regrex or config.data.pdt_data_file,
    }


def _uses_explicit_classification_split_files(config: Any) -> bool:
    data = config.data
    return (
        not bool(getattr(data, "rct_data_file", ""))
        and not bool(getattr(data, "rct_name_regrex", ""))
        and bool(getattr(data, "train_rct_data_file", ""))
        and bool(getattr(data, "val_rct_data_file", ""))
        and bool(getattr(data, "test_rct_data_file", ""))
    )


def build_classification_dataset(config: Any, *, split: str = "valid", specific_val: bool = False):
    split = split_name(split)
    explicit_files = specific_val or _uses_explicit_classification_split_files(config)
    files = classification_split_files(config, split, specific_val=specific_val)
    data_path = config.data.data_path
    trunck = config.data.data_trunck
    if not files["rct"] or not files["pdt"]:
        raise ValueError("Classification evaluation requires reactant and product dataset files")

    if explicit_files:
        rct = RXNDataset(root=data_path, name=files["rct"], trunck=trunck, task="classification")
        pdt = RXNDataset(root=data_path, name=files["pdt"], trunck=trunck, task="classification")
        return PairDataset(rct, pdt), files

    rct = MultiRXNDataset(
        root=data_path,
        name_regrex=files["rct"],
        trunck=trunck,
        task="classification",
        file_num_trunck=config.data.file_num_trunck,
        name_tag="rct",
    )
    pdt = MultiRXNDataset(
        root=data_path,
        name_regrex=files["pdt"],
        trunck=trunck,
        task="classification",
        file_num_trunck=config.data.file_num_trunck,
        name_tag="pdt",
    )
    if len(rct) != len(pdt):
        raise ValueError("The number of reactant and product classification data are not equal")
    split_ids = get_idx_split(
        len(rct),
        int(config.data.train_ratio * len(rct)),
        int(config.data.valid_ratio * len(rct)),
        config.data.seed,
    )[split]
    rct_subset = cast(SizedDataset[BaseData], rct[split_ids])
    pdt_subset = cast(SizedDataset[BaseData], pdt[split_ids])
    return PairDataset(rct_subset, pdt_subset), files


def build_classification_dataloader(
    config: Any,
    *,
    split: str = "valid",
    specific_val: bool = False,
    batch_size: int | None = None,
):
    dataset, files = build_classification_dataset(config, split=split, specific_val=specific_val)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size or config.data.batch_size,
        shuffle=False,
        collate_fn=pair_collate_fn,
    )
    return dataloader, files


def _evaluate_classification_split_loaded(
    config: Any,
    predictor: RXNGraphormerPredictor,
    settings: ClassificationEvaluationSettings,
) -> ClassificationSplitResult:
    config_path = settings.config_path or resolve_config_file(settings.model_path)
    dataloader, files = build_classification_dataloader(
        config,
        split=settings.split,
        specific_val=settings.specific_val,
        batch_size=settings.batch_size,
    )
    evaluation = evaluate_classification(
        predictor.model,
        dataloader,
        predictor.device,
        max_batches=settings.max_batches,
    )
    checkpoint_path = resolve_classification_checkpoint(
        settings.model_path,
        ckpt_file=settings.ckpt_file,
        checkpoint_path=settings.checkpoint_path,
    )
    return ClassificationSplitResult(
        split=split_name(settings.split),
        evaluation=evaluation,
        files=files,
        model_path=os.fspath(settings.model_path),
        config_path=os.fspath(config_path),
        checkpoint_path=os.fspath(checkpoint_path),
    )


def evaluate_classification_split(settings: ClassificationEvaluationSettings) -> ClassificationSplitResult:
    config_path = settings.config_path or resolve_config_file(settings.model_path)
    config = load_train_config(config_path)
    predictor = load_classification_predictor(settings)
    return _evaluate_classification_split_loaded(config, predictor, settings)


def evaluate_classification_splits(
    settings: ClassificationEvaluationSettings,
    splits: Iterable[str] = ("train", "valid", "test"),
) -> dict[str, ClassificationSplitResult]:
    config_path = settings.config_path or resolve_config_file(settings.model_path)
    config = load_train_config(config_path)
    predictor = load_classification_predictor(settings)
    results: dict[str, ClassificationSplitResult] = {}
    for split in splits:
        split_key = split_name(split)
        results[split_key] = _evaluate_classification_split_loaded(
            config,
            predictor,
            ClassificationEvaluationSettings(
                model_path=settings.model_path,
                ckpt_file=settings.ckpt_file,
                config_path=config_path,
                checkpoint_path=settings.checkpoint_path,
                split=split_key,
                specific_val=settings.specific_val,
                batch_size=settings.batch_size,
                max_batches=settings.max_batches,
                device=settings.device,
            ),
        )
    return results


def write_classification_json(
    results: ClassificationSplitResult | dict[str, ClassificationSplitResult],
    path: str | os.PathLike,
) -> None:
    if isinstance(results, ClassificationSplitResult):
        payload: Any = results.summary()
    else:
        payload = {split: result.summary() for split, result in results.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_classification_csv(
    results: ClassificationSplitResult | dict[str, ClassificationSplitResult],
    path: str | os.PathLike,
) -> None:
    items = [results] if isinstance(results, ClassificationSplitResult) else list(results.values())
    fields = [
        "split",
        "count",
        "accuracy",
        "loss",
        "mean_confidence",
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
