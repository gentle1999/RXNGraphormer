from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from .regression_workflow import (
    RegressionEvaluationSettings,
    RegressionSplitResult,
    evaluate_regression_splits,
    write_regression_csv,
    write_regression_json,
)


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {
            key: jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def git_snapshot(cwd: str | os.PathLike = ".") -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def trainer_output_dir(trainer: Any, fallback_root: str | os.PathLike) -> Path:
    logger = getattr(trainer, "logger", None)
    log_dir = getattr(logger, "log_dir", None)
    if isinstance(log_dir, (str, os.PathLike)):
        return Path(log_dir)
    return Path(fallback_root)


def best_checkpoint_path(trainer: Any) -> str | None:
    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    path = getattr(checkpoint_callback, "best_model_path", None)
    if path:
        return str(path)
    callbacks = getattr(trainer, "callbacks", []) or []
    for callback in callbacks:
        path = getattr(callback, "best_model_path", None)
        if path:
            return str(path)
    return None


def trainer_metrics(trainer: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in getattr(trainer, "callback_metrics", {}).items():
        if isinstance(value, torch.Tensor):
            metrics[str(key)] = float(value.detach().cpu()) if value.numel() == 1 else value.detach().cpu().tolist()
        else:
            metrics[str(key)] = value
    return metrics


def write_json(path: str | os.PathLike, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n")


def build_fit_manifest(
    *,
    artifacts: Any,
    settings: Any,
    started_at: float,
    ended_at: float,
    config_path: str | os.PathLike | None = None,
    eval_reports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trainer = artifacts.trainer
    best_ckpt = best_checkpoint_path(trainer)
    return {
        "schema_version": 1,
        "kind": "rxngraphormer.lightning_fit",
        "started_at": int(started_at),
        "ended_at": int(ended_at),
        "elapsed_sec": round(ended_at - started_at, 3),
        "root_dir": artifacts.root_dir,
        "output_dir": str(trainer_output_dir(trainer, artifacts.root_dir)),
        "config_path": str(config_path) if config_path is not None else None,
        "config": jsonable(artifacts.config),
        "settings": jsonable(settings),
        "checkpoint": {
            "best_model_path": best_ckpt,
            "last_model_path": getattr(getattr(trainer, "checkpoint_callback", None), "last_model_path", None),
        },
        "metrics": trainer_metrics(trainer),
        "init_checkpoint_report": jsonable(artifacts.init_checkpoint_report),
        "eval_reports": eval_reports or {},
        "environment": environment_snapshot(),
        "git": git_snapshot(),
    }


def write_fit_manifest(
    manifest: dict[str, Any],
    *,
    output_dir: str | os.PathLike,
    root_dir: str | os.PathLike,
) -> dict[str, str]:
    output_path = Path(output_dir) / "manifest.json"
    root_path = Path(root_dir) / "manifest.json"
    write_json(output_path, manifest)
    if output_path.resolve() != root_path.resolve():
        write_json(root_path, manifest)
    return {"output_manifest": str(output_path), "root_manifest": str(root_path)}


def run_post_fit_regression_eval(
    *,
    artifacts: Any,
    config_path: str | os.PathLike | None,
    checkpoint_path: str | os.PathLike,
    splits: tuple[str, ...],
    batch_size: int | None,
    scale: float,
    yield_constrain: bool,
    max_batches: int | None,
    specific_val: bool,
    output_dir: str | os.PathLike,
) -> dict[str, Any]:
    report_dir = Path(output_dir) / "reports"
    results = evaluate_regression_splits(
        RegressionEvaluationSettings(
            model_path=artifacts.root_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            specific_val=specific_val,
            batch_size=batch_size,
            scale=scale,
            yield_constrain=yield_constrain,
            max_batches=max_batches,
        ),
        splits=splits,
    )
    json_path = report_dir / "eval.json"
    csv_path = report_dir / "eval.csv"
    write_regression_json(results, json_path)
    write_regression_csv(results, csv_path)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "checkpoint_path": str(checkpoint_path),
        "splits": {split: result.summary() for split, result in results.items()},
    }
