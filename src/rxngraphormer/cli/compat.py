from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

from rxngraphormer import compatibility
from rxngraphormer.compatibility.classification import (
    ClassificationCheckpointCompatibilityReport,
    ClassificationCheckpointCompatibilitySettings,
)
from rxngraphormer.compatibility.regression import (
    RegressionPretrainCompatibilityReport,
    RegressionPretrainCompatibilitySettings,
)
from rxngraphormer.serialization import DEFAULT_CHECKPOINT_FILE


def compat_main() -> None:
    run_classification_check = cast(
        Callable[[ClassificationCheckpointCompatibilitySettings], ClassificationCheckpointCompatibilityReport],
        compatibility.check_classification_checkpoint_compatibility,
    )
    run_regression_check = cast(
        Callable[[RegressionPretrainCompatibilitySettings], RegressionPretrainCompatibilityReport],
        compatibility.check_regression_pretrain_compatibility,
    )

    parser = argparse.ArgumentParser(description="Run RXNGraphormer compatibility smoke checks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cls_parser = subparsers.add_parser("classification-checkpoint")
    cls_parser.add_argument("--model_path", required=True)
    cls_parser.add_argument("--ckpt_file", default=DEFAULT_CHECKPOINT_FILE)
    cls_parser.add_argument("--config_path", default=None)
    cls_parser.add_argument("--checkpoint_path", default=None)
    cls_parser.add_argument("--device", default=None)
    cls_parser.add_argument("--output_json", default=None)
    reg_parser = subparsers.add_parser("regression-pretrain")
    reg_parser.add_argument("--config", "--config_json", dest="config_path", required=True)
    reg_parser.add_argument("--device", default=None)
    reg_parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    if args.command == "classification-checkpoint":
        classification_report = run_classification_check(
            ClassificationCheckpointCompatibilitySettings(
                model_path=args.model_path,
                ckpt_file=args.ckpt_file,
                config_path=args.config_path,
                checkpoint_path=args.checkpoint_path,
                device=args.device,
            )
        )
        payload = classification_report.summary()
        text = json.dumps(payload, indent=2)
        print(text)
        if args.output_json is not None:
            path = Path(args.output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")
        if not classification_report.ok:
            raise SystemExit(1)
    elif args.command == "regression-pretrain":
        regression_report = run_regression_check(
            RegressionPretrainCompatibilitySettings(
                config_path=args.config_path,
                device=args.device,
            )
        )
        payload = regression_report.summary()
        text = json.dumps(payload, indent=2)
        print(text)
        if args.output_json is not None:
            path = Path(args.output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n")
        if not regression_report.ok:
            raise SystemExit(1)
