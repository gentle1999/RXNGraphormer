from __future__ import annotations

import argparse
from typing import Any

from rxngraphormer.cli.common import add_lightning_train_args, as_train_config, lightning_settings_from_args
from rxngraphormer.config import load_config


def _run_legacy_train(config: Any, *, local_rank: int = -1) -> None:
    if config.task == "regression":
        from rxngraphormer.training.legacy import SPLITRegressorTrainer

        SPLITRegressorTrainer(config).run()
    elif config.task == "sequence_generation":
        from rxngraphormer.training.legacy import SequenceTrainer

        config.others.local_rank = local_rank
        SequenceTrainer(config).run()
    elif config.task == "classification":
        from rxngraphormer.training.legacy import SPLITClassifierTrainer

        config.others.local_rank = local_rank
        SPLITClassifierTrainer(config).run()
    else:
        raise NotImplementedError


def train_main() -> None:
    from rxngraphormer.lightning import fit_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config_json", dest="config_path", type=str, default="./config/C_H_func_parameters.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--legacy_regression", action="store_true", help="Run the old regression trainer instead of the Lightning workflow.")
    parser.add_argument("--legacy_classification", action="store_true", help="Run the old classification trainer instead of the Lightning workflow.")
    add_lightning_train_args(parser)
    args = parser.parse_args()

    config = as_train_config(load_config(args.config_path))

    if config.task == "regression" and not args.legacy_regression:
        fit_config(config, lightning_settings_from_args(args))
    elif config.task == "classification" and not args.legacy_classification:
        fit_config(config, lightning_settings_from_args(args))
    else:
        _run_legacy_train(config, local_rank=args.local_rank)


def train_legacy_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config_json", dest="config_path", type=str, default="./config/C_H_func_parameters.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = as_train_config(load_config(args.config_path))
    _run_legacy_train(config, local_rank=args.local_rank)


def train_lit_main() -> None:
    from rxngraphormer.lightning import fit_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config_json", dest="config_path", type=str, default="./config/C_H_func_parameters.json")
    add_lightning_train_args(parser)
    args = parser.parse_args()

    config = as_train_config(load_config(args.config_path))
    fit_config(config, lightning_settings_from_args(args))
