from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from .compat import compat_main
from .eval import eval_legacy_main, eval_main
from .predict import predict_main, predict_sequence_main
from .preprocess import preprocess_main
from .train import _run_legacy_train, train_legacy_main, train_lit_main, train_main

__all__ = [
    "_run_legacy_train",
    "compat_main",
    "eval_legacy_main",
    "eval_main",
    "main",
    "predict_main",
    "predict_sequence_main",
    "preprocess_main",
    "train_legacy_main",
    "train_lit_main",
    "train_main",
]


def main() -> None:
    commands: dict[str, Callable[[], None]] = {
        "train": train_main,
        "train-legacy": train_legacy_main,
        "train-lit": train_lit_main,
        "eval": eval_main,
        "eval-legacy": eval_legacy_main,
        "compat": compat_main,
        "predict": predict_main,
        "predict-sequence": predict_sequence_main,
        "preprocess": preprocess_main,
    }
    parser = argparse.ArgumentParser(description="RXNGraphormer command line interface.")
    parser.add_argument("command", choices=sorted(commands))
    if len(sys.argv) <= 1 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    command = sys.argv[1]
    if command not in commands:
        parser.parse_args(sys.argv[1:2])
    sys.argv = [f"rxngraphormer {command}", *sys.argv[2:]]
    commands[command]()


if __name__ == "__main__":
    main()
