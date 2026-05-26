from __future__ import annotations

import argparse

from rxngraphormer.cli.common import as_eval_config
from rxngraphormer.config import load_config


def eval_main() -> None:
    import torch

    from rxngraphormer.evaluation.classification_workflow import (
        ClassificationEvaluationSettings,
        evaluate_classification_split,
        write_classification_csv,
        write_classification_json,
    )
    from rxngraphormer.evaluation.legacy_eval import SeqEval
    from rxngraphormer.evaluation.regression_workflow import (
        RegressionEvaluationSettings,
        evaluate_regression_split,
        write_regression_csv,
        write_regression_json,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config_json", dest="config_path", type=str, default="./config/uspto_50k_eval.json")
    parser.add_argument("--split", choices=["train", "valid", "test"], default=None)
    parser.add_argument("--specific_val", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()
    config = as_eval_config(load_config(args.config_path))

    if config.task == "sequence_generation":
        seq_eval = SeqEval(
            trained_model_path=config.trained_model_path,
            ckpt_file=config.ckpt_file,
            topk=config.topk,
            beam_size=config.beam_size,
            temperature=config.temperature,
            n_best=config.n_best,
            min_length=config.min_length,
            max_length=config.max_length,
            batch_size=config.batch_size,
        )
        seq_eval.eval(seq_eval.test_dataloader, seq_eval.test_ground_truth_smiles_lst)
    elif config.task == "regression":
        regression_result = evaluate_regression_split(
            RegressionEvaluationSettings(
                model_path=config.trained_model_path,
                ckpt_file=config.ckpt_file,
                split=args.split or getattr(config, "split", "test"),
                specific_val=args.specific_val or bool(getattr(config, "specific_val", False)),
                batch_size=args.batch_size,
                scale=config.scale,
                yield_constrain=config.yield_constrain,
                max_batches=args.max_batches if args.max_batches is not None else getattr(config, "max_batches", None),
            )
        )
        if args.output_json is not None:
            write_regression_json(regression_result, args.output_json)
        if args.output_csv is not None:
            write_regression_csv(regression_result, args.output_csv)
        regression_metrics = regression_result.evaluation.metrics
        print(f"R2: {regression_metrics.r2:.4f}, MAE: {regression_metrics.mae:.4f}")
    elif config.task == "classification":
        classification_result = evaluate_classification_split(
            ClassificationEvaluationSettings(
                model_path=config.trained_model_path,
                ckpt_file=config.ckpt_file,
                split=args.split or getattr(config, "split", "valid"),
                specific_val=args.specific_val or bool(getattr(config, "specific_val", False)),
                batch_size=args.batch_size,
                max_batches=args.max_batches if args.max_batches is not None else getattr(config, "max_batches", None),
            )
        )
        if args.output_json is not None:
            write_classification_json(classification_result, args.output_json)
        if args.output_csv is not None:
            write_classification_csv(classification_result, args.output_csv)
        classification_metrics = classification_result.evaluation.metrics
        print(f"Accuracy: {classification_metrics.accuracy:.4f}, loss: {classification_metrics.loss:.4f}")

    torch.cuda.empty_cache()


def eval_legacy_main() -> None:
    import torch

    from rxngraphormer.evaluation.legacy_eval import SeqEval, eval_regression_performance

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config_json", dest="config_path", type=str, default="./config/uspto_50k_eval.json")
    args = parser.parse_args()
    config = as_eval_config(load_config(args.config_path))

    if config.task == "sequence_generation":
        seq_eval = SeqEval(
            trained_model_path=config.trained_model_path,
            ckpt_file=config.ckpt_file,
            topk=config.topk,
            beam_size=config.beam_size,
            temperature=config.temperature,
            n_best=config.n_best,
            min_length=config.min_length,
            max_length=config.max_length,
            batch_size=config.batch_size,
        )
        seq_eval.eval(seq_eval.test_dataloader, seq_eval.test_ground_truth_smiles_lst)
    elif config.task == "regression":
        result = eval_regression_performance(
            config.trained_model_path,
            ckpt_file=config.ckpt_file,
            scale=config.scale,
            yield_constrain=config.yield_constrain,
            max_batches=getattr(config, "max_batches", None),
        )
        r2, mae = result[0], result[1]
        print(f"R2: {r2:.4f}, MAE: {mae:.4f}")

    torch.cuda.empty_cache()
