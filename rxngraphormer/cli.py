from __future__ import annotations

import argparse

from rxngraphormer.config import load_config


def _add_lightning_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--default_root_dir", type=str, default=None)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--split_manifest", type=str, default=None)
    parser.add_argument("--init_ckpt", type=str, default=None, help="Legacy/canonical checkpoint to load into LightningModule.model before training.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Lightning checkpoint used for Trainer resume.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--compile_mode", choices=["default", "reduce-overhead", "max-autotune"], default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--no_manifest", action="store_true", help="Disable writing experiment manifest.json files.")
    parser.add_argument("--eval_after_fit", action="store_true", help="Evaluate the best checkpoint after training.")
    parser.add_argument("--eval_splits", nargs="+", choices=["train", "valid", "test"], default=["test"])
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--eval_scale", type=float, default=1.0)
    parser.add_argument("--eval_yield_constrain", action="store_true")
    parser.add_argument("--eval_max_batches", type=int, default=None)
    parser.add_argument("--eval_specific_val", action="store_true")


def _lightning_settings_from_args(args):
    from .lightning import LightningFitSettings

    return LightningFitSettings(
        default_root_dir=args.default_root_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        split_manifest=args.split_manifest,
        init_ckpt=args.init_ckpt,
        resume_from_checkpoint=args.resume_from_checkpoint,
        deterministic=args.deterministic,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        early_stopping_patience=args.early_stopping_patience,
        config_path=args.config_path,
        write_manifest=not args.no_manifest,
        eval_after_fit=args.eval_after_fit,
        eval_splits=tuple(args.eval_splits),
        eval_batch_size=args.eval_batch_size,
        eval_scale=args.eval_scale,
        eval_yield_constrain=args.eval_yield_constrain,
        eval_max_batches=args.eval_max_batches,
        eval_specific_val=args.eval_specific_val,
    )


def _run_legacy_train(config, *, local_rank: int = -1) -> None:
    from .train import SPLITClassifierTrainer, SPLITRegressorTrainer, SequenceTrainer

    if config.task == "regression":
        trainer = SPLITRegressorTrainer(config)
    elif config.task == "sequence_generation":
        config.others.local_rank = local_rank
        trainer = SequenceTrainer(config)
    elif config.task == "classification":
        config.others.local_rank = local_rank
        trainer = SPLITClassifierTrainer(config)
    else:
        raise NotImplementedError
    trainer.run()


def train_main() -> None:
    from .lightning import fit_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config_json", dest="config_path", type=str, default="./config/C_H_func_parameters.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--legacy_regression", action="store_true", help="Run the old regression trainer instead of the Lightning workflow.")
    _add_lightning_train_args(parser)
    args = parser.parse_args()

    config = load_config(args.config_path)

    if config.task == "regression" and not args.legacy_regression:
        fit_config(config, _lightning_settings_from_args(args))
    else:
        _run_legacy_train(config, local_rank=args.local_rank)


def train_legacy_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config_json", dest="config_path", type=str, default="./config/C_H_func_parameters.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config_path)
    _run_legacy_train(config, local_rank=args.local_rank)


def train_lit_main() -> None:
    from .lightning import fit_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config_json", dest="config_path", type=str, default="./config/C_H_func_parameters.json")
    _add_lightning_train_args(parser)
    args = parser.parse_args()

    config = load_config(args.config_path)
    fit_config(config, _lightning_settings_from_args(args))


def eval_main() -> None:
    import torch

    from .eval import SeqEval
    from .regression_workflow import (
        RegressionEvaluationSettings,
        evaluate_regression_split,
        write_regression_csv,
        write_regression_json,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, default="./config/uspto_50k_eval.json")
    parser.add_argument("--split", choices=["train", "valid", "test"], default=None)
    parser.add_argument("--specific_val", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config_json)

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
        result = evaluate_regression_split(
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
            write_regression_json(result, args.output_json)
        if args.output_csv is not None:
            write_regression_csv(result, args.output_csv)
        metrics = result.evaluation.metrics
        print(f"R2: {metrics.r2:.4f}, MAE: {metrics.mae:.4f}")

    torch.cuda.empty_cache()


def eval_legacy_main() -> None:
    import torch

    from .eval import SeqEval, eval_regression_performance

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, default="./config/uspto_50k_eval.json")
    args = parser.parse_args()
    config = load_config(args.config_json)

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
        r2, mae, _preds, _targets = eval_regression_performance(
            config.trained_model_path,
            ckpt_file=config.ckpt_file,
            scale=config.scale,
            yield_constrain=config.yield_constrain,
            max_batches=getattr(config, "max_batches", None),
        )
        print(f"R2: {r2:.4f}, MAE: {mae:.4f}")

    torch.cuda.empty_cache()


def predict_main() -> None:
    from .predictor import RXNGraphormerPredictor, export_embeddings_csv, export_predictions_csv

    parser = argparse.ArgumentParser(description="Run RXNGraphormer prediction without a Trainer.")
    parser.add_argument("--model_path", required=True, help="Directory containing config and model/<ckpt_file>.")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--root", default=None, help="Dataset root containing input CSV files.")
    parser.add_argument("--rct_name_regrex", default=None)
    parser.add_argument("--pdt_name_regrex", default=None)
    parser.add_argument("--mid_name_regrex", default=None)
    parser.add_argument("--input_table", default=None, help="CSV/Parquet table with rxn_smiles or split SMILES columns.")
    parser.add_argument("--rxn_smiles_column", default="rxn_smiles")
    parser.add_argument("--rct_smiles_column", default="rct_smiles")
    parser.add_argument("--pdt_smiles_column", default="pdt_smiles")
    parser.add_argument("--mid_smiles_column", default="mid_smiles")
    parser.add_argument("--target_column", default=None)
    parser.add_argument("--ckpt_file", default="valid_checkpoint.pt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embeddings_output", default=None)
    parser.add_argument("--return_probabilities", action="store_true")
    parser.add_argument("--return_targets", action="store_true")
    parser.add_argument("--return_uncertainty", action="store_true")
    parser.add_argument("--use_mid_inf", choices=["auto", "true", "false"], default="auto")
    args = parser.parse_args()
    use_mid_inf = None
    if args.use_mid_inf != "auto":
        use_mid_inf = args.use_mid_inf == "true"
    if args.input_table is None:
        missing = [
            name
            for name, value in {
                "--root": args.root,
                "--rct_name_regrex": args.rct_name_regrex,
                "--pdt_name_regrex": args.pdt_name_regrex,
            }.items()
            if value is None
        ]
        if missing:
            parser.error("--input_table or dataset arguments are required; missing " + ", ".join(missing))

    predictor = RXNGraphormerPredictor(
        args.model_path,
        task=args.task,
        ckpt_file=args.ckpt_file,
        device=args.device,
    )
    if args.input_table is not None:
        result = predictor.predict_table(
            args.input_table,
            rxn_smiles_column=args.rxn_smiles_column,
            rct_smiles_column=args.rct_smiles_column,
            pdt_smiles_column=args.pdt_smiles_column,
            mid_smiles_column=args.mid_smiles_column,
            target_column=args.target_column,
            batch_size=args.batch_size,
            return_probabilities=args.return_probabilities,
            return_targets=args.return_targets,
            return_uncertainty=args.return_uncertainty,
            use_mid_inf=use_mid_inf,
        )
    elif args.task == "classification":
        result = predictor.predict_from_dataset(
            args.root,
            rct_name_regrex=args.rct_name_regrex,
            pdt_name_regrex=args.pdt_name_regrex,
            batch_size=args.batch_size,
            return_probabilities=args.return_probabilities,
            return_uncertainty=args.return_uncertainty,
        )
    else:
        result = predictor.predict_regression_from_dataset(
            args.root,
            rct_name_regrex=args.rct_name_regrex,
            pdt_name_regrex=args.pdt_name_regrex,
            mid_name_regrex=args.mid_name_regrex,
            batch_size=args.batch_size,
            use_mid_inf=use_mid_inf,
            return_targets=args.return_targets,
            return_uncertainty=args.return_uncertainty,
        )
    export_predictions_csv(result, args.output)
    print(f"[INFO] Predictions written to {args.output}")
    if args.embeddings_output is not None:
        if args.input_table is not None:
            embeddings = predictor.embed_table(
                args.input_table,
                rxn_smiles_column=args.rxn_smiles_column,
                rct_smiles_column=args.rct_smiles_column,
                pdt_smiles_column=args.pdt_smiles_column,
                batch_size=args.batch_size,
            )
        else:
            embeddings = predictor.embed_from_dataset(
                args.root,
                rct_name_regrex=args.rct_name_regrex,
                pdt_name_regrex=args.pdt_name_regrex,
                batch_size=args.batch_size,
            )
        export_embeddings_csv(embeddings, args.embeddings_output)
        print(f"[INFO] Embeddings written to {args.embeddings_output}")


def preprocess_main() -> None:
    from .preprocess.cli import main

    main()
