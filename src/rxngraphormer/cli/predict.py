from __future__ import annotations

import argparse
from pathlib import Path

from rxngraphormer.serialization import DEFAULT_CHECKPOINT_FILE

SEQUENCE_PREDICTION_TASKS = ("forward-synthesis", "retro-synthesis")


def _add_sequence_prediction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input_smiles", nargs="+", default=None, help="One or more source SMILES strings.")
    parser.add_argument("--input_file", default=None, help="Text file with one source SMILES string per line.")
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--n_best", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)


def _sequence_smiles_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    smiles = list(args.input_smiles or [])
    if args.input_file is not None:
        smiles.extend(
            line.strip()
            for line in Path(args.input_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not smiles:
        parser.error("--input_smiles or --input_file is required for sequence prediction")
    return smiles


def _run_sequence_prediction(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from rxngraphormer.evaluation.legacy_eval import reaction_prediction

    smiles = _sequence_smiles_from_args(args, parser)
    prediction = reaction_prediction(
        args.model_path,
        smiles,
        task_type=args.task,
        params={
            "batch_size": args.batch_size,
            "beam_size": args.beam_size,
            "n_best": args.n_best,
            "temperature": args.temperature,
            "min_length": args.min_length,
            "max_length": args.max_length,
        },
        ckpt_file=args.ckpt_file,
        device=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    to_csv = getattr(prediction, "to_csv", None)
    if callable(to_csv):
        to_csv(output)
    else:
        output.write_text(f"{prediction}\n", encoding="utf-8")
    print(f"[INFO] Sequence predictions written to {output}")


def predict_main() -> None:
    from rxngraphormer.inference import RXNGraphormerPredictor, export_embeddings_csv, export_predictions_csv

    parser = argparse.ArgumentParser(description="Run RXNGraphormer prediction without a Trainer.")
    parser.add_argument("--model_path", required=True, help="Directory containing config and model/<ckpt_file>.")
    parser.add_argument("--task", choices=["classification", "regression", *SEQUENCE_PREDICTION_TASKS], default="classification")
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
    parser.add_argument("--ckpt_file", default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embeddings_output", default=None)
    parser.add_argument("--return_probabilities", action="store_true")
    parser.add_argument("--return_targets", action="store_true")
    parser.add_argument("--return_uncertainty", action="store_true")
    parser.add_argument("--use_mid_inf", choices=["auto", "true", "false"], default="auto")
    _add_sequence_prediction_args(parser)
    args = parser.parse_args()
    if args.task in SEQUENCE_PREDICTION_TASKS:
        _run_sequence_prediction(args, parser)
        return

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


def predict_sequence_main() -> None:
    parser = argparse.ArgumentParser(description="Run RXNGraphormer sequence-generation prediction.")
    parser.add_argument("--model_path", required=True, help="Directory containing config and model/<ckpt_file>.")
    parser.add_argument("--task", "--task_type", dest="task", choices=SEQUENCE_PREDICTION_TASKS, required=True)
    parser.add_argument("--ckpt_file", default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    _add_sequence_prediction_args(parser)
    args = parser.parse_args()
    _run_sequence_prediction(args, parser)
