from __future__ import annotations

import os
import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.nn.init import xavier_uniform_
from tqdm import tqdm

from .checkpointing import CheckpointAdapter
from rxngraphormer.config import load_config, resolve_config_file
from .data import MultiRXNDataset, PairDataset, TripleDataset, pair_collate_fn, triple_collate_fn
from .evaluator import regression_batch_input
from .model import masked_sequence_mean
from .model_factory import build_classification_model, build_regression_model
from .reaction import canonicalize_reaction_side, split_reaction_smiles
from .utils import as_bool, canonical_smiles


@dataclass
class ClassificationPrediction:
    preds: torch.Tensor
    confidence: torch.Tensor
    probabilities: torch.Tensor | None = None
    uncertainty: torch.Tensor | None = None


@dataclass
class RegressionPrediction:
    preds: torch.Tensor
    targets: torch.Tensor | None = None
    uncertainty: torch.Tensor | None = None


@dataclass
class EmbeddingPrediction:
    embeddings: torch.Tensor


def export_predictions_csv(result: ClassificationPrediction | RegressionPrediction, path: str | os.PathLike) -> None:
    rows: list[dict[str, float | int]] = []
    if isinstance(result, ClassificationPrediction):
        probabilities = result.probabilities
        for idx in range(result.preds.shape[0]):
            row: dict[str, float | int] = {
                "prediction": int(result.preds[idx].item()),
                "confidence": float(result.confidence[idx].item()),
            }
            if probabilities is not None:
                for class_idx, probability in enumerate(probabilities[idx].tolist()):
                    row[f"prob_{class_idx}"] = float(probability)
            if result.uncertainty is not None:
                row["uncertainty"] = float(result.uncertainty[idx].item())
            rows.append(row)
    elif isinstance(result, RegressionPrediction):
        preds = result.preds.reshape(result.preds.shape[0], -1)
        targets = result.targets.reshape(result.targets.shape[0], -1) if result.targets is not None else None
        uncertainty = result.uncertainty.reshape(result.uncertainty.shape[0], -1) if result.uncertainty is not None else None
        for idx in range(preds.shape[0]):
            row = {"prediction": float(preds[idx, 0].item())}
            if preds.shape[1] > 1:
                for target_idx in range(preds.shape[1]):
                    row[f"prediction_{target_idx}"] = float(preds[idx, target_idx].item())
            if targets is not None:
                row["target"] = float(targets[idx, 0].item())
                if targets.shape[1] > 1:
                    for target_idx in range(targets.shape[1]):
                        row[f"target_{target_idx}"] = float(targets[idx, target_idx].item())
            if uncertainty is not None:
                row["uncertainty"] = float(uncertainty[idx, 0].item())
                if uncertainty.shape[1] > 1:
                    for target_idx in range(uncertainty.shape[1]):
                        row[f"uncertainty_{target_idx}"] = float(uncertainty[idx, target_idx].item())
            rows.append(row)
    else:
        raise TypeError("Unsupported prediction result type")

    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_embeddings_csv(result: EmbeddingPrediction | torch.Tensor, path: str | os.PathLike) -> None:
    embeddings = result.embeddings if isinstance(result, EmbeddingPrediction) else result
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        fieldnames = [f"embedding_{idx}" for idx in range(embeddings.shape[1])]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in embeddings.tolist():
            writer.writerow({field: float(value) for field, value in zip(fieldnames, row)})


class RXNGraphormerPredictor:
    def __init__(
        self,
        pretrained_model_path: str,
        *,
        task: str = "classification",
        ckpt_file: str = "valid_checkpoint.pt",
        config_path: str | os.PathLike | None = None,
        checkpoint_path: str | os.PathLike | None = None,
        device: torch.device | str | None = None,
        random_init: bool = False,
    ):
        if task not in {"classification", "regression"}:
            raise NotImplementedError("RXNGraphormerPredictor supports task='classification' or task='regression'")
        self.task = task
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = load_config(config_path or resolve_config_file(pretrained_model_path))
        if task == "classification":
            model = build_classification_model(config)
        else:
            model = build_regression_model(config)
        if not random_init:
            ckpt_path = checkpoint_path or os.path.join(pretrained_model_path, "model", ckpt_file)
            CheckpointAdapter().load_into_model(model, ckpt_path, map_location=self.device, mode="strict")
        else:
            for p in model.parameters():
                if p.dim() > 1 and p.requires_grad:
                    xavier_uniform_(p)
        model.to(self.device)
        model.eval()
        self.config = config
        self.model = model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | os.PathLike,
        *,
        config_path: str | os.PathLike,
        task: str = "regression",
        device: torch.device | str | None = None,
    ) -> "RXNGraphormerPredictor":
        return cls(
            os.fspath(Path(config_path).parent),
            task=task,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    def predict_from_dataset(
        self,
        root: str,
        *,
        rct_name_regrex: str,
        pdt_name_regrex: str,
        batch_size: int = 128,
        return_probabilities: bool = False,
        return_uncertainty: bool = False,
    ) -> ClassificationPrediction:
        if self.task != "classification":
            raise ValueError("predict_from_dataset requires task='classification'")
        rct_dataset = MultiRXNDataset(root=root, name_regrex=rct_name_regrex)
        pdt_dataset = MultiRXNDataset(root=root, name_regrex=pdt_name_regrex)
        pair_dataset = PairDataset(rct_dataset, pdt_dataset)
        pair_dataloader = torch.utils.data.DataLoader(
            pair_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pair_collate_fn,
        )

        pred_chunks: list[torch.Tensor] = []
        confidence_chunks: list[torch.Tensor] = []
        probability_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for rct_data, pdt_data in tqdm(pair_dataloader):
                probabilities = self.model([rct_data.to(self.device), pdt_data.to(self.device)])
                pred_chunks.append(probabilities.argmax(dim=1).detach().cpu())
                confidence_chunks.append(probabilities.max(dim=1).values.detach().cpu())
                if return_probabilities:
                    probability_chunks.append(probabilities.detach().cpu())

        if not pred_chunks:
            raise ValueError("Cannot predict an empty dataset")
        return ClassificationPrediction(
            preds=torch.cat(pred_chunks, dim=0),
            confidence=torch.cat(confidence_chunks, dim=0),
            probabilities=torch.cat(probability_chunks, dim=0) if return_probabilities else None,
            uncertainty=(1.0 - torch.cat(confidence_chunks, dim=0)) if return_uncertainty else None,
        )

    def embed_from_dataset(
        self,
        root: str,
        *,
        rct_name_regrex: str,
        pdt_name_regrex: str,
        batch_size: int = 128,
    ) -> EmbeddingPrediction:
        rct_dataset = MultiRXNDataset(root=root, name_regrex=rct_name_regrex, task=self.task)
        pdt_dataset = MultiRXNDataset(root=root, name_regrex=pdt_name_regrex, task=self.task)
        pair_dataset = PairDataset(rct_dataset, pdt_dataset)
        pair_dataloader = torch.utils.data.DataLoader(
            pair_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pair_collate_fn,
        )

        emb_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for rct_data, pdt_data in tqdm(pair_dataloader):
                emb = _reaction_pair_embeddings(
                    self.model,
                    rct_data.to(self.device),
                    pdt_data.to(self.device),
                )
                emb_chunks.append(emb.detach().cpu())

        if not emb_chunks:
            raise ValueError("Cannot embed an empty dataset")
        return EmbeddingPrediction(embeddings=torch.cat(emb_chunks, dim=0))

    def predict_reactions(
        self,
        rxn_smiles: Iterable[str],
        *,
        batch_size: int = 128,
        return_probabilities: bool = False,
        return_uncertainty: bool = False,
    ) -> ClassificationPrediction:
        rxn_smiles = list(rxn_smiles)
        if len(rxn_smiles) < 2:
            raise AssertionError("rxn_smiles_lst must contain at least 2 reactions")

        rct_smi_lst = []
        pdt_smi_lst = []
        for smi in rxn_smiles:
            reactant, product = split_reaction_smiles(smi)
            rct_smi_lst.append(f"{canonicalize_reaction_side(reactant)},0")
            pdt_smi_lst.append(f"{canonicalize_reaction_side(product)},0")
        with tempfile.TemporaryDirectory(prefix="rxngraphormer_predict_") as tmp_dir:
            rct_name = "rct_smiles_0.csv"
            pdt_name = "pdt_smiles_0.csv"
            with open(os.path.join(tmp_dir, rct_name), "w") as fw:
                fw.writelines("\n".join(rct_smi_lst))
            with open(os.path.join(tmp_dir, pdt_name), "w") as fw:
                fw.writelines("\n".join(pdt_smi_lst))
            return self.predict_from_dataset(
                tmp_dir,
                rct_name_regrex=rct_name,
                pdt_name_regrex=pdt_name,
                batch_size=batch_size,
                return_probabilities=return_probabilities,
                return_uncertainty=return_uncertainty,
            )

    def embed_reactions(
        self,
        rxn_smiles: Iterable[str],
        *,
        batch_size: int = 128,
    ) -> EmbeddingPrediction:
        rxn_smiles = list(rxn_smiles)
        if len(rxn_smiles) < 2:
            raise AssertionError("rxn_smiles_lst must contain at least 2 reactions")

        rct_smi_lst = []
        pdt_smi_lst = []
        for smi in rxn_smiles:
            reactant, product = split_reaction_smiles(smi)
            rct_smi_lst.append(f"{canonicalize_reaction_side(reactant)},0")
            pdt_smi_lst.append(f"{canonicalize_reaction_side(product)},0")
        with tempfile.TemporaryDirectory(prefix="rxngraphormer_embed_") as tmp_dir:
            rct_name = "rct_smiles_0.csv"
            pdt_name = "pdt_smiles_0.csv"
            _write_lines(Path(tmp_dir, rct_name), rct_smi_lst)
            _write_lines(Path(tmp_dir, pdt_name), pdt_smi_lst)
            return self.embed_from_dataset(
                tmp_dir,
                rct_name_regrex=rct_name,
                pdt_name_regrex=pdt_name,
                batch_size=batch_size,
            )

    def predict_table(
        self,
        path: str | os.PathLike,
        *,
        rxn_smiles_column: str = "rxn_smiles",
        rct_smiles_column: str = "rct_smiles",
        pdt_smiles_column: str = "pdt_smiles",
        mid_smiles_column: str = "mid_smiles",
        target_column: str | None = None,
        batch_size: int = 128,
        return_probabilities: bool = False,
        return_targets: bool = False,
        return_uncertainty: bool = False,
        use_mid_inf: bool | None = None,
    ) -> ClassificationPrediction | RegressionPrediction:
        rows = _read_prediction_table(path)
        if not rows:
            raise ValueError("Cannot predict an empty input table")

        if self.task == "classification":
            rxn_smiles = _reaction_smiles_from_rows(
                rows,
                rxn_smiles_column=rxn_smiles_column,
                rct_smiles_column=rct_smiles_column,
                pdt_smiles_column=pdt_smiles_column,
            )
            return self.predict_reactions(
                rxn_smiles,
                batch_size=batch_size,
                return_probabilities=return_probabilities,
                return_uncertainty=return_uncertainty,
            )

        if return_targets and target_column is None:
            raise ValueError("target_column is required when return_targets=True for table prediction")
        with tempfile.TemporaryDirectory(prefix="rxngraphormer_predict_table_") as tmp_dir:
            use_mid = as_bool(self.config.model.use_mid_inf) if use_mid_inf is None else use_mid_inf
            rct_lines, pdt_lines, mid_lines = _regression_table_lines(
                rows,
                rxn_smiles_column=rxn_smiles_column,
                rct_smiles_column=rct_smiles_column,
                pdt_smiles_column=pdt_smiles_column,
                mid_smiles_column=mid_smiles_column,
                target_column=target_column,
                require_mid=use_mid,
            )
            rct_name = "rct_smiles_0.csv"
            pdt_name = "pdt_smiles_0.csv"
            mid_name = "mid_smiles_0.csv"
            _write_lines(Path(tmp_dir, rct_name), rct_lines)
            _write_lines(Path(tmp_dir, pdt_name), pdt_lines)
            if use_mid:
                _write_lines(Path(tmp_dir, mid_name), mid_lines)
            return self.predict_regression_from_dataset(
                tmp_dir,
                rct_name_regrex=rct_name,
                pdt_name_regrex=pdt_name,
                mid_name_regrex=mid_name if use_mid else None,
                batch_size=batch_size,
                use_mid_inf=use_mid,
                return_targets=return_targets,
                return_uncertainty=return_uncertainty,
            )

    def embed_table(
        self,
        path: str | os.PathLike,
        *,
        rxn_smiles_column: str = "rxn_smiles",
        rct_smiles_column: str = "rct_smiles",
        pdt_smiles_column: str = "pdt_smiles",
        batch_size: int = 128,
    ) -> EmbeddingPrediction:
        rows = _read_prediction_table(path)
        if not rows:
            raise ValueError("Cannot embed an empty input table")
        rxn_smiles = _reaction_smiles_from_rows(
            rows,
            rxn_smiles_column=rxn_smiles_column,
            rct_smiles_column=rct_smiles_column,
            pdt_smiles_column=pdt_smiles_column,
        )
        return self.embed_reactions(rxn_smiles, batch_size=batch_size)

    def predict_regression_from_dataset(
        self,
        root: str,
        *,
        rct_name_regrex: str,
        pdt_name_regrex: str,
        mid_name_regrex: str | None = None,
        batch_size: int = 128,
        use_mid_inf: bool | None = None,
        return_targets: bool = False,
        return_uncertainty: bool = False,
    ) -> RegressionPrediction:
        if self.task != "regression":
            raise ValueError("predict_regression_from_dataset requires task='regression'")
        if use_mid_inf is None:
            use_mid_inf = as_bool(self.config.model.use_mid_inf)

        rct_dataset = MultiRXNDataset(root=root, name_regrex=rct_name_regrex, task="regression")
        pdt_dataset = MultiRXNDataset(root=root, name_regrex=pdt_name_regrex, task="regression")
        if use_mid_inf:
            if not mid_name_regrex:
                raise ValueError("mid_name_regrex is required when use_mid_inf=True")
            mid_dataset = MultiRXNDataset(root=root, name_regrex=mid_name_regrex, task="regression")
            dataset = TripleDataset(rct_dataset, pdt_dataset, mid_dataset)
            collate_fn = triple_collate_fn
        else:
            dataset = PairDataset(rct_dataset, pdt_dataset)
            collate_fn = pair_collate_fn

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        pred_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for batch_data in tqdm(dataloader):
                model_input, target = regression_batch_input(batch_data, self.device)
                preds = self.model(model_input)
                pred_chunks.append(preds.detach().cpu())
                if return_targets:
                    target_chunks.append(target.detach().cpu())

        if not pred_chunks:
            raise ValueError("Cannot predict an empty dataset")
        preds = torch.cat(pred_chunks, dim=0)
        return RegressionPrediction(
            preds=preds,
            targets=torch.cat(target_chunks, dim=0) if return_targets else None,
            uncertainty=torch.full_like(preds, float("nan")) if return_uncertainty else None,
        )


def _write_lines(path: Path, lines: list[str]) -> None:
    with open(path, "w") as handle:
        handle.writelines("\n".join(lines))


def _read_prediction_table(path: str | os.PathLike) -> list[dict[str, object]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pandas is required to read Parquet prediction inputs") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError("Prediction table must be .csv, .parquet, or .pq")


def _required_value(row: dict[str, object], column: str) -> str:
    if column not in row:
        raise ValueError(f"Input table is missing required column: {column}")
    value = row[column]
    if value is None or str(value) == "":
        raise ValueError(f"Input table contains an empty value in required column: {column}")
    return str(value)


def _optional_value(row: dict[str, object], column: str | None, default: str = "0") -> str:
    if column is None or column not in row or row[column] is None or str(row[column]) == "":
        return default
    return str(row[column])


def _reaction_smiles_from_rows(
    rows: list[dict[str, object]],
    *,
    rxn_smiles_column: str,
    rct_smiles_column: str,
    pdt_smiles_column: str,
) -> list[str]:
    if any(rxn_smiles_column in row and str(row[rxn_smiles_column]) != "" for row in rows):
        return [
            _required_value(row, rxn_smiles_column)
            if rxn_smiles_column in row and str(row[rxn_smiles_column]) != ""
            else f"{_required_value(row, rct_smiles_column)}>>{_required_value(row, pdt_smiles_column)}"
            for row in rows
        ]
    return [
        f"{_required_value(row, rct_smiles_column)}>>{_required_value(row, pdt_smiles_column)}"
        for row in rows
    ]


def _regression_table_lines(
    rows: list[dict[str, object]],
    *,
    rxn_smiles_column: str,
    rct_smiles_column: str,
    pdt_smiles_column: str,
    mid_smiles_column: str,
    target_column: str | None,
    require_mid: bool,
) -> tuple[list[str], list[str], list[str]]:
    rct_lines: list[str] = []
    pdt_lines: list[str] = []
    mid_lines: list[str] = []
    for row in rows:
        if rxn_smiles_column in row and str(row[rxn_smiles_column]) != "":
            reactant, product = split_reaction_smiles(_required_value(row, rxn_smiles_column))
        else:
            reactant = _required_value(row, rct_smiles_column)
            product = _required_value(row, pdt_smiles_column)
        target = _optional_value(row, target_column)
        rct_lines.append(f"{canonicalize_reaction_side(reactant)},{target}")
        pdt_lines.append(f"{canonicalize_reaction_side(product)},{target}")
        if require_mid:
            mid_lines.append(f"{canonical_smiles(_required_value(row, mid_smiles_column))},{target}")
    return rct_lines, pdt_lines, mid_lines


def _reaction_pair_embeddings(model: torch.nn.Module, rct_data, pdt_data) -> torch.Tensor:
    rct_padded_memory_bank, _rct_batch, rct_memory_lengths = model.rct_encoder(rct_data)
    pdt_padded_memory_bank, _pdt_batch, pdt_memory_lengths = model.pdt_encoder(pdt_data)
    rct_rxn_transf_emb = rct_padded_memory_bank.transpose(0, 1)
    pdt_rxn_transf_emb = pdt_padded_memory_bank.transpose(0, 1)
    if model.trans_readout != "mean":
        raise NotImplementedError(f"Unsupported trans_readout: {model.trans_readout}")
    rct_rxn_transf_emb_merg = masked_sequence_mean(rct_rxn_transf_emb, rct_memory_lengths)
    pdt_rxn_transf_emb_merg = masked_sequence_mean(pdt_rxn_transf_emb, pdt_memory_lengths)
    diff_emb = torch.abs(rct_rxn_transf_emb_merg - pdt_rxn_transf_emb_merg)
    if model.split_merge_method == "all":
        rxn_emb = torch.cat([rct_rxn_transf_emb_merg, pdt_rxn_transf_emb_merg, diff_emb], dim=-1)
    elif model.split_merge_method == "only_diff":
        rxn_emb = diff_emb
    elif model.split_merge_method == "rct_pdt":
        rxn_emb = torch.cat([rct_rxn_transf_emb_merg, pdt_rxn_transf_emb_merg], dim=-1)
    else:
        raise ValueError(f"Unknown split_merge_method: {model.split_merge_method}")
    hidden_layers = list(model.decoder.layers[:-1])
    if getattr(model.decoder, "batch_norm", False):
        norm_layers = list(model.decoder.batch_norms[:-1])
    else:
        norm_layers = [None] * len(hidden_layers)
    for lin_layer, norm_layer in zip(hidden_layers, norm_layers):
        rxn_emb = lin_layer(rxn_emb)
        if norm_layer is not None:
            rxn_emb = norm_layer(rxn_emb)
    return rxn_emb
