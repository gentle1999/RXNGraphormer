"""Legacy evaluation helpers hosted inside the evaluation boundary."""

import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader

from rxngraphormer.compatibility.checkpointing import CheckpointAdapter
from rxngraphormer.config import load_train_config, resolve_config_file
from rxngraphormer.data.collate import triple_collate_fn
from rxngraphormer.data.pairing import SizedDataset, TripleDataset
from rxngraphormer.data.reaction_dataset import RXNDataset
from rxngraphormer.data.sequence_dataset import RXNG2SDataset
from rxngraphormer.data.tokenization import load_vocab, smi_tokenizer
from rxngraphormer.evaluation.regression_workflow import (
    RegressionEvaluationSettings,
    evaluate_regression_split,
)
from rxngraphormer.models import RXNGRegressor, build_regression_model, build_sequence_model
from rxngraphormer.models.protocols import SequenceBatchProtocol, SequenceModelProtocol
from rxngraphormer.preprocessing.reactions import canonicalize_reaction_side, split_reaction_smiles
from rxngraphormer.runtime import DeviceManager

DEFAULT_DEVICE_MANAGER = DeviceManager.from_value(None)
device = DEFAULT_DEVICE_MANAGER.device


def _sequence_sample(dataset: object, index: int) -> SequenceBatchProtocol:
    sized_dataset = cast(SizedDataset[object], dataset)
    return cast(SequenceBatchProtocol, sized_dataset[index])


def _sequence_batch(batch: object) -> SequenceBatchProtocol:
    return cast(SequenceBatchProtocol, batch)


def _sequence_predictions(results: Mapping[str, object]) -> list[list[torch.Tensor]]:
    predictions = results.get("predictions")
    if predictions is None:
        return []
    return cast(list[list[torch.Tensor]], predictions)


def _loader_dataset(dataset: object) -> Sequence[BaseData]:
    return cast(Sequence[BaseData], dataset)


class SeqEval:
    def __init__(
        self,
        trained_model_path,
        topk=10,
        beam_size=10,
        temperature=1.0,
        n_best=10,
        min_length=1,
        max_length=512,
        batch_size=32,
        ckpt_file="valid_checkpoint.pt",
        device=None,
    ):
        self.device_manager = DeviceManager.from_value(device)
        self.device = self.device_manager.device
        self.trained_model_path = trained_model_path
        self.topk = topk
        self.beam_size = beam_size
        self.temperature = temperature
        self.n_best = n_best
        self.min_length = min_length
        self.max_length = max_length

        print(f"[INFO] Loading trained model from {trained_model_path}")
        trained_config = load_train_config(resolve_config_file(trained_model_path))
        vocab = load_vocab(
            f"{trained_config.data.data_path}/{trained_config.data.vocab_file}"
        )
        self.vocab_rev = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]
        model = build_sequence_model(trained_config, vocab)

        ckpt_file = f"{trained_model_path}/model/{ckpt_file}"
        self.device_manager.move_module(model)
        CheckpointAdapter().load_into_model(
            model, ckpt_file, map_location=self.device_manager.map_location, mode="strict"
        )
        self.model = model
        print(f"[INFO] Loading test dataset from {trained_config.data.data_path}")

        test_dataset = RXNG2SDataset(
            root=trained_config.data.data_path,
            src_file=trained_config.data.test_src_file,
            tgt_file=trained_config.data.test_tgt_file,
            vocab_file=trained_config.data.vocab_file,
            trunck=0,
            multi_process=False,
            oh=False,
        )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        self.test_ground_truth_smiles_lst = [
            "".join(
                [
                    self.vocab_rev[idx]
                    for idx in _sequence_sample(test_dataset, idx).tgt_token_ids[0][
                        : int(_sequence_sample(test_dataset, idx).tgt_lens[0]) - 1
                    ]
                ]
            )
            for idx in range(len(test_dataset))
        ]

    def eval(self, dataloader, ground_truth_smiles_lst):
        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = self.device_manager.move(batch_data)
                seq_batch = _sequence_batch(batch_data)
                results = self.model.infer(
                    reaction_batch=batch_data,
                    batch_size=len(seq_batch.tgt_lens),
                    beam_size=self.beam_size,
                    n_best=self.n_best,
                    temperature=self.temperature,
                    min_length=self.min_length,
                    max_length=self.max_length,
                )

                for predictions in _sequence_predictions(results):
                    smis = []
                    for prediction in predictions:
                        predicted_idx = prediction.detach().cpu().numpy()
                        predicted_tokens = [
                            self.vocab_rev[idx] for idx in predicted_idx[:-1]
                        ]
                        smi = " ".join(predicted_tokens)
                        smis.append(smi)
                    all_predictions.append(",".join(smis))

        accuracies = np.zeros(
            [len(ground_truth_smiles_lst), self.n_best], dtype=np.float32
        )
        for i in range(len(ground_truth_smiles_lst)):
            smi_tgt = ground_truth_smiles_lst[i]
            line_predict = all_predictions[i]
            line_predict = "".join(line_predict.split())
            smis_predict = line_predict.split(",")
            smis_predict = [
                Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                if Chem.MolFromSmiles(smi)
                else ""
                for smi in smis_predict
            ]
            for j, smi in enumerate(smis_predict):
                if smi == smi_tgt:
                    accuracies[i, j:] = 1.0
                    break
        self.accuracies = accuracies
        for i in range(self.topk):
            print(f"Top-{i + 1} Accuracy: {np.mean(self.accuracies[:, i])}")
        return accuracies


def eval_regression_performance(
    pretrained_model_path,
    ckpt_file="valid_checkpoint.pt",
    scale=1.0,
    specific_val=False,
    yield_constrain=False,
    return_train_results=False,
    max_batches=None,
    device=None,
):
    manager = DeviceManager.from_value(device)
    settings = RegressionEvaluationSettings(
        model_path=pretrained_model_path,
        ckpt_file=ckpt_file,
        split="test",
        specific_val=specific_val,
        scale=scale,
        yield_constrain=yield_constrain,
        max_batches=max_batches,
        device=manager.device,
    )
    test_result = evaluate_regression_split(settings).evaluation
    r2 = test_result.metrics.r2
    mae = test_result.metrics.mae
    preds = test_result.preds
    targets = test_result.targets

    if return_train_results:
        train_settings = RegressionEvaluationSettings(
            model_path=pretrained_model_path,
            ckpt_file=ckpt_file,
            split="train",
            specific_val=specific_val,
            scale=scale,
            yield_constrain=yield_constrain,
            max_batches=max_batches,
            device=manager.device,
        )
        train_result = evaluate_regression_split(train_settings).evaluation
        train_r2 = train_result.metrics.r2
        train_mae = train_result.metrics.mae
        train_preds = train_result.preds
        train_targets = train_result.targets
        return train_r2, train_mae, train_preds, train_targets, r2, mae, preds, targets

    return r2, mae, preds, targets


def load_pred_model(
    pretrained_model_path, ckpt_filename="valid_checkpoint.pt", task_type="reactivity", device=None
):
    manager = DeviceManager.from_value(device)
    assert task_type in [
        "reactivity",
        "selectivity",
        "retro-synthesis",
        "forward-synthesis",
    ]
    task_type = task_type.lower()
    pretrained_config = load_train_config(resolve_config_file(pretrained_model_path))

    ckpt_file = f"{pretrained_model_path}/model/{ckpt_filename}"
    if task_type in ["reactivity", "selectivity"]:
        model = build_regression_model(pretrained_config)
    else:
        vocab = load_vocab(
            f"{pretrained_config.data.data_path}/{pretrained_config.data.vocab_file}"
        )
        model = build_sequence_model(pretrained_config, vocab)
    CheckpointAdapter().load_into_model(
        model, ckpt_file, map_location=manager.map_location, mode="strict"
    )
    manager.move_module(model)
    model.eval()
    print("Model loaded successfully!")
    return model


def get_eval_dataloader(
    root, data_file_dict, pretrained_model_path, batch_size=4, task_type="reactivity"
) -> Iterable[object]:
    assert task_type in [
        "reactivity",
        "selectivity",
        "retro-synthesis",
        "forward-synthesis",
    ]
    task_type = task_type.lower()
    pretrained_config = load_train_config(resolve_config_file(pretrained_model_path))

    if task_type == "reactivity" or task_type == "selectivity":
        rct_dataset = RXNDataset(
            root=root,
            name=data_file_dict["rct"],
            trunck=pretrained_config.data.data_trunck,
        )
        pdt_dataset = RXNDataset(
            root=root,
            name=data_file_dict["pdt"],
            trunck=pretrained_config.data.data_trunck,
        )
        mid_dataset = RXNDataset(
            root=root,
            name=data_file_dict["delta-mol"],
            trunck=pretrained_config.data.data_trunck,
        )
        eval_dataset = TripleDataset(rct_dataset, pdt_dataset, mid_dataset)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=triple_collate_fn,
        )
        return eval_dataloader
    elif task_type == "forward-synthesis" or task_type == "retro-synthesis":
        eval_dataset = RXNG2SDataset(
            root=root,
            src_file=data_file_dict["src"],
            tgt_file=data_file_dict["tgt"],
            vocab_file=pretrained_config.data.vocab_file,
            trunck=0,
            multi_process=False,
            oh=False,
        )
        return DataLoader(_loader_dataset(eval_dataset), batch_size=batch_size, shuffle=False)
    raise ValueError(f"Unsupported task_type: {task_type}")


def reaction_prediction(
    model_path,
    rxn_smiles_lst,
    task_type,
    params={
        "batch_size": 4,
        "beam_size": 10,
        "n_best": 10,
        "temperature": 1.0,
        "min_length": 1,
        "max_length": 512,
    },
    ckpt_file="valid_checkpoint.pt",
    device=None,
):
    from rxngraphormer.midgen.midmol import gen_mech_mid_smi

    task_type = task_type.lower()
    assert len(rxn_smiles_lst) >= 2, (
        "'rxn_smiles_lst' must contain at least 2 reactions"
    )
    assert task_type in [
        "reactivity",
        "selectivity",
        "forward-synthesis",
        "retro-synthesis",
    ], (
        "task_type must be 'reactivity', 'selectivity', 'forward-synthesis' or 'retro-synthesis'"
    )
    manager = DeviceManager.from_value(device)
    model = load_pred_model(model_path, ckpt_filename=ckpt_file, task_type=task_type, device=manager.device)
    preds = None
    if task_type in ["reactivity", "selectivity"]:
        regression_model = cast(RXNGRegressor, model)
        rct_smi_lst = [
            f"{canonicalize_reaction_side(split_reaction_smiles(smi)[0])},0" for smi in rxn_smiles_lst
        ]
        pdt_smi_lst = [
            f"{canonicalize_reaction_side(split_reaction_smiles(smi)[1])},0" for smi in rxn_smiles_lst
        ]
        rct_pdt_pair_lst = [split_reaction_smiles(rxn_smiles) for rxn_smiles in rxn_smiles_lst]
        delta_mol_smi_lst = [
            f"{gen_mech_mid_smi(rct_pdt_pair)[0]},0"
            for rct_pdt_pair in rct_pdt_pair_lst
        ]
        with tempfile.TemporaryDirectory(
            prefix="rxngraphormer_reaction_pred_"
        ) as tmp_dir:
            with open(os.path.join(tmp_dir, "rct_smiles_0.csv"), "w") as fw:
                fw.writelines("\n".join(rct_smi_lst))
            with open(os.path.join(tmp_dir, "pdt_smiles_0.csv"), "w") as fw:
                fw.writelines("\n".join(pdt_smi_lst))
            with open(os.path.join(tmp_dir, "delta_mol_smiles_0.csv"), "w") as fw:
                fw.writelines("\n".join(delta_mol_smi_lst))

            eval_dataloader: Iterable[object] = get_eval_dataloader(
                root=tmp_dir,
                data_file_dict={
                    "rct": "rct_smiles_0.csv",
                    "pdt": "pdt_smiles_0.csv",
                    "delta-mol": "delta_mol_smiles_0.csv",
                },
                pretrained_model_path=model_path,
                task_type=task_type,
                batch_size=params["batch_size"],
            )
            pred_chunks = []
            with torch.no_grad():
                for batch_data in eval_dataloader:
                    rct_data, pdt_data, mid_data = cast(tuple[object, object, object], manager.move(batch_data))
                    out = regression_model([rct_data, pdt_data, mid_data])
                    pred_chunks.append(out.detach().cpu())
            regression_preds = torch.cat(pred_chunks, dim=0)
            if task_type == "reactivity":
                regression_preds = torch.where(regression_preds < 0, torch.tensor(0.0), regression_preds)
                regression_preds = torch.where(regression_preds > 1, torch.tensor(1.0), regression_preds)
                preds = regression_preds.numpy() * 100
            else:
                preds = regression_preds

    elif task_type in ["forward-synthesis", "retro-synthesis"]:
        sequence_model = cast(SequenceModelProtocol, model)
        vocab_rev = [k for k, v in sorted(sequence_model.vocab.items(), key=lambda tup: tup[1])]
        tokenized_smiles_lst = [smi_tokenizer(smi) for smi in rxn_smiles_lst]
        with tempfile.TemporaryDirectory(
            prefix="rxngraphormer_sequence_pred_"
        ) as tmp_dir:
            with open(os.path.join(tmp_dir, "src_tokenized_smiles.txt"), "w") as fw:
                fw.writelines("\n".join(tokenized_smiles_lst))
            with open(os.path.join(tmp_dir, "tgt_tokenized_smiles.txt"), "w") as fw:
                fw.writelines("\n".join(tokenized_smiles_lst))
            pretrained_config = load_train_config(resolve_config_file(model_path))
            shutil.copyfile(
                f"{pretrained_config.data.data_path}/{pretrained_config.data.vocab_file}",
                os.path.join(tmp_dir, pretrained_config.data.vocab_file),
            )
            sequence_eval_dataloader: Iterable[object] = get_eval_dataloader(
                tmp_dir,
                {"src": "src_tokenized_smiles.txt", "tgt": "tgt_tokenized_smiles.txt"},
                pretrained_model_path=model_path,
                batch_size=params["batch_size"],
                task_type=task_type,
            )

            all_predictions = []
            smis: list[str] = []
            with torch.no_grad():
                for batch_data in sequence_eval_dataloader:
                    batch_data = manager.move(batch_data)
                    seq_batch = _sequence_batch(batch_data)
                    results = sequence_model.infer(
                        reaction_batch=batch_data,
                        batch_size=len(seq_batch.tgt_lens),
                        beam_size=params["beam_size"],
                        n_best=params["n_best"],
                        temperature=params["temperature"],
                        min_length=params["min_length"],
                        max_length=params["max_length"],
                    )

                    for predictions in _sequence_predictions(results):
                        smis = []
                        for prediction in predictions:
                            predicted_idx = prediction.detach().cpu().numpy()
                            predicted_tokens = [
                                vocab_rev[idx] for idx in predicted_idx[:-1]
                            ]
                            smi = "".join(predicted_tokens)
                            smis.append(smi)
                        all_predictions.append(smis)
            import pandas as pd

            prediction_frame = pd.DataFrame(all_predictions).T
            prediction_frame.index = [f"Top-{i + 1}" for i in range(len(smis))]
            preds = prediction_frame
    if preds is None:
        raise RuntimeError("Prediction output was not produced")
    print("Done!")
    return preds


__all__ = [
    "SeqEval",
    "device",
    "eval_regression_performance",
    "get_eval_dataloader",
    "load_pred_model",
    "reaction_prediction",
]
