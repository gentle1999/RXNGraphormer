from __future__ import annotations

import argparse

from rxngraphormer.config import load_config
from rxngraphormer.preprocess.data import MultiRXNDataset, RXNDataset, RXNG2SDataset
from rxngraphormer.preprocess.table import write_reaction_table_files
from rxngraphormer.utils import as_bool


def preprocess_from_config(config_path: str) -> None:
    config = load_config(config_path)
    _materialize_input_table(config)

    if config.task == "classification":
        _preprocess_classification(config)
    elif config.task == "regression":
        _preprocess_regression(config)
    elif config.task == "sequence_generation":
        RXNG2SDataset(
            root=config.data.data_path,
            src_file=config.data.train_src_file,
            tgt_file=config.data.train_tgt_file,
            oh=False,
            multi_process=False,
            vocab_file=config.data.vocab_file,
            train=True,
        )
        RXNG2SDataset(
            root=config.data.data_path,
            src_file=config.data.valid_src_file,
            tgt_file=config.data.valid_tgt_file,
            oh=False,
            multi_process=False,
            vocab_file=config.data.vocab_file,
            train=False,
        )
        RXNG2SDataset(
            root=config.data.data_path,
            src_file=config.data.test_src_file,
            tgt_file=config.data.test_tgt_file,
            oh=False,
            multi_process=False,
            vocab_file=config.data.vocab_file,
            train=False,
        )
    else:
        raise ValueError("task must be classification, regression, or sequence_generation")


def _materialize_input_table(config) -> None:
    if not getattr(config.data, "input_table", ""):
        return

    table_files = write_reaction_table_files(
        config.data.input_table,
        output_dir=config.data.data_path,
        output_prefix=config.data.output_prefix or None,
        rxn_smiles_column=config.data.rxn_smiles_column,
        rct_smiles_column=config.data.rct_smiles_column,
        pdt_smiles_column=config.data.pdt_smiles_column,
        mid_smiles_column=config.data.mid_smiles_column,
        target_column=config.data.target_column or None,
        generate_mid=config.data.generate_mid,
        mapping_policy=config.data.mapping_policy,
    )
    config.data.rct_data_file = table_files.rct_data_file
    config.data.pdt_data_file = table_files.pdt_data_file
    config.data.rct_name_regrex = table_files.rct_name_regrex
    config.data.pdt_name_regrex = table_files.pdt_name_regrex
    config.data.mid_data_file = table_files.mid_data_file
    config.data.mid_name_regrex = table_files.mid_name_regrex
    if not table_files.mid_data_file and as_bool(getattr(config.model, "use_mid_inf", False)):
        raise ValueError(
            "model.use_mid_inf is enabled, but the input table did not provide "
            "mid_smiles and data.generate_mid is false"
        )


def _preprocess_classification(config) -> None:
    MultiRXNDataset(
        root=config.data.data_path,
        name_regrex=config.data.rct_name_regrex,
        trunck=config.data.data_trunck,
        task=config.data.task,
        file_num_trunck=config.data.file_num_trunck,
        name_tag="rct",
    )
    MultiRXNDataset(
        root=config.data.data_path,
        name_regrex=config.data.pdt_name_regrex,
        trunck=config.data.data_trunck,
        task=config.data.task,
        file_num_trunck=config.data.file_num_trunck,
        name_tag="pdt",
    )


def _preprocess_regression(config) -> None:
    for file_name in _regression_data_files(config):
        RXNDataset(
            root=config.data.data_path,
            name=file_name,
            trunck=config.data.data_trunck,
            task=config.data.task,
            multi_process=config.data.multi_process,
        )


def _regression_data_files(config) -> list[str]:
    use_mid = as_bool(getattr(config.model, "use_mid_inf", False))
    if config.data.rct_data_file:
        files = [config.data.rct_data_file, config.data.pdt_data_file]
        if use_mid:
            files.append(config.data.mid_data_file)
        return _require_files(files)

    files = [
        config.data.train_rct_data_file,
        config.data.train_pdt_data_file,
        config.data.val_rct_data_file,
        config.data.val_pdt_data_file,
        config.data.test_rct_data_file,
        config.data.test_pdt_data_file,
    ]
    if use_mid:
        files.extend(
            [
                config.data.train_mid_data_file,
                config.data.val_mid_data_file,
                config.data.test_mid_data_file,
            ]
        )
    return _require_files(files)


def _require_files(files: list[str]) -> list[str]:
    missing = [name for name in files if not name]
    if missing:
        raise ValueError("Regression preprocessing config is missing one or more data file names")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, default="./config/pretrain_parameters.json")
    args = parser.parse_args()
    preprocess_from_config(args.config_json)


if __name__ == "__main__":
    main()
