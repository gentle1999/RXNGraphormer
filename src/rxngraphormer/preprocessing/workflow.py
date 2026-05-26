from __future__ import annotations

import argparse

from rxngraphormer.config import Config, EvalConfig, PreprocessParallelMode, load_config
from rxngraphormer.config_utils import as_bool
from rxngraphormer.data import MultiRXNDataset, RXNDataset, RXNG2SDataset
from rxngraphormer.data.reaction_processing import reaction_preprocessing_kwargs_from_config
from rxngraphormer.preprocessing.materialization import write_reaction_table_files


def preprocess_from_config(
    config_path: str,
    *,
    multi_process: bool | None = None,
    preprocess_num_workers: int | None = None,
    preprocess_batch_size: int | None = None,
    preprocess_parallel_mode: PreprocessParallelMode | None = None,
) -> None:
    config = _load_preprocess_config(config_path)
    _apply_preprocess_overrides(
        config,
        multi_process=multi_process,
        preprocess_num_workers=preprocess_num_workers,
        preprocess_batch_size=preprocess_batch_size,
        preprocess_parallel_mode=preprocess_parallel_mode,
    )
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
            multi_process=config.data.multi_process,
            num_worker=config.data.preprocess_num_workers,
            vocab_file=config.data.vocab_file,
            train=True,
        )
        RXNG2SDataset(
            root=config.data.data_path,
            src_file=config.data.valid_src_file,
            tgt_file=config.data.valid_tgt_file,
            oh=False,
            multi_process=config.data.multi_process,
            num_worker=config.data.preprocess_num_workers,
            vocab_file=config.data.vocab_file,
            train=False,
        )
        RXNG2SDataset(
            root=config.data.data_path,
            src_file=config.data.test_src_file,
            tgt_file=config.data.test_tgt_file,
            oh=False,
            multi_process=config.data.multi_process,
            num_worker=config.data.preprocess_num_workers,
            vocab_file=config.data.vocab_file,
            train=False,
        )
    else:
        raise ValueError("task must be classification, regression, or sequence_generation")


def _load_preprocess_config(config_path: str) -> Config:
    config = load_config(config_path)
    if isinstance(config, EvalConfig):
        raise TypeError(f"Expected train config with data preprocessing settings, got eval config: {config_path}")
    return config


def _apply_preprocess_overrides(
    config: Config,
    *,
    multi_process: bool | None,
    preprocess_num_workers: int | None,
    preprocess_batch_size: int | None,
    preprocess_parallel_mode: PreprocessParallelMode | None,
) -> None:
    if multi_process is not None:
        config.data.multi_process = multi_process
    if preprocess_num_workers is not None:
        config.data.preprocess_num_workers = preprocess_num_workers
    if preprocess_batch_size is not None:
        config.data.preprocess_batch_size = preprocess_batch_size
    if preprocess_parallel_mode is not None:
        config.data.preprocess_parallel_mode = preprocess_parallel_mode


def _materialize_input_table(config: Config) -> None:
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


def _preprocess_classification(config: Config) -> None:
    preprocess_kwargs = reaction_preprocessing_kwargs_from_config(config)
    MultiRXNDataset(
        root=config.data.data_path,
        name_regrex=config.data.rct_name_regrex,
        trunck=config.data.data_trunck,
        task=config.data.task,
        file_num_trunck=config.data.file_num_trunck,
        name_tag="rct",
        **preprocess_kwargs,
    )
    MultiRXNDataset(
        root=config.data.data_path,
        name_regrex=config.data.pdt_name_regrex,
        trunck=config.data.data_trunck,
        task=config.data.task,
        file_num_trunck=config.data.file_num_trunck,
        name_tag="pdt",
        **preprocess_kwargs,
    )


def _preprocess_regression(config: Config) -> None:
    preprocess_kwargs = reaction_preprocessing_kwargs_from_config(config)
    for file_name in _regression_data_files(config):
        RXNDataset(
            root=config.data.data_path,
            name=file_name,
            trunck=config.data.data_trunck,
            task=config.data.task,
            **preprocess_kwargs,
        )


def _regression_data_files(config: Config) -> list[str]:
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
    parser.add_argument("--config", "--config_json", dest="config_path", type=str, default="./config/pretrain_parameters.json")
    parser.add_argument("--multi_process", action="store_true", help="Enable multiprocessing graph preprocessing.")
    parser.add_argument("--preprocess_num_workers", type=int, default=None)
    parser.add_argument("--preprocess_batch_size", type=int, default=None)
    parser.add_argument("--preprocess_parallel_mode", choices=("reaction", "file"), default=None)
    args = parser.parse_args()
    preprocess_from_config(
        args.config_path,
        multi_process=True if args.multi_process else None,
        preprocess_num_workers=args.preprocess_num_workers,
        preprocess_batch_size=args.preprocess_batch_size,
        preprocess_parallel_mode=args.preprocess_parallel_mode,
    )


__all__ = ["main", "preprocess_from_config"]
