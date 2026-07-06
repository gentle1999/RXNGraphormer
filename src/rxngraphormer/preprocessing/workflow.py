from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from itertools import zip_longest
from multiprocessing import Pool
from pathlib import Path

from rxngraphormer.config import Config, EvalConfig, PreprocessParallelMode, load_config
from rxngraphormer.config_utils import as_bool
from rxngraphormer.data import MultiRXNDataset, RXNDataset, RXNG2SDataset
from rxngraphormer.data.files import _matched_raw_files
from rxngraphormer.data.reaction_processing import reaction_preprocessing_kwargs_from_config
from rxngraphormer.preprocessing.materialization import (
    generate_mid_smiles_batch,
    validate_legacy_target,
    write_reaction_table_files,
)


@dataclass(frozen=True)
class _GeneratedMidFiles:
    rct_name_regrex: str
    pdt_name_regrex: str
    mid_name_regrex: str
    kept_count: int
    skipped_count: int


@dataclass
class _MidGenerationStats:
    kept_count: int = 0
    skipped_count: int = 0
    skipped_examples: list[str] | None = None

    def add(self, other: _MidGenerationStats) -> None:
        self.kept_count += other.kept_count
        self.skipped_count += other.skipped_count
        if other.skipped_examples:
            if self.skipped_examples is None:
                self.skipped_examples = []
            remaining = max(0, 5 - len(self.skipped_examples))
            self.skipped_examples.extend(other.skipped_examples[:remaining])


@dataclass(frozen=True)
class _MidGenerationFileTask:
    index: int
    rct_file: str
    pdt_file: str
    rct_valid_file: str
    pdt_valid_file: str
    mid_file: str
    mapping_policy: str
    batch_size: int


def _process_mid_generation_file(
    task: _MidGenerationFileTask,
) -> tuple[int, str, str, str, _MidGenerationStats]:
    stats = _write_midvalid_files_from_pair(
        Path(task.rct_file),
        Path(task.pdt_file),
        Path(task.rct_valid_file),
        Path(task.pdt_valid_file),
        Path(task.mid_file),
        mapping_policy=task.mapping_policy,
        batch_size=task.batch_size,
    )
    return task.index, task.rct_valid_file, task.pdt_valid_file, task.mid_file, stats


def _init_mid_generation_worker() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        import torch
    except Exception:
        return
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


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
    mid_name = _ensure_classification_mid_files(config) if as_bool(getattr(config.model, "use_mid_inf", False)) else ""
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
    if mid_name:
        MultiRXNDataset(
            root=config.data.data_path,
            name_regrex=mid_name,
            trunck=config.data.data_trunck,
            task=config.data.task,
            file_num_trunck=config.data.file_num_trunck,
            name_tag="mid",
            **preprocess_kwargs,
        )


def _ensure_classification_mid_files(config: Config) -> str:
    mid_name = config.data.mid_name_regrex or config.data.mid_data_file
    if mid_name and _raw_pattern_exists(config.data.data_path, mid_name):
        if "_midvalid_mid_" not in mid_name or _use_existing_midvalid_pair_files(config, mid_name):
            return mid_name
        if not as_bool(getattr(config.data, "generate_mid", False)):
            raise ValueError(
                f"Classification preprocessing matched mid files for {mid_name!r}, "
                "but the corresponding mid-valid rct/pdt files are missing."
            )
    if as_bool(getattr(config.data, "generate_mid", False)):
        generated = _generate_classification_mid_files(config)
        config.data.rct_name_regrex = generated.rct_name_regrex
        config.data.pdt_name_regrex = generated.pdt_name_regrex
        config.data.mid_name_regrex = generated.mid_name_regrex
        config.data.rct_data_file = generated.rct_name_regrex if "*" not in generated.rct_name_regrex else ""
        config.data.pdt_data_file = generated.pdt_name_regrex if "*" not in generated.pdt_name_regrex else ""
        config.data.mid_data_file = generated.mid_name_regrex if "*" not in generated.mid_name_regrex else ""
        print(
            "[INFO] Classification mid generation finished: "
            f"kept={generated.kept_count}, skipped={generated.skipped_count}, "
            f"rct={generated.rct_name_regrex}, pdt={generated.pdt_name_regrex}, mid={generated.mid_name_regrex}"
        )
        return generated.mid_name_regrex
    if mid_name:
        raise ValueError(
            f"Classification preprocessing with model.use_mid_inf=True matched no mid files for {mid_name!r}; "
            "set data.generate_mid=true to generate mid files from paired reactant/product CSVs."
        )
    else:
        raise ValueError(
            "Classification preprocessing with model.use_mid_inf=True requires "
            "data.mid_name_regrex/data.mid_data_file, or data.generate_mid=true "
            "to generate mid files from paired reactant/product CSVs."
        )


def _raw_pattern_exists(root: str, pattern: str) -> bool:
    return any(Path(path).is_file() for path in glob.glob(str(Path(root) / pattern)))


def _use_existing_midvalid_pair_files(config: Config, mid_name: str) -> bool:
    if "_midvalid_mid_" not in mid_name:
        return False
    rct_name = mid_name.replace("_midvalid_mid_", "_midvalid_rct_", 1)
    pdt_name = mid_name.replace("_midvalid_mid_", "_midvalid_pdt_", 1)
    if _expected_midvalid_files_exist(config):
        config.data.rct_name_regrex = rct_name
        config.data.pdt_name_regrex = pdt_name
        config.data.rct_data_file = rct_name if "*" not in rct_name else ""
        config.data.pdt_data_file = pdt_name if "*" not in pdt_name else ""
        print(f"[INFO] Using mid-valid filtered classification files: rct={rct_name}, pdt={pdt_name}, mid={mid_name}")
        return True
    return False


def _expected_midvalid_files_exist(config: Config) -> bool:
    data_path = Path(config.data.data_path)
    rct_name = config.data.rct_name_regrex or config.data.rct_data_file
    pdt_name = config.data.pdt_name_regrex or config.data.pdt_data_file
    if not rct_name or not pdt_name:
        return False

    rct_files = _matched_raw_files(data_path, rct_name)
    pdt_files = _matched_raw_files(data_path, pdt_name)
    if not rct_files or len(rct_files) != len(pdt_files):
        return False
    for rct_file_name, pdt_file_name in zip(rct_files, pdt_files, strict=True):
        rct_file = Path(rct_file_name)
        pdt_file = Path(pdt_file_name)
        if not _midvalid_triplet_exists(
            _midvalid_file_for_side(rct_file, "rct"),
            _midvalid_file_for_side(pdt_file, "pdt"),
            _midvalid_file_for_side(rct_file, "mid"),
        ):
            return False
    return True


def _generate_classification_mid_files(config: Config) -> _GeneratedMidFiles:
    data_path = Path(config.data.data_path)
    rct_name = config.data.rct_name_regrex or config.data.rct_data_file
    pdt_name = config.data.pdt_name_regrex or config.data.pdt_data_file
    if not rct_name or not pdt_name:
        raise ValueError(
            "data.generate_mid=true requires rct_name_regrex/rct_data_file and pdt_name_regrex/pdt_data_file"
        )

    rct_files = _matched_raw_files(data_path, rct_name)
    pdt_files = _matched_raw_files(data_path, pdt_name)
    if len(rct_files) != len(pdt_files):
        raise ValueError(
            f"Cannot generate mid files: matched {len(rct_files)} reactant files and {len(pdt_files)} product files"
        )

    mapping_policy = getattr(config.data, "mapping_policy", "auto")
    batch_size = max(1, int(getattr(config.data, "preprocess_batch_size", 128)))

    tasks: list[_MidGenerationFileTask] = []
    for idx, (rct_file_name, pdt_file_name) in enumerate(zip(rct_files, pdt_files, strict=True)):
        rct_file = Path(rct_file_name)
        pdt_file = Path(pdt_file_name)
        rct_valid_file = _midvalid_file_for_side(rct_file, "rct")
        pdt_valid_file = _midvalid_file_for_side(pdt_file, "pdt")
        mid_file = _midvalid_file_for_side(rct_file, "mid")
        tasks.append(
            _MidGenerationFileTask(
                index=idx,
                rct_file=str(rct_file),
                pdt_file=str(pdt_file),
                rct_valid_file=str(rct_valid_file),
                pdt_valid_file=str(pdt_valid_file),
                mid_file=str(mid_file),
                mapping_policy=mapping_policy,
                batch_size=batch_size,
            )
        )

    results = _run_mid_generation_tasks(config, tasks)

    generated_rct_files: list[Path] = []
    generated_pdt_files: list[Path] = []
    generated_mid_files: list[Path] = []
    total = _MidGenerationStats()
    for _idx, rct_valid_file, pdt_valid_file, mid_file, stats in results:
        total.add(stats)
        if stats.kept_count > 0:
            generated_rct_files.append(Path(rct_valid_file))
            generated_pdt_files.append(Path(pdt_valid_file))
            generated_mid_files.append(Path(mid_file))

    if total.kept_count == 0:
        raise ValueError("Cannot generate classification mid files: every row failed to produce a valid mid SMILES")

    return _GeneratedMidFiles(
        rct_name_regrex=_midvalid_pattern_for_side(rct_name, "rct", generated_rct_files),
        pdt_name_regrex=_midvalid_pattern_for_side(pdt_name, "pdt", generated_pdt_files),
        mid_name_regrex=_midvalid_pattern_for_side(rct_name, "mid", generated_mid_files),
        kept_count=total.kept_count,
        skipped_count=total.skipped_count,
    )


def _run_mid_generation_tasks(
    config: Config,
    tasks: list[_MidGenerationFileTask],
) -> list[tuple[int, str, str, str, _MidGenerationStats]]:
    parallel_mode = getattr(config.data, "preprocess_parallel_mode", "reaction")
    if parallel_mode not in {"reaction", "file"}:
        raise ValueError("preprocess_parallel_mode must be 'reaction' or 'file'")

    num_workers = int(getattr(config.data, "preprocess_num_workers", 1))
    use_file_parallel = (
        as_bool(getattr(config.data, "multi_process", False))
        and parallel_mode == "file"
        and num_workers > 1
        and len(tasks) > 1
    )
    if not use_file_parallel:
        return [_process_mid_generation_file(task) for task in tasks]

    workers = min(max(1, num_workers), len(tasks))
    print(f"[INFO] {workers} file workers are used to generate mid-valid classification files...")
    results: list[tuple[int, str, str, str, _MidGenerationStats]] = []
    with Pool(processes=workers, initializer=_init_mid_generation_worker) as pool:
        for result in pool.imap_unordered(_process_mid_generation_file, tasks):
            results.append(result)
    results.sort(key=lambda item: item[0])
    return results


def _midvalid_file_for_side(source_file: Path, side: str) -> Path:
    return source_file.with_name(_midvalid_name_for_side(source_file.name, side))


def _midvalid_pattern_for_side(source_pattern: str, side: str, generated_files: list[Path]) -> str:
    pattern = _midvalid_name_for_side(source_pattern, side)
    if "*" in pattern:
        return pattern
    if generated_files:
        return generated_files[0].name
    return pattern


def _midvalid_name_for_side(name: str, side: str) -> str:
    replacements = (
        ("_rct_", f"_midvalid_{side}_"),
        ("_pdt_", f"_midvalid_{side}_"),
        ("_rct.", f"_midvalid_{side}."),
        ("_pdt.", f"_midvalid_{side}."),
        ("rct_", f"midvalid_{side}_"),
        ("pdt_", f"midvalid_{side}_"),
        ("rct.", f"midvalid_{side}."),
        ("pdt.", f"midvalid_{side}."),
    )
    for old, new in replacements:
        if old in name:
            return name.replace(old, new, 1)
    path = Path(name)
    return f"{path.stem}_midvalid_{side}{path.suffix}"


def _write_midvalid_files_from_pair(
    rct_file: Path,
    pdt_file: Path,
    rct_valid_file: Path,
    pdt_valid_file: Path,
    mid_file: Path,
    *,
    mapping_policy: str,
    batch_size: int,
) -> _MidGenerationStats:
    if _midvalid_triplet_exists(rct_valid_file, pdt_valid_file, mid_file):
        kept_count = _count_lines(mid_file)
        print(f"[INFO] {mid_file.name} already exists with {kept_count} rows, skip mid generation...", flush=True)
        return _MidGenerationStats(kept_count=kept_count, skipped_count=0)

    stats = _MidGenerationStats()
    rct_part = _part_file(rct_valid_file)
    pdt_part = _part_file(pdt_valid_file)
    mid_part = _part_file(mid_file)
    for path in (rct_part, pdt_part, mid_part):
        if path.exists():
            path.unlink()

    processed_count = 0
    next_report_at = 1000
    batch: list[tuple[int, str, str, str, str, str]] = []
    with (
        rct_file.open(encoding="utf-8") as rct_handle,
        pdt_file.open(encoding="utf-8") as pdt_handle,
        rct_part.open("w", encoding="utf-8") as rct_out,
        pdt_part.open("w", encoding="utf-8") as pdt_out,
        mid_part.open("w", encoding="utf-8") as mid_out,
    ):
        for row_idx, pair in enumerate(zip_longest(rct_handle, pdt_handle), start=1):
            rct_raw, pdt_raw = pair
            if rct_raw is None or pdt_raw is None:
                raise ValueError(
                    f"Cannot generate {mid_file.name}: {rct_file.name} and {pdt_file.name} contain different row counts"
                )
            rct_line = rct_raw.strip()
            pdt_line = pdt_raw.strip()
            if not rct_line and not pdt_line:
                continue
            if not rct_line or not pdt_line:
                raise ValueError(
                    f"Cannot generate {mid_file.name}: one side is empty at row {row_idx} "
                    f"({rct_file.name}={bool(rct_line)}, {pdt_file.name}={bool(pdt_line)})"
                )
            processed_count += 1
            rct_smiles, rct_target = _split_legacy_smiles_target(rct_line, rct_file, row_idx)
            pdt_smiles, pdt_target = _split_legacy_smiles_target(pdt_line, pdt_file, row_idx)
            if rct_target != pdt_target:
                raise ValueError(
                    f"Cannot generate {mid_file.name}: target mismatch at row {row_idx} "
                    f"({rct_file.name}={rct_target!r}, {pdt_file.name}={pdt_target!r})"
                )
            batch.append((row_idx, rct_line, pdt_line, rct_smiles, pdt_smiles, rct_target))
            if len(batch) >= batch_size:
                _write_midvalid_batch(
                    batch,
                    rct_file,
                    mid_file,
                    rct_out,
                    pdt_out,
                    mid_out,
                    stats,
                    mapping_policy=mapping_policy,
                )
                batch.clear()
                while processed_count >= next_report_at:
                    print(
                        f"[INFO] {mid_file.name}: processed={processed_count}, "
                        f"kept={stats.kept_count}, skipped={stats.skipped_count}",
                        flush=True,
                    )
                    next_report_at += 1000

        if batch:
            _write_midvalid_batch(
                batch,
                rct_file,
                mid_file,
                rct_out,
                pdt_out,
                mid_out,
                stats,
                mapping_policy=mapping_policy,
            )
            while processed_count >= next_report_at:
                print(
                    f"[INFO] {mid_file.name}: processed={processed_count}, "
                    f"kept={stats.kept_count}, skipped={stats.skipped_count}",
                    flush=True,
                )
                next_report_at += 1000

    if stats.kept_count > 0:
        rct_part.replace(rct_valid_file)
        pdt_part.replace(pdt_valid_file)
        mid_part.replace(mid_file)
    else:
        for path in (rct_part, pdt_part, mid_part, rct_valid_file, pdt_valid_file, mid_file):
            if path.exists():
                path.unlink()

    print(
        f"[INFO] Generated {mid_file.name}: processed={processed_count}, "
        f"kept={stats.kept_count}, skipped={stats.skipped_count}",
        flush=True,
    )
    for example in stats.skipped_examples or []:
        print(f"[WARN] {example}")
    return stats


def _write_midvalid_batch(
    batch: list[tuple[int, str, str, str, str, str]],
    rct_file: Path,
    mid_file: Path,
    rct_out,
    pdt_out,
    mid_out,
    stats: _MidGenerationStats,
    *,
    mapping_policy: str,
) -> None:
    rxn_smiles = [f"{rct_smiles}>>{pdt_smiles}" for _idx, _rct, _pdt, rct_smiles, pdt_smiles, _target in batch]
    mid_results = generate_mid_smiles_batch(rxn_smiles, mapping_policy=mapping_policy)
    for (row_idx, rct_line, pdt_line, _rct_smiles, _pdt_smiles, target), mid_result in zip(
        batch,
        mid_results,
        strict=True,
    ):
        if isinstance(mid_result, Exception):
            _record_skipped_mid(stats, rct_file, row_idx, f"failed to generate mid SMILES ({mid_result})")
            continue
        if not mid_result:
            _record_skipped_mid(stats, rct_file, row_idx, "empty mid SMILES")
            continue
        rct_out.write(f"{rct_line}\n")
        pdt_out.write(f"{pdt_line}\n")
        mid_out.write(f"{mid_result},{target}\n")
        stats.kept_count += 1


def _record_skipped_mid(stats: _MidGenerationStats, rct_file: Path, row_idx: int, reason: str) -> None:
    stats.skipped_count += 1
    if stats.skipped_examples is None:
        stats.skipped_examples = []
    if len(stats.skipped_examples) < 5:
        stats.skipped_examples.append(f"Skipping {rct_file.name}:{row_idx}: {reason}")


def _part_file(path: Path) -> Path:
    return path.with_name(f"{path.name}.part")


def _split_legacy_smiles_target(line: str, path: Path, row_idx: int) -> tuple[str, str]:
    try:
        smiles, target = line.rsplit(",", 1)
    except ValueError as exc:
        raise ValueError(f"{path.name} row {row_idx} must contain 'smiles,target'") from exc
    if not smiles or not target:
        raise ValueError(f"{path.name} row {row_idx} contains an empty SMILES or target field")
    validate_legacy_target(target)
    return smiles, target


def _midvalid_triplet_exists(rct_valid_file: Path, pdt_valid_file: Path, mid_file: Path) -> bool:
    if not (rct_valid_file.is_file() and pdt_valid_file.is_file() and mid_file.is_file()):
        return False
    counts = (_count_lines(rct_valid_file), _count_lines(pdt_valid_file), _count_lines(mid_file))
    return counts[0] == counts[1] == counts[2] and counts[2] > 0


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _line in handle)


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
    parser.add_argument(
        "--config", "--config_json", dest="config_path", type=str, default="./config/pretrain_parameters.json"
    )
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
