"""Shared reaction graph preprocessing helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Literal, TypedDict, cast

from tqdm import tqdm

from .files import _ext_feat_tensor
from .graph_data import ReactionGraphData
from .reaction_graph import ExtFeatParams, PerformanceGraphInfo, get_rxn_pfm_info


@dataclass(frozen=True)
class ReactionProcessingSettings:
    task: str
    ext_feat: bool = False
    ext_feat_type: str = "morgan"
    ext_feat_param: ExtFeatParams | None = None
    mul_ext_readout: str = "mean"
    multi_process: bool = False
    num_workers: int = 8
    batch_size: int = 128
    include_atom_mass: bool = False
    raise_errors: bool = True
    log_invalid: bool = False


ReactionInfoTask = tuple[
    str,
    str,
    bool,
    str,
    dict[str, int | bool | float | str],
    str,
    bool,
    bool,
]
ReactionInfoResult = tuple[PerformanceGraphInfo | None, str | None]


def process_reaction_lines(
    lines: Sequence[str],
    settings: ReactionProcessingSettings,
    *,
    desc: str | None = None,
    show_progress: bool = True,
    progress_position: int | None = None,
    progress_leave: bool | None = None,
) -> list[ReactionGraphData]:
    if not lines:
        return []

    tasks = _reaction_info_tasks(lines, settings)
    total = len(lines)
    data_list: list[ReactionGraphData] = []
    if settings.multi_process and settings.num_workers > 1:
        print(f"[INFO] {settings.num_workers} workers are used to process data...")
        chunksize = max(1, int(settings.batch_size))
        with Pool(max(1, int(settings.num_workers))) as pool:
            iterator = pool.imap(_process_reaction_info_task, tasks, chunksize=chunksize)
            for result in tqdm(
                iterator,
                total=total,
                desc=desc,
                disable=not show_progress,
                position=progress_position,
                leave=progress_leave,
            ):
                _append_reaction_graph(data_list, result, settings)
    else:
        for task in tqdm(
            tasks,
            total=total,
            desc=desc,
            disable=not show_progress,
            position=progress_position,
            leave=progress_leave,
        ):
            _append_reaction_graph(data_list, _process_reaction_info_task(task), settings)
    return data_list


PreprocessParallelMode = Literal["reaction", "file"]


class ReactionPreprocessingKwargs(TypedDict):
    multi_process: bool
    num_worker: int
    batch_size: int
    parallel_mode: PreprocessParallelMode


def reaction_preprocessing_kwargs_from_config(config: object) -> ReactionPreprocessingKwargs:
    data = getattr(config, "data", config)
    parallel_mode = getattr(data, "preprocess_parallel_mode", "reaction")
    if parallel_mode not in {"reaction", "file"}:
        raise ValueError("preprocess_parallel_mode must be 'reaction' or 'file'")
    parallel_mode = cast(PreprocessParallelMode, parallel_mode)
    return {
        "multi_process": bool(getattr(data, "multi_process", False)),
        "num_worker": int(getattr(data, "preprocess_num_workers", 8)),
        "batch_size": int(getattr(data, "preprocess_batch_size", 128)),
        "parallel_mode": parallel_mode,
    }


def _reaction_info_tasks(lines: Sequence[str], settings: ReactionProcessingSettings) -> Iterable[ReactionInfoTask]:
    return (_reaction_info_task(line, settings) for line in lines)


def _reaction_info_task(line: str, settings: ReactionProcessingSettings) -> ReactionInfoTask:
    return (
        line,
        settings.task,
        settings.ext_feat,
        settings.ext_feat_type,
        dict(settings.ext_feat_param or {"radius": 2, "nBits": 2048, "useChirality": True}),
        settings.mul_ext_readout,
        settings.raise_errors,
        settings.log_invalid,
    )


def _process_reaction_info_task(task: ReactionInfoTask) -> ReactionInfoResult:
    line, task_name, ext_feat, ext_feat_type, ext_feat_param, mul_ext_readout, raise_errors, log_invalid = task
    try:
        rxn_inf = get_rxn_pfm_info((line, task_name, ext_feat, ext_feat_type, ext_feat_param, mul_ext_readout))
    except Exception:
        if raise_errors:
            raise
        return None, _error_message(line, task_name, ext_feat, ext_feat_type, ext_feat_param, mul_ext_readout)
    if rxn_inf is None and log_invalid:
        return None, _error_message(line, task_name, ext_feat, ext_feat_type, ext_feat_param, mul_ext_readout)
    return rxn_inf, None


def _append_reaction_graph(
    data_list: list[ReactionGraphData],
    result: ReactionInfoResult,
    settings: ReactionProcessingSettings,
) -> None:
    rxn_inf, error = result
    if error:
        print(error)
    if rxn_inf is None:
        return
    data_list.append(_reaction_graph_data_from_info(rxn_inf, include_atom_mass=settings.include_atom_mass))


def _reaction_graph_data_from_info(rxn_inf: PerformanceGraphInfo, *, include_atom_mass: bool) -> ReactionGraphData:
    (
        x_merge,
        edge_index_merge,
        edge_attr_merge,
        atom_mass_merge,
        _x_oh_merge,
        _edge_oh_attr_merge,
        _a_graphs_merge,
        _b_graphs_merge,
        mol_index,
        tgt_,
        ext_feat_desc,
    ) = rxn_inf
    data = ReactionGraphData(
        x=x_merge,
        edge_index=edge_index_merge,
        edge_attr=edge_attr_merge,
        mol_index=mol_index,
        y=tgt_,
        ext_feat=_ext_feat_tensor(ext_feat_desc),
    )
    if include_atom_mass:
        data.atom_mass = atom_mass_merge
    return data


def _error_message(
    line: str,
    task_name: str,
    ext_feat: bool,
    ext_feat_type: str,
    ext_feat_param: Mapping[str, int | bool | float | str],
    mul_ext_readout: str,
) -> str:
    return f"[ERROR] {line}, {task_name}, {ext_feat}, {ext_feat_type}, {ext_feat_param}, {mul_ext_readout}"


__all__ = [
    "ReactionProcessingSettings",
    "ReactionPreprocessingKwargs",
    "process_reaction_lines",
    "reaction_preprocessing_kwargs_from_config",
]
