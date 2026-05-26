"""Multi-file reaction PyG dataset."""

import os
from dataclasses import dataclass, replace
from multiprocessing import Pool, RLock

from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from ..serialization import (
    existing_processed_file_name,
    load_processed_graph_data,
    processed_file_exists,
    save_processed_graph_data,
)
from .files import (
    _collated_data_count,
    _matched_raw_files,
    _separate_collated_data,
)
from .graph_data import as_reaction_graph_data
from .reaction_processing import PreprocessParallelMode, ReactionProcessingSettings, process_reaction_lines


@dataclass(frozen=True)
class _MultiReactionFileTask:
    index: int
    progress_position: int
    raw_data_file: str
    processed_path: str
    trunck: int | None
    settings: ReactionProcessingSettings


def _process_reaction_file(task: _MultiReactionFileTask) -> tuple[int, str, int]:
    if processed_file_exists(task.processed_path):
        return task.index, task.processed_path, 0

    print(f"[INFO] {task.raw_data_file} is processing...")
    with open(task.raw_data_file, encoding="utf-8") as fh:
        rxn_smi_tgt_lst = [line.strip() for line in fh.readlines()]
    if task.trunck is not None:
        rxn_smi_tgt_lst = rxn_smi_tgt_lst[: task.trunck]

    settings = replace(task.settings, multi_process=False, num_workers=1)
    data_list = process_reaction_lines(
        rxn_smi_tgt_lst,
        settings,
        desc=os.path.basename(task.raw_data_file),
        show_progress=True,
        progress_position=task.progress_position,
        progress_leave=True,
    )
    if not data_list:
        raise ValueError(f"No valid reaction graph data was generated from {task.raw_data_file}")

    data, slices = InMemoryDataset.collate(data_list)
    save_processed_graph_data(data, slices, task.processed_path)
    return task.index, task.processed_path, len(data_list)


class MultiRXNDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        name_regrex="pretrain_rxn_dataset_test_0_*.csv",
        raw_data=[],
        transform=None,
        pre_transform=None,
        trunck=None,
        file_num_trunck=0,
        task="regression",
        num_worker=8,
        batch_size=128,
        multi_process=False,
        ext_feat=False,
        ext_feat_type="Morgan",
        ext_feat_param={"radius": 2, "nBits": 2048, "useChirality": True},
        oh=False,
        mul_ext_readout="mean",
        name_tag="",
        start_idx=None,
        end_idx=None,
        parallel_mode: PreprocessParallelMode = "reaction",
    ):
        """
        Multi-process is useless in this procedure
        """

        self.name_regrex = name_regrex
        # print(glob.glob(f"{root}/{self.name_regrex}"))
        self.raw_data_files = _matched_raw_files(root, self.name_regrex)
        print(f"[INFO] There are {len(self.raw_data_files)} data files in total")
        self.raw_data = raw_data
        self.trunck = trunck if trunck is not None and trunck != 0 else None
        self.file_num_trunck = file_num_trunck
        if self.file_num_trunck != 0:
            self.raw_data_files = self.raw_data_files[: self.file_num_trunck]
            print(f"[INFO] {self.file_num_trunck} data files will be used")
        else:
            print(f"[INFO] All data {len(self.raw_data_files)} files will be used")
        if start_idx is not None and end_idx is not None:
            self.raw_data_files = self.raw_data_files[start_idx:end_idx]
            print(
                f"[INFO] {len(self.raw_data_files)} (from {start_idx} to {end_idx}) data files will be used"
            )
        self.task = task
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.multi_process = multi_process
        self.oh = oh
        self.ext_feat = ext_feat
        self.ext_feat_type = ext_feat_type.lower()
        self.ext_feat_param = ext_feat_param
        self.mul_ext_readout = mul_ext_readout.lower()
        self.name_tag = name_tag
        if parallel_mode not in {"reaction", "file"}:
            raise ValueError("parallel_mode must be 'reaction' or 'file'")
        self.parallel_mode = parallel_mode
        super().__init__(root, transform, pre_transform)
        self.data_lst = []
        self.slices_lst = []
        self.data_num_lst = [0]
        for processed_path in self.processed_paths:
            data, slices = load_processed_graph_data(processed_path)
            self.data_lst.append(as_reaction_graph_data(data))
            self.slices_lst.append(slices)
            self.data_num_lst.append(
                self.data_num_lst[-1] + _collated_data_count(slices)
            )
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_num_lst = self.data_num_lst[1:]

    @property
    def raw_file_names(self):
        return [os.path.basename(file) for file in self.raw_data_files]

    @property
    def processed_file_names(self):
        if not self.ext_feat:
            names = [
                f"{os.path.basename(file).split('.')[0]}_{self.trunck}_{self.name_tag}.safetensors"
                for file in self.raw_data_files
            ]
        else:
            names = [
                f"{os.path.basename(file).split('.')[0]}_{self.trunck}_{self.ext_feat_type}_{self.mul_ext_readout}_{self.name_tag}.safetensors"
                for file in self.raw_data_files
            ]
        return [existing_processed_file_name(self.processed_dir, name) for name in names]

    def process(self):
        if self.multi_process and self.parallel_mode == "file" and self.num_worker > 1:
            self._process_files_in_parallel()
            return

        for idx, raw_data_f in enumerate(self.raw_data_files):
            if processed_file_exists(self.processed_paths[idx]):
                print(f"[INFO] {self.processed_paths[idx]} already exists, skip it...")
                continue
            print(f"[INFO] {raw_data_f} is processing...")
            data_list = []
            with open(raw_data_f, encoding="utf-8") as f:
                rxn_smi_tgt_lst = [line.strip() for line in f.readlines()]
            if self.trunck is not None:
                rxn_smi_tgt_lst = rxn_smi_tgt_lst[: self.trunck]

            data_list = process_reaction_lines(
                rxn_smi_tgt_lst,
                self._reaction_processing_settings(multi_process=self.multi_process, num_workers=self.num_worker),
                desc=os.path.basename(raw_data_f),
            )
            if not data_list:
                raise ValueError(
                    f"No valid reaction graph data was generated from {raw_data_f}"
                )
            data, slices = self.collate(data_list)
            print(f"[INFO] {len(data_list)} data index {idx} is saving...")
            save_processed_graph_data(data, slices, self.processed_paths[idx])

    def _process_files_in_parallel(self) -> None:
        tasks: list[_MultiReactionFileTask] = []
        for idx, raw_data_f in enumerate(self.raw_data_files):
            if processed_file_exists(self.processed_paths[idx]):
                continue
            tasks.append(
                _MultiReactionFileTask(
                    index=idx,
                    progress_position=len(tasks),
                    raw_data_file=raw_data_f,
                    processed_path=self.processed_paths[idx],
                    trunck=self.trunck,
                    settings=self._reaction_processing_settings(multi_process=False, num_workers=1),
                )
            )
        if not tasks:
            return

        workers = min(max(1, int(self.num_worker)), len(tasks))
        print(f"[INFO] {workers} file workers are used to process {len(tasks)} data files with per-file progress bars...")
        with Pool(processes=workers, initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
            iterator = pool.imap_unordered(_process_reaction_file, tasks)
            for idx, processed_path, data_count in iterator:
                if data_count == 0:
                    tqdm.write(f"[INFO] {processed_path} already exists, skip it...")
                else:
                    tqdm.write(f"[INFO] {data_count} data index {idx} is saved to {processed_path}")

    def _reaction_processing_settings(self, *, multi_process: bool, num_workers: int) -> ReactionProcessingSettings:
        return ReactionProcessingSettings(
            task=self.task,
            ext_feat=self.ext_feat,
            ext_feat_type=self.ext_feat_type,
            ext_feat_param=self.ext_feat_param,
            mul_ext_readout=self.mul_ext_readout,
            multi_process=multi_process,
            num_workers=num_workers,
            batch_size=self.batch_size,
            include_atom_mass=False,
            raise_errors=False,
            log_invalid=True,
        )

    def len(self):
        ct = 0
        for slices in self.slices_lst:
            ct += _collated_data_count(slices)
        return ct

    def get(self, idx):
        blk_i = 0
        for blk_i, data_num in enumerate(self.data_num_lst):
            if idx < data_num:
                break
        else:
            raise IndexError(idx)
        data_num_lst = [0] + self.data_num_lst
        idx -= data_num_lst[blk_i]
        return _separate_collated_data(
            self.data_lst[blk_i], self.slices_lst[blk_i], idx
        )

    def download(self):
        pass


__all__ = ["MultiRXNDataset"]
