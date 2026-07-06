from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, Subset

from ..config_utils import as_bool
from ..data.collate import identity_collate, pair_collate_fn, triple_collate_fn
from ..data.loader import DataLoaderSettings, dataloader_kwargs, dataloader_settings_from_config
from ..data.multi_reaction_dataset import MultiRXNDataset
from ..data.pairing import FastBatchPairDataset, FastBatchTripleDataset, PairDataset, TripleDataset
from ..data.reaction_dataset import RXNDataset
from ..data.reaction_processing import reaction_preprocessing_kwargs_from_config
from ..data.splits import get_idx_split


class RXNGraphormerDataModule(pl.LightningDataModule):
    """Lightning DataModule for legacy RXNDataset pair/triple regression data."""

    def __init__(
        self,
        config: Any,
        *,
        split_manifest: str | Path | None = None,
        dataloader: DataLoaderSettings | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.task = getattr(config, "task", "regression")
        self.split_manifest = Path(split_manifest) if split_manifest else None
        self.loader_settings = dataloader or dataloader_settings_from_config(config)
        self.use_mid_inf = self.task in {"regression", "classification"} and as_bool(config.model.use_mid_inf)

    def setup(self, stage: str | None = None) -> None:
        if self.task == "classification":
            self._setup_classification()
            self._preload_graph_cache_if_enabled()
            return
        if self.task != "regression":
            raise NotImplementedError("RXNGraphormerDataModule supports regression and classification tasks")
        self._setup_regression()
        self._preload_graph_cache_if_enabled()

    def _setup_regression(self) -> None:
        data_path = self.config.data.data_path
        trunck = self.config.data.data_trunck
        preprocess_kwargs = reaction_preprocessing_kwargs_from_config(self.config)
        if self.config.data.rct_data_file:
            rct = RXNDataset(root=data_path, name=self.config.data.rct_data_file, trunck=trunck, **preprocess_kwargs)
            pdt = RXNDataset(root=data_path, name=self.config.data.pdt_data_file, trunck=trunck, **preprocess_kwargs)
            mid = (
                RXNDataset(root=data_path, name=self.config.data.mid_data_file, trunck=trunck, **preprocess_kwargs)
                if self.use_mid_inf
                else None
            )
            split = self._load_or_make_split(len(rct))
            self.train_dataset = self._make_dataset(
                rct[split["train"]], pdt[split["train"]], None if mid is None else mid[split["train"]]
            )
            self.valid_dataset = self._make_dataset(
                rct[split["valid"]], pdt[split["valid"]], None if mid is None else mid[split["valid"]]
            )
            test_idx = split["test"] if len(split["test"]) > 10 else split["valid"]
            self.test_dataset = self._make_dataset(rct[test_idx], pdt[test_idx], None if mid is None else mid[test_idx])
        else:
            self.train_dataset = self._make_dataset(
                RXNDataset(
                    root=data_path, name=self.config.data.train_rct_data_file, trunck=trunck, **preprocess_kwargs
                ),
                RXNDataset(
                    root=data_path, name=self.config.data.train_pdt_data_file, trunck=trunck, **preprocess_kwargs
                ),
                (
                    RXNDataset(
                        root=data_path, name=self.config.data.train_mid_data_file, trunck=trunck, **preprocess_kwargs
                    )
                    if self.use_mid_inf
                    else None
                ),
            )
            self.valid_dataset = self._make_dataset(
                RXNDataset(root=data_path, name=self.config.data.val_rct_data_file, trunck=trunck, **preprocess_kwargs),
                RXNDataset(root=data_path, name=self.config.data.val_pdt_data_file, trunck=trunck, **preprocess_kwargs),
                (
                    RXNDataset(
                        root=data_path, name=self.config.data.val_mid_data_file, trunck=trunck, **preprocess_kwargs
                    )
                    if self.use_mid_inf
                    else None
                ),
            )
            self.test_dataset = self._make_dataset(
                RXNDataset(
                    root=data_path, name=self.config.data.test_rct_data_file, trunck=trunck, **preprocess_kwargs
                ),
                RXNDataset(
                    root=data_path, name=self.config.data.test_pdt_data_file, trunck=trunck, **preprocess_kwargs
                ),
                (
                    RXNDataset(
                        root=data_path, name=self.config.data.test_mid_data_file, trunck=trunck, **preprocess_kwargs
                    )
                    if self.use_mid_inf
                    else None
                ),
            )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True, drop_last=self.loader_settings.train_drop_last)

    def val_dataloader(self):
        return self._loader(self.valid_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False, drop_last=False)

    def _make_dataset(self, rct, pdt, mid=None):
        if mid is None:
            return PairDataset(rct, pdt)
        return TripleDataset(rct, pdt, mid)

    def _preload_graph_cache_if_enabled(self) -> None:
        if not as_bool(getattr(self.config.data, "preload_graph_cache", False)):
            return
        for dataset in (self.train_dataset, self.valid_dataset, self.test_dataset):
            self._preload_graph_cache(dataset)

    def _preload_graph_cache(self, dataset: Any) -> None:
        if isinstance(dataset, PairDataset):
            self._preload_graph_cache(dataset.rct_dataset)
            self._preload_graph_cache(dataset.pdt_dataset)
            return
        if isinstance(dataset, TripleDataset):
            self._preload_graph_cache(dataset.rct_dataset)
            self._preload_graph_cache(dataset.pdt_dataset)
            self._preload_graph_cache(dataset.mid_dataset)
            return
        if isinstance(dataset, Subset):
            self._preload_indexed_graph_cache(dataset.dataset, dataset.indices)
            return

        indices = getattr(dataset, "indices", None)
        if callable(indices) and hasattr(dataset, "get"):
            self._preload_indexed_graph_cache(dataset, indices())
            return
        for idx in range(len(dataset)):
            dataset[idx]

    def _preload_indexed_graph_cache(self, dataset: Any, indices: Any) -> None:
        index_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
        get = getattr(dataset, "get", None)
        if callable(get):
            for idx in index_list:
                get(int(idx))
            return
        for idx in index_list:
            dataset[int(idx)]

    def _setup_classification(self) -> None:
        data_path = self.config.data.data_path
        trunck = self.config.data.data_trunck
        preprocess_kwargs = reaction_preprocessing_kwargs_from_config(self.config)
        if self.config.data.train_rct_data_file:
            train_mid = (
                RXNDataset(
                    root=data_path,
                    name=self.config.data.train_mid_data_file,
                    trunck=trunck,
                    task="classification",
                    **preprocess_kwargs,
                )
                if self.use_mid_inf
                else None
            )
            self.train_dataset = self._make_dataset(
                RXNDataset(
                    root=data_path,
                    name=self.config.data.train_rct_data_file,
                    trunck=trunck,
                    task="classification",
                    **preprocess_kwargs,
                ),
                RXNDataset(
                    root=data_path,
                    name=self.config.data.train_pdt_data_file,
                    trunck=trunck,
                    task="classification",
                    **preprocess_kwargs,
                ),
                train_mid,
            )
            valid_mid = (
                RXNDataset(
                    root=data_path,
                    name=self.config.data.val_mid_data_file,
                    trunck=trunck,
                    task="classification",
                    **preprocess_kwargs,
                )
                if self.use_mid_inf
                else None
            )
            self.valid_dataset = self._make_dataset(
                RXNDataset(
                    root=data_path,
                    name=self.config.data.val_rct_data_file,
                    trunck=trunck,
                    task="classification",
                    **preprocess_kwargs,
                ),
                RXNDataset(
                    root=data_path,
                    name=self.config.data.val_pdt_data_file,
                    trunck=trunck,
                    task="classification",
                    **preprocess_kwargs,
                ),
                valid_mid,
            )
            test_rct_file = self.config.data.test_rct_data_file or self.config.data.val_rct_data_file
            test_pdt_file = self.config.data.test_pdt_data_file or self.config.data.val_pdt_data_file
            test_mid_file = self.config.data.test_mid_data_file or self.config.data.val_mid_data_file
            self.test_dataset = self._make_dataset(
                RXNDataset(
                    root=data_path, name=test_rct_file, trunck=trunck, task="classification", **preprocess_kwargs
                ),
                RXNDataset(
                    root=data_path, name=test_pdt_file, trunck=trunck, task="classification", **preprocess_kwargs
                ),
                (
                    RXNDataset(
                        root=data_path, name=test_mid_file, trunck=trunck, task="classification", **preprocess_kwargs
                    )
                    if self.use_mid_inf
                    else None
                ),
            )
            return

        rct_name = self.config.data.rct_name_regrex or self.config.data.rct_data_file
        pdt_name = self.config.data.pdt_name_regrex or self.config.data.pdt_data_file
        mid_name = getattr(self.config.data, "mid_name_regrex", "") or getattr(self.config.data, "mid_data_file", "")
        if not rct_name or not pdt_name:
            raise ValueError(
                "Classification data requires rct_name_regrex/pdt_name_regrex or rct_data_file/pdt_data_file"
            )
        if self.use_mid_inf and not mid_name:
            raise ValueError("Classification use_mid_inf=True requires mid_name_regrex or mid_data_file")
        rct = MultiRXNDataset(
            root=data_path,
            name_regrex=rct_name,
            trunck=trunck,
            task="classification",
            file_num_trunck=self.config.data.file_num_trunck,
            name_tag="rct",
            **preprocess_kwargs,
        )
        pdt = MultiRXNDataset(
            root=data_path,
            name_regrex=pdt_name,
            trunck=trunck,
            task="classification",
            file_num_trunck=self.config.data.file_num_trunck,
            name_tag="pdt",
            **preprocess_kwargs,
        )
        mid = (
            MultiRXNDataset(
                root=data_path,
                name_regrex=mid_name,
                trunck=trunck,
                task="classification",
                file_num_trunck=self.config.data.file_num_trunck,
                name_tag="mid",
                **preprocess_kwargs,
            )
            if self.use_mid_inf
            else None
        )
        if len(rct) != len(pdt) or (mid is not None and len(rct) != len(mid)):
            raise ValueError("The number of reactant, product, and mid classification data are not equal")
        split = self._load_or_make_split(len(rct))
        self.train_dataset = self._make_dataset(
            rct[split["train"]], pdt[split["train"]], None if mid is None else mid[split["train"]]
        )
        self.valid_dataset = self._make_dataset(
            rct[split["valid"]], pdt[split["valid"]], None if mid is None else mid[split["valid"]]
        )
        test_idx = split["test"] if len(split["test"]) > 10 else split["valid"]
        self.test_dataset = self._make_dataset(rct[test_idx], pdt[test_idx], None if mid is None else mid[test_idx])

    def _load_or_make_split(self, data_size: int) -> dict[str, torch.Tensor]:
        if self.split_manifest is None:
            return get_idx_split(
                data_size,
                int(self.config.data.train_ratio * data_size),
                int(self.config.data.valid_ratio * data_size),
                self.config.data.seed,
            )
        manifest = json.loads(self.split_manifest.read_text())
        indices = manifest.get("indices", manifest)
        return {
            name: torch.as_tensor(values, dtype=torch.long)
            for name, values in indices.items()
            if name in {"train", "valid", "test"}
        }

    def _loader(self, dataset, *, shuffle: bool, drop_last: bool):
        settings = self.loader_settings
        if as_bool(getattr(self.config.data, "fast_batch_collate", False)):
            return self._fast_loader(dataset, shuffle=shuffle, drop_last=drop_last)
        return torch.utils.data.DataLoader(
            dataset,
            **dataloader_kwargs(
                settings,
                shuffle=shuffle,
                collate_fn=triple_collate_fn if self.use_mid_inf else pair_collate_fn,
                drop_last=drop_last,
            ),
        )

    def _fast_loader(self, dataset, *, shuffle: bool, drop_last: bool):
        if isinstance(dataset, TripleDataset):
            fast_dataset = FastBatchTripleDataset(dataset)
        elif isinstance(dataset, PairDataset):
            fast_dataset = FastBatchPairDataset(dataset)
        else:
            raise TypeError(f"fast_batch_collate only supports PairDataset/TripleDataset, got {type(dataset).__name__}")
        sampler = RandomSampler(fast_dataset) if shuffle else SequentialSampler(fast_dataset)
        batch_sampler = BatchSampler(
            sampler,
            batch_size=self.loader_settings.batch_size,
            drop_last=drop_last,
        )
        return DataLoader(
            fast_dataset,
            sampler=batch_sampler,
            batch_size=None,
            num_workers=self.loader_settings.num_workers,
            pin_memory=self.loader_settings.pin_memory,
            persistent_workers=self.loader_settings.persistent_workers and self.loader_settings.num_workers > 0,
            prefetch_factor=self.loader_settings.prefetch_factor if self.loader_settings.num_workers > 0 else None,
            collate_fn=identity_collate,
        )
