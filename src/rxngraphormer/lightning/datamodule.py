from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch

from ..config_utils import as_bool
from ..data.collate import pair_collate_fn, triple_collate_fn
from ..data.loader import DataLoaderSettings, dataloader_kwargs, dataloader_settings_from_config
from ..data.multi_reaction_dataset import MultiRXNDataset
from ..data.pairing import PairDataset, TripleDataset
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
        self.use_mid_inf = self.task == "regression" and as_bool(config.model.use_mid_inf)

    def setup(self, stage: str | None = None) -> None:
        if self.task == "classification":
            self._setup_classification()
            return
        if self.task != "regression":
            raise NotImplementedError("RXNGraphormerDataModule supports regression and classification tasks")
        self._setup_regression()

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
            self.train_dataset = self._make_dataset(rct[split["train"]], pdt[split["train"]], None if mid is None else mid[split["train"]])
            self.valid_dataset = self._make_dataset(rct[split["valid"]], pdt[split["valid"]], None if mid is None else mid[split["valid"]])
            test_idx = split["test"] if len(split["test"]) > 10 else split["valid"]
            self.test_dataset = self._make_dataset(rct[test_idx], pdt[test_idx], None if mid is None else mid[test_idx])
        else:
            self.train_dataset = self._make_dataset(
                RXNDataset(root=data_path, name=self.config.data.train_rct_data_file, trunck=trunck, **preprocess_kwargs),
                RXNDataset(root=data_path, name=self.config.data.train_pdt_data_file, trunck=trunck, **preprocess_kwargs),
                (
                    RXNDataset(root=data_path, name=self.config.data.train_mid_data_file, trunck=trunck, **preprocess_kwargs)
                    if self.use_mid_inf
                    else None
                ),
            )
            self.valid_dataset = self._make_dataset(
                RXNDataset(root=data_path, name=self.config.data.val_rct_data_file, trunck=trunck, **preprocess_kwargs),
                RXNDataset(root=data_path, name=self.config.data.val_pdt_data_file, trunck=trunck, **preprocess_kwargs),
                (
                    RXNDataset(root=data_path, name=self.config.data.val_mid_data_file, trunck=trunck, **preprocess_kwargs)
                    if self.use_mid_inf
                    else None
                ),
            )
            self.test_dataset = self._make_dataset(
                RXNDataset(root=data_path, name=self.config.data.test_rct_data_file, trunck=trunck, **preprocess_kwargs),
                RXNDataset(root=data_path, name=self.config.data.test_pdt_data_file, trunck=trunck, **preprocess_kwargs),
                (
                    RXNDataset(root=data_path, name=self.config.data.test_mid_data_file, trunck=trunck, **preprocess_kwargs)
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

    def _setup_classification(self) -> None:
        data_path = self.config.data.data_path
        trunck = self.config.data.data_trunck
        preprocess_kwargs = reaction_preprocessing_kwargs_from_config(self.config)
        if self.config.data.train_rct_data_file:
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
            )
            test_rct_file = self.config.data.test_rct_data_file or self.config.data.val_rct_data_file
            test_pdt_file = self.config.data.test_pdt_data_file or self.config.data.val_pdt_data_file
            self.test_dataset = self._make_dataset(
                RXNDataset(root=data_path, name=test_rct_file, trunck=trunck, task="classification", **preprocess_kwargs),
                RXNDataset(root=data_path, name=test_pdt_file, trunck=trunck, task="classification", **preprocess_kwargs),
            )
            return

        rct_name = self.config.data.rct_name_regrex or self.config.data.rct_data_file
        pdt_name = self.config.data.pdt_name_regrex or self.config.data.pdt_data_file
        if not rct_name or not pdt_name:
            raise ValueError("Classification data requires rct_name_regrex/pdt_name_regrex or rct_data_file/pdt_data_file")
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
        if len(rct) != len(pdt):
            raise ValueError("The number of reactant and product classification data are not equal")
        split = self._load_or_make_split(len(rct))
        self.train_dataset = self._make_dataset(rct[split["train"]], pdt[split["train"]])
        self.valid_dataset = self._make_dataset(rct[split["valid"]], pdt[split["valid"]])
        test_idx = split["test"] if len(split["test"]) > 10 else split["valid"]
        self.test_dataset = self._make_dataset(rct[test_idx], pdt[test_idx])

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
        return torch.utils.data.DataLoader(
            dataset,
            **dataloader_kwargs(
                settings,
                shuffle=shuffle,
                collate_fn=triple_collate_fn if self.use_mid_inf else pair_collate_fn,
                drop_last=drop_last,
            ),
        )
