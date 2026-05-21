from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    pl = None

from ..data import (
    PairDataset,
    RXNDataset,
    TripleDataset,
    get_idx_split,
    pair_collate_fn,
    triple_collate_fn,
)
from ..dataloader import DataLoaderSettings, dataloader_kwargs, dataloader_settings_from_config
from ..utils import as_bool


class RXNGraphormerDataModule(pl.LightningDataModule if pl is not None else object):
    """Lightning DataModule for legacy RXNDataset pair/triple regression data."""

    def __init__(
        self,
        config: Any,
        *,
        split_manifest: str | Path | None = None,
        dataloader: DataLoaderSettings | None = None,
    ) -> None:
        if pl is None:
            raise ImportError("lightning is required to use RXNGraphormerDataModule")
        super().__init__()
        self.config = config
        self.split_manifest = Path(split_manifest) if split_manifest else None
        self.loader_settings = dataloader or dataloader_settings_from_config(config)
        self.use_mid_inf = as_bool(config.model.use_mid_inf)

    def setup(self, stage: str | None = None) -> None:
        data_path = self.config.data.data_path
        trunck = self.config.data.data_trunck
        if self.config.data.rct_data_file:
            rct = RXNDataset(root=data_path, name=self.config.data.rct_data_file, trunck=trunck)
            pdt = RXNDataset(root=data_path, name=self.config.data.pdt_data_file, trunck=trunck)
            mid = RXNDataset(root=data_path, name=self.config.data.mid_data_file, trunck=trunck) if self.use_mid_inf else None
            split = self._load_or_make_split(len(rct))
            self.train_dataset = self._make_dataset(rct[split["train"]], pdt[split["train"]], None if mid is None else mid[split["train"]])
            self.valid_dataset = self._make_dataset(rct[split["valid"]], pdt[split["valid"]], None if mid is None else mid[split["valid"]])
            test_idx = split["test"] if len(split["test"]) > 10 else split["valid"]
            self.test_dataset = self._make_dataset(rct[test_idx], pdt[test_idx], None if mid is None else mid[test_idx])
        else:
            self.train_dataset = self._make_dataset(
                RXNDataset(root=data_path, name=self.config.data.train_rct_data_file, trunck=trunck),
                RXNDataset(root=data_path, name=self.config.data.train_pdt_data_file, trunck=trunck),
                RXNDataset(root=data_path, name=self.config.data.train_mid_data_file, trunck=trunck) if self.use_mid_inf else None,
            )
            self.valid_dataset = self._make_dataset(
                RXNDataset(root=data_path, name=self.config.data.val_rct_data_file, trunck=trunck),
                RXNDataset(root=data_path, name=self.config.data.val_pdt_data_file, trunck=trunck),
                RXNDataset(root=data_path, name=self.config.data.val_mid_data_file, trunck=trunck) if self.use_mid_inf else None,
            )
            self.test_dataset = self._make_dataset(
                RXNDataset(root=data_path, name=self.config.data.test_rct_data_file, trunck=trunck),
                RXNDataset(root=data_path, name=self.config.data.test_pdt_data_file, trunck=trunck),
                RXNDataset(root=data_path, name=self.config.data.test_mid_data_file, trunck=trunck) if self.use_mid_inf else None,
            )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.valid_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)

    def _make_dataset(self, rct, pdt, mid=None):
        if mid is None:
            return PairDataset(rct, pdt)
        return TripleDataset(rct, pdt, mid)

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

    def _loader(self, dataset, *, shuffle: bool):
        settings = self.loader_settings
        return torch.utils.data.DataLoader(
            dataset,
            **dataloader_kwargs(
                settings,
                shuffle=shuffle,
                collate_fn=triple_collate_fn if self.use_mid_inf else pair_collate_fn,
            ),
        )
