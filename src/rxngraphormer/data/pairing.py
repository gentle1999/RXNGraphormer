"""Dataset adapters for paired and triplet graph inputs."""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from torch.utils.data import Dataset

T_co = TypeVar("T_co", covariant=True)
U_co = TypeVar("U_co", covariant=True)
V_co = TypeVar("V_co", covariant=True)


class SizedDataset(Protocol[T_co]):
    def __getitem__(self, idx: int, /) -> T_co:
        ...

    def __len__(self) -> int:
        ...


class SliceableDataset(Protocol[T_co]):
    def __getitem__(self, index: object, /) -> T_co:
        ...

    def __len__(self) -> int:
        ...


class PairDataset(Dataset[tuple[T_co, U_co]], Generic[T_co, U_co]):
    def __init__(self, rct_dataset: SizedDataset[T_co], pdt_dataset: SizedDataset[U_co]) -> None:
        self.rct_dataset = rct_dataset
        self.pdt_dataset = pdt_dataset

    def __getitem__(self, index: int) -> tuple[T_co, U_co]:
        return self.rct_dataset[index], self.pdt_dataset[index]

    def __len__(self) -> int:
        return len(self.rct_dataset)


class TripleDataset(Dataset[tuple[T_co, U_co, V_co]], Generic[T_co, U_co, V_co]):
    def __init__(
        self,
        rct_dataset: SizedDataset[T_co],
        pdt_dataset: SizedDataset[U_co],
        mid_dataset: SizedDataset[V_co],
    ) -> None:
        self.rct_dataset = rct_dataset
        self.pdt_dataset = pdt_dataset
        self.mid_dataset = mid_dataset

    def __getitem__(self, index: int) -> tuple[T_co, U_co, V_co]:
        return self.rct_dataset[index], self.pdt_dataset[index], self.mid_dataset[index]

    def __len__(self) -> int:
        return len(self.rct_dataset)


__all__ = ["PairDataset", "SizedDataset", "SliceableDataset", "TripleDataset"]
