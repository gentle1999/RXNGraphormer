"""Backward-compatible aggregate for legacy dataset classes.

Concrete implementations live in focused dataset modules. This import path is
kept for existing callers during the compatibility window.
"""

from .multi_reaction_dataset import MultiRXNDataset
from .pairing import PairDataset, TripleDataset
from .reaction_dataset import RXNDataset
from .sequence_dataset import RXNG2SDataset

__all__ = [
    "MultiRXNDataset",
    "PairDataset",
    "RXNDataset",
    "RXNG2SDataset",
    "TripleDataset",
]
