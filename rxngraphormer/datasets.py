"""Public dataset API.

The concrete legacy dataset implementations remain import-compatible from
``rxngraphormer.data``.  New code should type against ``data_protocol`` and use
these exports when it needs the existing PyG-backed datasets.
"""

from .data import (
    MultiRXNDataset,
    PairDataset,
    RXNDataset,
    RXNG2SDataset,
    TripleDataset,
    get_idx_split,
    load_vocab,
    pair_collate_fn,
    single_collate_fn,
    triple_collate_fn,
)
from .data_protocol import ReactionDatasetProtocol, ReactionDatasetSpec, validate_graph_data

__all__ = [
    "MultiRXNDataset",
    "PairDataset",
    "RXNDataset",
    "RXNG2SDataset",
    "TripleDataset",
    "ReactionDatasetProtocol",
    "ReactionDatasetSpec",
    "get_idx_split",
    "load_vocab",
    "pair_collate_fn",
    "single_collate_fn",
    "triple_collate_fn",
    "validate_graph_data",
]
