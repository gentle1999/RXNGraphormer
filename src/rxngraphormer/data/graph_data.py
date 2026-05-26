"""PyG data containers with RXNGraphormer-specific batching semantics."""

from __future__ import annotations

from typing import Any

from torch_geometric.data import Data
from torch_geometric.data.data import BaseData


class ReactionGraphData(Data):
    """Graph data item whose ``mol_index`` is data, not a graph index edge list.

    PyG treats attribute names containing ``index`` as index-like by default and
    offsets them during ``Batch.from_data_list``. RXNGraphormer ``mol_index`` is
    an atom-to-molecule label within each reaction, so it must be concatenated
    without PyG's automatic increment.
    """

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "mol_index":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "mol_index":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def as_reaction_graph_data(data: BaseData) -> ReactionGraphData:
    if isinstance(data, ReactionGraphData):
        return data
    graph_data = ReactionGraphData()
    for key in data.keys():
        graph_data[key] = data[key]
    return graph_data


__all__ = ["ReactionGraphData", "as_reaction_graph_data"]
