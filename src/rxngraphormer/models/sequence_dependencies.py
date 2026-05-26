from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

OnmtSymbols: TypeAlias = dict[str, Callable[..., Any]]
_ONMT_SYMBOLS: OnmtSymbols | None = None


def require_onmt() -> OnmtSymbols:
    global _ONMT_SYMBOLS
    if _ONMT_SYMBOLS is None:
        try:
            from onmt.decoders import TransformerDecoder
            from onmt.modules.embeddings import Embeddings, PositionalEncoding
            from onmt.modules.position_ffn import PositionwiseFeedForward
            from onmt.translate import BeamSearch, GNMTGlobalScorer, GreedySearch
            from onmt.utils.misc import sequence_mask
        except ImportError as exc:
            raise ImportError(
                "OpenNMT is required for sequence_generation models. "
                "Install the sequence extra with `uv sync --extra sequence`."
            ) from exc
        _ONMT_SYMBOLS = {
            "BeamSearch": BeamSearch,
            "Embeddings": Embeddings,
            "GNMTGlobalScorer": GNMTGlobalScorer,
            "GreedySearch": GreedySearch,
            "PositionalEncoding": PositionalEncoding,
            "PositionwiseFeedForward": PositionwiseFeedForward,
            "TransformerDecoder": TransformerDecoder,
            "sequence_mask": sequence_mask,
        }
    return _ONMT_SYMBOLS


_require_onmt = require_onmt
