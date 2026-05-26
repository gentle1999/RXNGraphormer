"""Backward-compatible aggregate for the legacy data API.

Implementation lives in focused `rxngraphormer.data.*` modules. This module is
kept only for existing imports during the compatibility window.
"""

from .collate import *  # noqa: F401,F403
from .collate import __all__ as _collate_all
from .constants import *  # noqa: F401,F403
from .constants import __all__ as _constants_all
from .datasets import *  # noqa: F401,F403
from .datasets import __all__ as _datasets_all
from .files import *  # noqa: F401,F403
from .files import __all__ as _files_all
from .graph import *  # noqa: F401,F403
from .graph import __all__ as _graph_all
from .reaction_graph import *  # noqa: F401,F403
from .reaction_graph import __all__ as _reaction_graph_all
from .splits import *  # noqa: F401,F403
from .splits import __all__ as _splits_all
from .tokenization import *  # noqa: F401,F403
from .tokenization import __all__ as _tokenization_all

__all__ = [
    *_collate_all,
    *_constants_all,
    *_datasets_all,
    *_files_all,
    *_graph_all,
    *_reaction_graph_all,
    *_splits_all,
    *_tokenization_all,
]
