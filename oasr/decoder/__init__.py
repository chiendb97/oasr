# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
CTC decoder package.

Provides Python wrapper classes backed by C++ compute cores, plus the
:class:`ContextGraph` phrase-boosting trie (C++ class, exposed via pybind11).

Public API::

    from oasr.decoder import (
        ContextGraph,
        CtcGreedySearch,    CtcGreedySearchOptions,
        CtcPrefixBeamSearch, CtcPrefixBeamSearchOptions,
        CtcWfstBeamSearch,  CtcWfstBeamSearchOptions,
        SearchType,
        k2_available,
    )
"""

from __future__ import annotations

import importlib as _importlib

from .search_interface import SearchInterface
from .options import (
    SearchType,
    CtcGreedySearchOptions,
    CtcPrefixBeamSearchOptions,
    CtcWfstBeamSearchOptions,
)
from .ctc_greedy_search import CtcGreedySearch
from .ctc_prefix_beam_search import CtcPrefixBeamSearch
from .ctc_wfst_beam_search import CtcWfstBeamSearch

# ---------------------------------------------------------------------------
# ContextGraph and k2_available come from the compiled C++ extension.
# ---------------------------------------------------------------------------

def _load_c_decoder():
    try:
        _C = _importlib.import_module("oasr._C")
        return _C.decoder
    except ImportError:
        return None


_decoder_c = _load_c_decoder()

if _decoder_c is not None:
    ContextGraph = _decoder_c.ContextGraph
    k2_available: bool = getattr(_decoder_c, "k2_available", False)
else:
    ContextGraph = None  # type: ignore[assignment,misc]
    k2_available: bool = False  # type: ignore[assignment]


__all__ = [
    "SearchInterface",
    "SearchType",
    "CtcGreedySearchOptions",
    "CtcPrefixBeamSearchOptions",
    "CtcWfstBeamSearchOptions",
    "CtcGreedySearch",
    "CtcPrefixBeamSearch",
    "CtcWfstBeamSearch",
    "ContextGraph",
    "k2_available",
]
