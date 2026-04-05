# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Python wrapper for the C++ CTC greedy search core."""

from __future__ import annotations

import importlib
from typing import List, Optional, Union

import torch

from .options import CtcGreedySearchOptions
from .search_interface import SearchInterface


def _get_core():
    """Lazily import and return the C++ _GreedySearchCore class."""
    _C = importlib.import_module("oasr._C")
    return _C.decoder._GreedySearchCore


class CtcGreedySearch(SearchInterface):
    """CTC greedy (best-path) decoder.

    At each timestep takes the most likely token (argmax), then collapses
    consecutive duplicates and removes blanks.  Supports offline and streaming
    (chunk-based) decoding.

    Args:
        opts_or_blank: Either a :class:`CtcGreedySearchOptions` dataclass or an
            integer blank token id.  When omitted, the ``blank`` keyword is used.
        blank: Blank token id (default 0).  Only used when *opts_or_blank* is
            not provided.

    Example::

        decoder = CtcGreedySearch(blank=0)
        decoder.search(logp)           # logp: Tensor [T, V]
        decoder.finalize_search()
        print(decoder.outputs[0])      # best token sequence
    """

    def __init__(
        self,
        opts_or_blank: Optional[Union[CtcGreedySearchOptions, int]] = None,
        *,
        blank: int = 0,
    ) -> None:
        if isinstance(opts_or_blank, CtcGreedySearchOptions):
            opts = opts_or_blank
        elif isinstance(opts_or_blank, int):
            opts = CtcGreedySearchOptions(blank=opts_or_blank)
        elif opts_or_blank is None:
            opts = CtcGreedySearchOptions(blank=blank)
        else:
            raise TypeError(
                f"Expected CtcGreedySearchOptions or int, got {type(opts_or_blank).__name__}"
            )
        self._core = _get_core()(blank=opts.blank)

    def search(self, logp: torch.Tensor) -> None:
        """Run CTC greedy search on a chunk of log-probability frames.

        Args:
            logp: Log-probability tensor of shape ``[T, V]``.
        """
        if logp.dim() != 2:
            raise ValueError(f"logp must be a 2-D tensor [T, V], got {logp.dim()}-D.")
        self._core.search(logp)

    def reset(self) -> None:
        """Reset internal decoder state."""
        self._core.reset()

    def finalize_search(self) -> None:
        """Finalize search (no-op for greedy; results are already final)."""
        self._core.finalize_search()

    def set_context_graph(self, context_graph) -> None:
        """Attach a ContextGraph for phrase boosting.

        Args:
            context_graph: A C++ :class:`ContextGraph` instance, or ``None``
                to disable phrase boosting.
        """
        self._core.set_context_graph(context_graph)

    @property
    def outputs(self) -> List[List[int]]:
        """Best output token ID sequence (size-1 N-best list)."""
        return self._core.outputs

    @property
    def likelihood(self) -> List[float]:
        """Cumulative log-likelihood of the best path (including context bonus)."""
        return self._core.likelihood

    @property
    def times(self) -> List[List[int]]:
        """Per-token frame timestamps (size-1 N-best list)."""
        return self._core.times
