# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Python wrapper for the C++ CTC prefix beam search core."""

from __future__ import annotations

import importlib
from typing import List, Optional, Union

import torch

from .options import CtcPrefixBeamSearchOptions
from .search_interface import SearchInterface


def _get_core():
    """Lazily import and return the C++ _PrefixBeamSearchCore class."""
    _C = importlib.import_module("oasr._C")
    return _C.decoder._PrefixBeamSearchCore


class CtcPrefixBeamSearch(SearchInterface):
    """CTC prefix beam search decoder.

    Maintains a beam of candidate prefix sequences, scoring them with the
    forward CTC probability.  Supports both offline and chunk-based streaming
    decoding, and optional phrase boosting via a :class:`ContextGraph`.

    Args:
        opts_or_blank: Either a :class:`CtcPrefixBeamSearchOptions` dataclass
            or an integer blank token id.  When omitted, keyword arguments
            ``blank``, ``first_beam_size``, and ``second_beam_size`` are used.
        blank: CTC blank token id (default 0).
        first_beam_size: Top-k vocabulary pruning per frame (default 10).
        second_beam_size: Hypothesis beam width (default 10).

    Example::

        decoder = CtcPrefixBeamSearch(blank=0, first_beam_size=10, second_beam_size=10)
        decoder.search(logp)           # logp: Tensor [T, V]
        decoder.finalize_search()
        print(decoder.outputs[0])      # best token sequence
    """

    def __init__(
        self,
        opts_or_blank: Optional[Union[CtcPrefixBeamSearchOptions, int]] = None,
        *,
        blank: int = 0,
        first_beam_size: int = 10,
        second_beam_size: int = 10,
    ) -> None:
        if isinstance(opts_or_blank, CtcPrefixBeamSearchOptions):
            opts = opts_or_blank
        elif isinstance(opts_or_blank, int):
            opts = CtcPrefixBeamSearchOptions(
                blank=opts_or_blank,
                first_beam_size=first_beam_size,
                second_beam_size=second_beam_size,
            )
        elif opts_or_blank is None:
            opts = CtcPrefixBeamSearchOptions(
                blank=blank,
                first_beam_size=first_beam_size,
                second_beam_size=second_beam_size,
            )
        else:
            raise TypeError(
                f"Expected CtcPrefixBeamSearchOptions or int, got {type(opts_or_blank).__name__}"
            )
        self._core = _get_core()(
            blank=opts.blank,
            first_beam_size=opts.first_beam_size,
            second_beam_size=opts.second_beam_size,
        )

    def search(self, logp: torch.Tensor) -> None:
        """Run one step of CTC prefix beam search on a chunk of log-probability frames.

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
        """Finalize beam search: apply context backoff and re-sort hypotheses."""
        self._core.finalize_search()

    def set_context_graph(self, context_graph) -> None:
        """Attach a ContextGraph for phrase boosting.

        Args:
            context_graph: A C++ :class:`ContextGraph` instance, or ``None``
                to disable phrase boosting.
        """
        self._core.set_context_graph(context_graph)

    @property
    def inputs(self) -> List[List[int]]:
        """N-best input token-ID sequences."""
        return self._core.inputs

    @property
    def outputs(self) -> List[List[int]]:
        """N-best output token-ID sequences."""
        return self._core.outputs

    @property
    def likelihood(self) -> List[float]:
        """Forward CTC log-likelihood scores for each hypothesis."""
        return self._core.likelihood

    @property
    def times(self) -> List[List[int]]:
        """Per-token Viterbi frame timestamps for each hypothesis."""
        return self._core.times

    @property
    def viterbi_likelihood(self) -> List[float]:
        """Viterbi path log-likelihood scores for each hypothesis."""
        return self._core.viterbi_likelihood
