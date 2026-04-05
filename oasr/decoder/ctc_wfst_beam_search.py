# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Python wrapper for the C++ CTC WFST beam search core (requires K2)."""

from __future__ import annotations

import importlib
from typing import List, Optional

import torch

from .options import CtcWfstBeamSearchOptions
from .search_interface import SearchInterface


def _get_core():
    """Lazily import and return the C++ _WfstBeamSearchCore class."""
    _C = importlib.import_module("oasr._C")
    if not getattr(_C.decoder, "k2_available", False):
        raise RuntimeError(
            "K2 WFST decoder is not available. "
            "Rebuild with OASR_USE_K2=1 to enable it."
        )
    return _C.decoder._WfstBeamSearchCore


class CtcWfstBeamSearch(SearchInterface):
    """WFST-based CTC beam search decoder (requires K2).

    Accumulates log-probability frames during :meth:`search` calls, then runs
    the full K2 WFST intersection and N-best extraction in :meth:`finalize_search`.

    Use the :meth:`from_file` class method to construct an instance from a
    decoding graph saved with ``torch.save(fsa.as_dict(), path)``.

    .. note::
        This decoder requires the C++ extension to be built with K2 support
        (``OASR_USE_K2=1 pip install -e .``).

    Example::

        decoder = CtcWfstBeamSearch.from_file("graph.pt")
        decoder.search(logp)        # accumulates frames
        decoder.finalize_search()   # runs K2 decode
        print(decoder.outputs[0])
    """

    def __init__(self, core) -> None:
        """Internal constructor — use :meth:`from_file` instead."""
        self._core = core

    @classmethod
    def from_file(
        cls,
        fst_path: str,
        options: Optional[CtcWfstBeamSearchOptions] = None,
    ) -> "CtcWfstBeamSearch":
        """Load a decoding graph from *fst_path* and create a WFST decoder.

        Args:
            fst_path: Path to a file saved with
                ``torch.save(fsa.as_dict(), path, _use_new_zipfile_serialization=True)``.
            options: Decoder options.  Defaults to :class:`CtcWfstBeamSearchOptions`.

        Returns:
            A ready-to-use :class:`CtcWfstBeamSearch` instance.
        """
        if options is None:
            options = CtcWfstBeamSearchOptions()
        core = _get_core().from_file(
            fst_path=fst_path,
            blank=options.blank,
            search_beam=options.search_beam,
            output_beam=options.output_beam,
            min_active_states=options.min_active_states,
            max_active_states=options.max_active_states,
            subsampling_factor=options.subsampling_factor,
            nbest=options.nbest,
            blank_skip_thresh=options.blank_skip_thresh,
        )
        return cls(core)

    def search(self, logp: torch.Tensor) -> None:
        """Accumulate a chunk of log-probability frames.

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
        """Run K2 WFST decoding over all accumulated frames and extract N-best paths."""
        self._core.finalize_search()

    @property
    def outputs(self) -> List[List[int]]:
        """N-best decoded token-ID sequences."""
        return self._core.outputs

    @property
    def likelihood(self) -> List[float]:
        """N-best path scores."""
        return self._core.likelihood

    @property
    def times(self) -> List[List[int]]:
        """Per-token timestamps (if available)."""
        return self._core.times
