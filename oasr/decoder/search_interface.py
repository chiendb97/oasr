# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for all CTC search decoders."""

from __future__ import annotations

import abc
from typing import List

import torch


class SearchInterface(abc.ABC):
    """Abstract interface for CTC search algorithms.

    All concrete decoders (greedy, prefix beam, WFST) implement this interface,
    enabling uniform offline and streaming usage through :class:`~oasr.decode.Decoder`.
    """

    @abc.abstractmethod
    def search(self, logp: torch.Tensor) -> None:
        """Feed one chunk of log-probability frames to the decoder.

        Args:
            logp: Log-probability tensor of shape ``[T, V]``.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset all internal decoder state for a new utterance."""

    @abc.abstractmethod
    def finalize_search(self) -> None:
        """Finalize decoding after all frames have been fed.

        For greedy search this is a no-op. For beam search and WFST it
        triggers context backoff and/or N-best extraction.
        """

    @property
    @abc.abstractmethod
    def outputs(self) -> List[List[int]]:
        """N-best output token-ID sequences."""

    @property
    @abc.abstractmethod
    def likelihood(self) -> List[float]:
        """Log-likelihood (or total score) for each hypothesis."""

    @property
    @abc.abstractmethod
    def times(self) -> List[List[int]]:
        """Per-token frame timestamps for each hypothesis."""
