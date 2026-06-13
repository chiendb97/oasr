# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive decoder base classes (transducer / AED / LLM).

These mirror :class:`~oasr.models.base.BaseHead` (the non-autoregressive CTC
output projection) but for **autoregressive** decode families.  A model sets
``self.decoder`` to a :class:`BaseDecoder`; the engine's matching
``DecodeStrategy`` (``oasr/engine/decode/``) drives it token-by-token, owning the
beam/greedy search and the decoder-side KV/state cache.

The contract is intentionally small and shape-explicit so the four ASR families
share one driver:

* **Transducer (RNNT)** — ``decoder`` composes a :class:`PredictionNetwork`
  (label-history → state) and a :class:`Joiner` (encoder frame ⊕ prediction →
  vocab logits).  Frame-synchronous: the strategy advances the encoder frame and
  emits/blanks per step.
* **AED** — ``decoder`` is a stack of self-attention + cross-attention layers
  over the full encoder output; label-synchronous with a decoder self-attention
  KV cache.
* **LLM-ASR** — ``decoder`` wraps an LLM that cross-attends / prefixes the
  acoustic embedding; label-synchronous with a KV cache.

``DecoderState`` is whatever per-request cache the decoder threads through
:meth:`BaseDecoder.step` (a tuple of RNN hidden tensors, a paged-KV handle, …).
The engine treats it opaquely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
from torch import nn

from ..base import DecodeType

#: Opaque per-request decoder cache threaded through :meth:`BaseDecoder.step`.
#: Concrete decoders pick the representation (RNN hidden tuple, paged-KV handle,
#: ``None`` for a stateless predictor).  The engine never inspects it.
DecoderState = Any


class BaseDecoder(nn.Module, ABC):
    """Autoregressive decode-side module for non-CTC ASR families.

    Concrete decoders ship with their model family; this defines the surface the
    engine's ``DecodeStrategy`` calls.  ``decode_type`` selects the strategy
    (mirrors :attr:`oasr.models.base.BaseHead.decode_type`).
    """

    decode_type: DecodeType = "transducer"

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> DecoderState:
        """Build the initial per-request decoder state for ``batch_size`` hyps."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        prev_tokens: torch.Tensor,
        encoder_out: torch.Tensor,
        state: DecoderState,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, DecoderState]:
        """One autoregressive step → ``(logits (B, V), new_state)``.

        Parameters
        ----------
        prev_tokens : Tensor
            ``(B,)`` last emitted token id per active hypothesis.
        encoder_out : Tensor
            For transducers, the current ``(B, D)`` encoder frame; for AED/LLM,
            the ``(B, T_enc, D)`` encoder output (cross-attention keys/values).
        state : DecoderState
            The cache returned by the previous :meth:`step` / :meth:`init_state`.
        """
        raise NotImplementedError


class PredictionNetwork(nn.Module, ABC):
    """Transducer label predictor: label history → prediction state/embedding.

    Either a stateful RNN (``state`` is the hidden/cell tuple) or a stateless
    convolutional predictor (``state`` is the last-``k`` label window).
    """

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> DecoderState:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, tokens: torch.Tensor, state: DecoderState
    ) -> Tuple[torch.Tensor, DecoderState]:
        """``(B,)`` tokens + state → ``(B, D_pred)`` prediction + new state."""
        raise NotImplementedError


class Joiner(nn.Module, ABC):
    """Transducer joiner: combine an encoder frame and a prediction → vocab logits."""

    @abstractmethod
    def forward(
        self, encoder_out: torch.Tensor, prediction_out: torch.Tensor
    ) -> torch.Tensor:
        """``(B, D_enc)`` ⊕ ``(B, D_pred)`` → ``(B, V)`` logits."""
        raise NotImplementedError
