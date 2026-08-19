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
from typing import Any, ClassVar, List, Optional, Sequence, Tuple

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

    #: Whether :meth:`prefill` accepts a ``kv_manager`` and will page its
    #: self-attention KV out of that pool instead of a dense capacity buffer.
    #: Declared rather than probed: a decoder that ignored the argument would
    #: silently keep allocating per batch while the engine reserved a pool for it.
    supports_paged_kv: ClassVar[bool] = False

    #: True only when every step input has stable storage or is copied into a
    #: static buffer. Per-group side caches make graph capture unsafe.
    supports_step_graphs: ClassVar[bool] = False

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> DecoderState:
        """Build the initial per-request decoder state for ``batch_size`` hyps."""
        raise NotImplementedError

    def step(
        self,
        prev_tokens: torch.Tensor,
        encoder_out: torch.Tensor,
        state: DecoderState,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, DecoderState]:
        """One autoregressive step → ``(logits (B, V), new_state)``.

        Optional generic AR entry point for label-synchronous drivers (AED /
        LLM).  Frame-synchronous families (transducer) drive their predictor +
        joiner directly from the decode strategy and need not implement this.

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
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the generic step(); its "
            "decode strategy drives it directly."
        )


class TransducerPredictor(BaseDecoder):
    """Label predictor driven by the frame-synchronous transducer strategy.

    Four operations, and between them they are the *whole* reason
    :class:`~oasr.engine.decode.TransducerDecodeStrategy` serves both a
    stateless convolutional predictor (icefall) and a recurrent one (NeMo's
    2-layer LSTM) with one greedy loop:

    ``init_state(B, device[, dtype])``
        The state before any label has been emitted.  For a recurrent predictor
        this is *not* zeros — it is the state after the start-of-sequence step,
        because NeMo/HF run the LSTM once on the blank embedding (which is the
        zero row) from a zero hidden state, and the resulting prediction is what
        the first frame's joint sees.
    ``predict(state) → (B, D_pred)``
        The prediction the joiner consumes.  A read for a recurrent predictor
        (the state carries it), a recompute for a stateless one (its state *is*
        the label window).
    ``advance(state, tokens, emit) → state``
        Fold ``tokens`` into the state for the rows where ``emit`` is true, and
        leave the others exactly as they were.  Row-wise masking rather than a
        gather because the strategy advances a whole batch per step.
    ``stack_states`` / ``unstack_states``
        Regroup per-stream states, which streaming needs because the cohort of
        ready streams changes every tick.

    Why the state is opaque to the strategy: a label window can be recomputed
    from the last ``k`` tokens, and an LSTM state cannot.  The strategy used to
    assume the former — it shifted a ``(B, context_size)`` int tensor with
    ``torch.cat`` and re-ran the predictor — so a recurrent predictor had no way
    in at all short of a second copy of the loop.
    """

    decode_type: DecodeType = "transducer"

    #: Whether ``state`` is a ``(B, context_size)`` label-window tensor.  Beam
    #: search (``oasr/engine/decode/transducer_beam.py``) keeps the beam's states
    #: in one ``(B, k, ctx)`` buffer and reorders them onto their parents with a
    #: ``gather``, which only works for that representation; a recurrent
    #: predictor leaves this ``False`` and greedy is the supported mode.
    label_window_state: ClassVar[bool] = False

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> DecoderState:
        """Per-hypothesis state before the first emission."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, state: DecoderState) -> torch.Tensor:
        """``state`` → ``(B, D_pred)`` prediction for the joiner."""
        raise NotImplementedError

    @abstractmethod
    def advance(
        self, state: DecoderState, tokens: torch.Tensor, emit: torch.Tensor
    ) -> DecoderState:
        """Fold ``tokens (B,)`` into ``state`` where ``emit (B,)`` is true."""
        raise NotImplementedError

    @abstractmethod
    def stack_states(self, states: Sequence[DecoderState]) -> DecoderState:
        """Concatenate per-stream states into one batched state."""
        raise NotImplementedError

    @abstractmethod
    def unstack_states(self, state: DecoderState) -> List[DecoderState]:
        """Split a batched state back into per-stream states (inverse of
        :meth:`stack_states`)."""
        raise NotImplementedError


class PredictionNetwork(nn.Module, ABC):
    """Transducer label predictor expressed as a plain step function.

    Retained as the minimal ``(tokens, state) -> (prediction, state)`` shape for
    a predictor used outside the engine's decode strategy;
    :class:`TransducerPredictor` is what the strategy drives.
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
    def forward(self, encoder_out: torch.Tensor, prediction_out: torch.Tensor) -> torch.Tensor:
        """``(B, D_enc)`` ⊕ ``(B, D_pred)`` → ``(B, V)`` logits."""
        raise NotImplementedError
