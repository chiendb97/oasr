# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive decode-strategy skeletons (transducer / AED / LLM).

These are **extension points**: registered so the engine resolves them by
``decode_type``, with the contract documented, but raising ``NotImplementedError``
until the family is implemented.  A full implementation drives the model's
:class:`~oasr.models.decoders.BaseDecoder` token-by-token with a decoder-side
KV/state cache (reuse :class:`~oasr.cache.block_pool.BlockPool` /
:class:`~oasr.cache.attention_cache.AttentionCacheManager` for AED/LLM
self/cross-attention KV).  ``consumes = "hidden"`` so the runner feeds raw
encoder hidden states rather than fused CTC log-probs.

Reference shape of an offline autoregressive decode (for the implementer):

#. encode the whole utterance → ``hidden (B, T_enc, D)`` (``ModelRunner.encode_offline``);
#. ``state = model.decoder.init_state(B, device)``;
#. loop greedy/beam, calling ``logits, state = model.decoder.step(prev, enc, state)``,
   emitting tokens until blank-budget (transducer) or EOS (AED/LLM);
#. detokenize the best hypothesis with the injected :class:`Detokenizer`.

Streaming additionally threads the decoder state across chunks via
:meth:`create_session` / :meth:`free_session`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, List

import torch

from ..request import Request, RequestOutput
from .base import DecodeStrategy, register_decode_strategy

if TYPE_CHECKING:
    from ..config import EngineConfig
    from .detokenize import Detokenizer


class _AutoregressiveSkeleton(DecodeStrategy):
    """Shared skeleton for autoregressive families (see module docstring)."""

    consumes: ClassVar[str] = "hidden"

    def __init__(self, config: "EngineConfig", detok: "Detokenizer") -> None:
        self._config = config
        self._detok = detok

    def _unimplemented(self) -> "RequestOutput":
        raise NotImplementedError(
            f"{type(self).__name__} (decode_type={self.decode_type!r}) is an "
            "extension point — the autoregressive decode loop is not implemented "
            "yet.  Implement decode_offline/decode_streaming_batch/finalize "
            "driving model.decoder.step(...) with a decoder-side KV/state cache."
        )

    def decode_offline(
        self, enc_out: torch.Tensor, enc_lengths: torch.Tensor
    ) -> List[RequestOutput]:
        self._unimplemented()

    def decode_streaming_batch(
        self, requests: List[Request], enc_out_map: Dict[str, torch.Tensor]
    ) -> List[RequestOutput]:
        self._unimplemented()

    def decode_streaming_chunk(self, request: Request, enc_out: torch.Tensor) -> RequestOutput:
        self._unimplemented()

    def finalize(self, request: Request) -> RequestOutput:
        self._unimplemented()


@register_decode_strategy("transducer")
class TransducerDecodeStrategy(_AutoregressiveSkeleton):
    """RNNT / transducer (frame-synchronous prediction-net + joiner beam)."""

    decode_type: ClassVar[str] = "transducer"


@register_decode_strategy("aed")
class AedDecodeStrategy(_AutoregressiveSkeleton):
    """Attention encoder-decoder (label-synchronous beam with cross-attention)."""

    decode_type: ClassVar[str] = "aed"


@register_decode_strategy("llm")
class LlmDecodeStrategy(_AutoregressiveSkeleton):
    """LLM-based ASR (acoustic-prefix / cross-attention LLM decode)."""

    decode_type: ClassVar[str] = "llm"
