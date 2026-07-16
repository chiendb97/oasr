# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Model execution + streaming cache lifecycle for the ASR engine.

``ModelRunner`` owns the offline forward path and delegates all streaming work
to a pluggable :class:`~oasr.engine.streaming_backend.StreamingEncoderBackend`
(selected from ``model.encoder.streaming_kind``).  This keeps the runner
architecture-agnostic: Conformer-style encoders use the paged-KV backend,
Zipformer-style encoders use the stateful backend, and new streaming runtimes
plug in without touching the runner or the executors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from oasr.cache import BlockPool, CacheConfig, StreamContext
from oasr.models.base import BaseAsrModel

from .config import EngineConfig
from .request import Request
from .streaming_backend import StreamingEncoderBackend, build_streaming_backend


class ModelRunner:
    """Wraps a :class:`~oasr.models.base.BaseAsrModel` execution.

    Provides the offline batch forward (``forward_offline`` /
    ``forward_offline_packed``) and delegates the streaming step
    (``allocate_stream`` / ``free_stream`` / ``forward_streaming_step`` /
    ``prewarm_encoder_graphs``) to a :class:`StreamingEncoderBackend`.

    Parameters
    ----------
    model : BaseAsrModel
        Loaded model already moved to the target device in eval mode.
    config : EngineConfig
        Engine configuration.
    cache_config : CacheConfig
        Cache configuration derived from the model.
    """

    def __init__(
        self,
        model: BaseAsrModel,
        config: EngineConfig,
        cache_config: CacheConfig,
        *,
        graph_pool: Optional[Tuple[int, int]] = None,
        consumes: str = "log_probs",
    ) -> None:
        self._model = model
        self._config = config
        self._cache_config = cache_config

        # Pick the streaming runtime from the encoder's declared cache model.
        # ``consumes`` (the active decode strategy's declared input) routes the
        # backend's per-chunk forward: fused head vs. raw hidden states.
        self._streaming_backend: StreamingEncoderBackend = build_streaming_backend(
            model.encoder.streaming_kind,
            model,
            config,
            cache_config,
            graph_pool=graph_pool,
            consumes=consumes,
        )

    # ------------------------------------------------------------------
    # Introspection / delegation helpers
    # ------------------------------------------------------------------

    @property
    def streaming_backend(self) -> StreamingEncoderBackend:
        """The active streaming-encoder backend."""
        return self._streaming_backend

    @property
    def decoding_window(self) -> int:
        """Input feature frames consumed per encoder chunk (from the backend)."""
        return self._streaming_backend.decoding_window

    @property
    def stride(self) -> int:
        """Feature-frame stride between consecutive chunk windows."""
        return self._streaming_backend.stride

    @property
    def _block_pool(self) -> Optional[BlockPool]:
        """Paged block pool, if the backend has one (used by memory tests)."""
        return getattr(self._streaming_backend, "block_pool", None)

    # ------------------------------------------------------------------
    # Offline forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_offline(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a batched offline forward pass.

        Parameters
        ----------
        features : Tensor
            ``(B, T, F)`` padded feature tensor on the model device.
        lengths : Tensor
            ``(B,)`` valid feature frame counts.

        Returns
        -------
        log_probs : Tensor
            ``(B, T_out, vocab_size)`` log-softmax probabilities.
        output_lengths : Tensor
            ``(B,)`` int32 valid encoder output frame counts.
        """
        return self._model.forward_offline(features, lengths)

    @torch.no_grad()
    def forward_offline_packed(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence-packing offline forward (gapless varlen attention).

        Same signature/return shapes as :meth:`forward_offline`; bit-exact to
        ``B=1`` inference.
        """
        return self._model.forward_offline_packed(features, lengths)

    @torch.no_grad()
    def encode_offline(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder-only offline forward → ``(hidden (B, T, D), out_lengths)``.

        For autoregressive decode strategies (``consumes == "hidden"``) that own
        their head/decoder (transducer / AED / LLM) instead of the fused CTC head.
        """
        return self._model.encode_offline(features, lengths)

    # ------------------------------------------------------------------
    # Streaming (delegated to the backend)
    # ------------------------------------------------------------------

    def prewarm_encoder_graphs(
        self,
        batch_sizes: Sequence[int],
        cache_t1_buckets: Optional[Sequence[int]] = None,
    ) -> None:
        """Pre-capture any per-shape streaming graphs (backend-specific)."""
        self._streaming_backend.prewarm(batch_sizes, cache_t1_buckets=cache_t1_buckets)

    def allocate_stream(self, request: Request) -> Optional[StreamContext]:
        """Allocate the per-request encoder cache for a streaming request."""
        return self._streaming_backend.allocate(request)

    def free_stream(self, request: Request) -> None:
        """Release the per-request encoder cache for a finished request."""
        self._streaming_backend.free(request)

    def forward_streaming_step(self, requests: List[Request]) -> Dict[str, torch.Tensor]:
        """Run at most one encoder chunk per ready request (delegated)."""
        return self._streaming_backend.forward_step(requests)
