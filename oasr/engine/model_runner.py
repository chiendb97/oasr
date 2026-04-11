# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Model execution and cache lifecycle management for the ASR engine."""

from __future__ import annotations

from typing import Dict, List

import torch

from oasr.cache import (
    AttentionCacheManager,
    BlockPool,
    CacheConfig,
    CnnCacheManager,
    CtcStateCacheManager,
    StreamContext,
)
from oasr.models.conformer.model import ConformerModel

from .config import EngineConfig
from .request import Request


class ModelRunner:
    """Wraps :class:`~oasr.models.conformer.ConformerModel` execution.

    Owns the shared paged KV-cache pool and all three stream cache managers.
    Provides a clean interface for both offline batch forwarding and the
    per-request streaming step.

    Parameters
    ----------
    model : ConformerModel
        Loaded model already moved to the target device in eval mode.
    config : EngineConfig
        Engine configuration.
    cache_config : CacheConfig
        Cache configuration derived from the model.

    Notes
    -----
    **Streaming batching constraint**: :meth:`ConformerEncoder.forward_chunk`
    and :meth:`ConformerEncoder.forward_chunk_paged` require ``batch_size=1``.
    The engine therefore iterates over concurrent streaming requests
    sequentially within a single ``step()`` rather than running a true
    batched forward.  The cache managers still track all streams concurrently
    and the scheduler amortises overhead across requests.
    """

    def __init__(
        self,
        model: ConformerModel,
        config: EngineConfig,
        cache_config: CacheConfig,
    ) -> None:
        self._model = model
        self._config = config
        self._cache_config = cache_config

        # Build shared cache infrastructure
        self._block_pool = BlockPool(cache_config)
        self._att_mgr = AttentionCacheManager(self._block_pool, cache_config)
        self._cnn_mgr = CnnCacheManager(cache_config)
        # Only create the CTC state manager when using the GPU decoder
        if config.decoder_type == "ctc_gpu":
            self._ctc_mgr: CtcStateCacheManager = CtcStateCacheManager(
                config.gpu_decoder_config
            )
        else:
            self._ctc_mgr = CtcStateCacheManager(config.gpu_decoder_config)

    # ------------------------------------------------------------------
    # Offline forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_offline(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Run a batched offline forward pass.

        Parameters
        ----------
        features : Tensor
            ``(B, T, F)`` padded feature tensor on the model device.
        lengths : Tensor
            ``(B,)`` valid feature frame counts.

        Returns
        -------
        Tensor
            ``(B, T_out, vocab_size)`` log-softmax probabilities.
        """
        return self._model(features, lengths)

    # ------------------------------------------------------------------
    # Streaming cache lifecycle
    # ------------------------------------------------------------------

    def allocate_stream(self, request: Request) -> StreamContext:
        """Allocate KV + CNN + CTC cache buffers for a streaming request.

        Assigns a :class:`~oasr.cache.StreamContext` to
        ``request.stream_context``.

        Parameters
        ----------
        request : Request
            The request being admitted to the running queue.  Must have
            ``request.stream_id`` set by the scheduler.

        Returns
        -------
        StreamContext
            The allocated context (also stored on the request).
        """
        sid = request.stream_id
        assert sid is not None, "stream_id must be assigned before allocate_stream"
        vocab_size = self._config._model_config.vocab_size or 5002  # type: ignore[union-attr]
        device = torch.device(self._config.device)

        self._att_mgr.allocate_stream(sid)
        self._cnn_mgr.allocate_stream(sid)
        self._ctc_mgr.allocate_stream(sid, batch=1, vocab_size=vocab_size, device=device)

        ctx = StreamContext(sid, self._att_mgr, self._cnn_mgr, self._ctc_mgr)
        request.stream_context = ctx
        return ctx

    def free_stream(self, request: Request) -> None:
        """Release all cache resources for a finished streaming request.

        Parameters
        ----------
        request : Request
            The request to clean up.  Its ``stream_context`` is freed and
            set to ``None``.
        """
        if request.stream_context is not None:
            request.stream_context.free()
            request.stream_context = None

    # ------------------------------------------------------------------
    # Streaming forward step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_streaming_step(
        self,
        requests: List[Request],
    ) -> Dict[str, torch.Tensor]:
        """Process one encoder chunk per request.

        Pops the next window from ``request.chunks_remaining``, runs
        ``forward_chunk_paged`` (or ``forward_chunk`` in dense mode), commits
        the updated caches, and advances ``request.offset``.

        Parameters
        ----------
        requests : List[Request]
            Active streaming requests with pre-chunked features and allocated
            stream contexts.

        Returns
        -------
        Dict[str, Tensor]
            Mapping ``request_id -> log_probs`` where ``log_probs`` has shape
            ``(1, chunk_size, vocab_size)``.  Requests that have no remaining
            chunks are omitted.
        """
        cfg = self._config
        use_paged = cfg.use_paged_cache
        chunk_size = cfg.chunk_size

        results: Dict[str, torch.Tensor] = {}

        for req in requests:
            if not req.chunks_remaining:
                continue

            chunk = req.chunks_remaining.pop(0)  # (1, window, F)
            ctx = req.stream_context
            assert ctx is not None

            if use_paged:
                ctx.prepare_chunk()
                att_caches = ctx.get_att_caches()
                cnn_cache = ctx.get_cnn_cache()
                log_probs, new_cnn = self._model.forward_chunk_paged(
                    chunk,
                    req.offset,
                    att_caches,
                    cnn_cache,
                )
                actual_frames = log_probs.size(1)
                ctx.commit_chunk_paged(actual_frames, new_cnn)
            else:
                att_cache = ctx.get_att_cache()
                cnn_cache = ctx.get_cnn_cache()
                required = cfg.required_cache_size
                log_probs, new_att, new_cnn = self._model.forward_chunk(
                    chunk,
                    req.offset,
                    required,
                    att_cache,
                    cnn_cache,
                )
                actual_frames = log_probs.size(1)
                new_kv_chunk = new_att[:, :, -actual_frames:, :]
                ctx.commit_chunk(new_kv_chunk, new_cnn)

            req.offset += actual_frames
            results[req.request_id] = log_probs

        return results
