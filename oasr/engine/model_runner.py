# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Model execution and cache lifecycle management for the ASR engine."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from oasr.cache import (
    AttentionCacheManager,
    BlockPool,
    CacheConfig,
    CnnCacheManager,
    CtcStateCacheManager,
    StreamContext,
)
from oasr.layers.attention.attention import PagedKVCache
from oasr.models.conformer.model import ConformerModel

from oasr.utils.nvtx import nvtx_pop, nvtx_push

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
            ``(B,)`` int32 valid encoder output frame counts, computed from
            the encoder's padding mask so downstream decoders ignore padding.
        """
        hidden, masks = self._model.encoder(features, lengths)
        log_probs = self._model.ctc(hidden)  # (B, T_out, V)
        # masks: (B, 1, T_out) with True = valid
        output_lengths = masks.squeeze(1).sum(dim=-1).to(torch.int32)
        return log_probs, output_lengths

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
        """Process at most one encoder chunk per request.

        Slices ``window`` frames out of ``request.feature_buffer`` starting
        at ``request.feature_cursor``, runs ``forward_chunk_paged`` (or
        ``forward_chunk`` in dense mode), commits the updated caches, and
        advances ``feature_cursor`` by ``stride`` frames.  Requests whose
        buffer doesn't yet hold a full window are skipped (they'll be
        revisited on the next step once more audio has been extracted).

        Requests that share the same ``(offset, chunk_window_size)`` are
        **batched into a single paged forward** — one encoder call over
        ``(B, window, F)`` stacked features and ``(B, max_blocks_per_seq)``
        stacked block-tables.  This is the single biggest streaming
        throughput lever because the per-layer matmuls were launch-bound
        at B=1; batching the lockstep pool amortises the ~200 kernel
        launches across all in-flight streams.  Streams with mismatched
        offsets or partial/final windows fall back to per-stream
        ``forward_chunk_paged``.

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
        window = cfg.decoding_window
        stride = cfg.stride
        context = cfg.right_context + 1

        results: Dict[str, torch.Tensor] = {}

        # Partition into (full-window + paged) and the catch-all fallback.
        # Full-window-paged streams with identical offsets are candidates
        # for batched forward.
        batchable: List[Request] = []
        fallback: List[Request] = []

        for req in requests:
            if not req.has_ready_encoder_chunk(window):
                continue
            if req.feature_buffer is None:
                continue

            available = req.feature_frames - req.feature_cursor
            is_final_window = (
                req.audio_final
                and not req.audio_chunks
                and (req.audio_tail is None or req.audio_tail.numel() == 0)
                and available <= window
            )
            # Only full-window paged forwards go through the batched path.
            if use_paged and not is_final_window and available >= window:
                batchable.append(req)
            else:
                fallback.append(req)

        if batchable:
            by_offset: Dict[int, List[Request]] = {}
            for req in batchable:
                by_offset.setdefault(req.offset, []).append(req)
            for off, group in by_offset.items():
                self._forward_batched_paged(group, off, window, stride, context, results)

        for req in fallback:
            self._forward_single(req, window, stride, context, results)

        return results

    # ------------------------------------------------------------------
    # Batched paged forward
    # ------------------------------------------------------------------

    def _forward_batched_paged(
        self,
        group: List[Request],
        offset: int,
        window: int,
        stride: int,
        context: int,
        results: Dict[str, torch.Tensor],
    ) -> None:
        """Run one paged forward on ``B = len(group)`` stacked streams.

        Pre-condition: every request in ``group`` shares ``offset``, has a
        full ``window`` frames ready in its feature buffer, and is using
        paged attention.
        """
        if len(group) == 1:
            # Skip the stack/unstack bookkeeping for a single stream.
            self._forward_single(group[0], window, stride, context, results)
            return

        nvtx_push(f"batched_paged[B={len(group)}]")
        # 1. Prepare per-stream write blocks (allocates a new physical
        #    block and updates each stream's block_table in place).  One
        #    batched allocator call instead of B per-stream calls.
        nvtx_push("prepare_chunk")
        stream_ids = [req.stream_id for req in group]
        # All streams in a group are paged; assert non-None to satisfy mypy.
        assert all(req.stream_context is not None for req in group)
        self._att_mgr.prepare_chunks_batched(stream_ids)  # type: ignore[arg-type]
        nvtx_pop()

        # 2. Stack per-stream paged descriptors into a (B, ...) view.
        #    All layers share the same block_table / cache_seqlens (one
        #    pair per stream), so we read those shared tensors via a
        #    cheap accessor (no per-layer dataclass construction) and
        #    stack once for the whole batched forward.
        block_tables: List[torch.Tensor] = []
        cache_seqlens: List[torch.Tensor] = []
        for req in group:
            bt, cs = req.stream_context.get_paged_state_views()
            block_tables.append(bt)
            cache_seqlens.append(cs)
        batched_bt = torch.cat(block_tables, dim=0)       # (B, blocks_per_seq)
        batched_cs = torch.cat(cache_seqlens, dim=0)      # (B,)

        # 3. Build batched PagedKVCache descriptors, one per layer, that
        #    share k_cache / v_cache with each stream (the block pool is
        #    global) and share the newly-stacked block_table / cache_seqlens.
        block_size = self._cache_config.block_size_frames
        num_layers = self._cache_config.num_layers
        batched_att_caches: List[PagedKVCache] = []
        for i in range(num_layers):
            k_view, v_view = self._block_pool.get_kv_view(i)
            batched_att_caches.append(PagedKVCache(
                k_cache=k_view,
                v_cache=v_view,
                block_table=batched_bt,
                cache_seqlens=batched_cs,
                block_size=block_size,
                host_seqlen=offset,
            ))

        # 4. Stack feature-chunk slices and cnn_cache across streams.
        feature_chunks = [
            req.feature_buffer[req.feature_cursor: req.feature_cursor + window]
            for req in group
        ]
        xs = torch.stack(feature_chunks, dim=0)  # (B, window, F)

        per_stream_cnn = [req.stream_context.get_cnn_cache() for req in group]
        # Each per-stream cnn_cache has shape
        # ``(num_layers, 1, cnn_cache_frames, hidden_dim)`` when populated
        # or a (0,0,0,0) placeholder on the first chunk.  Only stack when
        # every stream is past its first chunk; otherwise pass the
        # placeholder (the model handles it layer-wise).
        if all(c.size(0) > 0 for c in per_stream_cnn):
            cnn_cache = torch.cat(per_stream_cnn, dim=1)  # (L, B, T, H)
        else:
            cnn_cache = per_stream_cnn[0]  # placeholder (0,0,0,0)

        # 5. One batched encoder call.
        nvtx_push("encoder_call")
        log_probs, new_cnn = self._model.forward_chunk_paged(
            xs,
            offset,
            batched_att_caches,
            cnn_cache,
            cache_t1=offset,
        )
        actual_frames = log_probs.size(1)
        nvtx_pop()

        # 6. Split the new cnn_cache back into per-stream slices and
        #    commit per-stream attention-cache advancement.
        nvtx_push("commit")
        for b, req in enumerate(group):
            per_stream_new_cnn = new_cnn[:, b: b + 1]  # (L, 1, T, H)
            req.stream_context.commit_chunk_paged(actual_frames, per_stream_new_cnn)
            req.offset += actual_frames
            req.feature_cursor += stride
            results[req.request_id] = log_probs[b: b + 1]
        nvtx_pop()
        nvtx_pop()  # batched_paged

    # ------------------------------------------------------------------
    # Per-stream fallback forward
    # ------------------------------------------------------------------

    def _forward_single(
        self,
        req: Request,
        window: int,
        stride: int,
        context: int,
        results: Dict[str, torch.Tensor],
    ) -> None:
        """Run one paged-or-dense forward for a single request.

        Used for partial/final windows, for streams whose offsets differ
        from the batched group, and for the dense (non-paged) cache mode.
        """
        cfg = self._config
        use_paged = cfg.use_paged_cache
        if req.feature_buffer is None:
            return

        available = req.feature_frames - req.feature_cursor
        end = req.feature_cursor + min(window, available)
        chunk = req.feature_buffer[req.feature_cursor: end].unsqueeze(0)
        is_final_window = (
            req.audio_final
            and not req.audio_chunks
            and (req.audio_tail is None or req.audio_tail.numel() == 0)
            and available <= window
        )
        if chunk.size(1) < context:
            if is_final_window:
                req.feature_cursor = req.feature_frames
            return

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
                cache_t1=req.offset,
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
        if is_final_window:
            req.feature_cursor = req.feature_frames
        else:
            req.feature_cursor += stride
        results[req.request_id] = log_probs
