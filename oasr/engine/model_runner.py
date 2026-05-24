# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Model execution and cache lifecycle management for the ASR engine."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from oasr.cache import (
    AttentionCacheManager,
    BlockPool,
    CacheConfig,
    CnnCacheManager,
    CtcStateCacheManager,
    StreamContext,
    StreamSlotPool,
)
from oasr.cache.paged_kv import PagedKVCache
from oasr.cache.slot_cnn import SlotCnnCache
from oasr.models.conformer.model import ConformerModel

from oasr.utils.nvtx import nvtx_pop, nvtx_push

from .config import EngineConfig
from .graph_cache import GraphedEncoderForward, round_up_bucket
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
    Streaming runs the paged ``forward_chunk_paged`` path. Streams with full
    ready windows are batched together by :meth:`_forward_batched_paged`;
    partial/final windows fall back to :meth:`_forward_single`. The cache
    managers track all streams concurrently and the scheduler amortises
    overhead across requests.
    """

    def __init__(
        self,
        model: ConformerModel,
        config: EngineConfig,
        cache_config: CacheConfig,
        *,
        graph_pool: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._model = model
        self._config = config
        self._cache_config = cache_config
        self._graph_pool = graph_pool

        # Build shared cache infrastructure
        self._block_pool = BlockPool(cache_config)
        self._att_mgr = AttentionCacheManager(self._block_pool, cache_config)
        self._cnn_mgr = CnnCacheManager(cache_config)
        self._slot_pool = StreamSlotPool(cache_config.max_batch_size)
        # Only create the CTC state manager when using the GPU decoder
        ctc_graphs_enabled = (
            bool(getattr(config, "use_cuda_graphs", True))
            and bool(getattr(config, "use_ctc_cuda_graphs", True))
            and torch.device(config.device).type == "cuda"
        )
        if config.decoder_type == "ctc_gpu":
            self._ctc_mgr: CtcStateCacheManager = CtcStateCacheManager(
                config.gpu_decoder_config,
                use_cuda_graphs=ctc_graphs_enabled,
            )
        else:
            self._ctc_mgr = CtcStateCacheManager(
                config.gpu_decoder_config,
                use_cuda_graphs=ctc_graphs_enabled,
            )

        # CUDA Graph cache for the steady-state batched paged forward.
        # Captures lazily on first encounter of each (B_active, cache_t1
        # bucket) shape. Eager fallback is used for non-CUDA devices, when
        # graphs are disabled, or for partial/final windows.
        self._use_cuda_graphs = (
            config.use_cuda_graphs
            and torch.device(config.device).type == "cuda"
        )
        if self._use_cuda_graphs:
            self._graph_cache = GraphedEncoderForward(
                model,
                self._att_mgr,
                self._cnn_mgr,
                cache_dtype=cache_config.dtype,
                device=torch.device(config.device),
                window=config.decoding_window,
                feat_dim=config.feature_config.output_dim,
                cnn_cache_frames=cache_config.cnn_cache_frames,
                num_layers=cache_config.num_layers,
                hidden_dim=cache_config.hidden_dim,
                pool=self._graph_pool,
            )
        else:
            self._graph_cache = None

    # ------------------------------------------------------------------
    # Encoder graph pre-warm
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prewarm_encoder_graphs(self, batch_sizes: Sequence[int]) -> None:
        """Pre-capture ``GraphedEncoderForward`` at every B in ``batch_sizes``.

        Triggers the lazy capture path with dummy zero-filled inputs so the
        first real chunk at each preferred B replays instead of paying the
        capture latency on the request path.  All captures use
        ``cache_t1_bucket=0`` (empty cache) and ``T_input == window``; the
        cache_t1 ladder is populated lazily on first traffic.

        No-op when CUDA graphs are disabled, ``_graph_cache`` is ``None``,
        or ``batch_sizes`` is empty.  Must be called **before** any stream
        is allocated so the dummy ``slot_ids = arange(B)`` rows are
        guaranteed unused — the persistent ``block_table`` and CNN buffer
        rows default to zero, which is also what the warmup forward will
        read.  ``GraphedEncoderForward._capture`` snapshots and restores the
        CNN buffer rows it touches, so pre-warm is non-destructive.
        """
        if self._graph_cache is None or not batch_sizes:
            return
        seen: List[int] = sorted({int(b) for b in batch_sizes if int(b) >= 1})
        if not seen:
            return
        cap = self._cache_config.max_batch_size
        if seen[-1] > cap:
            raise ValueError(
                f"prewarm batch size {seen[-1]} exceeds max_batch_size {cap}"
            )

        device = self._att_mgr.block_table.device
        window = self._config.decoding_window
        feat_dim = self._config.feature_config.output_dim
        dtype = self._cache_config.dtype

        for B in seen:
            slot_ids = torch.arange(B, dtype=torch.long, device=device)
            offsets = torch.zeros(B, dtype=torch.int32, device=device)
            xs = torch.zeros(B, window, feat_dim, dtype=dtype, device=device)
            self._graph_cache.replay(
                B, window, 0,
                xs=xs, slot_ids=slot_ids, offsets=offsets,
            )

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

        slot_id = self._slot_pool.allocate()
        request.slot_id = slot_id

        self._att_mgr.allocate_stream(sid, slot_id=slot_id)
        self._cnn_mgr.allocate_stream(sid, slot_id=slot_id)
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
            set to ``None``; its slot is returned to the slot pool.
        """
        if request.stream_context is not None:
            request.stream_context.free()
            request.stream_context = None
        if request.slot_id is not None:
            self._slot_pool.free(request.slot_id)
            request.slot_id = None

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
        at ``request.feature_cursor``, runs ``forward_chunk_paged``, commits
        the updated caches, and advances ``feature_cursor`` by ``stride``
        frames.  Requests whose
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
        window = cfg.decoding_window
        stride = cfg.stride
        context = cfg.right_context + 1

        results: Dict[str, torch.Tensor] = {}

        # Partition into batchable (full-window paged) and the partial/final
        # fallback. Full-window streams go through the batched paged forward;
        # partial/final windows run one-at-a-time through ``_forward_single``.
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
            if not is_final_window and available >= window:
                batchable.append(req)
            else:
                fallback.append(req)

        if batchable:
            # Heterogeneous-offset batching: all batchable streams go into
            # one paged forward regardless of offset. FlexAttention's
            # block-mask is built from per-stream cache_seqlens, and the
            # encoder builds per-stream pos_emb when offsets differ.
            self._forward_batched_paged(
                batchable, window, stride, context, results,
            )

        for req in fallback:
            self._forward_single(req, window, stride, context, results)

        return results

    # ------------------------------------------------------------------
    # Batched paged forward
    # ------------------------------------------------------------------

    def _forward_batched_paged(
        self,
        group: List[Request],
        window: int,
        stride: int,
        context: int,
        results: Dict[str, torch.Tensor],
    ) -> None:
        """Run one paged forward on ``B = len(group)`` stacked streams.

        Streams may have **different** offsets (cohort-relaxed admission).
        FlexAttention enforces per-stream cache lengths via a block-mask
        derived from ``cache.cache_seqlens``; the encoder builds per-stream
        position embeddings when offsets differ.

        Pre-condition: every request in ``group`` has a full ``window``
        frames ready in its feature buffer and is using paged attention.
        Single-stream cohorts (B=1) also flow through this path so they
        can hit the CUDA graph cache instead of paying ~17 ms of eager
        launch overhead in :meth:`_forward_single`.
        """
        nvtx_push(f"batched_paged[B={len(group)}]")
        B_active = len(group)
        stream_ids = [req.stream_id for req in group]
        slot_ids_host = [req.slot_id for req in group]
        assert all(s is not None for s in slot_ids_host), (
            "all batched streams must have an allocated slot_id"
        )
        device = self._att_mgr.block_table.device
        slot_ids_gpu = torch.tensor(slot_ids_host, dtype=torch.long, device=device)

        # 1. Prepare per-stream write blocks — one allocator call + one
        #    scatter onto the persistent block_table.
        nvtx_push("prepare_chunk")
        self._att_mgr.prepare_chunks_batched(stream_ids)  # type: ignore[arg-type]
        nvtx_pop()

        # 2. Gather feature-chunk slices. CNN left-context is now read by
        #    the encoder itself via the SlotCnnCache descriptor (mirroring
        #    how K/V are written through PagedKVCache).
        max_offset = max(req.offset for req in group)
        feature_chunks = [
            req.feature_buffer[req.feature_cursor: req.feature_cursor + window]
            for req in group
        ]
        xs = torch.stack(feature_chunks, dim=0)  # (B, window, F)
        cnn_cache = SlotCnnCache(buffer=self._cnn_mgr.buffer, slot_ids=slot_ids_gpu)

        # 3. Per-stream encoder-frame offsets (always a tensor for the
        #    graphed path; the eager fallback accepts the same tensor).
        offsets_gpu = torch.tensor(
            [req.offset for req in group],
            dtype=torch.int32, device=device,
        )

        # 4. Encoder forward — graph replay for any captured (B, bucket)
        #    shape; eager fallback when the graph cache is saturated or
        #    disabled. Captures are lazy and per-(B, cache_t1_bucket); a
        #    typical workload sees a small set of (B, bucket) combos
        #    because the scheduler keeps streams in lockstep cohorts and
        #    cache_t1 grows in N_BLOCK-sized steps.
        nvtx_push("encoder_call")
        cache_t1_bucket = round_up_bucket(max_offset)
        log_probs = None
        if self._use_cuda_graphs and self._graph_cache is not None:
            log_probs = self._graph_cache.replay(
                B_active, xs.size(1), cache_t1_bucket,
                xs=xs, slot_ids=slot_ids_gpu, offsets=offsets_gpu,
            )
        if log_probs is None:
            batched_att_caches, _, _ = self._att_mgr.get_batched_paged_caches(slot_ids_gpu)
            for c in batched_att_caches:
                c.host_seqlen_max = max_offset
            log_probs = self._model.forward_chunk_paged(
                xs, offsets_gpu, batched_att_caches, cnn_cache,
                cache_t1=max_offset,
            )
        actual_frames = log_probs.size(1)
        nvtx_pop()

        # 5. Commit: KV cache_seqlens scatter + host-side cursor / result
        #    updates. CNN cache was already scattered in place by the
        #    encoder through ``cnn_cache.scatter()``.
        nvtx_push("commit")
        self._att_mgr.commit_chunks_paged_batched(stream_ids, actual_frames)  # type: ignore[arg-type]
        for b, req in enumerate(group):
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
        """Run one paged forward for a single request.

        Used for partial/final windows that the batched paged forward
        cannot accommodate.
        """
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
        assert req.slot_id is not None

        nvtx_push(f"single[off={req.offset},T_q={chunk.size(1)}]")
        # Allocate the next physical block for this stream and refresh the
        # persistent block_table row before we hand the slot into either
        # the graph-cached or eager forward.
        self._att_mgr.prepare_chunks_batched([req.stream_id])  # type: ignore[arg-type]

        device = self._att_mgr.block_table.device
        slot_ids_gpu = torch.tensor([req.slot_id], dtype=torch.long, device=device)
        offsets_gpu = torch.tensor([req.offset], dtype=torch.int32, device=device)
        cnn_cache = SlotCnnCache(buffer=self._cnn_mgr.buffer, slot_ids=slot_ids_gpu)

        cache_t1_bucket = round_up_bucket(req.offset)
        nvtx_push("single.encoder_call")
        log_probs = None
        if self._use_cuda_graphs and self._graph_cache is not None:
            log_probs = self._graph_cache.replay(
                1, chunk.size(1), cache_t1_bucket,
                xs=chunk, slot_ids=slot_ids_gpu, offsets=offsets_gpu,
            )
        if log_probs is None:
            batched_att_caches, _, _ = self._att_mgr.get_batched_paged_caches(slot_ids_gpu)
            for c in batched_att_caches:
                c.host_seqlen_max = req.offset
            log_probs = self._model.forward_chunk_paged(
                chunk, offsets_gpu, batched_att_caches, cnn_cache,
                cache_t1=req.offset,
            )
        nvtx_pop()
        actual_frames = log_probs.size(1)

        # Commit cache state — KV cache_seqlens scatter only; CNN cache
        # was scattered in place by the encoder.
        self._att_mgr.commit_chunks_paged_batched([req.stream_id], actual_frames)  # type: ignore[arg-type]
        nvtx_pop()  # single

        req.offset += actual_frames
        if is_final_window:
            req.feature_cursor = req.feature_frames
        else:
            req.feature_cursor += stride
        results[req.request_id] = log_probs
