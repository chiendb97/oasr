# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paged-KV streaming backend (Conformer-style encoders).

Owns the shared paged KV-cache pool + slot-CNN cache and runs the
``forward_chunk_paged`` (encoder + CTC head fused) path with CUDA-graph capture.
This is the engine's original streaming runtime, extracted behind the
:class:`~oasr.engine.streaming_backend.base.StreamingEncoderBackend` seam so other
encoder streaming models can plug in alongside it — behaviour is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Sequence, Tuple

import torch

from oasr.cache import (
    AttentionCacheManager,
    BlockPool,
    CnnCacheManager,
    StreamContext,
    StreamSlotPool,
)
from oasr.cache.slot_cnn import SlotCnnCache
from oasr.utils.nvtx import nvtx_pop, nvtx_push

from ..graph_cache import GraphedEncoderForward, round_up_bucket
from ..request import Request
from .base import StreamingEncoderBackend, register_streaming_backend

if TYPE_CHECKING:
    from oasr.cache.types import CacheConfig
    from oasr.models.base import BaseAsrModel

    from ..config import EngineConfig


@register_streaming_backend("paged")
class PagedStreamingBackend(StreamingEncoderBackend):
    """Paged-KV + slot-CNN streaming runtime with CUDA-graph capture.

    Streams with full ready windows are batched together by
    :meth:`_forward_batched_paged`; partial/final windows fall back to
    :meth:`_forward_single`.  The cache managers track all streams concurrently
    and the scheduler amortises overhead across requests.
    """

    streaming_kind: ClassVar[str] = "paged"

    def __init__(
        self,
        model: "BaseAsrModel",
        config: "EngineConfig",
        cache_config: "CacheConfig",
        *,
        graph_pool: Optional[Tuple[int, int]] = None,
        consumes: str = "log_probs",
    ) -> None:
        self._model = model
        self._config = config
        self._cache_config = cache_config
        self._graph_pool = graph_pool
        # What the active decode strategy consumes.  "log_probs" runs the fused
        # encoder+head ``forward_chunk_paged`` (today's CUDA-graph fast path);
        # "hidden" runs the encoder-only ``encode_chunk_paged`` **eagerly** —
        # graph capture for hidden mode is deferred until a shape proves hot
        # (the capture machinery bakes in a (B, chunk, V) log-probs buffer).
        self._consumes = consumes
        self._chunk_forward = (
            model.encode_chunk_paged if consumes == "hidden" else model.forward_chunk_paged
        )

        # Window geometry derived from the *encoder* (not hardcoded): a chunk of
        # ``chunk_size`` encoder frames needs ``(chunk_size-1)*sub + right_context
        # + 1`` input frames, advancing ``sub*chunk_size`` per step.
        enc = model.encoder
        sub = int(enc.subsampling_rate)
        rc = int(enc.right_context)
        cs = int(config.chunk_size)
        self._window = (cs - 1) * sub + rc + 1
        self._stride = sub * cs
        self._context = rc + 1

        # Build shared cache infrastructure.
        self._block_pool = BlockPool(cache_config)
        self._att_mgr = AttentionCacheManager(self._block_pool, cache_config)
        self._cnn_mgr = CnnCacheManager(cache_config)
        self._slot_pool = StreamSlotPool(cache_config.max_batch_size)

        # CUDA Graph cache for the steady-state batched paged forward.
        # Captures lazily on first encounter of each (B_active, cache_t1
        # bucket) shape. Eager fallback is used for non-CUDA devices, when
        # graphs are disabled, for partial/final windows, or for hidden-mode
        # strategies (the captured graph bakes in the fused-head output).
        self._use_cuda_graphs = (
            config.use_cuda_graphs
            and torch.device(config.device).type == "cuda"
            and consumes == "log_probs"
        )
        if self._use_cuda_graphs:
            self._graph_cache: Optional[GraphedEncoderForward] = GraphedEncoderForward(
                model,
                self._att_mgr,
                self._cnn_mgr,
                cache_dtype=cache_config.dtype,
                device=torch.device(config.device),
                window=self._window,
                feat_dim=config.feature_config.output_dim,
                cnn_cache_frames=cache_config.cnn_cache_frames,
                num_layers=cache_config.num_layers,
                hidden_dim=cache_config.hidden_dim,
                pool=self._graph_pool,
            )
        else:
            self._graph_cache = None

    # ------------------------------------------------------------------
    # Window geometry / introspection
    # ------------------------------------------------------------------

    @property
    def decoding_window(self) -> int:
        return self._window

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def block_pool(self) -> BlockPool:
        """The shared paged-KV block pool (used by memory-cleanup tests)."""
        return self._block_pool

    # ------------------------------------------------------------------
    # Encoder graph pre-warm
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prewarm(
        self,
        batch_sizes: Sequence[int],
        cache_t1_buckets: Optional[Sequence[int]] = None,
    ) -> None:
        """Pre-capture ``GraphedEncoderForward`` over a (B, cache_t1) ladder.

        Triggers the lazy capture path with dummy zero-filled inputs so the
        first real chunk at each ``(B, cache_t1_bucket)`` shape **replays**
        instead of paying the ~capture latency on the request path.  This is
        an interactive-latency win: without it a live stream pays a blocking
        ``cudaGraphInstantiate`` the first time each new ``cache_t1`` bucket
        appears mid-request (every ~64 encoder frames as the stream grows),
        producing a bimodal step-latency tail.

        ``cache_t1_buckets`` is the list of host-side cache_t1 values to
        capture (each rounded up to the kernel ``N_BLOCK`` multiple).  ``None``
        keeps the legacy behaviour of capturing only ``cache_t1_bucket=0``
        (empty cache) — the rest of the ladder then captures lazily.

        No-op when CUDA graphs are disabled, ``_graph_cache`` is ``None``,
        or ``batch_sizes`` is empty.  Must be called **before** any stream
        is allocated so the dummy ``slot_ids = arange(B)`` rows are
        guaranteed unused — the persistent ``block_table`` and CNN buffer
        rows default to zero, which is also what the warmup forward will
        read.  ``GraphedEncoderForward._capture`` snapshots and restores the
        CNN buffer rows it touches, so pre-warm is non-destructive.  The
        captured graph reads ``offset`` from a buffer at replay, so one
        capture per bucket serves every real offset within that bucket.
        """
        if self._graph_cache is None or not batch_sizes:
            return
        seen: List[int] = sorted({int(b) for b in batch_sizes if int(b) >= 1})
        if not seen:
            return
        cap = self._cache_config.max_batch_size
        if seen[-1] > cap:
            raise ValueError(f"prewarm batch size {seen[-1]} exceeds max_batch_size {cap}")

        if cache_t1_buckets is None:
            buckets: List[int] = [0]
        else:
            buckets = sorted(
                {round_up_bucket(int(c)) for c in cache_t1_buckets if int(c) >= 0}
            ) or [0]

        device = self._att_mgr.block_table.device
        window = self._window
        feat_dim = self._config.feature_config.output_dim
        dtype = self._cache_config.dtype

        for B in seen:
            slot_ids = torch.arange(B, dtype=torch.long, device=device)
            xs = torch.zeros(B, window, feat_dim, dtype=dtype, device=device)
            for bucket in buckets:
                offsets = torch.full((B,), bucket, dtype=torch.int32, device=device)
                self._graph_cache.replay(
                    B,
                    window,
                    bucket,
                    xs=xs,
                    slot_ids=slot_ids,
                    offsets=offsets,
                )

    # ------------------------------------------------------------------
    # Streaming cache lifecycle
    # ------------------------------------------------------------------

    def allocate(self, request: Request) -> StreamContext:
        """Allocate the encoder KV + CNN cache buffers for a streaming request.

        Assigns an encoder-only :class:`~oasr.cache.StreamContext` to
        ``request.stream_context``.  The CTC beam state is allocated separately
        by the decode strategy (``OutputProcessor.create_session``).
        """
        sid = request.stream_id
        assert sid is not None, "stream_id must be assigned before allocate"

        slot_id = self._slot_pool.allocate()
        request.slot_id = slot_id

        self._att_mgr.allocate_stream(sid, slot_id=slot_id)
        self._cnn_mgr.allocate_stream(sid, slot_id=slot_id)

        ctx = StreamContext(sid, self._att_mgr, self._cnn_mgr)
        request.stream_context = ctx
        return ctx

    def free(self, request: Request) -> None:
        """Release all encoder cache resources for a finished streaming request."""
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
    def forward_step(
        self,
        requests: List[Request],
    ) -> Dict[str, torch.Tensor]:
        """Process at most one encoder chunk per request.

        Slices ``window`` frames out of ``request.feature_buffer`` starting
        at ``request.feature_cursor``, runs ``forward_chunk_paged``, commits
        the updated caches, and advances ``feature_cursor`` by ``stride``
        frames.  Requests whose buffer doesn't yet hold a full window are
        skipped (revisited next step once more audio has been extracted).

        Requests that share the same ``(offset, chunk_window_size)`` are
        **batched into a single paged forward**; streams with mismatched
        offsets or partial/final windows fall back to per-stream
        ``forward_chunk_paged``.

        Returns ``{request_id: log_probs (1, chunk_size, V)}``; requests with
        no remaining chunks are omitted.
        """
        window = self._window
        stride = self._stride
        context = self._context

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
                batchable,
                window,
                stride,
                context,
                results,
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
        assert all(
            s is not None for s in slot_ids_host
        ), "all batched streams must have an allocated slot_id"
        device = self._att_mgr.block_table.device
        slot_ids_device = torch.tensor(slot_ids_host, dtype=torch.long, device=device)

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
            req.feature_buffer[req.feature_cursor : req.feature_cursor + window] for req in group
        ]
        xs = torch.stack(feature_chunks, dim=0)  # (B, window, F)
        cnn_cache = SlotCnnCache(buffer=self._cnn_mgr.buffer, slot_ids=slot_ids_device)

        # 3. Per-stream encoder-frame offsets (always a tensor for the
        #    graphed path; the eager fallback accepts the same tensor).
        offsets_device = torch.tensor(
            [req.offset for req in group],
            dtype=torch.int32,
            device=device,
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
                B_active,
                xs.size(1),
                cache_t1_bucket,
                xs=xs,
                slot_ids=slot_ids_device,
                offsets=offsets_device,
            )
        if log_probs is None:
            batched_att_caches, _, _ = self._att_mgr.get_batched_paged_caches(slot_ids_device)
            for c in batched_att_caches:
                c.host_seqlen_max = max_offset
            log_probs = self._chunk_forward(
                xs,
                offsets_device,
                batched_att_caches,
                cnn_cache,
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
            results[req.request_id] = log_probs[b : b + 1]
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
        chunk = req.feature_buffer[req.feature_cursor : end].unsqueeze(0)
        is_final_window = (
            req.audio_final
            and not req.audio_chunks
            and (req.audio_tail is None or req.audio_tail.numel() == 0)
            and available <= window
        )
        # Skip the all-silence trailing partial.  With finalize silence-padding
        # (``EngineConfig.finalize_silence_pad``) the last real-audio window is
        # a FULL window (decoded on the encoder CUDA-graph fast path), so any
        # sub-window final chunk lies entirely within the appended silence —
        # forwarding it would only emit blanks while taking the slow eager
        # sub-window path (which the streaming graph also mis-encodes at B>1).
        # Skipping it recovers full streaming throughput with no transcript
        # change.  When padding is disabled the sub-window tail carries real
        # audio, so it must still be forwarded (eager, via the guard below).
        if (
            getattr(self._config, "finalize_silence_pad", True)
            and is_final_window
            and chunk.size(1) < window
        ):
            req.feature_cursor = req.feature_frames
            return
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
        slot_ids_device = torch.tensor([req.slot_id], dtype=torch.long, device=device)
        offsets_device = torch.tensor([req.offset], dtype=torch.int32, device=device)
        cnn_cache = SlotCnnCache(buffer=self._cnn_mgr.buffer, slot_ids=slot_ids_device)

        cache_t1_bucket = round_up_bucket(req.offset)
        nvtx_push("single.encoder_call")
        log_probs = None
        # Only graph **full-window** chunks. Sub-window (partial/final-tail)
        # chunks must run eager: the cute attention reads the relative-position
        # bias tile unpredicated along T_q (padded to the kernel's M_BLOCK), and
        # for a short T_q the over-read past the real bias rows is benign in
        # eager mode (adjacent allocations are mapped) but reads stale data once
        # the captured graph carves its own memory pool — corrupting the last
        # chunk's attention (observed: a ~33-magnitude log-prob error on the
        # final sub-window chunk that flips borderline decodes vs eager). These
        # chunks are at most one per stream, so eager costs nothing.
        if self._use_cuda_graphs and self._graph_cache is not None and chunk.size(1) == window:
            log_probs = self._graph_cache.replay(
                1,
                chunk.size(1),
                cache_t1_bucket,
                xs=chunk,
                slot_ids=slot_ids_device,
                offsets=offsets_device,
            )
        if log_probs is None:
            batched_att_caches, _, _ = self._att_mgr.get_batched_paged_caches(slot_ids_device)
            for c in batched_att_caches:
                c.host_seqlen_max = req.offset
            log_probs = self._chunk_forward(
                chunk,
                offsets_device,
                batched_att_caches,
                cnn_cache,
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
