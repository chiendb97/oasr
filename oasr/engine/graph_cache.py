# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph cache for the streaming encoder forward.

A standalone PyTorch :class:`torch.cuda.CUDAGraph` is captured per
``(B_active, cache_t1_bucket)`` shape. Captures are lazy on first
encounter; replays reuse pre-allocated input buffers (``xs``,
``slot_ids``, ``offset``) and the captured output buffer.
Fixed-extent stream state (the convolutional left-context, plus any further
tensors the encoder declared via ``CacheSpec.stream_states``) is read/written in
place inside the captured forward through
:class:`~oasr.cache.SlotTensor` descriptors, so the persistent
:class:`~oasr.cache.SlotStateCache` buffers are updated directly without a
separate post-replay scatter.

The captured callable is injected (``chunk_forward``), so the same machinery
serves both streaming shapes: the fused ``forward_chunk_paged``
(encoder + CTC head → ``(B, chunk, V)`` log-probs) that one-shot families
consume, and the encoder-only ``encode_chunk_paged``
(→ ``(B, chunk, D)`` hidden) that ``consumes="hidden"`` families such as the
transducer consume.  Nothing here inspects the output beyond its identity as a
tensor, so a new streaming decode family needs no change in this file.

Steady-state streaming is launch-bound — the model's 12-layer conformer
encoder issues ~200 small kernels per chunk, and at 32 streams ×
25 chunks/sec the Python/launch overhead per chunk swamps the actual
compute. Replaying the captured graph collapses the entire forward into
a single launch, which is a ~20× speedup at fixed shape on this model.

Capture constraints
-------------------
* The persistent batched ``block_table`` / ``cache_seqlens`` and every
  ``SlotStateCache`` buffer must be allocated **before** the first capture and
  never reallocated; the graph captures the read sites by address.
* ``cache_t1`` is rounded up to a multiple of 64 (kernel ``N_BLOCK`` tile)
  per bucket so the kernel's per-tile block-table reads stay in-bounds.
* The encoder is invoked with ``offset`` as a per-stream int32 tensor
  (heterogeneous-offset code path) regardless of whether the actual batch
  is homogeneous — the captured code path must be stable.
* The stream-state rows referenced by ``slot_ids`` are snapshotted before
  capture and restored after; the warmup and captured ``_run`` both
  mutate those rows, but the first real replay must read the engine's
  pre-chunk state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import tvm_ffi

from oasr.cache.attention_cache import AttentionCacheManager
from oasr.cache.cnn_cache import CONV_STATE
from oasr.cache.paged_kv import PagedKVCache
from oasr.cache.state import SlotStateCache
from oasr.features import FeatureConfig, build_extractor
from oasr.features.batched import supports_batched_fbank, supports_batched_mfcc

# N_BLOCK tile size of the FMHA kernel; T_kv must be a multiple of this.
_KERNEL_N_BLOCK = 64


def round_up_bucket(cache_t1: int, granularity: int = _KERNEL_N_BLOCK) -> int:
    """Round ``cache_t1`` up to the next ``granularity`` multiple."""
    return ((cache_t1 + granularity - 1) // granularity) * granularity


#: Below this many cached frames the ladder stays ``_KERNEL_N_BLOCK``-granular,
#: because that is where a coarse bucket would be a large *relative* over-read.
CACHE_BUCKET_KNEE = 512

#: Above the knee each rung is this much larger than the last.  1.5 keeps the
#: worst-case over-read at 50% of the true cache length, which costs ~4% of a
#: replay -- doubling a bucket was measured at 7-14%, and a replay is about half
#: a streaming tick.
CACHE_BUCKET_GROWTH = 1.5


def cache_bucket_ladder(
    capacity: int,
    *,
    knee: int = CACHE_BUCKET_KNEE,
    growth: float = CACHE_BUCKET_GROWTH,
) -> List[int]:
    """Every ``cache_t1`` bucket a stream can reach, smallest first.

    The streaming graph is keyed on ``(B_active, T_input, cache_t1_bucket)`` and
    the last axis grows with stream age.  Rounding it to a flat 64 frames makes
    that axis *unbounded*: with the default ``num_left_chunks=-1`` a long stream
    reaches a new bucket every 64 encoder frames and captures a fresh graph each
    time, for as long as it runs.  Measured on a 120 s stream, that was 48
    captures over 191 ticks -- a capture on **a quarter of all ticks**, ~30 ms
    each, and the p99 was 33.3 ms against a 2.7 ms p50.  It also never settles:
    the captures were split 24/24 between the first and second half of the run,
    and the cache eventually saturates at ``_max_captures`` and drops the stream
    to eager permanently.

    Growing the rungs geometrically above ``knee`` turns that unbounded axis into
    ~10-16 rungs that cover a stream's whole life, so the ladder can be pre-warmed
    exhaustively and no capture ever lands on a live tick.  The trade is a larger
    over-read of the paged K/V: the kernel is handed ``host_seqlen_max`` = the
    rung, and the frames past a stream's real ``cache_seqlens`` are masked.  That
    is the same over-read the flat rounding already performed, only wider -- and
    it is cheap, because a replay grows just 1.04 ms -> 1.41 ms from an empty
    cache to 82 s of history while the eager forward stays flat at ~12 ms.

    ``capacity`` is ``CacheConfig.max_stream_frames``: a stream cannot cache more
    than its block table can address, so the ladder stops there and the last rung
    is exactly that bound.  A bucket past it would walk the block table off its
    end, which is the paged-loader trap in AGENTS.md.
    """
    cap = max(0, int(capacity))
    rungs = [0]
    step = _KERNEL_N_BLOCK
    b = step
    while b < min(knee, cap):
        rungs.append(b)
        b += step
    b = max(step, round_up_bucket(min(knee, cap) if cap else step))
    while b < cap:
        rungs.append(b)
        nxt = round_up_bucket(int(b * growth)) if growth > 1.0 else b + step
        b = nxt if nxt > b else b + step
    if cap > 0:
        # Floor, never ceil: ``capacity`` is a hard ceiling (the block table's
        # addressable frames, and the encoder's relative-position table), so a
        # rung rounded *up* past it is exactly the out-of-bounds read the
        # ceiling exists to prevent.
        rungs.append((cap // _KERNEL_N_BLOCK) * _KERNEL_N_BLOCK)
    return sorted({r for r in rungs if 0 <= r <= cap})


def pick_cache_bucket(cache_t1: int, ladder: Sequence[int]) -> int:
    """Smallest rung ``>= cache_t1``; a flat 64-round when it is off the ladder.

    Off-ladder means the stream is longer than the capacity the ladder was built
    for, which the cache manager is supposed to prevent.  Rounding rather than
    clamping keeps the contract that the returned bucket is never *below*
    ``cache_t1`` -- handing the kernel a shorter ``host_seqlen_max`` than the
    stream's real ``cache_seqlens`` would silently truncate its attention history.
    """
    want = int(cache_t1)
    for rung in ladder:
        if rung >= want:
            return rung
    return round_up_bucket(want)


@dataclass
class _CapturedShape:
    """One captured CUDA graph + the pre-allocated input/output buffers."""

    graph: "torch.cuda.CUDAGraph"
    xs_buf: torch.Tensor
    slot_ids_buf: torch.Tensor
    offset_buf: torch.Tensor
    # Persistent gather buffers that mirror the persistent state for the
    # active slots. The captured kernel reads from these addresses; before
    # each replay we refresh them with the current state via copy_().
    batched_block_table: torch.Tensor
    batched_cache_seqlens: torch.Tensor
    output_buf: torch.Tensor


class GraphedEncoderForward:
    """Lazy CUDA-Graph cache for the batched paged chunk forward.

    Parameters
    ----------
    chunk_forward : callable
        The chunk forward to capture, invoked as
        ``chunk_forward(xs, offset, att_caches, cnn_cache, cache_t1=...)``.
        Pass ``model.forward_chunk_paged`` for the fused encoder+head path
        (``(B, chunk, V)`` log-probs) or ``model.encode_chunk_paged`` for the
        encoder-only path (``(B, chunk, D)`` hidden).  The capture is
        output-shape agnostic; only the frame count ``size(1)`` is read back by
        the caller.
    att_mgr : AttentionCacheManager
        Provides the persistent batched paging tensors the graph reads from.
    state_mgr : SlotStateCache
        Provides the persistent fixed-extent stream-state buffers (the
        convolutional left-context under ``"conv"``, plus whatever else the
        encoder declared).  Their addresses are captured by reference, so this
        object must outlive every capture and never reallocate.
    device : torch.device
        Device of the persistent state (used to synchronise around capture).
    pool : tuple of int, optional
        Shared CUDA Graph memory-pool handle (from
        ``torch.cuda.graph_pool_handle()``) used by every engine-level
        capture (encoder, feature extraction, CTC). Defaults to a private
        pool when ``None`` so direct instantiation in tests still works.
    """

    def __init__(
        self,
        chunk_forward: Callable[..., torch.Tensor],
        att_mgr: AttentionCacheManager,
        state_mgr: SlotStateCache,
        *,
        device: torch.device,
        pool: Optional[Tuple[int, int]] = None,
        extra_states: Sequence[str] = (),
    ) -> None:
        self._chunk_forward = chunk_forward
        self._att_mgr = att_mgr
        self._state_mgr = state_mgr
        # Non-empty only for an encoder that declared state beyond the conv cache;
        # the captured call then carries a ``states=`` kwarg; otherwise it uses
        # the narrower compatibility signature.
        self._extra_states = tuple(extra_states)
        self._device = device

        # Captures share one memory pool to avoid allocator fragmentation.
        # Outputs remain valid only until that pool is reused by another replay.
        # Tests may omit ``pool`` when constructing this cache directly.
        self._pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        # Keyed by batch, input length, and cache-length bucket.
        self._captured: Dict[Tuple[int, int, int], _CapturedShape] = {}
        # Bound graph-pool growth; uncached shapes fall back to eager execution.
        self._max_captures = 512

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def have(self, B: int, T_input: int, cache_t1_bucket: int) -> bool:
        return (B, T_input, cache_t1_bucket) in self._captured

    @torch.no_grad()
    def replay(
        self,
        B: int,
        T_input: int,
        cache_t1_bucket: int,
        *,
        xs: torch.Tensor,
        slot_ids: torch.Tensor,
        offsets: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Replay (or first-capture) the encoder forward at the given shape.

        On the first call for a new ``(B, cache_t1_bucket)`` shape the graph
        is captured using the caller-provided ``slot_ids`` / ``offsets`` (so
        the capture-time warmup runs on the **actual** persistent paging
        state rather than dummy state). Subsequent replays just refresh the
        pre-allocated input buffers and trigger ``cudaGraphLaunch``. The
        captured forward reads its left-context from the persistent
        ``SlotStateCache`` buffers (at rows ``slot_ids``) and scatters new tails
        back into them in place.

        Parameters
        ----------
        B : int
            Active batch size for this chunk.
        cache_t1_bucket : int
            Bucket-aligned host-side cache_t1 (multiple of ``_KERNEL_N_BLOCK``).
        xs : Tensor
            ``(B, window, feat_dim)`` encoder input.
        slot_ids : Tensor
            ``(B,)`` int64 slot ids (gather into persistent tables; also
            selects the CNN buffer rows read/written by the captured
            forward).
        offsets : Tensor
            ``(B,)`` int32 per-stream encoder-frame offsets.

        Returns
        -------
        out : Tensor or None
            Whatever ``chunk_forward`` produces — ``(B, chunk_size, vocab_size)``
            log-probs for the fused path, ``(B, chunk_size, hidden_dim)`` for the
            encoder-only path. **Aliases the captured buffer**, and is invalidated
            by two things, not one: the next replay at the same shape key, *and* the
            next **capture** at any key.  Captures share one memory pool — that is
            where the fragmentation win lives — and a new capture may be handed the
            block an earlier capture's output buffer occupies.  A caller that hands
            out a result and then triggers a capture in the same step must clone
            first; see ``PagedStreamingBackend.forward_step``, where getting this
            wrong cost the trailing words of every stream that finalized in the same
            step as a fresh capture.
            Returns ``None`` when the per-shape capture cache is saturated
            (caller falls back to eager mode).
        """
        key = (B, T_input, cache_t1_bucket)
        state = self._captured.get(key)
        if state is None:
            if len(self._captured) >= self._max_captures:
                # Refuse to capture once the cache is saturated; the caller
                # will fall back to eager mode for this chunk.
                return None
            state = self._capture(
                B,
                T_input,
                cache_t1_bucket,
                xs,
                slot_ids,
                offsets,
            )
            self._captured[key] = state

        # Refresh captured input buffers with current per-chunk state.
        # The captured forward reads ``slot_ids_buf`` to gather/scatter
        # CNN cache rows and to drive the gather buffers below.
        state.xs_buf.copy_(xs)
        state.slot_ids_buf.copy_(slot_ids)
        state.offset_buf.copy_(offsets)
        # Refresh the gather buffers from the current persistent state
        # (post-``prepare_chunks_batched`` / ``commit_chunks_paged_batched``)
        # before replay. The captured graph reads from these specific
        # buffer addresses; without the refresh the kernel would see
        # the pre-capture snapshot.
        torch.index_select(self._att_mgr.block_table, 0, slot_ids, out=state.batched_block_table)
        torch.index_select(
            self._att_mgr.cache_seqlens, 0, slot_ids, out=state.batched_cache_seqlens
        )

        state.graph.replay()
        return state.output_buf

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _capture(
        self,
        B: int,
        T_input: int,
        cache_t1_bucket: int,
        xs: torch.Tensor,
        slot_ids: torch.Tensor,
        offsets: torch.Tensor,
    ) -> _CapturedShape:
        """Allocate persistent buffers, warm up, then capture the forward.

        The captured graph reads from **pre-allocated** gather buffers for
        ``block_table`` / ``cache_seqlens`` (stable addresses) instead of
        an in-graph ``index_select`` result (whose destination address
        could move across captures). Before each replay the caller
        refreshes these gather buffers from the persistent state.

        Stream state is read **in place** from the ``SlotStateCache`` buffers
        (whose addresses are stable) at rows ``slot_ids_buf``. Both the warmup
        ``_run`` and the captured ``_run`` mutate those rows, so we
        snapshot them before warmup and restore them after capture so the
        first real ``state.graph.replay()`` reads the engine's pre-chunk
        state.
        """
        # Pre-allocate input buffers, primed with the caller's tensors so
        # the warmup run operates on the real persistent paging state.
        xs_buf = xs.clone()
        slot_ids_buf = slot_ids.clone()
        offset_buf = offsets.clone()

        # Gather buffers — read by the captured kernel through the
        # PagedKVCache descriptors. Refreshed via ``index_select(out=...)``
        # before each replay so the kernel sees the latest persistent
        # paging state without an in-graph index_select.
        batched_bt = self._att_mgr.block_table.index_select(0, slot_ids_buf)
        batched_cs = self._att_mgr.cache_seqlens.index_select(0, slot_ids_buf)

        caches: List[PagedKVCache] = []
        for layer in range(self._att_mgr.num_layers):
            base = self._att_mgr._persistent_caches[layer]  # noqa: SLF001
            caches.append(
                PagedKVCache(
                    k_cache=base.k_cache,
                    v_cache=base.v_cache,
                    block_table=batched_bt,
                    cache_seqlens=batched_cs,
                    block_size=base.block_size,
                    host_seqlen_max=cache_t1_bucket,
                )
            )

        states = self._state_mgr.views(slot_ids_buf)
        cnn_cache = states[CONV_STATE]
        extra = {"states": states} if self._extra_states else {}

        def _run() -> torch.Tensor:
            return self._chunk_forward(
                xs_buf,
                offset_buf,
                caches,
                cnn_cache,
                cache_t1=cache_t1_bucket,
                **extra,
            )

        # Snapshot the CNN buffer rows for the active slots so we can
        # restore them after capture. The warmup ``_run`` and the captured
        # ``_run`` both write to these rows; without restore, the first
        # real ``state.graph.replay()`` would read the post-capture state
        # instead of the engine's pre-chunk state.
        saved = {name: view.gather().clone() for name, view in states.items()}

        def _restore() -> None:
            for name, snapshot in saved.items():
                states[name].scatter(snapshot)

        # Warmup once on the default stream so cuBLAS / cuDNN finalise any
        # one-time workspace allocations before we open capture. Real
        # streaming writes K/V to the chunk's logical position; the
        # subsequent ``replay()`` overwrites that warmup write with the
        # caller's actual ``xs``.
        _run()
        torch.cuda.synchronize(self._device)
        # Restore so the captured ``_run`` reads the same pre-chunk state that
        # the first real replay will see.
        _restore()

        graph = torch.cuda.CUDAGraph()
        # ``tvm_ffi.use_torch_stream(torch.cuda.graph(g))`` is the documented
        # path for capturing TVM-FFI kernel launches into a CUDA graph: it
        # opens the torch graph context and sets the FFI environment stream
        # to the active capture stream so the cute kernel's launch is
        # captured into ``graph`` (rather than escaping to the default
        # stream and tripping ``cudaErrorIllegalAddress`` on replay).
        with tvm_ffi.use_torch_stream(torch.cuda.graph(graph, pool=self._pool)):
            output_buf = _run()

        # Restore again so the first real replay (line below in caller)
        # reads the engine's pre-chunk state.
        _restore()

        return _CapturedShape(
            graph=graph,
            xs_buf=xs_buf,
            slot_ids_buf=slot_ids_buf,
            offset_buf=offset_buf,
            batched_block_table=batched_bt,
            batched_cache_seqlens=batched_cs,
            output_buf=output_buf,
        )


# =============================================================================
# Streaming feature extraction (batched fbank / mfcc) graph cache
# =============================================================================


@dataclass
class _CapturedFeatureShape:
    """One captured CUDA graph + pinned/device buffers for a single B bucket."""

    graph: "torch.cuda.CUDAGraph"
    # Pinned host buffers: caller writes the current chunk into these before
    # ``graph.replay()``. Addresses are stable for the cache's lifetime
    # because the buffers are allocated once.
    padded_host_buf: torch.Tensor  # (B_bucket, T_pad) float32, pinned
    lengths_host_buf: torch.Tensor  # (B_bucket,)       int64,   pinned
    # Device buffers — captured-graph destinations for the H2D copies.
    wav_device_buf: torch.Tensor  # (B_bucket, T_pad) float32, cuda
    lengths_device_buf: torch.Tensor  # (B_bucket,)       int64,   cuda
    # Captured output. Aliases the graph pool's output allocation; callers
    # must consume (or copy) before the next replay.
    feats_out: torch.Tensor  # (B_bucket, num_frames_max, feat_dim)
    # Completion of the previous replay, recorded on the issuing stream.  The
    # captured graph's *first* ops are async H2Ds **out of the pinned host
    # buffers above**, whose addresses are baked into the graph and so cannot be
    # rotated the way the eager staging pair is.  The host must therefore wait
    # here before refilling them.  ``None`` until the first replay.
    ready: Optional["torch.cuda.Event"] = None


class GraphedFeatureExtraction:
    """Lazy CUDA-Graph cache for batched ``fbank`` / ``mfcc`` feature extraction.

    The per-step batched feature extraction path in
    :meth:`~oasr.engine.InputProcessor.extract_streaming_batch` launches a
    handful of small kernels (unfold, dc-remove, pre-emphasis, Povey window,
    rfft, mel matmul, log) on a ``(B_active, T)`` waveform. At streaming
    cadence this is launch-bound; capturing it into one CUDA Graph per
    ``B`` bucket collapses the whole sequence into a single ``cudaGraphLaunch``.

    Each captured shape pre-allocates pinned host buffers (``padded_host_buf``,
    ``lengths_host_buf``) at a fixed ``(B_bucket, T_pad)`` shape. ``T_pad`` is
    the worst-case combined waveform length per step — one chunk's audio plus
    the maximum carried-over remainder (``frame_length_samples - 1`` samples)
    — so a single capture covers every steady-state call. The caller pads
    its variable-B / variable-T input into the bucket's pinned buffer before
    triggering replay; rows past ``B_active`` and samples past the actual
    combined length are zero-filled and their outputs are discarded
    host-side via the host-computed ``feat_lens_cpu`` formula.

    Buckets default to powers of two up to ``max_batch_size`` so a service
    with many active streams hits at most ``log2(max_batch_size) + 1``
    captures. The optional ``batch_buckets`` override (wired through
    :attr:`oasr.engine.EngineConfig.feature_graph_batch_buckets`) lets a
    deployment pin a smaller, fixed set when a preferred batch size is
    enforced upstream.

    Capture constraints
    -------------------
    * The pinned host buffers and the captured device input/output buffers
      must be allocated **before** the first capture and never reallocated
      — the graph captures the addresses.
    * Only the steady-state path is captured. ``extract_streaming_batch``
      keeps an eager fallback for the per-request ``flush`` path
      (irregular tail shapes), for cohorts larger than the biggest bucket,
      and for combined waveforms longer than ``T_pad`` (shouldn't occur in
      steady state but is defended against).
    """

    def __init__(
        self,
        *,
        pool: Optional[Tuple[int, int]],
        device: torch.device,
        feature_config: FeatureConfig,
        output_dtype: torch.dtype,
        chunk_samples: int,
        max_batch_size: int,
        batch_buckets: Optional[List[int]] = None,
    ) -> None:
        self._pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        self._device = device
        self._fcfg = feature_config
        self._output_dtype = output_dtype
        # Resolved through the feature registry so a new frontend needs no edit
        # here; the captured graph only cares that it is one callable.
        self._extractor = build_extractor(feature_config)

        frame_len = int(feature_config.frame_length_samples)
        frame_shift = int(feature_config.frame_shift_samples)
        # Worst-case combined waveform: this step's chunk_samples plus the
        # maximum carry-over remainder from the previous step (a partial
        # frame, ``frame_len - 1`` samples).
        self._t_pad = int(chunk_samples) + frame_len - 1
        self._frame_len = frame_len
        self._frame_shift = frame_shift
        self._feat_dim = int(feature_config.output_dim)

        if batch_buckets is None:
            buckets: List[int] = []
            b = 1
            while b < max_batch_size:
                buckets.append(b)
                b *= 2
            if max_batch_size >= 1:
                buckets.append(int(max_batch_size))
            self._buckets = sorted(set(buckets))
        else:
            cleaned = sorted({int(x) for x in batch_buckets if int(x) >= 1})
            if not cleaned:
                raise ValueError(
                    "feature_graph_batch_buckets must contain at least one " "positive integer"
                )
            self._buckets = cleaned

        self._max_bucket = self._buckets[-1] if self._buckets else 0
        self._captured: Dict[int, _CapturedFeatureShape] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def t_pad(self) -> int:
        """Maximum combined waveform length the captured graphs accept."""
        return self._t_pad

    @property
    def buckets(self) -> List[int]:
        """Sorted list of ``B_active`` buckets covered by this cache."""
        return list(self._buckets)

    def pick_bucket(self, B_active: int) -> Optional[int]:
        """Smallest captured bucket ``>= B_active``, or ``None`` when oversized."""
        if B_active < 1:
            return None
        for b in self._buckets:
            if b >= B_active:
                return b
        return None

    @torch.no_grad()
    def replay(
        self,
        B_active: int,
        padded_cpu: torch.Tensor,
        lengths_cpu: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Replay the captured feature graph for ``B_active`` streams.

        Parameters
        ----------
        B_active : int
            Number of streams in this step. Must be ``<= max(buckets)`` and
            ``>= 1``.
        padded_cpu : Tensor
            ``(B_active, T)`` float32 padded waveform batch. May be on CPU
            (pinned or not); contents are copied into the captured pinned
            buffer before replay. ``T`` must be ``<= self.t_pad``.
        lengths_cpu : Tensor
            ``(B_active,)`` int64 valid sample counts per stream. Copied
            into the captured pinned lengths buffer.

        Returns
        -------
        feats_out : Tensor or None
            ``(B_bucket, num_frames_max, feat_dim)`` view of the captured
            output buffer in ``output_dtype``. Callers should slice the
            first ``B_active`` rows and the first ``feat_lens_cpu[i]``
            frames per row. Returns ``None`` when ``B_active`` or
            ``padded_cpu.size(1)`` exceeds the captured shape (caller
            falls back to the eager path).
        """
        bucket = self.pick_bucket(B_active)
        if bucket is None:
            return None
        T = int(padded_cpu.size(1))
        if T > self._t_pad:
            return None

        state = self._captured.get(bucket)
        if state is None:
            state = self._capture(bucket)
            self._captured[bucket] = state

        # Zero only the per-row tail (samples [T, t_pad)) of the active rows;
        # the previous full-buffer ``zero_()`` was ~2.7 MB of pure-CPU work
        # per step at steady-state shapes and dominated the per-replay cost
        # for large N.  Rows past ``B_active`` are never read downstream
        # (host-side ``feat_lens_cpu`` discards their outputs) so they don't
        # need zeroing either, even if stale data persists between replays.
        if B_active > 0:
            # The previous replay's captured H2D reads these pinned buffers, and
            # the launch returns before the DMA runs.  Rewriting them without
            # waiting corrupts the *previous* step's features whenever the copy
            # has not drained.  Load on another stream makes this race easier to
            # trigger, but the ordering is required even when it usually finishes first.
            # Unlike the eager staging pair this cannot be double-buffered away:
            # the addresses are captured inside the graph.  In steady state a
            # full step of encoder work separates two replays, so the event has
            # long since fired and this does not park.
            if state.ready is not None:
                state.ready.synchronize()
                state.ready = None
            if T < self._t_pad:
                state.padded_host_buf[:B_active, T:].zero_()
            if T > 0:
                state.padded_host_buf[:B_active, :T].copy_(padded_cpu)
            state.lengths_host_buf[:B_active].copy_(lengths_cpu)
        state.graph.replay()
        event = torch.cuda.Event()
        event.record()
        state.ready = event
        return state.feats_out

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _capture(self, bucket: int) -> _CapturedFeatureShape:
        device = self._device
        T_pad = self._t_pad

        padded_host = torch.zeros(bucket, T_pad, dtype=torch.float32)
        lengths_host = torch.zeros(bucket, dtype=torch.int64)
        if device.type == "cuda":
            padded_host = padded_host.pin_memory()
            lengths_host = lengths_host.pin_memory()

        wav_device = torch.zeros(bucket, T_pad, dtype=torch.float32, device=device)
        lengths_device = torch.zeros(bucket, dtype=torch.int64, device=device)

        # Seed lengths_host with T_pad so the warmup hits the same kernel
        # tile shapes the captured replay will hit.
        lengths_host.fill_(T_pad)

        batched_fn = self._extractor
        fcfg = self._fcfg
        out_dtype = self._output_dtype

        def _run() -> torch.Tensor:
            wav_device.copy_(padded_host, non_blocking=True)
            lengths_device.copy_(lengths_host, non_blocking=True)
            feats_f32, _ = batched_fn(wav_device, lengths_device, fcfg)
            return feats_f32.to(dtype=out_dtype)

        # Warmup once on the default stream so any one-shot kernel/workspace
        # initialisation finishes before capture opens.
        _run()
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            feats_out = _run()

        return _CapturedFeatureShape(
            graph=graph,
            padded_host_buf=padded_host,
            lengths_host_buf=lengths_host,
            wav_device_buf=wav_device,
            lengths_device_buf=lengths_device,
            feats_out=feats_out,
        )


# Re-export so callers can probe backend support without reaching into
# ``oasr.features.batched`` directly.
__all__ = [
    "CACHE_BUCKET_GROWTH",
    "CACHE_BUCKET_KNEE",
    "cache_bucket_ladder",
    "pick_cache_bucket",
    "round_up_bucket",
    "GraphedEncoderForward",
    "GraphedFeatureExtraction",
    "supports_batched_fbank",
    "supports_batched_mfcc",
]
