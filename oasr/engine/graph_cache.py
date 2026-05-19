# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph cache for the streaming encoder forward.

A standalone PyTorch :class:`torch.cuda.CUDAGraph` is captured per
``(B_active, cache_t1_bucket)`` shape. Captures are lazy on first
encounter; replays reuse pre-allocated input buffers (``xs``,
``slot_ids``, ``offset``) and the captured output buffer ``log_probs``.
The CNN cache is read/written in place inside the captured forward via
a :class:`~oasr.cache.SlotCnnCache` descriptor, so the persistent
``CnnCacheManager`` buffer is updated directly without a separate
post-replay scatter.

Steady-state streaming is launch-bound — the model's 12-layer conformer
encoder issues ~200 small kernels per chunk, and at 32 streams ×
25 chunks/sec the Python/launch overhead per chunk swamps the actual
compute. Replaying the captured graph collapses the entire forward into
a single launch, which is a ~20× speedup at fixed shape on this model.

Capture constraints
-------------------
* The persistent batched ``block_table`` / ``cache_seqlens`` /
  ``cnn_cache`` tensors must be allocated **before** the first capture and
  never reallocated; the graph captures the read sites by address.
* ``cache_t1`` is rounded up to a multiple of 64 (kernel ``N_BLOCK`` tile)
  per bucket so the kernel's per-tile block-table reads stay in-bounds.
* The encoder is invoked with ``offset`` as a per-stream int32 tensor
  (heterogeneous-offset code path) regardless of whether the actual batch
  is homogeneous — the captured code path must be stable.
* The CNN buffer rows referenced by ``slot_ids`` are snapshotted before
  capture and restored after; the warmup and captured ``_run`` both
  mutate those rows, but the first real replay must read the engine's
  pre-chunk state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import tvm_ffi

from oasr.cache.attention_cache import AttentionCacheManager
from oasr.cache.cnn_cache import CnnCacheManager
from oasr.cache.paged_kv import PagedKVCache
from oasr.cache.slot_cnn import SlotCnnCache
from oasr.features import FeatureConfig
from oasr.features.batched import (
    batched_fbank,
    batched_mfcc,
    supports_batched_fbank,
    supports_batched_mfcc,
)


# N_BLOCK tile size of the FMHA kernel; T_kv must be a multiple of this.
_KERNEL_N_BLOCK = 64


def round_up_bucket(cache_t1: int, granularity: int = _KERNEL_N_BLOCK) -> int:
    """Round ``cache_t1`` up to the next ``granularity`` multiple."""
    return ((cache_t1 + granularity - 1) // granularity) * granularity


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
    log_probs_out: torch.Tensor


class GraphedEncoderForward:
    """Lazy CUDA-Graph cache for the batched paged encoder forward.

    Parameters
    ----------
    model : ConformerModel
        The Conformer model whose ``forward_chunk_paged`` is captured.
    att_mgr, cnn_mgr : cache managers
        Provide the persistent batched paging / CNN-cache tensors that the
        graph reads from.
    cache_dtype, device :
        Dtype / device of the persistent state.
    window : int
        Input feature frames per chunk (``cfg.decoding_window``).
    feat_dim : int
        Input feature dimensionality.
    cnn_cache_frames : int
        CNN cache frames (== ``kernel_size - 1``).
    num_layers, hidden_dim : int
        Encoder dimensions used to allocate the persistent CNN buffer.
    pool : tuple of int, optional
        Shared CUDA Graph memory-pool handle (from
        ``torch.cuda.graph_pool_handle()``) used by every engine-level
        capture (encoder, feature extraction, CTC). Defaults to a private
        pool when ``None`` so direct instantiation in tests still works.
    """

    def __init__(
        self,
        model,
        att_mgr: AttentionCacheManager,
        cnn_mgr: CnnCacheManager,
        *,
        cache_dtype: torch.dtype,
        device: torch.device,
        window: int,
        feat_dim: int,
        cnn_cache_frames: int,
        num_layers: int,
        hidden_dim: int,
        pool: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._model = model
        self._att_mgr = att_mgr
        self._cnn_mgr = cnn_mgr
        self._cache_dtype = cache_dtype
        self._device = device
        self._window = window
        self._feat_dim = feat_dim
        self._cnn_cache_frames = cnn_cache_frames
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim

        # Captured graph cache, keyed by (B_active, cache_t1_bucket).
        # All captures share one CUDA Graph memory pool so that the
        # intermediate-tensor allocations from earlier captures don't
        # cumulatively fragment the regular caching allocator. The
        # caller must consume any captured output (log_probs) before the
        # next replay — that's already the case in the engine step loop
        # (CTC decode + commit happen synchronously after ``replay()``
        # returns).
        #
        # ``pool`` is optionally injected so this cache shares a single
        # ``torch.cuda.graph_pool_handle()`` with the other engine-level
        # graph caches (feature extraction, CTC). Passing ``None`` keeps
        # the legacy standalone behaviour for tests that instantiate this
        # cache directly without an ``ASREngine``.
        self._pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        # Keyed by ``(B, T_input, cache_t1_bucket)``. ``T_input`` is the
        # encoder input frame count for the chunk — the batched path uses
        # ``T_input == window`` for every cohort, and the B=1 fallback
        # ``_forward_single`` path lands here too with smaller ``T_input``
        # for partial / final windows.
        self._captured: Dict[Tuple[int, int, int], _CapturedShape] = {}
        # Steady-state streaming hits at most ``max_batch_size`` distinct B
        # values × ``ceil(max_offset / N_BLOCK)`` cache_t1 buckets. The B=1
        # single fallback adds a handful of ``T_input`` variants on top.
        # Cap generously so the typical workload is fully captured without
        # ever falling back to eager mode; ``_capture`` short-circuits
        # cleanly when the cap is hit.
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
        captured forward reads CNN left-context from the persistent
        ``cnn_mgr.buffer`` (at rows ``slot_ids``) and scatters new tails
        back into it in place.

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
        log_probs : Tensor or None
            ``(B, chunk_size, vocab_size)`` log-softmax output. **Aliases
            the captured buffer**; callers must consume it before the next
            replay or clone. Returns ``None`` when the per-shape capture
            cache is saturated (caller falls back to eager mode).
        """
        key = (B, T_input, cache_t1_bucket)
        state = self._captured.get(key)
        if state is None:
            if len(self._captured) >= self._max_captures:
                # Refuse to capture once the cache is saturated; the caller
                # will fall back to eager mode for this chunk.
                return None
            state = self._capture(
                B, T_input, cache_t1_bucket, xs, slot_ids, offsets,
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
        torch.index_select(self._att_mgr.block_table, 0, slot_ids,
                           out=state.batched_block_table)
        torch.index_select(self._att_mgr.cache_seqlens, 0, slot_ids,
                           out=state.batched_cache_seqlens)

        state.graph.replay()
        return state.log_probs_out

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

        The CNN cache is read **in place** from ``cnn_mgr.buffer`` (whose
        address is stable) at rows ``slot_ids_buf``. Both the warmup
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

        cnn_cache = SlotCnnCache(buffer=self._cnn_mgr.buffer, slot_ids=slot_ids_buf)

        def _run() -> torch.Tensor:
            return self._model.forward_chunk_paged(
                xs_buf, offset_buf, caches, cnn_cache,
                cache_t1=cache_t1_bucket,
            )

        # Snapshot the CNN buffer rows for the active slots so we can
        # restore them after capture. The warmup ``_run`` and the captured
        # ``_run`` both write to these rows; without restore, the first
        # real ``state.graph.replay()`` would read the post-capture state
        # instead of the engine's pre-chunk state.
        saved_cnn = self._cnn_mgr.buffer.index_select(1, slot_ids_buf).clone()

        # Warmup once on the default stream so cuBLAS / cuDNN finalise any
        # one-time workspace allocations before we open capture. Real
        # streaming writes K/V to the chunk's logical position; the
        # subsequent ``replay()`` overwrites that warmup write with the
        # caller's actual ``xs``.
        _run()
        torch.cuda.synchronize(self._device)
        # Restore so the captured ``_run`` reads the same pre-chunk CNN
        # state that the first real replay will see.
        self._cnn_mgr.buffer.index_copy_(1, slot_ids_buf, saved_cnn)

        graph = torch.cuda.CUDAGraph()
        # ``tvm_ffi.use_torch_stream(torch.cuda.graph(g))`` is the documented
        # path for capturing TVM-FFI kernel launches into a CUDA graph: it
        # opens the torch graph context and sets the FFI environment stream
        # to the active capture stream so the cute kernel's launch is
        # captured into ``graph`` (rather than escaping to the default
        # stream and tripping ``cudaErrorIllegalAddress`` on replay).
        with tvm_ffi.use_torch_stream(torch.cuda.graph(graph, pool=self._pool)):
            log_probs_out = _run()

        # Restore again so the first real replay (line below in caller)
        # reads the engine's pre-chunk CNN state.
        self._cnn_mgr.buffer.index_copy_(1, slot_ids_buf, saved_cnn)

        return _CapturedShape(
            graph=graph,
            xs_buf=xs_buf,
            slot_ids_buf=slot_ids_buf,
            offset_buf=offset_buf,
            batched_block_table=batched_bt,
            batched_cache_seqlens=batched_cs,
            log_probs_out=log_probs_out,
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
    padded_host_buf: torch.Tensor   # (B_bucket, T_pad) float32, pinned
    lengths_host_buf: torch.Tensor  # (B_bucket,)       int64,   pinned
    # Device buffers — captured-graph destinations for the H2D copies.
    wav_gpu_buf: torch.Tensor       # (B_bucket, T_pad) float32, cuda
    lengths_gpu_buf: torch.Tensor   # (B_bucket,)       int64,   cuda
    # Captured output. Aliases the graph pool's output allocation; callers
    # must consume (or copy) before the next replay.
    feats_out: torch.Tensor         # (B_bucket, num_frames_max, feat_dim)


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
        self._is_mfcc = (feature_config.feature_type == "mfcc")

        frame_len = int(feature_config.frame_length_samples)
        frame_shift = int(feature_config.frame_shift_samples)
        # Worst-case combined waveform: this step's chunk_samples plus the
        # maximum carry-over remainder from the previous step (a partial
        # frame, ``frame_len - 1`` samples).
        self._t_pad = int(chunk_samples) + frame_len - 1
        self._frame_len = frame_len
        self._frame_shift = frame_shift
        self._feat_dim = int(feature_config.output_dim)

        # Build the bucket list.
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
                    "feature_graph_batch_buckets must contain at least one "
                    "positive integer"
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
            if T < self._t_pad:
                state.padded_host_buf[:B_active, T:].zero_()
            if T > 0:
                state.padded_host_buf[:B_active, :T].copy_(padded_cpu)
            state.lengths_host_buf[:B_active].copy_(lengths_cpu)
        state.graph.replay()
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

        wav_gpu = torch.zeros(bucket, T_pad, dtype=torch.float32, device=device)
        lengths_gpu = torch.zeros(bucket, dtype=torch.int64, device=device)

        # Seed lengths_host with T_pad so the warmup hits the same kernel
        # tile shapes the captured replay will hit.
        lengths_host.fill_(T_pad)

        batched_fn = batched_mfcc if self._is_mfcc else batched_fbank
        fcfg = self._fcfg
        out_dtype = self._output_dtype

        def _run() -> torch.Tensor:
            wav_gpu.copy_(padded_host, non_blocking=True)
            lengths_gpu.copy_(lengths_host, non_blocking=True)
            feats_f32, _ = batched_fn(wav_gpu, lengths_gpu, fcfg)
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
            wav_gpu_buf=wav_gpu,
            lengths_gpu_buf=lengths_gpu,
            feats_out=feats_out,
        )


# Re-export so callers can probe backend support without reaching into
# ``oasr.features.batched`` directly.
__all__ = [
    "round_up_bucket",
    "GraphedEncoderForward",
    "GraphedFeatureExtraction",
    "supports_batched_fbank",
    "supports_batched_mfcc",
]
