# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph cache for the streaming encoder forward.

A standalone PyTorch :class:`torch.cuda.CUDAGraph` is captured per
``(B_active, cache_t1_bucket)`` shape. Captures are lazy on first
encounter; replays reuse pre-allocated input buffers (``xs``, ``cnn``,
``slot_ids``, ``offset``) plus the captured output buffers
(``log_probs``, ``new_cnn``).

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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import tvm_ffi

from oasr.cache.attention_cache import AttentionCacheManager
from oasr.cache.cnn_cache import CnnCacheManager
from oasr.cache.paged_kv import PagedKVCache


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
    cnn_in_buf: torch.Tensor
    slot_ids_buf: torch.Tensor
    offset_buf: torch.Tensor
    # Persistent gather buffers that mirror the persistent state for the
    # active slots. The captured kernel reads from these addresses; before
    # each replay we refresh them with the current state via copy_().
    batched_block_table: torch.Tensor
    batched_cache_seqlens: torch.Tensor
    log_probs_out: torch.Tensor
    new_cnn_out: torch.Tensor


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
        # caller must consume any captured output (log_probs / new_cnn)
        # before the next replay — that's already the case in the engine
        # step loop (CTC decode + commit happen synchronously after
        # ``replay()`` returns).
        self._pool = torch.cuda.graph_pool_handle()
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
        cnn_in: torch.Tensor,
        slot_ids: torch.Tensor,
        offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replay (or first-capture) the encoder forward at the given shape.

        On the first call for a new ``(B, cache_t1_bucket)`` shape the graph
        is captured using the caller-provided ``slot_ids`` / ``offsets`` (so
        the capture-time warmup runs on the **actual** persistent paging
        state rather than dummy state). Subsequent replays just refresh the
        pre-allocated input buffers and trigger ``cudaGraphLaunch``.

        Parameters
        ----------
        B : int
            Active batch size for this chunk.
        cache_t1_bucket : int
            Bucket-aligned host-side cache_t1 (multiple of ``_KERNEL_N_BLOCK``).
        xs : Tensor
            ``(B, window, feat_dim)`` encoder input.
        cnn_in : Tensor
            ``(num_layers, B, cnn_cache_frames, hidden_dim)`` CNN cache.
        slot_ids : Tensor
            ``(B,)`` int64 slot ids (host-side gather into persistent tables).
        offsets : Tensor
            ``(B,)`` int32 per-stream encoder-frame offsets.

        Returns
        -------
        log_probs : Tensor
            ``(B, chunk_size, vocab_size)`` log-softmax output. **Aliases
            the captured buffer**; callers must consume it before the next
            replay or clone.
        new_cnn : Tensor
            ``(num_layers, B, cnn_cache_frames, hidden_dim)`` new CNN cache.
            Same aliasing caveat as ``log_probs``.
        """
        key = (B, T_input, cache_t1_bucket)
        state = self._captured.get(key)
        if state is None:
            if len(self._captured) >= self._max_captures:
                # Refuse to capture once the cache is saturated; the caller
                # will fall back to eager mode for this chunk.
                return None  # type: ignore[return-value]
            state = self._capture(
                B, T_input, cache_t1_bucket, xs, cnn_in, slot_ids, offsets,
            )
            self._captured[key] = state

        # Refresh captured input buffers with current per-chunk state.
        # ``slot_ids`` / ``offsets`` are kept for downstream code that may
        # read them but the kernel doesn't see them at replay — it reads
        # the gather buffers below. The graph captured ``index_select``
        # writes into ``batched_block_table`` / ``batched_cache_seqlens``
        # whose addresses we keep stable across replays; refreshing those
        # gather buffers from the persistent state is the only way the
        # captured kernel sees post-``prepare_chunks_batched`` /
        # ``commit_chunks_paged_batched`` updates.
        state.xs_buf.copy_(xs)
        state.cnn_in_buf.copy_(cnn_in)
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
        return state.log_probs_out, state.new_cnn_out

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
        cnn_in: torch.Tensor,
        slot_ids: torch.Tensor,
        offsets: torch.Tensor,
    ) -> _CapturedShape:
        """Allocate persistent buffers, warm up, then capture the forward.

        The captured graph reads from **pre-allocated** gather buffers for
        ``block_table`` / ``cache_seqlens`` (stable addresses) instead of
        an in-graph ``index_select`` result (whose destination address
        could move across captures). Before each replay the caller
        refreshes these gather buffers from the persistent state.
        """
        # Pre-allocate input buffers, primed with the caller's tensors so
        # the warmup run operates on the real persistent paging state.
        xs_buf = xs.clone()
        cnn_in_buf = cnn_in.clone()
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

        def _run() -> Tuple[torch.Tensor, torch.Tensor]:
            return self._model.forward_chunk_paged(
                xs_buf, offset_buf, caches, cnn_in_buf,
                cache_t1=cache_t1_bucket,
            )

        # Warmup once on the default stream so cuBLAS / cuDNN finalise any
        # one-time workspace allocations before we open capture. Real
        # streaming writes K/V to the chunk's logical position; the
        # subsequent ``replay()`` overwrites that warmup write with the
        # caller's actual ``xs``.
        _run()
        torch.cuda.synchronize(self._device)

        graph = torch.cuda.CUDAGraph()
        # ``tvm_ffi.use_torch_stream(torch.cuda.graph(g))`` is the documented
        # path for capturing TVM-FFI kernel launches into a CUDA graph: it
        # opens the torch graph context and sets the FFI environment stream
        # to the active capture stream so the cute kernel's launch is
        # captured into ``graph`` (rather than escaping to the default
        # stream and tripping ``cudaErrorIllegalAddress`` on replay).
        with tvm_ffi.use_torch_stream(torch.cuda.graph(graph, pool=self._pool)):
            log_probs_out, new_cnn_out = _run()

        return _CapturedShape(
            graph=graph,
            xs_buf=xs_buf,
            cnn_in_buf=cnn_in_buf,
            slot_ids_buf=slot_ids_buf,
            offset_buf=offset_buf,
            batched_block_table=batched_bt,
            batched_cache_seqlens=batched_cs,
            log_probs_out=log_probs_out,
            new_cnn_out=new_cnn_out,
        )
