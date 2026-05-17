# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paged attention KV cache manager for streaming conformer inference.

Each active stream owns a **slot id** in ``[0, max_batch_size)``; the manager
holds two persistent batched tensors indexed by slot:

* ``block_table`` — ``(max_batch_size, max_blocks_per_seq)`` int32, logical
  block ids per stream.
* ``cache_seqlens`` — ``(max_batch_size,)`` int32, committed encoder frames
  per stream.

The persistent layout lets the batched paged forward fetch all B_active
streams' paging metadata via two ``index_select`` calls — no per-stream
Python loop, no per-chunk ``torch.cat``. Per-stream descriptors are still
exposed (``get_paged_caches``, ``get_paged_state_views``) as zero-copy
views into the persistent tensors.

Streaming is paged-only; dense ``forward_chunk`` and its accompanying
``commit`` / ``get_stacked_cache`` API were removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from oasr.cache.block_pool import BlockPool
from oasr.cache.paged_kv import PagedKVCache
from oasr.cache.types import CacheConfig


@dataclass
class _StreamKVState:
    """Internal per-stream paging state.

    Attributes
    ----------
    slot_id : int
        Row index in the persistent batched block_table / cache_seqlens.
    logical_blocks : list[int]
        Ordered list of physical block IDs (oldest → newest).
    num_committed_frames : int
        Total encoder-output frames written into this stream's K/V pool.
    """

    slot_id: int
    logical_blocks: List[int] = field(default_factory=list)
    num_committed_frames: int = 0


class AttentionCacheManager:
    """Manages paged attention KV cache for all active streams.

    Holds one persistent ``(max_batch_size, max_blocks_per_seq)`` int32
    ``block_table`` and ``(max_batch_size,)`` int32 ``cache_seqlens``
    on the cache device. Each admitted stream is bound to a slot id; the
    persistent rows for that slot store its paging metadata. Per-stream
    views are zero-copy slices of the global tensors.

    Parameters
    ----------
    block_pool : BlockPool
        Shared physical block pool.
    config : CacheConfig
        Cache configuration (must define ``max_batch_size``,
        ``max_blocks_per_seq``).
    """

    def __init__(self, block_pool: BlockPool, config: CacheConfig) -> None:
        self._pool = block_pool
        self._config = config
        self._streams: Dict[int, _StreamKVState] = {}

        # Persistent batched paging tensors. Allocated once at construction
        # so the batched paged forward only needs two ``index_select`` calls
        # to pull active-batch metadata, not a Python loop over B streams.
        self._block_table = torch.zeros(
            config.max_batch_size, config.max_blocks_per_seq,
            dtype=torch.int32, device=config.device,
        )
        self._cache_seqlens = torch.zeros(
            config.max_batch_size, dtype=torch.int32, device=config.device,
        )

        # Pre-built per-layer PagedKVCache descriptors pointing at the FULL
        # persistent block_table / cache_seqlens. The batched paged forward
        # builds B_active-row views from these on each chunk.
        self._persistent_caches: List[PagedKVCache] = [
            PagedKVCache(
                k_cache=self._pool.get_kv_view(layer)[0],
                v_cache=self._pool.get_kv_view(layer)[1],
                block_table=self._block_table,
                cache_seqlens=self._cache_seqlens,
                block_size=config.block_size_frames,
                host_seqlen_max=0,
            )
            for layer in range(config.num_layers)
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def block_table(self) -> torch.Tensor:
        """The persistent ``(max_batch_size, max_blocks_per_seq)`` block_table."""
        return self._block_table

    @property
    def cache_seqlens(self) -> torch.Tensor:
        """The persistent ``(max_batch_size,)`` cache_seqlens."""
        return self._cache_seqlens

    @property
    def num_layers(self) -> int:
        return self._config.num_layers

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def allocate_stream(self, stream_id: int, slot_id: int) -> None:
        """Register a new stream at a given slot id.

        Parameters
        ----------
        stream_id : int
            Unique stream identifier.
        slot_id : int
            Row index in the persistent batched block_table / cache_seqlens.
            Must be in ``[0, max_batch_size)`` and not already in use.

        Raises
        ------
        ValueError
            If ``stream_id`` is already allocated or ``slot_id`` is out of
            range / in use.
        """
        if stream_id in self._streams:
            raise ValueError(f"Attention cache for stream {stream_id} already allocated.")
        if not (0 <= slot_id < self._config.max_batch_size):
            raise ValueError(
                f"slot_id {slot_id} out of range [0, {self._config.max_batch_size})"
            )
        for s in self._streams.values():
            if s.slot_id == slot_id:
                raise ValueError(f"slot_id {slot_id} already in use")
        self._streams[stream_id] = _StreamKVState(slot_id=slot_id)
        # Reset persistent rows for the new stream.
        self._block_table[slot_id].zero_()
        self._cache_seqlens[slot_id] = 0

    def free_stream(self, stream_id: int) -> None:
        """Release all physical blocks for a stream and remove it.

        The slot is left tied to the freed state until the caller (engine)
        releases it back to the StreamSlotPool.
        """
        state = self._get_state(stream_id)
        if state.logical_blocks:
            self._pool.free(state.logical_blocks)
        del self._streams[stream_id]

    def slot_of(self, stream_id: int) -> int:
        """Return the slot id bound to ``stream_id``."""
        return self._get_state(stream_id).slot_id

    # ------------------------------------------------------------------
    # Paged-mode access and mutation
    # ------------------------------------------------------------------

    def prepare_chunk(self, stream_id: int) -> None:
        """Allocate the next physical block and update the block table.

        Must be called **before** ``get_paged_caches`` and
        ``forward_chunk_paged`` so the block table contains a valid entry
        for the frames about to be written.
        """
        state = self._get_state(stream_id)
        (block_id,) = self._pool.allocate(1)
        state.logical_blocks.append(block_id)
        logical_idx = len(state.logical_blocks) - 1
        self._block_table[state.slot_id, logical_idx] = block_id

    def prepare_chunks_batched(self, stream_ids: List[int]) -> None:
        """Allocate one new physical block for each of ``stream_ids``.

        Issues a single ``BlockPool.allocate(B)`` plus a batched scatter
        into the persistent block_table — replacing the per-stream scalar
        stores that the old per-stream-tensor layout required.
        """
        if not stream_ids:
            return

        # One allocator call for all B blocks.
        block_ids = self._pool.allocate(len(stream_ids))

        # Gather the per-stream (slot, logical_idx) targets host-side; one
        # batched scatter onto the persistent block_table replaces B scalar
        # writes.
        slots: List[int] = []
        logical_indices: List[int] = []
        for sid, block_id in zip(stream_ids, block_ids):
            state = self._streams[sid]
            state.logical_blocks.append(block_id)
            logical_indices.append(len(state.logical_blocks) - 1)
            slots.append(state.slot_id)

        device = self._block_table.device
        slots_t = torch.tensor(slots, dtype=torch.long, device=device)
        logical_t = torch.tensor(logical_indices, dtype=torch.long, device=device)
        block_ids_t = torch.tensor(block_ids, dtype=torch.int32, device=device)
        self._block_table[slots_t, logical_t] = block_ids_t

    def get_paged_caches(self, stream_id: int) -> List[PagedKVCache]:
        """Return one :class:`PagedKVCache` per encoder layer for the stream.

        Used by per-stream fallback paths (e.g. partial/final windows). The
        returned descriptors are ``(1, max_blocks_per_seq)`` / ``(1,)``
        zero-copy slices of the persistent batched tensors.

        Raises
        ------
        RuntimeError
            If ``stream_id`` has not been admitted.
        """
        state = self._get_state(stream_id)
        cfg = self._config
        slot = state.slot_id
        block_table_view = self._block_table[slot: slot + 1]
        cache_seqlens_view = self._cache_seqlens[slot: slot + 1]
        host_seqlen = state.num_committed_frames
        caches: List[PagedKVCache] = []
        for layer in range(cfg.num_layers):
            k_view, v_view = self._pool.get_kv_view(layer)
            caches.append(
                PagedKVCache(
                    k_cache=k_view,
                    v_cache=v_view,
                    block_table=block_table_view,
                    cache_seqlens=cache_seqlens_view,
                    block_size=cfg.block_size_frames,
                    host_seqlen_max=host_seqlen,
                )
            )
        return caches

    def get_paged_state_views(
        self, stream_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(block_table_row, cache_seqlens_row)`` views for the stream.

        Both views are zero-copy slices of the persistent batched tensors.
        """
        state = self._get_state(stream_id)
        slot = state.slot_id
        return (
            self._block_table[slot: slot + 1],
            self._cache_seqlens[slot: slot + 1],
        )

    def get_batched_paged_caches(
        self, slot_ids_gpu: torch.Tensor,
    ) -> Tuple[List[PagedKVCache], torch.Tensor, torch.Tensor]:
        """Return per-layer paged caches indexed by an active-batch slot tensor.

        Parameters
        ----------
        slot_ids_gpu : Tensor
            ``(B_active,)`` int64 / int32 tensor on the cache device. Picks
            the rows of the persistent batched ``block_table`` /
            ``cache_seqlens`` to expose to the kernel.

        Returns
        -------
        caches : list[PagedKVCache]
            One :class:`PagedKVCache` per encoder layer. ``block_table`` and
            ``cache_seqlens`` are shared across layers; only ``k_cache`` /
            ``v_cache`` differ.
        batched_block_table : Tensor
            The ``(B_active, max_blocks_per_seq)`` gather; returned so the
            caller can reuse it for kernel-size trimming / debugging.
        batched_cache_seqlens : Tensor
            The ``(B_active,)`` gather.
        """
        cfg = self._config
        # Single index_select call replaces the old per-stream ``torch.cat``
        # of B ``(1, max_blocks_per_seq)`` rows.
        batched_bt = self._block_table.index_select(0, slot_ids_gpu)
        batched_cs = self._cache_seqlens.index_select(0, slot_ids_gpu)
        caches: List[PagedKVCache] = []
        for layer in range(cfg.num_layers):
            base = self._persistent_caches[layer]
            caches.append(
                PagedKVCache(
                    k_cache=base.k_cache,
                    v_cache=base.v_cache,
                    block_table=batched_bt,
                    cache_seqlens=batched_cs,
                    block_size=cfg.block_size_frames,
                    host_seqlen_max=0,  # the caller passes cache_t1 host-side
                )
            )
        return caches, batched_bt, batched_cs

    def commit_chunk_paged(self, stream_id: int, chunk_frames: int) -> None:
        """Advance ``cache_seqlens`` after a paged forward pass and evict if needed.

        The attention layer wrote K/V directly into the pool via
        :meth:`PagedKVCache.write_kv_chunk`; this method only updates the
        host-side counter, the persistent ``cache_seqlens`` row, and runs
        per-stream eviction.
        """
        state = self._get_state(stream_id)
        state.num_committed_frames += chunk_frames
        self._cache_seqlens[state.slot_id] = state.num_committed_frames
        self._evict_oldest(stream_id)

    def commit_chunks_paged_batched(
        self,
        stream_ids: List[int],
        chunk_frames: int,
    ) -> None:
        """Batched ``commit_chunk_paged`` for a group of streams.

        Advances ``cache_seqlens`` for ``B`` streams via a single scatter,
        then runs per-stream eviction host-side.
        """
        if not stream_ids:
            return
        slots: List[int] = []
        for sid in stream_ids:
            state = self._streams[sid]
            state.num_committed_frames += chunk_frames
            slots.append(state.slot_id)
        slots_t = torch.tensor(slots, dtype=torch.long, device=self._block_table.device)
        # In-place batched advance — one kernel for all B updates.
        self._cache_seqlens[slots_t] += chunk_frames
        for sid in stream_ids:
            self._evict_oldest(sid)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_oldest(self, stream_id: int) -> None:
        """Evict the oldest block if the stream exceeds ``max_logical_blocks``."""
        state = self._get_state(stream_id)
        cfg = self._config
        max_blocks = cfg.max_logical_blocks
        if max_blocks is None:
            return  # unlimited history

        slot = state.slot_id
        while len(state.logical_blocks) > max_blocks:
            evicted = state.logical_blocks.pop(0)
            self._pool.free([evicted])
            state.num_committed_frames = len(state.logical_blocks) * cfg.block_size_frames
            self._cache_seqlens[slot] = state.num_committed_frames
            # Shift this stream's block_table row left by one entry.
            n = len(state.logical_blocks)
            self._block_table[slot, :n] = self._block_table[slot, 1: n + 1].clone()
            self._block_table[slot, n] = 0

    def _get_state(self, stream_id: int) -> _StreamKVState:
        try:
            return self._streams[stream_id]
        except KeyError:
            raise KeyError(
                f"Attention cache for stream {stream_id} not allocated."
            ) from None
