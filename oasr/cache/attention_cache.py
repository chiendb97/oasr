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
from typing import Dict, List, Tuple

import torch

from oasr.cache.block_pool import BlockPool
from oasr.cache.paged_kv import PagedKVCache
from oasr.cache.types import CacheConfig
from oasr.utils.staging import to_device


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
            config.max_batch_size,
            config.max_blocks_per_seq,
            dtype=torch.int32,
            device=config.device,
        )
        self._cache_seqlens = torch.zeros(
            config.max_batch_size,
            dtype=torch.int32,
            device=config.device,
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
            raise ValueError(f"slot_id {slot_id} out of range [0, {self._config.max_batch_size})")
        for s in self._streams.values():
            if s.slot_id == slot_id:
                raise ValueError(f"slot_id {slot_id} already in use")
        self._streams[stream_id] = _StreamKVState(slot_id=slot_id)
        # Reset persistent rows for the new stream.
        self._block_table[slot_id].zero_()
        self._cache_seqlens[slot_id] = 0
        if self._config.prefill_kv_window:
            self._prefill_window(stream_id)

    def _prefill_window(self, stream_id: int) -> None:
        """Give a new stream its whole retained window up front, zero-filled.

        For an encoder whose attention span is a trained constant
        (``CacheConfig.prefill_kv_window``).  Without this a young stream reports a
        shorter ``cache_seqlens`` than its peers, and a Transformer-XL
        relative-position table's distances are ``cache_seqlens + i - j`` — so the
        cohort would need one table per distinct cache length, which at ``B = 32``
        costs more than the encoder layer it feeds.  Prefilling makes the length
        uniform from chunk one, at the price of attending over zeros the encoder's
        own bias masks out.

        The blocks are **zeroed**, not merely reserved: ``oasr.fmha`` gives a
        past-the-length column zero softmax weight but still multiplies it into
        ``P @ V``, so a ``NaN`` there would propagate where no mask can intercept
        it.  Finite stale data is inert; uninitialised memory is not.
        """
        state = self._get_state(stream_id)
        blocks = self._config.max_logical_blocks
        assert blocks is not None  # guaranteed by CacheConfig.__post_init__
        block_ids = self._pool.allocate(blocks)
        for logical_idx, block_id in enumerate(block_ids):
            state.logical_blocks.append(block_id)
            self._block_table[state.slot_id, logical_idx] = block_id
        self._pool.zero_blocks(block_ids)
        state.num_committed_frames = blocks * self._config.block_size_frames
        self._cache_seqlens[state.slot_id] = state.num_committed_frames

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
    # Capacity
    # ------------------------------------------------------------------

    def at_capacity(self, stream_id: int) -> bool:
        """Whether ``stream_id`` cannot accept another chunk's worth of cache.

        With eviction enabled a stream is never at capacity — the oldest block
        is recycled instead.  With unlimited history (``num_left_chunks < 0``,
        the default) growth is bounded by
        :attr:`~oasr.cache.CacheConfig.blocks_per_stream`, and additionally by
        the pool actually having a free block.  The streaming backend consults
        this **before** dispatching a chunk so an exhausted stream is finalized
        cleanly instead of raising ``BlockPool exhausted`` (or indexing past the
        block table) from inside the forward.
        """
        if self._config.max_logical_blocks is not None:
            return False  # eviction recycles a block; growth is bounded already
        state = self._get_state(stream_id)
        held = len(state.logical_blocks)
        if held + 1 > self._config.blocks_per_stream:
            return True
        if held + 1 > self._block_table.size(1):
            return True
        return self._pool.num_free_blocks < 1

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

        Evicts **first** when history is capped.  Eviction used to run only at
        commit time, i.e. *after* this allocation, so a stream already holding
        its full ``max_logical_blocks`` had to be handed a block before the one
        it was about to give back — meaning the pool silently needed
        ``max_batch_size`` blocks of headroom beyond the invariant the config
        documents, and a pool sized exactly to
        ``max_batch_size * max_logical_blocks`` raised ``BlockPool exhausted``
        the moment the cap was reached.  Reclaiming before allocating makes the
        documented sizing correct and is the natural order anyway.

        Steady state is unchanged: evicting to ``max_logical_blocks - 1`` and
        then appending leaves exactly ``max_logical_blocks``, and the block
        evicted is the same oldest one commit-time eviction would have taken.
        """
        if not stream_ids:
            return

        # Make room before asking for a block (see above).
        self.evict_oldest_batched(stream_ids, headroom=1)

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
        slots_t = to_device(slots, dtype=torch.long, device=device)
        logical_t = to_device(logical_indices, dtype=torch.long, device=device)
        block_ids_t = to_device(block_ids, dtype=torch.int32, device=device)
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
        block_table_view = self._block_table[slot : slot + 1]
        cache_seqlens_view = self._cache_seqlens[slot : slot + 1]
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
        self,
        stream_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(block_table_row, cache_seqlens_row)`` views for the stream.

        Both views are zero-copy slices of the persistent batched tensors.
        """
        state = self._get_state(stream_id)
        slot = state.slot_id
        return (
            self._block_table[slot : slot + 1],
            self._cache_seqlens[slot : slot + 1],
        )

    def get_batched_paged_caches(
        self,
        slot_ids_gpu: torch.Tensor,
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
        # Staged through pinned memory (:func:`oasr.utils.staging.to_device`):
        # this is the first host->device copy after the encoder forward is
        # issued, so a pageable one waits out the whole forward and the host can
        # never run a step ahead of the device.  Measured at 1.7 ms per call on
        # a 32-stream pool — 26% of streaming wall time across all such sites.
        slots_t = to_device(slots, dtype=torch.long, device=self._block_table.device)
        # In-place batched advance — one kernel for all B updates.
        self._cache_seqlens[slots_t] += chunk_frames
        self.evict_oldest_batched(stream_ids)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_oldest_batched(self, stream_ids: List[int], headroom: int = 0) -> None:
        """Evict over-cap blocks for a whole group in a constant number of kernels.

        ``headroom`` reserves that many logical slots below the cap — the
        allocation path passes 1 so a stream at its cap gives a block back
        *before* asking for the next one (see :meth:`prepare_chunks_batched`).

        The per-stream version below ran a GPU ``.clone()`` of the block-table
        row plus three scalar GPU writes **per stream per chunk**, so a finite
        ``num_left_chunks`` cost ~4B tiny launches every streaming step — which
        is why nobody enabled it.  Measured on a 16-stream pool: 10.7% at
        ``num_left_chunks=8``, 15.4% at 4.

        Here the *decision* is pure host-side bookkeeping (no sync — the block
        lists are Python), and the device work collapses to a fixed handful of
        batched ops per eviction round: one gather-scatter for the row shift, one
        scatter to blank the vacated column, one scatter for ``cache_seqlens``.
        Steady state needs exactly one round, since the cap is re-checked after
        every committed chunk.

        Deliberately **not** the ring block table the review proposed.  A ring
        (per-stream ``first_logical`` with the kernel indexing
        ``block_table[(first + i) % width]``) would remove the shift entirely,
        but it is a *kernel* change to the paged CuteDSL FMHA — a path that
        currently has two known defects (the masked-tile NaN and the head_dim-32
        stale read).  Batching removes the launch count, which is what the cost
        actually was, at no kernel risk.  The ring stays worthwhile only if the
        remaining shift ever shows up in a profile.
        """
        max_blocks = self._config.max_logical_blocks
        if max_blocks is None or not stream_ids:
            return  # unlimited history

        block_size = self._config.block_size_frames
        device = self._block_table.device
        limit = max(0, max_blocks - int(headroom))
        while True:
            # -- host-side round: who is over cap, and by how much --------
            freed: List[int] = []
            slots: List[int] = []
            kept: List[int] = []
            for sid in stream_ids:
                state = self._streams[sid]
                if len(state.logical_blocks) <= limit:
                    continue
                freed.append(state.logical_blocks.pop(0))
                state.num_committed_frames = len(state.logical_blocks) * block_size
                slots.append(state.slot_id)
                kept.append(len(state.logical_blocks))
            if not slots:
                return

            self._pool.free(freed)
            slots_t = to_device(slots, dtype=torch.long, device=device)
            kept_t = to_device(kept, dtype=torch.long, device=device)
            width = self._block_table.size(1)
            # Shift every affected row left by one.  Advanced indexing on the
            # right builds a copy, so source and destination cannot alias.
            self._block_table[slots_t, : width - 1] = self._block_table[slots_t, 1:width]
            # Blank the column each row just vacated (its new logical end).
            self._block_table[slots_t, kept_t] = 0
            self._cache_seqlens[slots_t] = (kept_t * block_size).to(self._cache_seqlens.dtype)

    def _evict_oldest(self, stream_id: int) -> None:
        """Single-stream eviction — kept for the single-stream commit path."""
        self.evict_oldest_batched([stream_id])

    def _get_state(self, stream_id: int) -> _StreamKVState:
        try:
            return self._streams[stream_id]
        except KeyError:
            raise KeyError(f"Attention cache for stream {stream_id} not allocated.") from None
