# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paged attention KV cache manager for streaming conformer inference.

Each active stream maintains a logical-to-physical block mapping backed by
a shared :class:`BlockPool`. Per-stream usage:

* :meth:`AttentionCacheManager.prepare_chunk` allocates the next physical
  block in advance (or :meth:`prepare_chunks_batched` for B streams).
* :meth:`AttentionCacheManager.get_paged_caches` returns one
  :class:`PagedKVCache` per encoder layer; ``forward_chunk_paged`` writes
  K/V directly into the pool via :meth:`PagedKVCache.write_kv_chunk`.
* :meth:`AttentionCacheManager.commit_chunk_paged` increments
  ``cache_seqlens`` and evicts the oldest block when the left-context limit
  is reached.

The dense-cache path (``forward_chunk`` / ``commit`` / ``get_stacked_cache``)
was removed: streaming is paged-only.
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
    """Internal per-stream logical-to-physical block mapping.

    Attributes
    ----------
    logical_blocks : list[int]
        Ordered list of physical block IDs (oldest → newest).
    num_committed_frames : int
        Total encoder-output frames written into this stream's K/V pool.
    block_table : Tensor or None
        ``(1, max_blocks_per_seq)`` int32 on the cache device. Allocated on
        first use; ``None`` until then.
    cache_seqlens : Tensor or None
        ``(1,)`` int32 scalar tracking committed frames for paged attention.
        Shared across all layers of the same stream.
    """

    logical_blocks: List[int] = field(default_factory=list)
    num_committed_frames: int = 0
    block_table: Optional[torch.Tensor] = None
    cache_seqlens: Optional[torch.Tensor] = None


class AttentionCacheManager:
    """Manages paged attention KV cache for all active streams.

    Parameters
    ----------
    block_pool : BlockPool
        Shared physical block pool.
    config : CacheConfig
        Cache configuration.

    Examples
    --------
    >>> mgr = AttentionCacheManager(pool, config)
    >>> mgr.allocate_stream(42)
    >>> mgr.prepare_chunk(42)                  # allocate block for next write
    >>> att_caches = mgr.get_paged_caches(42)  # List[PagedKVCache]
    >>> xs, cnn = model.forward_chunk_paged(xs, offset, att_caches, cnn)
    >>> mgr.commit_chunk_paged(42, chunk_size)
    >>> mgr.free_stream(42)
    """

    def __init__(self, block_pool: BlockPool, config: CacheConfig) -> None:
        self._pool = block_pool
        self._config = config
        self._streams: Dict[int, _StreamKVState] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def allocate_stream(self, stream_id: int) -> None:
        """Register a new stream with an empty KV cache.

        Parameters
        ----------
        stream_id : int
            Unique stream identifier.

        Raises
        ------
        ValueError
            If ``stream_id`` is already allocated.
        """
        if stream_id in self._streams:
            raise ValueError(f"Attention cache for stream {stream_id} already allocated.")
        self._streams[stream_id] = _StreamKVState()

    def free_stream(self, stream_id: int) -> None:
        """Release all physical blocks for a stream and remove it.

        Parameters
        ----------
        stream_id : int
            Stream to release.

        Raises
        ------
        KeyError
            If ``stream_id`` is not allocated.
        """
        state = self._get_state(stream_id)
        if state.logical_blocks:
            self._pool.free(state.logical_blocks)
        del self._streams[stream_id]

    # ------------------------------------------------------------------
    # Paged-mode access and mutation
    # ------------------------------------------------------------------

    def prepare_chunk(self, stream_id: int) -> None:
        """Allocate the next physical block and update the block table.

        Must be called **before** ``get_paged_caches`` and
        ``forward_chunk_paged`` so that the block table contains a valid entry
        for the frames about to be written.

        Parameters
        ----------
        stream_id : int
            Stream identifier.

        Raises
        ------
        RuntimeError
            If the pool has no free blocks.
        """
        state = self._get_state(stream_id)
        cfg = self._config

        # Lazily allocate block_table and cache_seqlens on first use.
        if state.block_table is None:
            state.block_table = torch.zeros(
                1, cfg.max_blocks_per_seq, dtype=torch.int32, device=cfg.device
            )
            state.cache_seqlens = torch.zeros(1, dtype=torch.int32, device=cfg.device)

        (block_id,) = self._pool.allocate(1)
        state.logical_blocks.append(block_id)
        logical_idx = len(state.logical_blocks) - 1
        state.block_table[0, logical_idx] = block_id

    def get_paged_caches(self, stream_id: int) -> List[PagedKVCache]:
        """Return one :class:`PagedKVCache` per encoder layer for the stream.

        All returned objects share the same ``block_table`` and
        ``cache_seqlens`` tensors.  Only ``k_cache`` / ``v_cache`` differ.

        Parameters
        ----------
        stream_id : int
            Stream identifier.  Must have been prepared with
            :meth:`prepare_chunk` at least once.

        Returns
        -------
        list[PagedKVCache]
            One entry per encoder layer (length ``config.num_layers``).

        Raises
        ------
        RuntimeError
            If :meth:`prepare_chunk` was never called for this stream.
        """
        state = self._get_state(stream_id)
        if state.block_table is None:
            raise RuntimeError(
                f"Stream {stream_id}: call prepare_chunk() before get_paged_caches()."
            )
        cfg = self._config
        host_seqlen = state.num_committed_frames
        caches = []
        for layer in range(cfg.num_layers):
            k_view, v_view = self._pool.get_kv_view(layer)
            caches.append(
                PagedKVCache(
                    k_cache=k_view,
                    v_cache=v_view,
                    block_table=state.block_table,
                    cache_seqlens=state.cache_seqlens,
                    block_size=cfg.block_size_frames,
                    host_seqlen_max=host_seqlen,
                )
            )
        return caches

    def prepare_chunks_batched(self, stream_ids: List[int]) -> None:
        """Allocate one new physical block for each of ``stream_ids``.

        Cheaper than calling :meth:`prepare_chunk` once per stream when
        the engine is dispatching a batched paged forward — does ONE
        :meth:`BlockPool.allocate` call for ``B`` blocks and writes the
        resulting IDs into each stream's GPU block_table via a single
        scatter-style update.
        """
        if not stream_ids:
            return
        cfg = self._config
        # Lazily allocate block_table / cache_seqlens for any new streams.
        for sid in stream_ids:
            state = self._get_state(sid)
            if state.block_table is None:
                state.block_table = torch.zeros(
                    1, cfg.max_blocks_per_seq, dtype=torch.int32,
                    device=cfg.device,
                )
                state.cache_seqlens = torch.zeros(
                    1, dtype=torch.int32, device=cfg.device,
                )

        block_ids = self._pool.allocate(len(stream_ids))
        # Update host + GPU state per stream.  GPU writes are one-element
        # scalar stores into each stream's own block_table tensor — small
        # but unavoidable with per-stream block tables.
        for sid, block_id in zip(stream_ids, block_ids):
            state = self._streams[sid]
            state.logical_blocks.append(block_id)
            logical_idx = len(state.logical_blocks) - 1
            state.block_table[0, logical_idx] = block_id  # type: ignore[index]

    def get_paged_state_views(
        self, stream_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(block_table, cache_seqlens)`` shared across all layers.

        Cheaper than :meth:`get_paged_caches` for callers that only need
        the per-stream paging tensors (the batched paged forward stacks
        these across streams to build a single ``B``-row block-table).
        """
        state = self._get_state(stream_id)
        if state.block_table is None or state.cache_seqlens is None:
            raise RuntimeError(
                f"Stream {stream_id}: call prepare_chunk() before "
                f"get_paged_state_views()."
            )
        return state.block_table, state.cache_seqlens

    def commit_chunk_paged(self, stream_id: int, chunk_frames: int) -> None:
        """Advance ``cache_seqlens`` after a paged forward pass and evict if needed.

        The attention layer has already written K/V directly into the
        pool via :meth:`PagedKVCache.write_kv_chunk`. This method only
        updates the frame counter and handles eviction.

        Parameters
        ----------
        stream_id : int
            Stream identifier.
        chunk_frames : int
            Number of encoder-output frames written in the latest chunk.
        """
        state = self._get_state(stream_id)
        state.num_committed_frames += chunk_frames
        if state.cache_seqlens is not None:
            state.cache_seqlens[0] = state.num_committed_frames
        self._evict_oldest(stream_id)

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

        while len(state.logical_blocks) > max_blocks:
            evicted = state.logical_blocks.pop(0)
            self._pool.free([evicted])
            # Adjust committed frame counter.
            state.num_committed_frames = len(state.logical_blocks) * cfg.block_size_frames
            if state.cache_seqlens is not None:
                state.cache_seqlens[0] = state.num_committed_frames
            # Shift the block table left by one entry.
            if state.block_table is not None:
                n = len(state.logical_blocks)
                state.block_table[0, :n] = state.block_table[0, 1: n + 1].clone()
                state.block_table[0, n] = 0

    def _get_state(self, stream_id: int) -> _StreamKVState:
        try:
            return self._streams[stream_id]
        except KeyError:
            raise KeyError(
                f"Attention cache for stream {stream_id} not allocated."
            ) from None
