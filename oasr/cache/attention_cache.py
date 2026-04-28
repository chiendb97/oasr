# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paged attention KV cache manager for streaming conformer inference.

Each active stream maintains a logical-to-physical block mapping backed by
a shared ``BlockPool``.  Two usage modes are supported:

**Dense mode** (backward-compatible with ``forward_chunk``)
    ``commit(stream_id, new_kv_chunk)`` allocates a block, copies packed K+V
    data into it, and ``get_stacked_cache`` gathers them back into the dense
    ``(elayers, head, cache_t1, d_k * 2)`` tensor expected by
    ``ConformerEncoder.forward_chunk``.

**Paged mode** (used with ``forward_chunk_paged``)
    ``prepare_chunk(stream_id)`` allocates the next physical block in advance.
    ``get_paged_caches(stream_id)`` returns one :class:`PagedKVCache` per
    encoder layer.  ``forward_chunk_paged`` writes K/V directly into the pool.
    ``commit_chunk_paged(stream_id, chunk_frames)`` increments ``cache_seqlens``
    and evicts the oldest block when the left-context limit is reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from oasr.cache.block_pool import BlockPool
from oasr.cache.types import CacheConfig
from oasr.layers.attention.attention import PagedKVCache


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
        ``(1, max_blocks_per_seq)`` int32 on the cache device.  Allocated on
        first use in paged mode; ``None`` until then.
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
    Dense mode::

        mgr = AttentionCacheManager(pool, config)
        mgr.allocate_stream(42)
        cache = mgr.get_stacked_cache(42)      # (L, H, 0, d_k*2) initially
        mgr.commit(42, new_kv_chunk)
        cache = mgr.get_stacked_cache(42)      # (L, H, chunk_sz, d_k*2)
        mgr.free_stream(42)

    Paged mode::

        mgr.allocate_stream(42)
        mgr.prepare_chunk(42)                  # allocate block for next write
        att_caches = mgr.get_paged_caches(42)  # List[PagedKVCache]
        xs, cnn = model.forward_chunk_paged(xs, offset, att_caches, cnn)
        mgr.commit_chunk_paged(42, chunk_size)
        mgr.free_stream(42)
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
    # Dense-mode access
    # ------------------------------------------------------------------

    def get_cache_view(self, stream_id: int, layer: int) -> torch.Tensor:
        """Return the gathered KV cache for one layer as a packed tensor.

        Parameters
        ----------
        stream_id : int
            Stream identifier.
        layer : int
            Encoder layer index ``[0, num_layers)``.

        Returns
        -------
        torch.Tensor
            Shape ``(1, n_kv_head, committed_frames, head_dim * 2)``.
        """
        state = self._get_state(stream_id)
        cfg = self._config
        if not state.logical_blocks:
            return torch.empty(
                1, cfg.n_kv_head, 0, cfg.kv_last_dim,
                dtype=cfg.dtype, device=cfg.device,
            )
        # Gather K and V separately, each (n*block_size, n_kv_head, head_dim).
        k_flat, v_flat = self._pool.gather_kv_blocks(layer, state.logical_blocks)
        k_flat = k_flat[: state.num_committed_frames]
        v_flat = v_flat[: state.num_committed_frames]
        # Pack into (committed_frames, n_kv_head, head_dim * 2).
        kv_flat = torch.cat([k_flat, v_flat], dim=-1)
        # → (1, n_kv_head, committed_frames, head_dim * 2)
        return kv_flat.permute(1, 0, 2).unsqueeze(0)

    def get_stacked_cache(self, stream_id: int) -> torch.Tensor:
        """Return the full packed KV cache stacked across all layers.

        Shape matches the ``att_cache`` parameter of
        ``ConformerEncoder.forward_chunk``:
        ``(num_layers, n_kv_head, cache_t1, d_k * 2)``.

        Parameters
        ----------
        stream_id : int
            Stream identifier.

        Returns
        -------
        torch.Tensor
            ``(num_layers, n_kv_head, committed_frames, kv_last_dim)`` or
            ``(0, 0, 0, 0)`` when no frames have been committed.
        """
        state = self._get_state(stream_id)
        cfg = self._config
        if not state.logical_blocks:
            return torch.zeros(0, 0, 0, 0, dtype=cfg.dtype, device=cfg.device)
        views = [self.get_cache_view(stream_id, l) for l in range(cfg.num_layers)]
        return torch.cat(views, dim=0)

    # ------------------------------------------------------------------
    # Dense-mode mutation
    # ------------------------------------------------------------------

    def commit(self, stream_id: int, new_kv_chunk: torch.Tensor) -> None:
        """Write new packed K+V frames from the latest ``forward_chunk`` call.

        Parameters
        ----------
        stream_id : int
            Stream identifier.
        new_kv_chunk : torch.Tensor
            Shape ``(num_layers, n_kv_head, chunk_frames, kv_last_dim)``
            where ``kv_last_dim = head_dim * 2``.

        Raises
        ------
        ValueError
            If shape is inconsistent with the config.
        RuntimeError
            If the pool has no free blocks.
        """
        state = self._get_state(stream_id)
        cfg = self._config
        chunk_frames = new_kv_chunk.size(2)

        if new_kv_chunk.size(0) != cfg.num_layers:
            raise ValueError(
                f"new_kv_chunk.shape[0]={new_kv_chunk.size(0)} != "
                f"num_layers={cfg.num_layers}."
            )
        if new_kv_chunk.size(3) != cfg.kv_last_dim:
            raise ValueError(
                f"new_kv_chunk.shape[3]={new_kv_chunk.size(3)} != "
                f"kv_last_dim={cfg.kv_last_dim}."
            )
        if chunk_frames > cfg.block_size_frames:
            raise ValueError(
                f"chunk_frames={chunk_frames} exceeds "
                f"block_size_frames={cfg.block_size_frames}."
            )

        # Allocate one new physical block.
        (block_id,) = self._pool.allocate(1)
        state.logical_blocks.append(block_id)

        # Write split K and V into the block for each layer.
        # Pool block view shape: (block_size, n_kv_head, head_dim).
        # new_kv_chunk[l] shape: (n_kv_head, chunk_frames, kv_last_dim).
        head_dim = cfg.head_dim
        for l in range(cfg.num_layers):
            k_view, v_view = self._pool.get_kv_block_view(l, block_id)
            layer_kv = new_kv_chunk[l]  # (n_kv_head, chunk_frames, kv_last_dim)
            # Permute to (chunk_frames, n_kv_head, kv_last_dim) and split.
            layer_kv_t = layer_kv.permute(1, 0, 2)  # (chunk_frames, n_kv_head, kv_last_dim)
            k_view[:chunk_frames].copy_(layer_kv_t[..., :head_dim])
            v_view[:chunk_frames].copy_(layer_kv_t[..., head_dim:])

        state.num_committed_frames += chunk_frames

        # Also keep block_table / cache_seqlens in sync for paged mode.
        if state.block_table is not None:
            logical_idx = len(state.logical_blocks) - 1
            state.block_table[0, logical_idx] = block_id
            state.cache_seqlens[0] = state.num_committed_frames

        self._evict_oldest(stream_id)

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
                    # Single-stream is trivially homogeneous; enables the
                    # scalar-offset write and zero-pad-bias fast paths.
                    host_seqlen_homogeneous=host_seqlen,
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

        The model has already written K/V directly into the pool via
        :func:`~oasr.layers.attention.attention._paged_write_kv`.  This method
        only updates the frame counter and handles eviction.

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

    # Keep old name for backward compatibility.
    def evict_oldest(self, stream_id: int) -> None:
        """Alias for :meth:`_evict_oldest` (backward-compatible name)."""
        self._evict_oldest(stream_id)

    def _get_state(self, stream_id: int) -> _StreamKVState:
        try:
            return self._streams[stream_id]
        except KeyError:
            raise KeyError(
                f"Attention cache for stream {stream_id} not allocated."
            ) from None
