# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paged attention KV cache manager for streaming conformer inference.

Each active stream maintains a logical-to-physical block mapping backed by
a shared ``BlockPool``. New chunks append one new physical block; the oldest
block is evicted when the stream exceeds its configured left-context limit.

The attention cache format matches ``ConformerEncoder.forward_chunk``:

- **Input** ``att_cache``: ``(elayers, head, cache_t1, d_k * 2)``
- **Output** ``new_att_cache`` (trimmed): ``(elayers, head, cache_t1', d_k * 2)``

``cache_t1`` grows by ``chunk_size`` each chunk and is trimmed to
``chunk_size * num_left_chunks`` when ``num_left_chunks >= 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from oasr.cache.block_pool import BlockPool
from oasr.cache.types import CacheConfig


@dataclass
class _StreamKVState:
    """Internal per-stream logical-to-physical block mapping.

    Attributes
    ----------
    logical_blocks : list[int]
        Ordered list of physical block IDs. ``logical_blocks[0]`` is the
        oldest (earliest in time); ``logical_blocks[-1]`` is the most recent.
    num_committed_frames : int
        Total encoder output frames written into this stream's cache.
    """

    logical_blocks: List[int] = field(default_factory=list)
    num_committed_frames: int = 0


class AttentionCacheManager:
    """Manages paged attention KV cache for all active streams.

    Each stream maintains a ``_StreamKVState`` that maps logical block
    indices to physical block IDs in the shared ``BlockPool``. On every
    chunk:

    1. A new physical block is allocated from the pool.
    2. The ``chunk_frames`` of new KV data are written into it.
    3. If the stream exceeds ``max_logical_blocks``, the oldest block is
       freed and removed from the mapping.

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
    >>> cache = mgr.get_stacked_cache(42)      # (L, H, 0, d_k*2) initially
    >>> mgr.commit(42, new_kv_chunk)           # (L, H, chunk_sz, d_k*2)
    >>> cache = mgr.get_stacked_cache(42)      # (L, H, chunk_sz, d_k*2)
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

        No physical blocks are allocated until the first ``commit()``.

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
    # Access
    # ------------------------------------------------------------------

    def get_cache_view(self, stream_id: int, layer: int) -> torch.Tensor:
        """Return the gathered KV cache for one layer as a contiguous tensor.

        Parameters
        ----------
        stream_id : int
            Stream identifier.
        layer : int
            Encoder layer index ``[0, num_layers)``.

        Returns
        -------
        torch.Tensor
            Shape ``(1, n_kv_head, committed_frames, kv_last_dim)`` where
            ``committed_frames`` equals the total frames written so far
            (clamped to ``block_size_frames * len(logical_blocks)``).
        """
        state = self._get_state(stream_id)
        cfg = self._config
        if not state.logical_blocks:
            return torch.empty(
                1, cfg.n_kv_head, 0, cfg.kv_last_dim,
                dtype=cfg.dtype, device=cfg.device,
            )
        flat = self._pool.gather_blocks(layer, state.logical_blocks)
        # flat shape: (N * block_size_frames, n_kv_head, kv_last_dim)
        # Trim to exact committed frames (last block may be partially filled).
        flat = flat[: state.num_committed_frames]
        # Reshape to match forward_chunk's att_cache[i:i+1] slice format.
        return flat.permute(1, 0, 2).unsqueeze(0)
        # Result: (1, n_kv_head, committed_frames, kv_last_dim)

    def get_stacked_cache(self, stream_id: int) -> torch.Tensor:
        """Return the full KV cache stacked across all layers.

        The returned shape matches the ``att_cache`` parameter of
        ``ConformerEncoder.forward_chunk``:
        ``(num_layers, n_kv_head, cache_t1, d_k * 2)``.

        Parameters
        ----------
        stream_id : int
            Stream identifier.

        Returns
        -------
        torch.Tensor
            Shape ``(num_layers, n_kv_head, committed_frames, kv_last_dim)``.
            An empty tensor ``(0, 0, 0, 0)`` is returned when no frames
            have been committed yet, matching ``forward_chunk``'s default.
        """
        state = self._get_state(stream_id)
        cfg = self._config
        if not state.logical_blocks:
            return torch.zeros(0, 0, 0, 0, dtype=cfg.dtype, device=cfg.device)
        views = [self.get_cache_view(stream_id, l) for l in range(cfg.num_layers)]
        return torch.cat(views, dim=0)
        # Shape: (num_layers, n_kv_head, committed_frames, kv_last_dim)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def commit(self, stream_id: int, new_kv_chunk: torch.Tensor) -> None:
        """Write new KV data from the latest encoder chunk into the cache.

        Caller should pass only the **new** frames produced by the current
        chunk — typically the last ``chunk_size`` frames of the full trimmed
        ``new_att_cache`` tensor returned by ``forward_chunk``::

            new_kv_chunk = new_att_cache[:, :, -chunk_size:, :]

        This method:

        1. Allocates a new physical block from the pool.
        2. Copies ``new_kv_chunk`` into that block (one layer at a time).
        3. Appends the block to the stream's logical mapping.
        4. Calls ``evict_oldest()`` if the stream has exceeded its
           ``max_logical_blocks`` limit.

        Parameters
        ----------
        stream_id : int
            Stream identifier.
        new_kv_chunk : torch.Tensor
            New KV data with shape
            ``(num_layers, n_kv_head, chunk_frames, kv_last_dim)``
            where ``chunk_frames <= block_size_frames``.

        Raises
        ------
        KeyError
            If ``stream_id`` is not allocated.
        ValueError
            If ``new_kv_chunk`` has an unexpected shape.
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
                f"chunk_frames={chunk_frames} exceeds block_size_frames={cfg.block_size_frames}."
            )

        # Allocate one new physical block.
        (block_id,) = self._pool.allocate(1)
        state.logical_blocks.append(block_id)

        # Write new KV into the block for each layer.
        # block view shape: (block_size_frames, n_kv_head, kv_last_dim)
        # new_kv_chunk[l] shape: (n_kv_head, chunk_frames, kv_last_dim)
        for l in range(cfg.num_layers):
            view = self._pool.get_block_view(l, block_id)
            # Permute new_kv_chunk[l] from (n_kv_head, chunk_frames, kv_last_dim)
            # to (chunk_frames, n_kv_head, kv_last_dim) to match view layout.
            view[:chunk_frames].copy_(new_kv_chunk[l].permute(1, 0, 2))

        state.num_committed_frames += chunk_frames

        # Evict oldest block if we've exceeded the left-context limit.
        self.evict_oldest(stream_id)

    def evict_oldest(self, stream_id: int) -> None:
        """Evict the oldest block if the stream exceeds ``max_logical_blocks``.

        When ``num_left_chunks < 0`` (unlimited history) this is a no-op.

        Parameters
        ----------
        stream_id : int
            Stream identifier.
        """
        state = self._get_state(stream_id)
        cfg = self._config
        max_blocks = cfg.max_logical_blocks
        if max_blocks is None:
            return  # unlimited history
        while len(state.logical_blocks) > max_blocks:
            evicted = state.logical_blocks.pop(0)
            self._pool.free([evicted])
            # Adjust the committed frame counter to reflect the eviction.
            state.num_committed_frames = (
                len(state.logical_blocks) * cfg.block_size_frames
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self, stream_id: int) -> _StreamKVState:
        try:
            return self._streams[stream_id]
        except KeyError:
            raise KeyError(f"Attention cache for stream {stream_id} not allocated.") from None
