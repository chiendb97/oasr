# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared paged block pool for attention KV cache.

Inspired by the vLLM v1 ``BlockPool`` design:
https://github.com/vllm-project/vllm/tree/main/vllm/v1/core

Two large GPU tensors (one for K, one for V) are pre-allocated and divided
into fixed-size physical blocks (pages).  A CPU-side free list tracks
available block IDs.  All ``AttentionCacheManager`` instances share one
``BlockPool``.

The backing tensors have shape::

    (num_layers, max_num_blocks, block_size_frames, n_kv_head, head_dim)

Keeping K and V in separate contiguous tensors is required by
``flash_attn_with_kvcache``, which expects
``k_cache: (max_num_blocks, block_size, n_kv_head, head_dim)`` per layer.
"""

from __future__ import annotations

import collections
import threading
from typing import List, Tuple

import torch

from oasr.cache.types import CacheConfig


class BlockPool:
    """Shared pool of physical GPU memory blocks for paged KV cache.

    Pre-allocates two contiguous GPU tensors (K pool and V pool) and
    manages allocation via a thread-safe free list.  Multiple
    ``AttentionCacheManager`` instances share a single pool.

    Each pool has shape::

        (num_layers, max_num_blocks, block_size_frames, n_kv_head, head_dim)

    Indexing ``pool[layer, block_id]`` yields a
    ``(block_size_frames, n_kv_head, head_dim)`` slab for one physical block
    in one encoder layer.

    Parameters
    ----------
    config : CacheConfig
        Cache configuration controlling pool dimensions, device, and dtype.

    Examples
    --------
    >>> pool = BlockPool(config)
    >>> ids = pool.allocate(2)
    >>> k_view, v_view = pool.get_kv_block_view(layer=0, block_id=ids[0])
    >>> pool.free(ids)
    """

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._lock = threading.Lock()

        cfg = config
        pool_shape = (
            cfg.num_layers,
            cfg.max_num_blocks,
            cfg.block_size_frames,
            cfg.n_kv_head,
            cfg.head_dim,
        )
        self._k_pool = torch.zeros(*pool_shape, dtype=cfg.dtype, device=cfg.device)
        self._v_pool = torch.zeros(*pool_shape, dtype=cfg.dtype, device=cfg.device)

        # Free list: all block IDs are initially free.
        self._free_list: collections.deque[int] = collections.deque(
            range(cfg.max_num_blocks)
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        """Number of currently available physical blocks."""
        with self._lock:
            return len(self._free_list)

    @property
    def num_total_blocks(self) -> int:
        """Total number of physical blocks in the pool."""
        return self._config.max_num_blocks

    # ------------------------------------------------------------------
    # Allocation / free
    # ------------------------------------------------------------------

    def allocate(self, num_blocks: int) -> List[int]:
        """Allocate ``num_blocks`` physical block IDs from the free list.

        Parameters
        ----------
        num_blocks : int
            Number of blocks to allocate.

        Returns
        -------
        list[int]
            Allocated physical block IDs.

        Raises
        ------
        RuntimeError
            If the pool does not have enough free blocks.
        """
        with self._lock:
            if len(self._free_list) < num_blocks:
                raise RuntimeError(
                    f"BlockPool exhausted: requested {num_blocks} blocks but only "
                    f"{len(self._free_list)} are free (total={self._config.max_num_blocks})."
                )
            return [self._free_list.popleft() for _ in range(num_blocks)]

    def free(self, block_ids: List[int]) -> None:
        """Return physical blocks to the free list.

        Parameters
        ----------
        block_ids : list[int]
            Physical block IDs to release.  Passing an empty list is a no-op.
        """
        with self._lock:
            self._free_list.extend(block_ids)

    # ------------------------------------------------------------------
    # Tensor access
    # ------------------------------------------------------------------

    def get_kv_view(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero-copy views of the full K and V pools for one layer.

        The returned tensors have shape
        ``(max_num_blocks, block_size_frames, n_kv_head, head_dim)`` and are
        passed directly into ``PagedKVCache`` / ``flash_attn_with_kvcache``.

        Parameters
        ----------
        layer : int
            Encoder layer index ``[0, num_layers)``.

        Returns
        -------
        k_view, v_view : Tensor
            Views into ``_k_pool[layer]`` and ``_v_pool[layer]``.
        """
        return self._k_pool[layer], self._v_pool[layer]

    def get_kv_block_view(
        self, layer: int, block_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return direct views into K and V slabs for one layer and block.

        The returned tensors are *views* — writes are reflected in the pool.

        Parameters
        ----------
        layer : int
            Encoder layer index ``[0, num_layers)``.
        block_id : int
            Physical block ID ``[0, max_num_blocks)``.

        Returns
        -------
        k_view, v_view : Tensor
            Both shaped ``(block_size_frames, n_kv_head, head_dim)``.
        """
        return self._k_pool[layer, block_id], self._v_pool[layer, block_id]

    def gather_kv_blocks(
        self, layer: int, block_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather multiple blocks for one layer into contiguous K and V tensors.

        Uses ``torch.index_select`` on each pool separately.

        Parameters
        ----------
        layer : int
            Encoder layer index.
        block_ids : list[int]
            Ordered list of physical block IDs to gather.

        Returns
        -------
        k_flat, v_flat : Tensor
            Both shaped ``(len(block_ids) * block_size_frames, n_kv_head, head_dim)``.
            Data is contiguous and freshly allocated.
        """
        cfg = self._config
        if not block_ids:
            empty = torch.empty(
                0, cfg.n_kv_head, cfg.head_dim,
                dtype=cfg.dtype, device=cfg.device,
            )
            return empty, empty.clone()

        idx = torch.tensor(block_ids, dtype=torch.long, device=self._k_pool.device)
        n, bsf = len(block_ids), cfg.block_size_frames
        k = torch.index_select(self._k_pool[layer], 0, idx).reshape(n * bsf, cfg.n_kv_head, cfg.head_dim)
        v = torch.index_select(self._v_pool[layer], 0, idx).reshape(n * bsf, cfg.n_kv_head, cfg.head_dim)
        return k, v
