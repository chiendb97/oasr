# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared paged block pool for attention KV cache.

Inspired by the vLLM v1 ``BlockPool`` design:
https://github.com/vllm-project/vllm/tree/main/vllm/v1/core

A single large GPU tensor is pre-allocated and divided into fixed-size
physical blocks (pages). A CPU-side free list tracks available block IDs.
All ``AttentionCacheManager`` instances share one ``BlockPool``.
"""

from __future__ import annotations

import collections
import threading
from typing import List

import torch

from oasr.cache.types import CacheConfig


class BlockPool:
    """Shared pool of physical GPU memory blocks for paged KV cache.

    Pre-allocates one contiguous GPU tensor and manages allocation via a
    thread-safe free list. Multiple ``AttentionCacheManager`` instances
    share a single pool.

    The backing tensor has shape::

        (num_layers, max_num_blocks, block_size_frames, n_kv_head, kv_last_dim)

    Indexing ``pool[layer, block_id]`` yields a
    ``(block_size_frames, n_kv_head, kv_last_dim)`` slab for one physical block
    in one encoder layer.

    Parameters
    ----------
    config : CacheConfig
        Cache configuration controlling pool dimensions, device, and dtype.

    Examples
    --------
    >>> pool = BlockPool(config)
    >>> ids = pool.allocate(2)
    >>> pool.get_block_view(layer=0, block_id=ids[0])  # write KV into slab
    >>> pool.free(ids)
    """

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._lock = threading.Lock()

        # Pre-allocate the full pool tensor on the target device.
        # Shape: (num_layers, max_num_blocks, block_size_frames, n_kv_head, kv_last_dim)
        cfg = config
        self._pool = torch.zeros(
            cfg.num_layers,
            cfg.max_num_blocks,
            cfg.block_size_frames,
            cfg.n_kv_head,
            cfg.kv_last_dim,
            dtype=cfg.dtype,
            device=cfg.device,
        )

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
            Physical block IDs to release. Passing an empty list is a no-op.
        """
        with self._lock:
            self._free_list.extend(block_ids)

    # ------------------------------------------------------------------
    # Tensor access
    # ------------------------------------------------------------------

    def get_block_view(self, layer: int, block_id: int) -> torch.Tensor:
        """Return a direct view into the pool for one layer and block.

        The returned tensor is a *view* — writes are reflected in the pool.

        Parameters
        ----------
        layer : int
            Encoder layer index ``[0, num_layers)``.
        block_id : int
            Physical block ID ``[0, max_num_blocks)``.

        Returns
        -------
        torch.Tensor
            Shape ``(block_size_frames, n_kv_head, kv_last_dim)``, same
            dtype and device as the pool.
        """
        return self._pool[layer, block_id]  # view, no copy

    def gather_blocks(self, layer: int, block_ids: List[int]) -> torch.Tensor:
        """Gather multiple blocks for one layer into a contiguous tensor.

        Uses ``torch.index_select`` to collect the requested blocks from the
        pool, then reshapes to merge the block and frame dimensions.

        Parameters
        ----------
        layer : int
            Encoder layer index.
        block_ids : list[int]
            Ordered list of physical block IDs to gather.

        Returns
        -------
        torch.Tensor
            Shape ``(len(block_ids) * block_size_frames, n_kv_head, kv_last_dim)``.
            Data is contiguous and freshly allocated.
        """
        if not block_ids:
            cfg = self._config
            return torch.empty(
                0, cfg.n_kv_head, cfg.kv_last_dim,
                dtype=cfg.dtype, device=cfg.device,
            )
        idx = torch.tensor(block_ids, dtype=torch.long, device=self._pool.device)
        # pool[layer] shape: (max_num_blocks, block_size_frames, n_kv_head, kv_last_dim)
        selected = torch.index_select(self._pool[layer], dim=0, index=idx)
        # selected shape: (num_blocks, block_size_frames, n_kv_head, kv_last_dim)
        n = selected.size(0)
        bsf = self._config.block_size_frames
        return selected.reshape(n * bsf, self._config.n_kv_head, self._config.kv_last_dim)
