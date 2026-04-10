# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-stream CNN cache manager for conformer convolution modules.

The CNN module in a causal conformer stores the last ``kernel_size - 1``
input frames as left-context padding for the next chunk. This cache is
fixed-size per stream (no paging required), so a simple per-stream tensor
allocation is used.

The cache tensor shape matches the ``cnn_cache`` argument accepted and
returned by ``ConformerEncoder.forward_chunk``:
``(num_layers, 1, cnn_cache_frames, hidden_dim)``.
"""

from __future__ import annotations

from typing import Dict

import torch

from oasr.cache.types import CacheConfig


class CnnCacheManager:
    """Manages fixed-size CNN cache tensors for all active streams.

    Each stream holds one tensor of shape
    ``(num_layers, 1, cnn_cache_frames, hidden_dim)`` that is overwritten
    in place after every chunk.

    Parameters
    ----------
    config : CacheConfig
        Cache configuration. The CNN cache only uses ``num_layers``,
        ``cnn_cache_frames``, ``hidden_dim``, ``device``, and ``dtype``.

    Examples
    --------
    >>> mgr = CnnCacheManager(config)
    >>> mgr.allocate_stream(stream_id=0)
    >>> cache = mgr.get_cache(0)            # shape (L, 1, K-1, D)
    >>> mgr.update(0, new_cache)            # overwrite after forward_chunk
    >>> mgr.free_stream(0)
    """

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._caches: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def allocate_stream(self, stream_id: int) -> None:
        """Allocate a zero-initialized CNN cache tensor for a new stream.

        Parameters
        ----------
        stream_id : int
            Unique stream identifier.

        Raises
        ------
        ValueError
            If ``stream_id`` is already allocated.
        """
        if stream_id in self._caches:
            raise ValueError(f"CNN cache for stream {stream_id} already allocated.")
        cfg = self._config
        self._caches[stream_id] = torch.zeros(
            cfg.num_layers,
            1,
            cfg.cnn_cache_frames,
            cfg.hidden_dim,
            dtype=cfg.dtype,
            device=cfg.device,
        )

    def free_stream(self, stream_id: int) -> None:
        """Release the CNN cache tensor for a stream.

        Parameters
        ----------
        stream_id : int
            Stream to release.

        Raises
        ------
        KeyError
            If ``stream_id`` is not allocated.
        """
        if stream_id not in self._caches:
            raise KeyError(f"CNN cache for stream {stream_id} not found.")
        del self._caches[stream_id]

    # ------------------------------------------------------------------
    # Access / update
    # ------------------------------------------------------------------

    def get_cache(self, stream_id: int) -> torch.Tensor:
        """Return the CNN cache tensor for a stream.

        The returned tensor is the live storage — callers should not modify
        it directly; use ``update()`` instead.

        Parameters
        ----------
        stream_id : int
            Stream identifier.

        Returns
        -------
        torch.Tensor
            Shape ``(num_layers, 1, cnn_cache_frames, hidden_dim)`` matching
            the ``cnn_cache`` input expected by ``forward_chunk``.

        Raises
        ------
        KeyError
            If ``stream_id`` is not allocated.
        """
        if stream_id not in self._caches:
            raise KeyError(f"CNN cache for stream {stream_id} not found.")
        return self._caches[stream_id]

    def update(self, stream_id: int, new_cnn_cache: torch.Tensor) -> None:
        """Overwrite the CNN cache with the output from the latest chunk.

        Parameters
        ----------
        stream_id : int
            Stream identifier.
        new_cnn_cache : torch.Tensor
            New cache tensor, shape ``(num_layers, 1, cnn_cache_frames, hidden_dim)``
            as returned by ``ConformerEncoder.forward_chunk``.

        Raises
        ------
        KeyError
            If ``stream_id`` is not allocated.
        ValueError
            If ``new_cnn_cache`` shape is incompatible with the stored cache.
        """
        if stream_id not in self._caches:
            raise KeyError(f"CNN cache for stream {stream_id} not found.")
        stored = self._caches[stream_id]
        if new_cnn_cache.shape != stored.shape:
            raise ValueError(
                f"CNN cache shape mismatch for stream {stream_id}: "
                f"expected {tuple(stored.shape)}, got {tuple(new_cnn_cache.shape)}."
            )
        stored.copy_(new_cnn_cache)
