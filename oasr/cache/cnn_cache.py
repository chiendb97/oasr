# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-stream CNN cache manager for conformer convolution modules.

The causal Conformer CNN module stores the last ``kernel_size - 1`` input
frames per layer as left-context padding for the next chunk. The cache is
fixed-size per stream (no paging required), so we keep one **persistent
batched tensor** of shape
``(num_layers, max_batch_size, cnn_cache_frames, hidden_dim)`` and assign
each admitted stream a slot id. Per-stream views are zero-copy slices.

The batched paged forward reads / writes this buffer in place via a
:class:`~oasr.cache.SlotCnnCache` descriptor (gather at the top of the
encoder, scatter at the bottom), mirroring how K/V are written through
:class:`~oasr.cache.PagedKVCache`.
"""

from __future__ import annotations

from typing import Dict

import torch

from oasr.cache.types import CacheConfig


class CnnCacheManager:
    """Slot-indexed CNN cache for all active streams.

    Parameters
    ----------
    config : CacheConfig
        Cache configuration. The CNN cache uses ``num_layers``,
        ``cnn_cache_frames``, ``hidden_dim``, ``max_batch_size``,
        ``device``, and ``dtype``.

    Examples
    --------
    >>> mgr = CnnCacheManager(config)
    >>> mgr.allocate_stream(stream_id=0, slot_id=0)
    >>> # The encoder usually wraps this with a SlotCnnCache descriptor and
    >>> # scatters new tails back in place via ``forward_chunk_paged``; the
    >>> # accessors below remain for direct per-stream inspection / tests.
    >>> cache = mgr.get_cache(0)            # shape (L, 1, K-1, D)
    >>> mgr.update(0, new_cache)
    >>> mgr.free_stream(0)
    """

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._slots: Dict[int, int] = {}
        # Persistent batched cache: (L, max_batch_size, K-1, D).
        self._buffer = torch.zeros(
            config.num_layers,
            config.max_batch_size,
            config.cnn_cache_frames,
            config.hidden_dim,
            dtype=config.dtype,
            device=config.device,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def allocate_stream(self, stream_id: int, slot_id: int) -> None:
        """Register a stream at the given slot id with a zeroed cache row.

        Raises
        ------
        ValueError
            If ``stream_id`` is already allocated, or ``slot_id`` is out of
            range / in use.
        """
        if stream_id in self._slots:
            raise ValueError(f"CNN cache for stream {stream_id} already allocated.")
        if not (0 <= slot_id < self._config.max_batch_size):
            raise ValueError(
                f"slot_id {slot_id} out of range [0, {self._config.max_batch_size})"
            )
        if slot_id in self._slots.values():
            raise ValueError(f"slot_id {slot_id} already in use")
        self._slots[stream_id] = slot_id
        # Zero this slot's column across all layers.
        self._buffer[:, slot_id].zero_()

    def free_stream(self, stream_id: int) -> None:
        """Release the slot mapping for a stream."""
        if stream_id not in self._slots:
            raise KeyError(f"CNN cache for stream {stream_id} not found.")
        del self._slots[stream_id]

    # ------------------------------------------------------------------
    # Access / update
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> torch.Tensor:
        """The persistent ``(L, max_batch_size, K-1, D)`` buffer."""
        return self._buffer

    def slot_of(self, stream_id: int) -> int:
        if stream_id not in self._slots:
            raise KeyError(f"CNN cache for stream {stream_id} not found.")
        return self._slots[stream_id]

    def get_cache(self, stream_id: int) -> torch.Tensor:
        """Return the CNN cache view for a single stream.

        Returns
        -------
        torch.Tensor
            Shape ``(num_layers, 1, cnn_cache_frames, hidden_dim)`` — a
            zero-copy slice of the persistent buffer matching the
            ``cnn_cache`` input expected by ``forward_chunk_paged``.

        Raises
        ------
        KeyError
            If ``stream_id`` is not allocated.
        """
        slot = self.slot_of(stream_id)
        return self._buffer[:, slot: slot + 1]

    def update(self, stream_id: int, new_cnn_cache: torch.Tensor) -> None:
        """Overwrite the CNN cache for a stream with the new chunk output.

        Streaming forwards normally update the buffer in place via a
        :class:`~oasr.cache.SlotCnnCache` descriptor passed into
        ``forward_chunk_paged``; this accessor remains for direct
        per-stream inspection and tests.

        Parameters
        ----------
        stream_id : int
            Stream identifier.
        new_cnn_cache : torch.Tensor
            Shape ``(num_layers, 1, cnn_cache_frames, hidden_dim)``.
        """
        slot = self.slot_of(stream_id)
        expected = (
            self._config.num_layers, 1,
            self._config.cnn_cache_frames, self._config.hidden_dim,
        )
        if tuple(new_cnn_cache.shape) != expected:
            raise ValueError(
                f"CNN cache shape mismatch for stream {stream_id}: "
                f"expected {expected}, got {tuple(new_cnn_cache.shape)}."
            )
        self._buffer[:, slot: slot + 1].copy_(new_cnn_cache)
