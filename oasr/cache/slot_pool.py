# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Free-slot allocator for streaming caches.

Streams are indexed by a fixed slot id in ``[0, max_batch_size)`` so the
attention block_table, cache_seqlens, CNN cache and feature buffer can live
in single persistent batched tensors. The slot id is reused once a stream is
freed; consumers must release the slot before re-using it.
"""

from __future__ import annotations

from collections import deque
from typing import Deque


class StreamSlotPool:
    """Fixed-capacity slot allocator.

    Parameters
    ----------
    capacity : int
        Maximum number of concurrent slots (== ``max_batch_size``).

    Examples
    --------
    >>> pool = StreamSlotPool(capacity=4)
    >>> s0 = pool.allocate()
    >>> s1 = pool.allocate()
    >>> pool.free(s0)
    >>> pool.allocate() == s0  # freed slots are reused
    True
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._free: Deque[int] = deque(range(capacity))

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        """Return the next free slot id.

        Raises
        ------
        RuntimeError
            If no slots are free.
        """
        if not self._free:
            raise RuntimeError(
                f"StreamSlotPool exhausted (capacity={self._capacity})."
            )
        return self._free.popleft()

    def free(self, slot_id: int) -> None:
        """Release ``slot_id`` back to the free list.

        Raises
        ------
        ValueError
            If ``slot_id`` is out of range or already free.
        """
        if not (0 <= slot_id < self._capacity):
            raise ValueError(
                f"slot_id {slot_id} out of range [0, {self._capacity})"
            )
        if slot_id in self._free:
            raise ValueError(f"slot_id {slot_id} is already free")
        self._free.append(slot_id)
