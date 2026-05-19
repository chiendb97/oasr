# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Slot-indexed CNN cache descriptor + gather/scatter helpers.

Symmetric with :class:`~oasr.cache.PagedKVCache`: wraps the persistent
:class:`~oasr.cache.CnnCacheManager` buffer plus the ``slot_ids`` tensor
selecting the active rows for one batched forward. The encoder calls
:meth:`gather` once at the top to materialise per-batch left-context and
:meth:`scatter` once at the bottom to write the post-chunk tail back into
the persistent buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SlotCnnCache:
    """Slot-indexed CNN cache descriptor.

    Attributes
    ----------
    buffer : Tensor
        ``(num_layers, max_batch_size, cnn_cache_frames, hidden_dim)``
        persistent buffer owned by :class:`~oasr.cache.CnnCacheManager`.
        Mutated in place at rows ``slot_ids`` by :meth:`scatter`.
    slot_ids : Tensor
        ``(B,)`` int64 slot ids selecting the active rows for this
        forward. Must live on the same device as ``buffer``.
    """

    buffer: torch.Tensor
    slot_ids: torch.Tensor

    def gather(self) -> torch.Tensor:
        """Read the per-batch left-context.

        Returns
        -------
        Tensor
            ``(num_layers, B, cnn_cache_frames, hidden_dim)`` view of the
            persistent buffer at rows ``slot_ids``.
        """
        return self.buffer.index_select(1, self.slot_ids)

    def scatter(self, new_cache: torch.Tensor) -> None:
        """Write the post-chunk tail back into the persistent buffer.

        Parameters
        ----------
        new_cache : Tensor
            ``(num_layers, B, cnn_cache_frames, hidden_dim)`` new cache
            produced by the encoder.
        """
        self.buffer.index_copy_(1, self.slot_ids, new_cache)


__all__ = ["SlotCnnCache"]
