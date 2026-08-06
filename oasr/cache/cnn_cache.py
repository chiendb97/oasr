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

Implementation note: this is a **single-spec** :class:`~oasr.cache.SlotStateCache`
— the convolutional left-context, under the name it has always had, with the
accessors callers already use.  The generic form is what an encoder with more than
one fixed-extent state (Nemotron's per-subsampling-stage tails) declares instead;
see ``oasr/cache/state.py``.  Subclassing rather than wrapping keeps this the
*same tensor object*, which matters because the CUDA-graph cache captures the
buffer by address.
"""

from __future__ import annotations

import torch

from oasr.cache.state import SlotStateCache, StreamStateSpec
from oasr.cache.types import CacheConfig

#: Name the convolutional left-context is declared and read back under.
CONV_STATE = "conv"


def conv_state_spec(config: CacheConfig) -> StreamStateSpec:
    """The Conformer conv cache as a :class:`StreamStateSpec`.

    ``slot_axis = 1`` is what preserves the historical
    ``(layers, slots, frames, dim)`` layout — and therefore the gather/scatter
    the encoder already performs and the buffer address the graph captures.
    """
    return StreamStateSpec(
        name=CONV_STATE,
        shape=(config.num_layers, config.cnn_cache_frames, config.hidden_dim),
        slot_axis=1,
    )


class CnnCacheManager(SlotStateCache):
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
        super().__init__(
            [conv_state_spec(config)],
            max_batch_size=config.max_batch_size,
            device=config.device,
            dtype=config.dtype,
        )
        self._config = config

    # ------------------------------------------------------------------
    # Single-tensor accessors
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> torch.Tensor:
        """The persistent ``(L, max_batch_size, K-1, D)`` buffer."""
        return self.buffer_of(CONV_STATE)

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
        return self.get_state(CONV_STATE, stream_id)

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
        expected = (
            self._config.num_layers,
            1,
            self._config.cnn_cache_frames,
            self._config.hidden_dim,
        )
        if tuple(new_cnn_cache.shape) != expected:
            raise ValueError(
                f"CNN cache shape mismatch for stream {stream_id}: "
                f"expected {expected}, got {tuple(new_cnn_cache.shape)}."
            )
        self.get_state(CONV_STATE, stream_id).copy_(new_cnn_cache)


__all__ = ["CONV_STATE", "CnnCacheManager", "conv_state_spec"]
