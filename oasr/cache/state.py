# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Slot-addressed fixed-extent streaming state — the generic form of the CNN cache.

Every streaming cache in OASR is one of two storage disciplines:

**Fixed extent, slot-addressed** (this module).  The size is known when the engine
is built and never grows: an encoder's convolutional left-context, a subsampling
stage's tail, a recurrent per-layer state.  One persistent tensor per declaration,
with the stream's :class:`~oasr.cache.StreamSlotPool` slot id as one of its axes;
a chunk gathers the active rows, computes, and scatters the new tails back in
place.  Nothing is ever evicted because nothing accumulates.

**Growing, block-addressed** (:mod:`oasr.cache.attention_cache`).  Attention K/V
grows with the stream, so it needs a shared :class:`~oasr.cache.BlockPool`, a
per-stream block table, and eviction.

This module is the first discipline, declared rather than hardcoded.  It began as
``CnnCacheManager`` + ``SlotCnnCache`` — a perfectly good slot cache whose name,
shape and place in the encoder signature were all fixed to *the Conformer
depthwise convolution*, which is why an encoder needing a **second** fixed-extent
tensor (Nemotron's three subsampling-stage tails) had nowhere to put one.

Two things are worth naming about :attr:`StreamStateSpec.slot_axis`, because it is
what makes this general rather than merely renamed:

* it is why the Conformer conv cache keeps its exact ``(L, B, K-1, D)`` layout —
  ``slot_axis = 1`` — and therefore its buffer address, which the CUDA-graph cache
  captures by reference;
* it is *the same declaration* as a Zipformer-style encoder's per-kind batch dims
  (icefall's convention: embed and conv caches batch on dim 0, key/nonlin/value
  caches on dim 1), which live today as a hand-written
  ``stack_streaming_states`` / ``unstack_streaming_states`` pair on one encoder.
  Moving that to this axis is a table plus a deletion, not a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

__all__ = ["StreamStateSpec", "SlotTensor", "SlotStateCache"]


@dataclass(frozen=True)
class StreamStateSpec:
    """One fixed-extent per-stream tensor an encoder carries across chunks.

    Attributes
    ----------
    name : str
        Key the encoder reads it back under (``"conv"``, ``"subsample.0"``, ...).
        Reserved: ``"conv"`` is the convolutional left-context every paged
        encoder already declares through ``CacheSpec.conv_kernel_size``.
    shape : tuple of int
        Per-stream shape, **excluding** the slot axis.
    slot_axis : int
        Where the slot axis is inserted in the persistent buffer.  ``1`` gives the
        Conformer conv cache's ``(layers, slots, frames, dim)``; ``0`` gives a
        plain ``(slots, ...)``.
    dtype : torch.dtype, optional
        ``None`` uses the cache's dtype (the model's).  Set it for a state that
        must stay fp32 regardless — a counter, or an accumulator whose precision
        is load-bearing.
    """

    name: str
    shape: Tuple[int, ...]
    slot_axis: int = 0
    dtype: Optional[torch.dtype] = None

    def buffer_shape(self, max_batch_size: int) -> Tuple[int, ...]:
        """Persistent buffer shape: :attr:`shape` with the slot axis inserted."""
        axis = self.slot_axis
        if not (0 <= axis <= len(self.shape)):
            raise ValueError(
                f"state {self.name!r}: slot_axis {axis} is outside "
                f"[0, {len(self.shape)}] for shape {self.shape}"
            )
        return (*self.shape[:axis], max_batch_size, *self.shape[axis:])


@dataclass
class SlotTensor:
    """Slot-indexed view descriptor: gather the active rows, scatter them back.

    The encoder is handed one of these per declared state and uses it exactly the
    way the paged K/V descriptor is used — read at the top of the chunk, write at
    the bottom — so the persistent buffer is updated in place and no per-chunk
    allocation happens on the streaming path.

    Attributes
    ----------
    buffer : Tensor
        The persistent tensor owned by :class:`SlotStateCache`.  Mutated in place
        at rows ``slot_ids`` by :meth:`scatter`.
    slot_ids : Tensor
        ``(B,)`` int64 slot ids selecting the active rows for this forward.  Must
        live on the same device as ``buffer``.
    slot_axis : int
        Axis of ``buffer`` the slot ids index.  Defaults to ``1``, the historical
        CNN-cache layout.
    """

    buffer: torch.Tensor
    slot_ids: torch.Tensor
    slot_axis: int = 1

    def gather(self) -> torch.Tensor:
        """The active rows — ``buffer`` with the slot axis narrowed to ``B``."""
        return self.buffer.index_select(self.slot_axis, self.slot_ids)

    def scatter(self, new_state: torch.Tensor) -> None:
        """Write the post-chunk state back into the persistent buffer."""
        self.buffer.index_copy_(self.slot_axis, self.slot_ids, new_state)


class SlotStateCache:
    """Owns one persistent tensor per :class:`StreamStateSpec`, addressed by slot.

    Parameters
    ----------
    specs : sequence of StreamStateSpec
        The states to allocate.  Names must be unique.  An empty sequence is
        legal and allocates nothing — an encoder with no fixed-extent state pays
        nothing for this axis existing.
    max_batch_size : int
        Slot capacity (== ``CacheConfig.max_batch_size``).
    device, dtype :
        Defaults for every spec that does not override ``dtype``.

    Examples
    --------
    >>> cache = SlotStateCache(
    ...     [StreamStateSpec("conv", (12, 14, 256), slot_axis=1)],
    ...     max_batch_size=4, device=torch.device("cpu"), dtype=torch.float32)
    >>> cache.allocate_stream(stream_id=7, slot_id=0)
    >>> views = cache.views(torch.tensor([0]))
    >>> views["conv"].gather().shape
    torch.Size([12, 1, 14, 256])
    """

    def __init__(
        self,
        specs: Sequence[StreamStateSpec],
        max_batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        names = [s.name for s in specs]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate stream-state names: {sorted(dupes)}")

        self._specs: Tuple[StreamStateSpec, ...] = tuple(specs)
        self._max_batch_size = max_batch_size
        self._slots: Dict[int, int] = {}
        self._buffers: Dict[str, torch.Tensor] = {
            spec.name: torch.zeros(
                spec.buffer_shape(max_batch_size),
                dtype=spec.dtype or dtype,
                device=device,
            )
            for spec in specs
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def specs(self) -> Tuple[StreamStateSpec, ...]:
        return self._specs

    @property
    def names(self) -> List[str]:
        return [s.name for s in self._specs]

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    def buffer_of(self, name: str) -> torch.Tensor:
        """The persistent tensor for ``name``.

        Its address is stable for the cache's lifetime, which is what lets a CUDA
        graph capture the read/write sites by reference.
        """
        try:
            return self._buffers[name]
        except KeyError:
            raise KeyError(f"no stream state named {name!r}; declared: {self.names}") from None

    def spec_of(self, name: str) -> StreamStateSpec:
        for spec in self._specs:
            if spec.name == name:
                return spec
        raise KeyError(f"no stream state named {name!r}; declared: {self.names}")

    def nbytes_per_stream(self) -> int:
        """Bytes one stream occupies across every declared state."""
        total = 0
        for spec in self._specs:
            buf = self._buffers[spec.name]
            total += buf.element_size() * (buf.numel() // self._max_batch_size)
        return total

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def allocate_stream(self, stream_id: int, slot_id: int) -> None:
        """Bind ``stream_id`` to ``slot_id`` and zero that slot in every state.

        Zeroing is not hygiene — it *is* the initial state.  A convolutional
        left-context of zeros is exactly the padding an offline pass applies, so a
        stream's first chunk sees the same left context it would have seen
        offline.

        Raises
        ------
        ValueError
            If ``stream_id`` is already allocated, or ``slot_id`` is out of range
            or in use.
        """
        if stream_id in self._slots:
            raise ValueError(f"stream state for stream {stream_id} already allocated.")
        if not (0 <= slot_id < self._max_batch_size):
            raise ValueError(f"slot_id {slot_id} out of range [0, {self._max_batch_size})")
        if slot_id in self._slots.values():
            raise ValueError(f"slot_id {slot_id} already in use")
        self._slots[stream_id] = slot_id
        self.reset_slot(slot_id)

    def reset_slot(self, slot_id: int) -> None:
        """Zero ``slot_id``'s rows in every declared state."""
        for spec in self._specs:
            buf = self._buffers[spec.name]
            buf.select(spec.slot_axis, slot_id).zero_()

    def free_stream(self, stream_id: int) -> None:
        """Release the slot mapping for a stream."""
        if stream_id not in self._slots:
            raise KeyError(f"stream state for stream {stream_id} not found.")
        del self._slots[stream_id]

    def slot_of(self, stream_id: int) -> int:
        if stream_id not in self._slots:
            raise KeyError(f"stream state for stream {stream_id} not found.")
        return self._slots[stream_id]

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def views(self, slot_ids: torch.Tensor) -> Dict[str, SlotTensor]:
        """One :class:`SlotTensor` per declared state, for the given active slots.

        ``slot_ids`` is an int64 device tensor; the same one drives every state,
        so a chunk forward gathers and scatters all of them with one index each.
        """
        return {
            spec.name: SlotTensor(
                buffer=self._buffers[spec.name],
                slot_ids=slot_ids,
                slot_axis=spec.slot_axis,
            )
            for spec in self._specs
        }

    def view_of(self, name: str, slot_ids: torch.Tensor) -> SlotTensor:
        """A single state's descriptor."""
        spec = self.spec_of(name)
        return SlotTensor(buffer=self._buffers[name], slot_ids=slot_ids, slot_axis=spec.slot_axis)

    def get_state(self, name: str, stream_id: int) -> torch.Tensor:
        """One stream's slice of ``name`` — a zero-copy view, slot axis kept at 1."""
        spec = self.spec_of(name)
        slot = self.slot_of(stream_id)
        return self._buffers[name].narrow(spec.slot_axis, slot, 1)
