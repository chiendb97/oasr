# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Slot-addressed LSTM/RNN state, and a timestep-granular continuous batcher.

A recurrence carries no history beyond ``(h, c)``.  That single fact is what makes
continuous batching cheap here and expensive for attention: any set of rows at any
mix of timesteps is a legal batch, because nothing in the step depends on how a
row got to its current state.  There is no KV history to keep contiguous, no
block table, and no prefill/decode distinction -- only fixed-extent per-stream
state, which is the slot discipline :mod:`oasr.cache.state` already owns.

So the cache here is *data*, not a new manager: :class:`RecurrentStateCache`
declares one ``StreamStateSpec`` per layer per state and inherits allocation,
zeroing and slot addressing from :class:`~oasr.cache.state.SlotStateCache`.

:class:`RecurrentContinuousBatcher` is the scheduler on top.  Its unit of work is
one *timestep over the currently active rows*, not one request: every admitted
stream advances one frame per tick, finished streams release their slot inside the
same tick, and waiting streams take the freed slots -- so the batch stays full
without draining, exactly as token-level continuous batching keeps a decode batch
full.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Sequence, Set, Tuple

import torch

from oasr.cache.state import SlotStateCache, StreamStateSpec
from oasr.utils.staging import to_device

__all__ = ["RecurrentStateCache", "RecurrentContinuousBatcher", "RecurrentStepPlan"]


class RecurrentStateCache(SlotStateCache):
    """Per-stream LSTM/RNN hidden (and cell) state for ``num_layers`` layers.

    ``h.{layer}`` is a ``(2, slots, hidden)`` ring and ``c.{layer}`` is
    ``(slots, hidden)``.  The ring is not redundancy: in
    :func:`oasr.lstm_slot_step` a CTA owning hidden unit *j* reads its row's whole
    ``h`` vector while other CTAs write their own elements of it, so ``h`` must be
    double-buffered to avoid a read/write race, whereas each cell element is
    touched only by the thread that owns it.

    Which slice holds a stream's current ``h`` is a property of *that stream* --
    it is the parity of how many timesteps the stream has taken -- not of the
    tick.  A stream that sits out a tick keeps its ``h`` where it last wrote it,
    so the parity handed to the kernel is per row.  :meth:`step_indices` builds
    the ``(slot_ids, read_parity)`` pair from the per-stream step counts this
    class already tracks, in one staged host-to-device copy.

    Parameters
    ----------
    num_layers, hidden_size : int
        Geometry of the recurrent stack this cache serves.
    max_batch_size : int
        Slot capacity.
    device, dtype :
        Where the state lives and in what precision.
    cell : bool
        ``True`` for LSTM (allocates ``c``), ``False`` for a vanilla RNN.

    Examples
    --------
    >>> cache = RecurrentStateCache(1, 4, max_batch_size=2,
    ...                             device=torch.device("cpu"), dtype=torch.float32)
    >>> cache.allocate_stream(stream_id=9, slot_id=0)
    >>> cache.hidden(0).shape
    torch.Size([2, 2, 4])
    >>> cache.cell(0).shape
    torch.Size([2, 4])
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        max_batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        cell: bool = True,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        specs: List[StreamStateSpec] = []
        for layer in range(num_layers):
            specs.append(StreamStateSpec(f"h.{layer}", (2, hidden_size), slot_axis=1))
            if cell:
                specs.append(StreamStateSpec(f"c.{layer}", (hidden_size,), slot_axis=0))
        super().__init__(specs, max_batch_size, device, dtype)
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.has_cell = bool(cell)
        self._device = device
        # Timesteps each stream has taken.  Doubles as its ring parity.
        self._steps: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def hidden(self, layer: int) -> torch.Tensor:
        """The ``(2, slots, hidden)`` hidden ring for ``layer``."""
        return self.buffer_of(f"h.{layer}")

    def cell(self, layer: int) -> torch.Tensor:
        """The ``(slots, hidden)`` cell buffer for ``layer``."""
        if not self.has_cell:
            raise RuntimeError("this cache was built without cell state (cell=False)")
        return self.buffer_of(f"c.{layer}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def allocate_stream(self, stream_id: int, slot_id: int) -> None:
        super().allocate_stream(stream_id, slot_id)
        self._steps[stream_id] = 0

    def reset_stream(self, stream_id: int) -> None:
        super().reset_stream(stream_id)
        # The step count is this stream's ring parity, so a reset that left it
        # alone would read the new turn's first state out of the half the old
        # turn wrote — stale left context wearing a fresh slot.
        self._steps[stream_id] = 0

    def free_stream(self, stream_id: int) -> None:
        super().free_stream(stream_id)
        self._steps.pop(stream_id, None)

    def steps_taken(self, stream_id: int) -> int:
        """How many timesteps this stream has advanced -- its current ring parity."""
        if stream_id not in self._steps:
            raise KeyError(f"stream state for stream {stream_id} not found.")
        return self._steps[stream_id]

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step_indices(self, stream_ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(slot_ids, read_parity)`` for one tick over ``stream_ids``.

        Both are built from host state in a single staged copy each, so the step
        adds no device-side index arithmetic.  Row order is ``stream_ids`` order,
        which is the row order the caller must use for its input tensor.
        """
        slots = [self.slot_of(sid) for sid in stream_ids]
        parity = [self._steps[sid] & 1 for sid in stream_ids]
        return (
            to_device(slots, dtype=torch.int64, device=self._device),
            to_device(parity, dtype=torch.int32, device=self._device),
        )

    def commit_step(self, stream_ids: Sequence[int]) -> None:
        """Record that ``stream_ids`` each advanced one timestep.

        Call this after the step kernels for *every* layer have been issued: the
        parity handed to layer ``l`` must be the same one handed to layer 0 within
        a tick, since each layer's ring advances once per tick.
        """
        for sid in stream_ids:
            if sid not in self._steps:
                raise KeyError(f"stream state for stream {sid} not found.")
            self._steps[sid] += 1

    def current_hidden(self, layer: int, stream_id: int) -> torch.Tensor:
        """A stream's live ``h`` row for ``layer`` -- a view, not a copy."""
        return self.hidden(layer)[self.steps_taken(stream_id) & 1, self.slot_of(stream_id)]

    def current_cell(self, layer: int, stream_id: int) -> torch.Tensor:
        """A stream's live ``c`` row for ``layer`` -- a view, not a copy."""
        return self.cell(layer)[self.slot_of(stream_id)]


@dataclass
class RecurrentStepPlan:
    """What one tick of :class:`RecurrentContinuousBatcher` will do.

    Attributes
    ----------
    stream_ids : list of hashable
        The rows of this tick, in row order.
    slot_ids, read_parity : Tensor
        Ready to hand to :func:`oasr.lstm_slot_step` / :func:`oasr.rnn_slot_step`.
        Both are *reused* across ticks while membership is unchanged, so a caller
        must not retain them past :meth:`RecurrentContinuousBatcher.commit`.
    frames : Tensor
        ``(rows, input_size)`` -- one frame per active stream, gathered in row
        order out of the packed submission buffer.
    active_rows : int
        How many leading rows are real.  Equal to ``len(stream_ids)``.  In padded
        mode the tensors are always ``max_batch_size - 1`` wide and the rows past
        this point are filler aimed at a reserved dummy slot, so shapes -- and
        therefore a CUDA-graph capture -- stay valid as membership changes.
    admitted, finished : list of hashable
        Streams that joined at the top of this tick and that retired at the
        bottom of it.  Reported so a caller can emit results without polling.
    membership_changed : bool
        ``True`` when this tick's row set differs from the previous tick's.  In
        variable-width mode a caller holding a CUDA graph must re-capture when
        this is set; in padded mode it never has to, because the row count is
        fixed and only the index tensors' contents move.
    """

    stream_ids: List[Hashable]
    slot_ids: torch.Tensor
    read_parity: torch.Tensor
    frames: torch.Tensor
    active_rows: int = 0
    admitted: List[Hashable] = field(default_factory=list)
    finished: List[Hashable] = field(default_factory=list)
    membership_changed: bool = True


class RecurrentContinuousBatcher:
    """Timestep-granular continuous batching over a recurrent stack.

    The scheduled unit is one timestep over the active rows.  Each tick:

    1. retire streams that consumed their last frame and free their slots;
    2. admit waiting streams into free slots (zeroed state == zero initial state);
    3. gather one frame per active stream and return the slot/parity indices.

    Because a recurrent step depends only on ``(h, c)``, rows at wildly different
    timesteps batch together with no padding and no masking.  That is the whole
    advantage over sequence-major batching: a padded cohort runs at the length of
    its longest member, while here a finished row is replaced immediately and
    every row-step does useful work.

    **This loop is host-bound at ASR sizes and the driver is built around that.**
    One tick's GPU work for a 2-layer H=640 stack over 32 rows measures ~17 us, so
    a naive per-tick driver -- stacking a Python list of row views, allocating a
    pinned buffer per index copy, rescanning membership -- costs an order of
    magnitude more than the work it schedules.  Three things follow:

    * submitted frames are concatenated into one packed device buffer, so the
      per-tick gather is a single ``index_select`` rather than a Python loop;
    * the slot/parity/frame index tensors are built once per *membership change*,
      not once per tick, and the frame cursor advances with an in-place add;
    * ``read_parity`` alternates between two preallocated tensors, because every
      row of a stable cohort flips together.

    What remains is the per-layer functional call.  Capture *that* -- the
    :meth:`~oasr.layers.LSTM.step` call, not :meth:`next_step`, which plans on the
    host and stages index copies a replay could not repeat -- and in ``padded``
    mode one capture serves every tick, because a retiring stream then changes
    only the contents of device index tensors and never a shape.

    Parameters
    ----------
    cache : RecurrentStateCache
        Owns the state and the slot mapping.
    input_size : int
        Width of one frame.

    Examples
    --------
    >>> cache = RecurrentStateCache(1, 4, 2, torch.device("cpu"), torch.float32)
    >>> batcher = RecurrentContinuousBatcher(cache, input_size=3)
    >>> batcher.submit("a", torch.zeros(5, 3))
    >>> batcher.submit("b", torch.zeros(2, 3))
    >>> plan = batcher.next_step()
    >>> plan.stream_ids, plan.frames.shape, plan.membership_changed
    (['a', 'b'], torch.Size([2, 3]), True)
    """

    def __init__(
        self, cache: RecurrentStateCache, input_size: int, *, padded: bool = False
    ) -> None:
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if padded and cache.max_batch_size < 2:
            raise ValueError("padded mode reserves one slot as filler, so it needs 2+ slots")
        self.cache = cache
        self.input_size = int(input_size)
        self.padded = bool(padded)
        self._device = cache.hidden(0).device
        # Padded mode reserves the last slot as the filler target so that rows
        # past `active_rows` compute a throwaway state instead of corrupting a
        # live stream's.
        self.width = cache.max_batch_size - 1 if padded else cache.max_batch_size
        self._dummy_slot = cache.max_batch_size - 1 if padded else -1
        self._free: List[int] = list(range(self.width))
        self._waiting: List[Hashable] = []
        self._frames: Dict[Hashable, torch.Tensor] = {}
        self._length: Dict[Hashable, int] = {}
        self._cursor: Dict[Hashable, int] = {}
        self._order: List[Hashable] = []
        self._slots: Dict[Hashable, int] = {}
        self._internal: Dict[Hashable, int] = {}
        self._next_internal_id = 0
        # Per-tick device state, rebuilt only when the row set changes.
        self._slot_ids: Optional[torch.Tensor] = None
        self._frame_index: Optional[torch.Tensor] = None
        self._parity_pair: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]] = (None, None)
        self._dirty = True
        self._packed_dirty = False
        self._packed_order: List[Hashable] = []
        self._base: Dict[Hashable, int] = {}
        self._retired_frames = 0
        self._packed_frames = 0
        self._retired: Set[Hashable] = set()

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def submit(self, stream_id: Hashable, frames: torch.Tensor) -> None:
        """Queue a whole sequence, ``(length, input_size)``, for streaming."""
        if frames.dim() != 2 or frames.shape[1] != self.input_size:
            raise ValueError(
                f"frames must be (length, {self.input_size}), got {tuple(frames.shape)}"
            )
        if frames.shape[0] <= 0:
            raise ValueError(f"stream {stream_id!r} has no frames")
        if stream_id in self._frames:
            raise ValueError(f"stream {stream_id!r} is already submitted")
        self._frames[stream_id] = frames.contiguous()
        self._length[stream_id] = int(frames.shape[0])
        self._waiting.append(stream_id)
        self._packed_dirty = True

    @property
    def pending(self) -> int:
        """Streams still waiting for a slot."""
        return len(self._waiting)

    @property
    def active(self) -> int:
        """Streams currently holding a slot."""
        return len(self._order)

    def __bool__(self) -> bool:
        return bool(self._waiting or self._order)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def next_step(self) -> Optional[RecurrentStepPlan]:
        """Plan the next tick, or ``None`` when nothing is left to do."""
        finished = self._retire()
        admitted = self._admit()
        if not self._order:
            return None
        if self._packed_dirty:
            self._pack()
        if self._dirty:
            self._rebuild()
        slot_ids = cast_tensor(self._slot_ids)
        frame_index = cast_tensor(self._frame_index)
        # Every row of a stable cohort has taken the same number of steps since
        # the last membership change, so one of two parity vectors always fits.
        parity = self._parity_for_tick()
        frames = torch.index_select(cast_tensor(self._packed), 0, frame_index)
        return RecurrentStepPlan(
            stream_ids=list(self._order),
            slot_ids=slot_ids,
            read_parity=parity,
            frames=frames,
            active_rows=self._active_rows,
            admitted=admitted,
            finished=finished,
            membership_changed=bool(admitted or finished),
        )

    def commit(self, plan: RecurrentStepPlan) -> None:
        """Advance every row of ``plan`` by one timestep."""
        self.cache.commit_step([self._internal[sid] for sid in plan.stream_ids])
        for sid in plan.stream_ids:
            self._cursor[sid] += 1
        # One device-side increment replaces rebuilding the index list.  In
        # padded mode the filler tail must not walk off the packed buffer, so it
        # is pinned back to frame zero.
        index = cast_tensor(self._frame_index)
        index.add_(1)
        if self.padded and self._active_rows < self.width:
            index[self._active_rows :].zero_()
        self._tick += 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parity_for_tick(self) -> torch.Tensor:
        even, odd = self._parity_pair
        return cast_tensor(even if self._tick % 2 == 0 else odd)

    def _pack(self) -> None:
        """Concatenate every submitted sequence once, with fixed base offsets.

        Packing is per *submission*, not per membership change: a row's frame is
        then ``base[stream] + cursor[stream]``, a single flat index, and retiring
        a stream costs nothing but a smaller index list.  Repacking on every
        retirement would copy the whole active corpus once per finished stream.
        """
        order = [sid for sid in self._packed_order if sid in self._frames]
        order += [sid for sid in self._frames if sid not in self._packed_order]
        base = 0
        self._base = {}
        for sid in order:
            self._base[sid] = base
            base += self._length[sid]
        self._packed = torch.cat([self._frames[sid] for sid in order])
        self._packed_order = order
        self._packed_frames = base
        self._retired_frames = 0
        self._packed_dirty = False
        self._dirty = True

    def _rebuild(self) -> None:
        """Rebuild the per-tick device indices for the current row set.

        Runs on membership change only; between changes the frame cursor advances
        with one in-place add and the parity alternates between two tensors.
        """
        rows = self._order
        n = len(rows)
        flat = [self._base[sid] + self._cursor[sid] for sid in rows]
        slots = [self._slots[sid] for sid in rows]
        parity = [self.cache.steps_taken(self._internal[sid]) & 1 for sid in rows]
        if self.padded:
            # Fixed shapes, updated contents: the kernel reads slots and parity
            # out of device tensors, so refilling them costs nothing a graph
            # capture can see.  One capture therefore serves the whole run, which
            # is what a variable-width plan cannot offer -- there, every retiring
            # stream changes the row count and forces a re-capture whose warm-up
            # forward costs more than the ticks it serves.
            pad = self.width - n
            slots += [self._dummy_slot] * pad
            flat += [0] * pad
            parity += [0] * pad
        if self._slot_ids is None or self._slot_ids.numel() != len(slots):
            self._slot_ids = torch.empty(len(slots), dtype=torch.int64, device=self._device)
            self._frame_index = torch.empty(len(flat), dtype=torch.int64, device=self._device)
            self._parity_pair = (
                torch.empty(len(parity), dtype=torch.int32, device=self._device),
                torch.empty(len(parity), dtype=torch.int32, device=self._device),
            )
        cast_tensor(self._slot_ids).copy_(to_device(slots, dtype=torch.int64, device=self._device))
        cast_tensor(self._frame_index).copy_(
            to_device(flat, dtype=torch.int64, device=self._device)
        )
        cast_tensor(self._parity_pair[0]).copy_(
            to_device(parity, dtype=torch.int32, device=self._device)
        )
        cast_tensor(self._parity_pair[1]).copy_(
            to_device([1 - p for p in parity], dtype=torch.int32, device=self._device)
        )
        self._tick = 0
        self._active_rows = n
        self._dirty = False

    def _retire(self) -> List[Hashable]:
        done = [sid for sid in self._order if self._cursor[sid] >= self._length[sid]]
        for sid in done:
            self.cache.free_stream(self._internal[sid])
            # Reuse in retirement order so the pool behaves as a queue.
            self._free.append(self._slots[sid])
            self._order.remove(sid)
            del self._slots[sid], self._internal[sid], self._cursor[sid]
            # A retired stream's frames stay addressable until the next compaction:
            # its base offset is what keeps every *other* stream's offset valid, so
            # dropping it one at a time would repack the whole corpus per finished
            # stream.  They are released in one pass below instead.
            self._retired_frames += self._length[sid]
            self._retired.add(sid)
            self._dirty = True
        if self._retired_frames * 2 >= self._packed_frames and self._retired:
            # Half the packed buffer is now dead weight.  Compact once, which also
            # bounds memory for a batcher that runs indefinitely -- without this,
            # every sequence ever submitted stays resident.
            for sid in self._retired:
                del self._frames[sid], self._length[sid]
            self._retired.clear()
            self._packed_dirty = True
        return done

    def _admit(self) -> List[Hashable]:
        admitted: List[Hashable] = []
        while self._waiting and self._free:
            stream_id = self._waiting.pop(0)
            slot = self._free.pop(0)
            internal = self._next_internal_id
            self._next_internal_id += 1
            # allocate_stream zeroes the slot, and zero *is* the initial state --
            # the same h0/c0 an offline call would default to.
            self.cache.allocate_stream(internal, slot)
            self._cursor[stream_id] = 0
            self._internal[stream_id] = internal
            self._slots[stream_id] = slot
            self._order.append(stream_id)
            admitted.append(stream_id)
            self._dirty = True
        return admitted

    _packed: Optional[torch.Tensor] = None
    _tick: int = 0
    _active_rows: int = 0


def cast_tensor(value: Optional[torch.Tensor]) -> torch.Tensor:
    """Narrow an ``Optional[Tensor]`` that the caller knows is populated."""
    if value is None:  # pragma: no cover - a rebuild always precedes a read
        raise RuntimeError("per-tick index tensors were not built")
    return value
