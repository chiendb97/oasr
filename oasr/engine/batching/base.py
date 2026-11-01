# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Batching-policy + partition-policy contracts, registries, and shared helpers.

Two orthogonal seams used by :class:`~oasr.engine.scheduler.Scheduler`:

* :class:`BatchingPolicy` — *selects* one offline batch from the waiting queue
  (fcfs / bucket / sjf), keyed on ``config.schedule_policy``.
* :class:`PartitionPolicy` — *partitions* a selected batch into encoder
  micro-batches (count / frames / packing), keyed on engine config flags.

Adding a new batching or partition strategy is a subclass + ``@register_*`` — no
scheduler edits.  The length/preferred-size helpers live here because both the
streaming admission path (in the scheduler) and the offline policies use them.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from bisect import bisect_left
from operator import attrgetter
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Deque,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

from ..request import Request

if TYPE_CHECKING:
    from ..config import EngineConfig

#: Result of partitioning: ``(chunks, orig_indices)`` where ``orig_indices[pos]``
#: is the input index of the request at flat output position ``pos`` (``None``
#: when no reordering was applied).
PartitionResult = Tuple[List[List[Request]], Optional[List[int]]]

_ARRIVAL = attrgetter("arrival_time")


# ----------------------------------------------------------------------------
# Shared length / preferred-size helpers (used by streaming admission too)
# ----------------------------------------------------------------------------


def request_cost_frames(request: Request, config: "EngineConfig") -> int:
    """Feature frames this request will actually cost the encoder.

    Normally that is its own length (``num_frames``).  But a **fixed-window**
    frontend pads *and trims* every utterance to one size — ``whisper_logmel``
    forces the 30 s Whisper window, shared by Qwen2-Audio — so every row costs the
    same no matter how long it was, and the encoder discards the real lengths
    entirely (``WhisperEncoder.forward`` does ``del xs_lens``).

    Reading ``num_frames`` there makes every length-aware decision wrong in the
    same direction: ``length_bucket_ratio`` and ``max_offline_pad_ratio`` split
    batches to avoid padding waste that does not exist (a 2 s and a 30 s clip cost
    *the same*), and ``max_batch_frames`` under-counts the real padded width by up
    to ~30x, so it stops bounding memory. Asking the config for the cost fixes all
    three at once, and any future fixed-window frontend inherits the fix.
    """
    fcfg = getattr(config, "feature_config", None)
    fixed = getattr(fcfg, "fixed_window_frames", None) if fcfg is not None else None
    if fixed:
        return int(fixed)
    return max(1, int(request.num_frames))


def sort_by_length(queue: Deque[Request]) -> None:
    """In-place stable sort of ``queue`` by ``(priority, num_frames)``."""
    ordered = sorted(queue, key=lambda r: (r.priority, r.num_frames))
    queue.clear()
    queue.extend(ordered)


def promote_overdue(queue: Deque[Request], max_wait_time: float) -> float:
    """Move every request that has waited ``max_wait_time`` to the head, oldest first.

    The rest keep their order behind them.  Returns the ``time.monotonic()``
    instant at which the next waiting request becomes overdue (``math.inf`` if
    none will), before which another call finds nothing new to move -- unless
    a request has since been filed ahead of others, or the overdue ones were
    reordered.

    This is the starvation bound, and it outranks priority.  The queue is kept
    in priority order, so its head is the best class's oldest request, and a
    bound read off the head alone never fired for a deferred request while
    better work kept arriving.  Run before a selection, it makes the head the
    oldest overdue request whenever there is one, which is what each policy's
    ``q[0].waited_for >= max_wait_time`` check assumes.
    """
    if not queue:
        return math.inf
    # The queue can be thousands deep, so the common outcomes (nothing overdue,
    # or every overdue request already at the head) are decided on the arrival
    # times alone, at C speed.
    now = time.monotonic()
    times: List[float] = list(map(_ARRIVAL, queue))
    first = min(times)
    if now - first < max_wait_time:
        return first + max_wait_time
    # The wait falls as the arrival time rises, so the overdue requests are the
    # oldest ``k``.
    oldest = sorted(times)
    k = bisect_left(oldest, True, key=lambda t: now - t < max_wait_time)
    if times[:k] != oldest[:k]:  # else already there: a single-class backlog
        limit = oldest[k - 1]
        overdue = sorted((r for r in queue if r.arrival_time <= limit), key=_ARRIVAL)
        rest = [r for r in queue if r.arrival_time > limit]
        queue.clear()
        queue.extend(overdue)
        queue.extend(rest)
    return oldest[k] + max_wait_time if k < len(oldest) else math.inf


def sort_streams_for_admission(queue: Deque[Request], max_wait_time: float) -> None:
    """In-place sort of the streaming waiting queue: overdue streams first, oldest first.

    Streams that have waited ``max_wait_time`` or longer come first in arrival
    order, whatever their priority; the rest follow by priority, then by
    ``num_frames``.

    Sorting a waiting *stream* by length alone is last-in, first-out. A waiting
    stream's ``num_frames`` is the audio buffered so far, which grows the whole
    time it waits, so the stream that has waited longest sorts last. Once more
    streams wait than there are free slots, it is passed over on every tick by
    each newer arrival and its backlog grows without bound. The overdue group
    is the same starvation bound the offline policies apply (:func:`promote_overdue`),
    and for the same reason it spans the priority classes: applied within each
    one, a deferred stream waited for as long as better ones kept arriving.
    """
    now = time.monotonic()

    def key(r: Request):
        if now - r.arrival_time >= max_wait_time:
            return (0, r.arrival_time, 0)
        return (1, r.priority, r.num_frames)

    ordered = sorted(queue, key=key)
    queue.clear()
    queue.extend(ordered)


def snap_to_preferred(candidate: int, preferred: Optional[List[int]]) -> int:
    """Largest configured preferred batch size ``<= candidate``.

    Returns ``0`` when ``preferred`` is unset/empty or ``candidate`` is below the
    smallest preferred value.  ``preferred`` is assumed sorted ascending (the
    config normaliser guarantees it).
    """
    if not preferred or candidate < preferred[0]:
        return 0
    snapped = 0
    for v in preferred:
        if v <= candidate:
            snapped = v
        else:
            break
    return snapped


# ----------------------------------------------------------------------------
# Policy contracts
# ----------------------------------------------------------------------------


class BatchingPolicy(ABC):
    """Selects one offline batch from the waiting queue (mutating it)."""

    name: ClassVar[str]

    @abstractmethod
    def select_offline_batch(
        self,
        queue: Deque[Request],
        config: "EngineConfig",
        limit: Optional[int] = None,
    ) -> List[Request]:
        """Pop and return one offline batch; leftover requests stay in ``queue``.

        ``queue`` arrives in priority order (lower first, FIFO within a class),
        except that the scheduler has moved every request that has waited
        ``max_wait_time`` to the head, oldest first (:func:`promote_overdue`).
        So ``q[0].waited_for >= max_wait_time`` exactly when some request is
        overdue, and ``q[0]`` is then the one that has waited longest.

        While ``q[0]`` is overdue, leave the requests you do not take in the
        order they came in.  The scheduler re-runs the bound only when a
        request turns overdue or is filed ahead of others, so an overdue head
        reordered here would stay out of order until then.  With nothing
        overdue, reorder freely (``sjf`` sorts).

        ``limit`` caps the batch below ``config.max_batch_size`` for this call
        only.  The incremental AR executor passes its remaining decode slots:
        without it, a tick with one free slot would still pull a full
        ``max_batch_size`` batch and prefill all of it, overshooting
        ``max_decode_slots`` by up to ``max_batch_size - 1`` requests' worth of
        decoder KV.
        """
        raise NotImplementedError


class PartitionPolicy(ABC):
    """Partitions a selected offline batch into encoder micro-batches."""

    name: ClassVar[str]

    @abstractmethod
    def split(self, batch: List[Request], config: "EngineConfig") -> PartitionResult:
        """Return ``(chunks, orig_indices)`` for ``batch`` (non-empty)."""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Registries + builders
# ----------------------------------------------------------------------------

_BATCHING: Dict[str, Type[BatchingPolicy]] = {}
_PARTITION: Dict[str, Type[PartitionPolicy]] = {}


def register_batching_policy(name: str):
    """Class decorator registering a :class:`BatchingPolicy` under ``name``
    (a ``config.schedule_policy`` value)."""

    def _wrap(cls: Type[BatchingPolicy]) -> Type[BatchingPolicy]:
        _BATCHING[name] = cls
        return cls

    return _wrap


def register_partition_policy(name: str):
    """Class decorator registering a :class:`PartitionPolicy` under ``name``."""

    def _wrap(cls: Type[PartitionPolicy]) -> Type[PartitionPolicy]:
        _PARTITION[name] = cls
        return cls

    return _wrap


def build_batching_policy(config: "EngineConfig") -> BatchingPolicy:
    """Construct the batching policy for ``config.schedule_policy``."""
    name = config.schedule_policy
    cls = _BATCHING.get(name)
    if cls is None:
        raise NotImplementedError(
            f"No batching policy registered for schedule_policy={name!r}. "
            f"Registered: {sorted(_BATCHING)}."
        )
    return cls()


def build_partition_policy(config: "EngineConfig") -> PartitionPolicy:
    """Pick the partition policy from engine config flags.

    Dispatch order matches the original scheduler: sequence packing →
    padded-frame budget → count / preferred-size.
    """
    if config.enable_sequence_packing:
        name = "packing"
    elif config.max_batch_frames is not None:
        name = "frames"
    else:
        name = "count"
    return _PARTITION[name]()
