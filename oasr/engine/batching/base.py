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

from abc import ABC, abstractmethod
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


# ----------------------------------------------------------------------------
# Shared length / preferred-size helpers (used by streaming admission too)
# ----------------------------------------------------------------------------


def sort_by_length(queue: Deque[Request]) -> None:
    """In-place stable sort of ``queue`` by ``(priority, num_frames)``."""
    ordered = sorted(queue, key=lambda r: (r.priority, r.num_frames))
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
    def select_offline_batch(self, queue: Deque[Request], config: "EngineConfig") -> List[Request]:
        """Pop and return one offline batch; leftover requests stay in ``queue``."""
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
