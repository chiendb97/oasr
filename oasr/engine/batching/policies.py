# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Offline batch-selection policies: fcfs / bucket / sjf.

Each selects one offline batch from the waiting deque (mutating it), preserving
the scheduler's original length-bucketing, padded-compute guard, preferred-size
snapping, and ``max_wait_time`` forced-flush semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Deque, List

from ..request import Request
from .base import BatchingPolicy, register_batching_policy, snap_to_preferred, sort_by_length

if TYPE_CHECKING:
    from ..config import EngineConfig


def snap_offline_batch(
    batch: List[Request], q: "Deque[Request]", force_flush: bool, config: "EngineConfig"
) -> List[Request]:
    """Trim a built batch to a preferred batch size, when configured.

    Returns the overflow to the head of ``q`` (order preserved) so the next step
    picks them up.  Skipped when ``preferred_batch_size`` is unset or
    ``force_flush`` is set (the wait deadline overrides the preferred-size cap).
    """
    if config.preferred_batch_size is None or force_flush or not batch:
        return batch
    target = snap_to_preferred(len(batch), config.preferred_batch_size)
    if target == 0:
        # Below the smallest preferred — hold everything and wait.
        q.extendleft(reversed(batch))
        return []
    if target < len(batch):
        overflow = batch[target:]
        batch = batch[:target]
        q.extendleft(reversed(overflow))
    return batch


def fill_batch_fifo(batch: List[Request], q: "Deque[Request]", cap: int) -> List[Request]:
    """Fill a forced-flush batch with strict FIFO order up to ``cap``."""
    while q and len(batch) < cap:
        batch.append(q.popleft())
    return batch


@register_batching_policy("fcfs")
class FcfsPolicy(BatchingPolicy):
    """Strict first-come-first-served — preserves arrival order, no bucketing."""

    name: ClassVar[str] = "fcfs"

    def select_offline_batch(
        self, queue: "Deque[Request]", config: "EngineConfig"
    ) -> List[Request]:
        q = queue
        if not q:
            return []
        cap = max(1, config.max_batch_size)
        force_flush = q[0].waited_for >= config.max_wait_time
        batch: List[Request] = []
        while q and len(batch) < cap:
            batch.append(q.popleft())
        return snap_offline_batch(batch, q, force_flush, config)


class _LengthAwarePolicy(BatchingPolicy):
    """Anchor + greedy length-similar fill (shared by bucket and sjf)."""

    def _preorder(self, q: "Deque[Request]", config: "EngineConfig", force_flush: bool) -> None:
        """Reorder the queue before anchor selection.  Default: no-op."""

    def select_offline_batch(
        self, queue: "Deque[Request]", config: "EngineConfig"
    ) -> List[Request]:
        q = queue
        if not q:
            return []
        cap = max(1, config.max_batch_size)
        # Forced-flush anchor if the oldest request has waited too long.
        force_flush = q[0].waited_for >= config.max_wait_time
        self._preorder(q, config, force_flush)

        anchor = q.popleft()
        anchor_len = max(1, anchor.num_frames)
        batch = [anchor]
        min_len = anchor_len
        max_len = anchor_len

        if force_flush:
            # Keep strict FIFO for this batch — don't reorder just because we've
            # exceeded the wait deadline.
            batch = fill_batch_fifo(batch, q, cap)
            return snap_offline_batch(batch, q, True, config)

        ratio = config.length_bucket_ratio
        pad_cap = config.max_offline_pad_ratio
        frame_cap = config.max_batch_frames

        i = 0
        while i < len(q) and len(batch) < cap:
            cand = q[i]
            cand_len = max(1, cand.num_frames)
            new_min = min(min_len, cand_len)
            new_max = max(max_len, cand_len)

            if ratio > 0 and new_min / new_max < ratio:
                i += 1
                continue

            # Padded-frame budget: would adding this push the padded width
            # ``new_max * (batch_size + 1)`` over ``max_batch_frames``?  The
            # anchor always ships even if it alone exceeds the budget.
            if frame_cap is not None and new_max * (len(batch) + 1) > frame_cap:
                i += 1
                continue

            # Pad-waste guard: would adding this push total padded compute above
            # ``pad_cap`` × useful compute?
            useful = sum(max(1, r.num_frames) for r in batch) + cand_len
            padded = new_max * (len(batch) + 1)
            if pad_cap > 0 and padded / useful > pad_cap:
                i += 1
                continue

            batch.append(cand)
            min_len = new_min
            max_len = new_max
            del q[i]

        return snap_offline_batch(batch, q, False, config)


@register_batching_policy("bucket")
class BucketPolicy(_LengthAwarePolicy):
    """Oldest request as anchor, greedily add arrival-ordered length-similar peers."""

    name: ClassVar[str] = "bucket"


@register_batching_policy("sjf")
class SjfPolicy(_LengthAwarePolicy):
    """Shortest-job-first — sort the queue by length, then anchor + greedy fill."""

    name: ClassVar[str] = "sjf"

    def _preorder(self, q: "Deque[Request]", config: "EngineConfig", force_flush: bool) -> None:
        if not force_flush:
            sort_by_length(q)
