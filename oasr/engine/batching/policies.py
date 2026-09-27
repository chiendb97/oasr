# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Offline batch-selection policies: fcfs / bucket / sjf.

Each selects one offline batch from the waiting deque (mutating it), preserving
the scheduler's original length-bucketing, padded-compute guard, preferred-size
snapping, and ``max_wait_time`` forced-flush semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Deque, List, Optional

from ..request import Request
from .base import (
    BatchingPolicy,
    register_batching_policy,
    request_cost_frames,
    snap_to_preferred,
    sort_by_length,
)

if TYPE_CHECKING:
    from ..config import EngineConfig


def _batch_cap(config: "EngineConfig", limit: "Optional[int]") -> int:
    """Batch width for one selection: ``max_batch_size``, narrowed by ``limit``.

    Always at least 1 — a caller with zero slots free is expected not to call at
    all, and returning 0 here would stall the queue silently.
    """
    cap = max(1, config.max_batch_size)
    if limit is not None:
        cap = max(1, min(cap, int(limit)))
    return cap


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
        self, queue: "Deque[Request]", config: "EngineConfig", limit: Optional[int] = None
    ) -> List[Request]:
        q = queue
        if not q:
            return []
        cap = _batch_cap(config, limit)
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
        self, queue: "Deque[Request]", config: "EngineConfig", limit: Optional[int] = None
    ) -> List[Request]:
        q = queue
        if not q:
            return []
        cap = _batch_cap(config, limit)
        # Forced-flush anchor if the oldest request has waited too long.
        force_flush = q[0].waited_for >= config.max_wait_time
        self._preorder(q, config, force_flush)

        anchor = q.popleft()
        anchor_len = request_cost_frames(anchor, config)
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
            cand_len = request_cost_frames(cand, config)
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
            useful = sum(request_cost_frames(r, config) for r in batch) + cand_len
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


@register_batching_policy("window")
class WindowPolicy(BatchingPolicy):
    """Length-sorted batches from a bounded window of the oldest requests.

    A padded batch costs its *longest* row times its width, so arrival-order
    batching pays for the length spread of whatever happened to arrive together.
    Measured on LJSpeech at ``B = 32`` (1.5-10 s utterances): FIFO batches
    compute **1.48x** the useful frames.  Sorting the *whole* queue (``sjf``)
    removes nearly all of it — **1.35x** offline Conformer throughput, 1.32x on
    the transducer — but under continuous arrivals a long request loses to every
    newer short one until ``max_wait_time`` forces a FIFO flush, which brings the
    padding straight back.

    This bounds the reordering instead.  Of the oldest ``cap *
    length_window_factor`` requests, sort by cost and take the contiguous run of
    ``cap`` with the least padding **that contains the oldest request** — so the
    anchor always ships (no starvation, no flush needed), and a request can be
    passed over only while it is still inside the window.  Simulated on 2048
    LJSpeech arrivals: a 4x window leaves 1.107x padding against FIFO's 1.484x,
    and delays no request by more than 8 batches.  Below ``cap`` waiting requests
    nothing is reordered at all, so the policy only acts under load, which is
    also the only time padding costs throughput.

    Batches stay in arrival order internally and only one priority class is mixed
    (the anchor's).  The guards the other length-aware policies apply to a batch
    hold here too — ``max_batch_frames``, ``max_offline_pad_ratio`` and
    ``length_bucket_ratio``: a run that breaks one is not a candidate, and the
    batch narrows until one fits (the anchor alone always ships).  At their
    defaults they leave every LJSpeech benchmark batch as it is; set explicitly,
    they mean what they mean under ``bucket``, rather than being dropped when
    the default policy changed.  ``preferred_batch_size`` snaps the result as it
    does for the other policies.
    """

    name: ClassVar[str] = "window"

    def select_offline_batch(
        self, queue: "Deque[Request]", config: "EngineConfig", limit: Optional[int] = None
    ) -> List[Request]:
        q = queue
        if not q:
            return []
        cap = _batch_cap(config, limit)
        force_flush = q[0].waited_for >= config.max_wait_time
        factor = max(1, int(getattr(config, "length_window_factor", 4)))
        anchor = q[0]

        # The window: the oldest requests of the anchor's priority, arrival order.
        window: List[Request] = []
        for r in q:
            if len(window) >= cap * factor or r.priority != anchor.priority:
                break
            window.append(r)
        costs = [request_cost_frames(r, config) for r in window]

        n = min(cap, len(window))
        chosen = self._best_run(
            costs,
            n,
            config.max_batch_frames,
            pad_cap=float(config.max_offline_pad_ratio or 0.0),
            ratio=float(config.length_bucket_ratio or 0.0),
        )
        picked = set(chosen)
        batch = [window[i] for i in sorted(chosen)]
        rest = [r for i, r in enumerate(window) if i not in picked]
        for _ in range(len(window)):
            q.popleft()
        q.extendleft(reversed(rest))
        return snap_offline_batch(batch, q, force_flush, config)

    @staticmethod
    def _best_run(
        costs: List[int],
        n: int,
        frame_cap: Optional[int],
        pad_cap: float = 0.0,
        ratio: float = 0.0,
    ) -> List[int]:
        """Window indices of the least-padded length-run of ``n`` holding index 0.

        A run is a candidate only if it passes the guards the other length-aware
        policies apply to a batch, with the same meaning: ``frame_cap`` bounds its
        padded width (``max_batch_frames``), ``pad_cap`` its padded / useful
        frames (``max_offline_pad_ratio``) and ``ratio`` its shortest / longest
        row (``length_bucket_ratio``); ``None`` / ``0`` disables one.  With no
        candidate at ``n`` the run narrows by one, down to the anchor alone, which
        ships even when it breaks a guard by itself.
        """
        order = sorted(range(len(costs)), key=lambda i: (costs[i], i))
        p = order.index(0)
        prefix = [0]
        for i in order:
            prefix.append(prefix[-1] + costs[i])
        while n > 1:
            best_start: Optional[int] = None
            best_waste = 0
            for st in range(max(0, p - n + 1), min(p, len(order) - n) + 1):
                longest = costs[order[st + n - 1]]
                padded = longest * n
                useful = prefix[st + n] - prefix[st]
                if frame_cap is not None and padded > frame_cap:
                    continue
                if pad_cap > 0 and padded > pad_cap * useful:
                    continue
                if ratio > 0 and costs[order[st]] < ratio * longest:
                    continue
                if best_start is None or padded - useful < best_waste:
                    best_start, best_waste = st, padded - useful
            if best_start is not None:
                return order[best_start : best_start + n]
            n -= 1
        return [0]
