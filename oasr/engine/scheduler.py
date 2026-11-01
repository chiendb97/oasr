# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Dynamic-batching request scheduler for the ASR engine.

The scheduler is length-aware: it admits waiting requests into streaming and
offline batches chosen to minimise padded-compute waste.  Four policies are
supported (set via ``EngineConfig.schedule_policy``):

``"fcfs"``
    Strict first-come-first-served — preserves arrival order; no bucketing.
``"window"`` (default)
    Length-sorted batches from the oldest ``max_batch_size *
    length_window_factor`` requests, always including the oldest: most of
    ``"sjf"``'s padding reduction with the reordering bounded by the window.
    Identical to FIFO while fewer than ``max_batch_size`` requests wait.
``"bucket"``
    Pick the oldest waiting request as an "anchor", then greedily add
    arrival-ordered peers whose feature length is within
    ``length_bucket_ratio`` of the anchor.
``"sjf"``
    Shortest-job-first — sort the waiting queue by feature length and pick
    the shortest group.  Best throughput, but relies on
    ``max_wait_time`` to bound starvation of long requests.

Priority: lower ``Request.priority`` is sooner, FIFO within a class, and the
default (0) is the most urgent class.

Starvation bound: any request whose ``waited_for`` exceeds ``max_wait_time``
becomes an immediate-flush anchor regardless of policy or priority.  Before a
selection the overdue requests move to the head of their queue, oldest first --
for offline, only when one has turned overdue or a request was filed ahead of
others since the last time.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .batching import (
    build_batching_policy,
    build_partition_policy,
    promote_overdue,
    snap_to_preferred,
    sort_streams_for_admission,
)
from .config import EngineConfig
from .request import Request, RequestState


@dataclass
class SchedulerOutput:
    """Output of a single scheduling step.

    Attributes
    ----------
    running_streams : List[Request]
        All admitted streaming requests currently in the running pool.  The
        engine step-loop decides per-request whether to run feature
        extraction, an encoder chunk, or finalisation; the scheduler no
        longer distinguishes those states because a single request may need
        several of them within one step.
    newly_admitted : List[Request]
        Requests that were just promoted from WAITING to RUNNING this step.
        For streaming requests, this is where the engine allocates their KV
        cache; offline batches are admitted-then-finalised in the same step
        and never enter ``_running``.
    offline_batch : List[Request]
        Length-bucketed batch of offline requests to forward in a single
        padded batch.  Empty when no offline work is scheduled this step.
    """

    running_streams: List[Request] = field(default_factory=list)
    newly_admitted: List[Request] = field(default_factory=list)
    offline_batch: List[Request] = field(default_factory=list)


class Scheduler:
    """Dynamic-batching scheduler with length bucketing.

    Maintains three queues:

    * ``_streaming_waiting`` — streaming requests awaiting admission to the
      running pool.
    * ``_offline_waiting`` — offline requests awaiting a batched forward
      pass.
    * ``_running`` — admitted streaming requests with allocated KV cache.

    Each :meth:`schedule` call emits at most one offline batch (sized by
    ``max_batch_size`` and bucketed by feature length), admits as many
    streaming requests as ``max_batch_size`` allows, and tags running
    requests for processing or finalisation based on remaining chunk state.

    Parameters
    ----------
    config : EngineConfig
        Engine configuration.
    """

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._streaming_waiting: Deque[Request] = deque()
        self._offline_waiting: Deque[Request] = deque()
        self._running: "OrderedDict[str, Request]" = OrderedDict()
        # O(1) lookup index — kept in sync with the three queues above so
        # ``feed_chunk`` doesn't pay an O(N_pending) scan per chunk on
        # backlog-style workloads.
        self._index: Dict[str, Request] = {}
        self._next_stream_id: int = 0
        # When the offline wait bound next has work (``promote_overdue``): the
        # instant a waiting request turns overdue, or -inf once a request has
        # been filed ahead of others.  Until then the bound would only re-read
        # every request's arrival time, which is most of a selection's cost on a
        # deep queue.
        self._offline_promote_at: float = math.inf

        # Pluggable offline batch-selection + micro-batch partition policies.
        # New policies plug in via the batching registries — no scheduler edits.
        self._batching_policy = build_batching_policy(config)
        self._partition_policy = build_partition_policy(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Add a new request to the appropriate waiting queue.

        Streaming requests enter ``_streaming_waiting``; offline requests
        enter ``_offline_waiting``.  The split lets each policy apply
        independently: streaming admission is cache-bounded while offline
        batching is length-bounded.
        """
        request.state = RequestState.WAITING
        if request.streaming:
            self._insert_ordered(self._streaming_waiting, request)
        else:
            appended = self._insert_ordered(self._offline_waiting, request)
            # Filed ahead of others, it may be ahead of an overdue request.
            due = request.arrival_time + self._config.max_wait_time if appended else -math.inf
            if due < self._offline_promote_at:
                self._offline_promote_at = due
        self._index[request.request_id] = request

    def schedule_offline(self, limit: Optional[int] = None) -> List[Request]:
        """Pick one length-bucketed offline batch from the offline queue.

        Marks each picked request RUNNING and returns the batch.  Returns
        an empty list when no batch is ready.  Each batch is admitted-then-
        finalised in the same step; offline requests never enter
        ``_running`` (that pool tracks streaming admissions only).

        Batch *selection* is delegated to the configured
        :class:`~oasr.engine.batching.BatchingPolicy` (fcfs / bucket / sjf /
        window), after the wait bound has moved every overdue request to the
        head of the queue, whatever its priority.  The bound runs only when it
        can have work; removing requests, or leaving them in order as every
        policy does while one is overdue, gives it none.
        """
        if time.monotonic() >= self._offline_promote_at:
            self._offline_promote_at = promote_overdue(
                self._offline_waiting, self._config.max_wait_time
            )
        batch = self._batching_policy.select_offline_batch(
            self._offline_waiting, self._config, limit
        )
        for req in batch:
            req.state = RequestState.RUNNING
            # A selected offline request never returns to the scheduler: no
            # ``finish_request`` follows, and the executor owns it from here
            # (its incremental pool answers ``find_request`` for the ones that
            # outlive this tick).  Left indexed, every offline request -- with
            # its output, and the waveform of one rejected before collate --
            # stayed reachable for the life of the process.
            self._index.pop(req.request_id, None)
        return batch

    def split_offline_batch(
        self, batch: List[Request]
    ) -> Tuple[List[List[Request]], Optional[List[int]]]:
        """Partition a selected offline batch into encoder micro-batches.

        :meth:`schedule_offline` decides *which* requests form this step's
        batch; this decides *how* they are grouped for the encoder forward.
        Dispatch order: sequence packing (one gapless packed row per chunk) →
        padded-frame budget → count / preferred-size.  All paths length-sort
        the batch to keep each micro-batch tight.

        Returns ``(chunks, orig_indices)`` where ``orig_indices[pos]`` is the
        input index of the request at flat output position ``pos`` — the caller
        uses it to restore arrival order after the length sort.  ``orig_indices``
        is ``None`` when no reordering was applied (single-chunk fast path).

        The *how* (count / frames / packing) is delegated to the configured
        :class:`~oasr.engine.batching.PartitionPolicy`.
        """
        if not batch:
            return [], None
        return self._partition_policy.split(batch, self._config)

    def schedule_streaming(self) -> Tuple[List[Request], List[Request]]:
        """Admit waiting streaming requests up to ``max_batch_size``.

        Returns ``(newly_admitted, running_streams)`` — freshly-promoted
        streams (the engine/executor allocates their KV cache on the first
        list) and the full running pool the caller should iterate this
        step.
        """
        budget = self._config.max_batch_size - len(self._running)
        admitted = self._admit_streaming(budget)
        for req in admitted:
            self._running[req.request_id] = req
        return admitted, list(self._running.values())

    def schedule(self) -> SchedulerOutput:
        """Legacy compositional helper — combines :meth:`schedule_offline`
        and :meth:`schedule_streaming` into one :class:`SchedulerOutput`.

        Mode-specific executors call the finer-grained methods directly so
        they don't pay for the other mode's queue scan.  Retained for
        tests that exercise both queues in one shared scheduler instance.
        """
        output = SchedulerOutput()
        offline_batch = self.schedule_offline()
        if offline_batch:
            output.offline_batch = offline_batch
            output.newly_admitted.extend(offline_batch)
        admitted_streaming, running_streams = self.schedule_streaming()
        output.newly_admitted.extend(admitted_streaming)
        output.running_streams = running_streams
        return output

    def finish_request(self, request_id: str) -> Request:
        """Move a running streaming request to the FINISHED state."""
        req = self._running.pop(request_id)
        req.state = RequestState.FINISHED
        self._index.pop(request_id, None)
        return req

    def abort_request(self, request_id: str) -> Optional[Request]:
        """Remove a request from any queue.  Returns ``None`` if not found."""
        if request_id in self._running:
            req = self._running.pop(request_id)
            req.state = RequestState.FINISHED
            self._index.pop(request_id, None)
            return req
        for queue in (self._streaming_waiting, self._offline_waiting):
            for i, req in enumerate(queue):
                if req.request_id == request_id:
                    del queue[i]
                    req.state = RequestState.FINISHED
                    self._index.pop(request_id, None)
                    return req
        return None

    def has_pending(self) -> bool:
        """Return ``True`` if there are waiting or running requests."""
        return bool(self._streaming_waiting or self._offline_waiting or self._running)

    def find_request(self, request_id: str) -> Optional[Request]:
        """Look up a request by id across the running pool and waiting queues.

        Used by :meth:`ASREngine.feed_chunk` to push an audio chunk onto a
        streaming request that may be running, waiting, or already gone
        (in which case ``None`` is returned and the caller must decide
        whether to raise or silently drop the chunk).  O(1) via the
        scheduler's id index.
        """
        return self._index.get(request_id)

    # ------------------------------------------------------------------
    # Internal — streaming admission
    # ------------------------------------------------------------------

    def _admit_streaming(self, budget: int) -> List[Request]:
        """Pick up to ``budget`` streaming requests to promote to RUNNING.

        Heterogeneous-offset batching (FlexAttention-paged) batches all
        running streams in one encoder forward regardless of each stream's
        ``offset``, so admission carries **no** cohort-alignment gate —
        new requests can join an in-flight pool at any time, and the
        per-stream ``cache_seqlens`` + pos_emb handle the variable
        lengths inside the kernel.

        ``streaming_cohort_admit=True`` now only controls **length
        bucketing**: the waiting queue is sorted by length before
        admission so each batched paged forward sees length-similar
        utterances (lower padded-compute waste). With ``False``, admission
        follows ``schedule_policy`` ordering (FCFS / bucket / SJF).

        The length sort is bounded by ``max_wait_time``, as the offline
        policies are: a stream that has waited that long is admitted ahead
        of the sorted rest, oldest first, whatever its priority.  Without the
        bound the sort is last-in, first-out under load, because a waiting
        stream's ``num_frames`` grows with the audio it buffers (see
        :func:`~oasr.engine.batching.sort_streams_for_admission`).  Without
        the sort, the queue is in priority order and the same bound applies
        (:func:`~oasr.engine.batching.promote_overdue`).

        When ``preferred_batch_size`` is set, the admit count is snapped
        so that the resulting running pool size lands on a preferred
        value (largest preferred ``<= len(_running) + n_avail``).  If no
        preferred grouping is reachable, admit 0 and wait for more
        arrivals — unless the oldest waiting stream has been queued
        longer than ``max_wait_time``, in which case force-flush all
        available so progress is bounded.
        """
        if budget <= 0 or not self._streaming_waiting:
            return []

        cfg = self._config
        policy = cfg.schedule_policy
        if cfg.streaming_cohort_admit or policy == "sjf":
            sort_streams_for_admission(self._streaming_waiting, cfg.max_wait_time)
        else:
            promote_overdue(self._streaming_waiting, cfg.max_wait_time)

        n_avail = min(budget, len(self._streaming_waiting))
        if cfg.preferred_batch_size is not None:
            candidate = len(self._running) + n_avail
            target = snap_to_preferred(candidate, cfg.preferred_batch_size)
            n_admit = max(0, target - len(self._running))
            if n_admit == 0:
                # Either ordering above puts the overdue streams first, oldest
                # first, so the head has waited longest whenever any is overdue.
                if self._streaming_waiting[0].waited_for >= cfg.max_wait_time:
                    n_admit = n_avail
                else:
                    return []
        else:
            n_admit = n_avail

        admitted: List[Request] = []
        while self._streaming_waiting and len(admitted) < n_admit:
            req = self._streaming_waiting.popleft()
            req.stream_id = self._next_stream_id
            self._next_stream_id += 1
            req.state = RequestState.RUNNING
            admitted.append(req)
        return admitted

    # ------------------------------------------------------------------
    # Internal — priority/length ordering
    # ------------------------------------------------------------------

    def _insert_ordered(self, queue: Deque[Request], request: Request) -> bool:
        """Insert behind every request of the same or a better (lower) priority.

        Returns whether the request went to the back.  Within a class the order
        is FIFO.  When every request has the same priority -- the default --
        this is an ``append`` and O(1).

        The scan runs from the back.  Requests the wait bound moved to the head
        (:func:`~oasr.engine.batching.promote_overdue`) are in arrival order, not
        priority order, so a scan from the front could stop at a deferred one
        there and file this request ahead of better ones waiting behind it.
        """
        if not queue or queue[-1].priority <= request.priority:
            queue.append(request)
            return True
        for i in range(len(queue) - 2, -1, -1):
            if queue[i].priority <= request.priority:
                queue.insert(i + 1, request)
                return False
        queue.appendleft(request)
        return False

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def num_waiting(self) -> int:
        """Total number of requests across both waiting queues."""
        return len(self._streaming_waiting) + len(self._offline_waiting)

    @property
    def num_waiting_streaming(self) -> int:
        return len(self._streaming_waiting)

    @property
    def num_waiting_offline(self) -> int:
        return len(self._offline_waiting)

    def oldest_offline_wait(self) -> Optional[float]:
        """Seconds the longest-waiting offline request has been queued.

        ``None`` when the offline queue is empty.  Read by the offline executor's
        AR admission window, which trades a bounded wait for a wider decode batch.
        """
        if not self._offline_waiting:
            return None
        return max(req.waited_for for req in self._offline_waiting)

    @property
    def num_running(self) -> int:
        """Number of currently active streaming requests."""
        return len(self._running)
