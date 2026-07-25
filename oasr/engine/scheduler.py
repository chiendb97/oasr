# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Dynamic-batching request scheduler for the ASR engine.

The scheduler is length-aware: it admits waiting requests into streaming and
offline batches chosen to minimise padded-compute waste.  Three policies are
supported (set via ``EngineConfig.schedule_policy``):

``"fcfs"``
    Strict first-come-first-served — preserves arrival order; no bucketing.
``"bucket"`` (default)
    Pick the oldest waiting request as an "anchor", then greedily add
    arrival-ordered peers whose feature length is within
    ``length_bucket_ratio`` of the anchor.  Good trade-off between latency and
    throughput.
``"sjf"``
    Shortest-job-first — sort the waiting queue by feature length and pick
    the shortest group.  Best throughput, but relies on
    ``max_wait_time`` to bound starvation of long requests.

Starvation bound: any request whose ``waited_for`` exceeds ``max_wait_time``
becomes an immediate-flush anchor regardless of policy.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .batching import (
    build_batching_policy,
    build_partition_policy,
    snap_to_preferred,
    sort_by_length,
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
            self._insert_ordered(self._offline_waiting, request)
        self._index[request.request_id] = request

    def schedule_offline(self) -> List[Request]:
        """Pick one length-bucketed offline batch from the offline queue.

        Marks each picked request RUNNING and returns the batch.  Returns
        an empty list when no batch is ready.  Each batch is admitted-then-
        finalised in the same step; offline requests never enter
        ``_running`` (that pool tracks streaming admissions only).

        Batch *selection* is delegated to the configured
        :class:`~oasr.engine.batching.BatchingPolicy` (fcfs / bucket / sjf).
        """
        batch = self._batching_policy.select_offline_batch(self._offline_waiting, self._config)
        for req in batch:
            req.state = RequestState.RUNNING
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
            sort_by_length(self._streaming_waiting)

        n_avail = min(budget, len(self._streaming_waiting))
        if cfg.preferred_batch_size is not None:
            candidate = len(self._running) + n_avail
            target = snap_to_preferred(candidate, cfg.preferred_batch_size)
            n_admit = max(0, target - len(self._running))
            if n_admit == 0:
                oldest = self._streaming_waiting[0]
                if oldest.waited_for >= cfg.max_wait_time:
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

    def _insert_ordered(self, queue: Deque[Request], request: Request) -> None:
        """Append respecting priority.  Lower priority value inserts earlier.

        Within the same priority the insertion is FIFO, preserving arrival
        order.  For the default case (all priority=0) this degenerates to
        ``append`` and is O(1).
        """
        if request.priority == 0 or not queue or queue[-1].priority <= request.priority:
            queue.append(request)
            return
        # Find the first slot whose priority is strictly worse (higher) and
        # insert before it.  Queues are expected to be small (< 10³).
        for i, existing in enumerate(queue):
            if existing.priority > request.priority:
                queue.insert(i, request)
                return
        queue.append(request)

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
