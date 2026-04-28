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
from typing import Deque, Dict, List, Optional

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
    ``max_offline_batch_size`` and bucketed by feature length), admits as
    many streaming requests as ``max_batch_size`` allows, and tags running
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

    def schedule(self) -> SchedulerOutput:
        """Compute the next scheduling step."""
        output = SchedulerOutput()

        # 1. Offline batching — length-bucketed, size-capped.
        offline_batch = self._build_offline_batch()
        if offline_batch:
            for req in offline_batch:
                req.state = RequestState.RUNNING
            output.offline_batch = offline_batch
            output.newly_admitted.extend(offline_batch)

        # 2. Streaming admission — capped by running pool size.
        budget = self._config.max_batch_size - len(self._running)
        admitted_streaming = self._admit_streaming(budget)
        for req in admitted_streaming:
            self._running[req.request_id] = req
            output.newly_admitted.append(req)

        # 3. Hand the engine all active streams; it dispatches per-stream
        #    work (feature extraction, encoder chunk, finalisation) itself
        #    because one step may need to do several of those in sequence.
        output.running_streams = list(self._running.values())

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
        """
        if budget <= 0 or not self._streaming_waiting:
            return []

        cfg = self._config
        policy = cfg.schedule_policy
        if cfg.streaming_cohort_admit or policy == "sjf":
            self._sort_by_length(self._streaming_waiting)

        admitted: List[Request] = []
        while self._streaming_waiting and len(admitted) < budget:
            req = self._streaming_waiting.popleft()
            req.stream_id = self._next_stream_id
            self._next_stream_id += 1
            req.state = RequestState.RUNNING
            admitted.append(req)
        return admitted

    # ------------------------------------------------------------------
    # Internal — offline batching
    # ------------------------------------------------------------------

    def _build_offline_batch(self) -> List[Request]:
        """Construct one length-bucketed offline batch.

        Policy dispatch lives here.  The batch cap is
        ``max_offline_batch_size``; bucket tolerance is
        ``length_bucket_ratio``.  Requests whose ``waited_for`` exceeds
        ``max_wait_time`` become forced anchors — they ship next step
        regardless of whether length-similar peers are available.
        """
        q = self._offline_waiting
        if not q:
            return []

        cfg = self._config
        cap = max(1, cfg.max_offline_batch_size)
        policy = cfg.schedule_policy

        # Forced-flush anchor if the oldest request has waited too long.
        force_flush = q[0].waited_for >= cfg.max_wait_time

        if policy == "fcfs":
            batch: List[Request] = []
            while q and len(batch) < cap:
                batch.append(q.popleft())
            return batch

        if policy == "sjf" and not force_flush:
            self._sort_by_length(q)

        # "bucket" (default) and "sjf" both do length-aware selection.
        anchor = q.popleft()
        anchor_len = max(1, anchor.num_frames)
        batch = [anchor]
        min_len = anchor_len
        max_len = anchor_len

        if force_flush:
            # Keep strict FIFO for this batch — don't reorder the queue just
            # because we've exceeded the wait deadline.
            return self._fill_batch_fifo(batch, q, cap)

        ratio = cfg.length_bucket_ratio
        pad_cap = cfg.max_offline_pad_ratio

        i = 0
        while i < len(q) and len(batch) < cap:
            cand = q[i]
            cand_len = max(1, cand.num_frames)
            new_min = min(min_len, cand_len)
            new_max = max(max_len, cand_len)

            if ratio > 0 and new_min / new_max < ratio:
                i += 1
                continue

            # Pad-waste guard: would adding this request push total padded
            # compute above ``pad_cap`` × useful compute?
            useful = sum(max(1, r.num_frames) for r in batch) + cand_len
            padded = new_max * (len(batch) + 1)
            if pad_cap > 0 and padded / useful > pad_cap:
                i += 1
                continue

            batch.append(cand)
            min_len = new_min
            max_len = new_max
            del q[i]

        return batch

    @staticmethod
    def _fill_batch_fifo(
        batch: List[Request],
        q: Deque[Request],
        cap: int,
    ) -> List[Request]:
        """Fill a forced-flush batch with strict FIFO order up to ``cap``."""
        while q and len(batch) < cap:
            batch.append(q.popleft())
        return batch

    # ------------------------------------------------------------------
    # Internal — priority/length ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_by_length(queue: Deque[Request]) -> None:
        """In-place stable sort of ``queue`` by (priority, num_frames)."""
        ordered = sorted(queue, key=lambda r: (r.priority, r.num_frames))
        queue.clear()
        queue.extend(ordered)

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

    @property
    def num_running(self) -> int:
        """Number of currently active streaming requests."""
        return len(self._running)
