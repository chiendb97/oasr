# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""FCFS request scheduler for the ASR engine."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import EngineConfig
from .request import Request, RequestState


@dataclass
class SchedulerOutput:
    """Output of a single scheduling step.

    Attributes
    ----------
    scheduled_requests : List[Request]
        All running requests that have at least one chunk ready to process.
    newly_admitted : List[Request]
        Requests that were just promoted from WAITING to RUNNING this step.
    to_finalize : List[Request]
        Running requests that have exhausted their chunks and should be
        finalized (decode + cleanup) this step.
    """

    scheduled_requests: List[Request] = field(default_factory=list)
    newly_admitted: List[Request] = field(default_factory=list)
    to_finalize: List[Request] = field(default_factory=list)


class Scheduler:
    """First-come-first-served request scheduler.

    Maintains two queues:

    * ``_waiting`` — requests that have been added but not yet started.
    * ``_running`` — active streaming requests (admitted to engine step).

    Each :meth:`schedule` call:

    1. Admits as many waiting requests as possible without exceeding
       ``max_batch_size``.
    2. Collects all running requests that have pre-chunked features to
       process into ``scheduled_requests``.
    3. Identifies running requests with no remaining chunks and marks them
       for finalization in ``to_finalize``.

    Parameters
    ----------
    config : EngineConfig
        Engine configuration (uses ``max_batch_size``).
    """

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._waiting: deque[Request] = deque()
        self._running: OrderedDict[str, Request] = OrderedDict()
        self._next_stream_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Add a new request to the waiting queue.

        Parameters
        ----------
        request : Request
            Request with state WAITING.  The scheduler assigns a ``stream_id``
            when it is admitted to the running queue.
        """
        request.state = RequestState.WAITING
        self._waiting.append(request)

    def schedule(self) -> SchedulerOutput:
        """Compute the next scheduling step.

        Returns
        -------
        SchedulerOutput
            The sets of requests to admit, process, and finalize this step.
        """
        output = SchedulerOutput()

        # Admit waiting requests up to max_batch_size
        budget = self._config.max_batch_size - len(self._running)
        while self._waiting and budget > 0:
            req = self._waiting.popleft()
            req.stream_id = self._next_stream_id
            self._next_stream_id += 1
            req.state = RequestState.RUNNING
            self._running[req.request_id] = req
            output.newly_admitted.append(req)
            budget -= 1

        # Collect requests with work to do vs. requests ready to finalize
        for req in list(self._running.values()):
            if req.has_chunks:
                output.scheduled_requests.append(req)
            else:
                output.to_finalize.append(req)

        return output

    def finish_request(self, request_id: str) -> Request:
        """Move a request from running to finished state.

        Parameters
        ----------
        request_id : str
            The ID of the request to finish.

        Returns
        -------
        Request
            The finished request.
        """
        req = self._running.pop(request_id)
        req.state = RequestState.FINISHED
        return req

    def abort_request(self, request_id: str) -> Optional[Request]:
        """Remove a request from either queue.

        Parameters
        ----------
        request_id : str
            The ID of the request to abort.

        Returns
        -------
        Request or None
            The aborted request, or ``None`` if not found.
        """
        # Check running first
        if request_id in self._running:
            req = self._running.pop(request_id)
            req.state = RequestState.FINISHED
            return req
        # Check waiting
        for i, req in enumerate(self._waiting):
            if req.request_id == request_id:
                del self._waiting[i]
                req.state = RequestState.FINISHED
                return req
        return None

    def has_pending(self) -> bool:
        """Return ``True`` if there are waiting or running requests."""
        return bool(self._waiting or self._running)

    @property
    def num_waiting(self) -> int:
        """Number of requests in the waiting queue."""
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        """Number of currently active streaming requests."""
        return len(self._running)
