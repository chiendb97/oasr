# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Executor abstract base class.

The engine runs in exactly one mode per service lifecycle — streaming OR
offline, never both.  An :class:`Executor` encapsulates everything
mode-specific: admission, per-tick orchestration (fbank, forward, decode,
finalise), and cache lifecycle.  The engine itself is mode-agnostic and
delegates each public entry point to ``self._executor``.

Concrete implementations:

* :class:`oasr.engine.executor.OfflineExecutor` — length-bucketed
  micro-batches (optionally sequence-packed); one final output per request
  when its micro-batch drains.
* :class:`oasr.engine.executor.StreamingExecutor` — chunk-by-chunk per
  stream with paged KV cache; partial outputs per tick, final on drain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional, Union

import numpy as np
import torch

from ..metrics import EngineMetrics, NullMetrics
from ..request import Request, RequestOutput


class Executor(ABC):
    """Abstract per-tick executor for streaming or offline inference."""

    #: Class-level mode tag.  ``True`` for streaming executors, ``False``
    #: for offline.  Used by :meth:`ASREngine.add_request` to validate that
    #: incoming requests match the configured service mode.
    streaming: ClassVar[bool]

    #: Per-engine collector with a stateless default for minimally constructed
    #: executors. This remains an instance attribute, not a ``ClassVar``.
    _metrics: EngineMetrics = NullMetrics()

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    @abstractmethod
    def admit(self, request: Request) -> None:
        """Enqueue a freshly-built :class:`Request` for processing.

        Implementations are responsible for any mode-specific preparation
        (e.g. ``InputProcessor.prepare_offline`` vs ``prepare_streaming``)
        and for inserting the request into their scheduler's waiting queue.
        """

    @abstractmethod
    def feed_chunk(
        self,
        request_id: str,
        chunk: Union[torch.Tensor, "np.ndarray"],
        is_last: bool = False,
    ) -> None:
        """Push one audio chunk into a streaming request.

        Streaming executors append the chunk to the request's audio deque.
        Offline executors raise :class:`NotImplementedError`.
        """

    @abstractmethod
    def abort(self, request_id: str) -> None:
        """Remove a request, freeing any allocated cache resources."""

    # ------------------------------------------------------------------
    # Per-tick step
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self) -> List[RequestOutput]:
        """Execute one engine tick worth of work; return any outputs.

        Streaming executors emit partial outputs per active stream and
        final outputs for streams whose audio has been fully consumed.
        Offline executors emit one final output per request when its
        micro-batch drains.  May return an empty list when no work is
        ready.
        """

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @abstractmethod
    def has_pending(self) -> bool:
        """``True`` while any request is waiting, running, or in-flight."""

    @abstractmethod
    def num_running(self) -> int:
        """Currently-admitted (or in-flight) requests."""

    @abstractmethod
    def num_waiting(self) -> int:
        """Requests in the waiting queue."""

    @abstractmethod
    def find_request(self, request_id: str) -> Optional[Request]:
        """Look up a request by id; ``None`` if unknown or finished."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Best-effort shutdown hook.  Default is a no-op."""
        return None

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def record_gauges(self, metrics) -> None:
        """Refresh the point-in-time gauges this executor owns.

        Called at *drain* time rather than per tick, so an executor with
        nothing to report costs nothing and one that must take a lock (the
        block pool's free list) takes it a few times a second instead of a few
        thousand.  Non-abstract because occupancy is a property of the
        execution model: an offline one-shot executor has no decode slots and
        an all-offline engine has no paged pool, and neither should have to
        write an empty override to say so.
        """
        return None
