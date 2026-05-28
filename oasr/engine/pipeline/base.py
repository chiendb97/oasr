# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Pipeline abstract base class.

The engine runs in exactly one mode per service lifecycle — streaming OR
offline, never both.  A :class:`Pipeline` encapsulates everything
mode-specific: admission, per-tick orchestration (fbank, forward, decode,
finalise), and cache lifecycle.  The engine itself is mode-agnostic and
delegates each public entry point to ``self._pipeline``.

Concrete implementations:

* :class:`oasr.engine.pipeline.OfflinePipeline` — length-bucketed
  micro-batches with depth-N CPU/GPU overlap; one final output per request
  when its micro-batch drains.
* :class:`oasr.engine.pipeline.StreamingPipeline` — chunk-by-chunk per
  stream with paged KV cache; partial outputs per tick, final on drain.
  (Added in step 3 of the refactor.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional, Union

import numpy as np
import torch

from ..request import Request, RequestOutput


class Pipeline(ABC):
    """Abstract per-tick pipeline for streaming or offline inference."""

    #: Class-level mode tag.  ``True`` for streaming pipelines, ``False``
    #: for offline.  Used by :meth:`ASREngine.add_request` to validate that
    #: incoming requests match the configured service mode.
    streaming: ClassVar[bool]

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

        Streaming pipelines append the chunk to the request's audio deque.
        Offline pipelines raise :class:`NotImplementedError`.
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

        Streaming pipelines emit partial outputs per active stream and
        final outputs for streams whose audio has been fully consumed.
        Offline pipelines emit one final output per request when its
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
