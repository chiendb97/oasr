# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The backend contract every inference framework is measured through.

The contract is asynchronous because the load generator is: one event loop
issues requests on a schedule and timestamps completions, whether the backend
is an HTTP server, a WebSocket, or an in-process engine pumped from the same
loop.  A backend reports what it can do (:class:`Capabilities`) and the suite
refuses -- records ``unsupported`` -- rather than emulating what it cannot: a
"batch" that is really N sequential calls, or "streaming" that is an offline
call on accumulated audio, would make two frameworks look comparable when they
are not.

Timestamps are ``time.perf_counter()`` values taken by the backend at the point
it *observes* each event (a response's last byte, the engine step that emitted
an output).  The runner converts them to the run clock; it never times a
backend from outside an ``await``, where another coroutine's resumption could
be billed to it.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.asrbench.workloads import WorkItem

#: How a backend produces streaming output.  Never interchangeable in a comparison.
STREAMING_MODES: Dict[str, str] = {
    "native": "incremental: the model consumes audio as it arrives and emits partials",
    "buffered": "audio is buffered and decoded once the stream is committed",
    "segmented": "audio is decoded per detected speech segment (VAD/endpoint driven)",
}


class Unsupported(Exception):
    """The backend cannot do what a benchmark asked; recorded, never emulated."""


class RequestFailed(Exception):
    """One request did not produce a transcript.

    ``kind`` is ``"rejected"`` for explicit back-pressure (HTTP 429/503, gRPC
    RESOURCE_EXHAUSTED) -- the server is protecting itself, which a load test
    must count separately from ``"failed"``.
    """

    def __init__(self, message: str, kind: str = "failed", status: Optional[int] = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclass
class Capabilities:
    offline: bool = True
    #: A native multi-input call, executed as one batch.
    batch: bool = False
    max_batch_size: Optional[int] = None
    #: Overlapping requests are scheduled by the backend (server-side batching).
    concurrent: bool = False
    #: One of :data:`STREAMING_MODES`, or ``None``.
    streaming: Optional[str] = None
    partials: bool = False
    word_timestamps: bool = False
    #: ``"whole"`` (decodes arbitrarily long input in one pass), ``"windowed"``
    #: (internal chunking), or ``None`` (bounded input length).
    long_form: Optional[str] = None
    #: Inference runs in this process: allocator peaks are observable, and the
    #: resource monitor watches this PID.
    in_process: bool = False
    server_metrics: bool = False
    notes: str = ""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Transcript:
    text: str
    t_submit: float
    t_done: float
    t_first: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    kind: str  # "partial" | "final" | "error"
    t: float
    text: str = ""
    words: Optional[List[Dict[str, Any]]] = None


class StreamSession(ABC):
    """One streaming request.  Events accumulate in :attr:`events` as they arrive."""

    def __init__(self) -> None:
        self.events: List[StreamEvent] = []
        self._final = asyncio.Event()
        self.error: Optional[str] = None

    def push(self, event: StreamEvent) -> None:
        self.events.append(event)
        if event.kind == "error":
            self.error = event.text
        if event.kind in ("final", "error"):
            self._final.set()

    @abstractmethod
    async def send(self, chunk: Any, is_last: bool) -> None:
        """Hand one chunk (from :meth:`Backend.prepare_chunk`) to the backend."""

    async def wait_final(self, timeout: Optional[float]) -> StreamEvent:
        await asyncio.wait_for(self._final.wait(), timeout)
        if self.error is not None:
            raise RequestFailed(self.error)
        return next(e for e in reversed(self.events) if e.kind == "final")

    async def abort(self) -> None:  # noqa: B027 -- optional hook, no-op by default
        """Release backend resources for an abandoned session."""


@dataclass
class BackendInfo:
    name: str
    version: str = ""
    mode: str = "offline"
    model: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    device: str = ""
    dtype: str = ""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


class Backend(ABC):
    """One inference framework, configured for one service mode."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    #: ``option -> help`` -- what ``-o key=value`` accepts.
    OPTIONS: ClassVar[Dict[str, str]] = {}

    def __init__(self, options: Optional[Dict[str, Any]] = None, mode: str = "offline") -> None:
        self.options: Dict[str, Any] = dict(options or {})
        self.mode = mode
        unknown = set(self.options) - set(self.OPTIONS) - {"sample_rate"}
        if unknown:
            raise ValueError(
                f"backend {self.name!r}: unknown option(s) {sorted(unknown)}; "
                f"accepted: {sorted(self.OPTIONS)}"
            )

    @classmethod
    def available(cls) -> Tuple[bool, str]:
        """Whether the backend's dependencies import, and why not."""
        return True, ""

    @classmethod
    def mode_unsupported(cls, mode: str, options: Dict[str, Any]) -> Optional[str]:
        """Why this backend cannot serve *mode* at all, known before loading.

        Lets the runner skip loading a second copy of a model only to learn
        that every benchmark of that mode is unsupported.
        """
        return None

    @property
    def sample_rate(self) -> int:
        return int(self.options.get("sample_rate", 16000))

    # -- lifecycle --------------------------------------------------------------

    @abstractmethod
    async def load(self) -> None:
        """Initialise (load the model / start the server).  Timed as ``init_s``."""

    async def close(self) -> None:
        return None

    @abstractmethod
    def info(self) -> BackendInfo: ...

    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    # -- inputs (untimed) ------------------------------------------------------

    def prepare(self, audio: np.ndarray, item: WorkItem) -> Any:
        """Convert a decoded waveform to this backend's request payload."""
        return audio

    def prepare_chunk(self, chunk: np.ndarray) -> Any:
        """Convert one streaming chunk to the backend's wire form."""
        return chunk

    # -- inference (timed) -------------------------------------------------------

    @abstractmethod
    async def transcribe(self, payload: Any) -> Transcript: ...

    async def transcribe_batch(self, payloads: Sequence[Any]) -> List[Transcript]:
        raise Unsupported(f"{self.name} has no native batch call")

    async def open_stream(self, *, word_timestamps: bool = False) -> StreamSession:
        """Open one streaming request; ``word_timestamps`` asks for final word timings."""
        raise Unsupported(f"{self.name} does not stream")

    async def cancel(self, payload: Any) -> None:  # noqa: B027 -- optional hook
        """Best-effort cleanup after a timed-out request."""

    # -- observability ---------------------------------------------------------

    async def synchronize(self) -> None:  # noqa: B027 -- optional hook
        """Wait for any device work still queued (benchmark boundaries only)."""

    def monitored_pid(self) -> Optional[int]:
        """The process doing inference, for the resource monitor."""
        return os.getpid()

    def reset_peak_memory(self) -> None:
        return None

    def peak_memory(self) -> Dict[str, float]:
        """In-process allocator peaks since :meth:`reset_peak_memory` (MiB)."""
        return {}

    def server_metrics_begin(self) -> None:
        return None

    def server_metrics_end(self) -> Dict[str, Any]:
        """Server-side observations for the window since :meth:`server_metrics_begin`."""
        return {}
