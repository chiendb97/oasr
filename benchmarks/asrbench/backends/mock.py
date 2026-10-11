# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""A simulated backend -- for testing the suite, never for measuring.

It models a GPU server just closely enough to exercise every code path the
real backends reach: a dynamic batcher (requests queued while a "forward" runs
are batched into the next), a cost that grows with the padded batch, native
streaming with per-chunk partials and word timestamps, and injectable
failures, back-pressure and hangs.  Timings are ``asyncio.sleep`` -- every run
using it is flagged as simulated by ``report``.

The transcript echoes the item's reference (when it has one) so the
correctness path can be tested end to end; ``drop_word_every`` removes words
deterministically to produce a known WER.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from benchmarks.asrbench.backends.base import (
    Backend,
    BackendInfo,
    Capabilities,
    RequestFailed,
    StreamEvent,
    StreamSession,
    Transcript,
    Unsupported,
)
from benchmarks.asrbench.workloads import WorkItem

DEFAULTS: Dict[str, Any] = {
    "init_s": 0.0,
    "base_ms": 4.0,
    "ms_per_audio_s": 2.0,
    "batch_discount": 0.3,
    "max_batch": 8,
    "fail_every": 0,
    "reject_every": 0,
    "hang_every": 0,
    "drop_word_every": 0,
    "streaming": "native",
    "word_timestamps": True,
    "words_per_s": 2.5,
}


@dataclass
class MockPayload:
    seq: int
    duration_s: float
    text: str


def _words_for(item: WorkItem, words_per_s: float) -> str:
    if item.text:
        return item.text
    n = max(1, int(item.duration_s * words_per_s))
    return " ".join(f"w{k}" for k in range(n))


class MockBackend(Backend):
    name = "mock"
    description = "Simulated dynamic-batching server (suite self-tests; timings are not real)"
    OPTIONS = {
        "init_s": "simulated initialisation time, s (0)",
        "base_ms": "fixed cost of one forward, ms (4)",
        "ms_per_audio_s": "cost per padded audio second, ms (2)",
        "batch_discount": "relative cost of each row beyond the first (0.3)",
        "max_batch": "widest server-side batch for concurrent requests (8)",
        "fail_every": "fail every Nth request (0 = never)",
        "reject_every": "reject (back-pressure) every Nth request (0 = never)",
        "hang_every": "never answer every Nth request -- exercises timeouts (0 = never)",
        "drop_word_every": "drop every Nth word of the echoed reference (0 = never)",
        "streaming": "native | buffered | none (native)",
        "word_timestamps": "emit word timings on final stream results (true)",
        "words_per_s": "pseudo-words per audio second when there is no reference (2.5)",
    }

    def __init__(self, options: Optional[Dict[str, Any]] = None, mode: str = "offline") -> None:
        super().__init__(options, mode)
        self.cfg = {**DEFAULTS, **self.options}
        self._seq = 0
        self._queue: "asyncio.Queue[tuple]" = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self._served = 0
        self._batch_widths: List[int] = []

    # -- lifecycle --------------------------------------------------------------

    async def load(self) -> None:
        await asyncio.sleep(float(self.cfg["init_s"]))
        self._worker = asyncio.get_running_loop().create_task(self._run())

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            version="simulated",
            mode=self.mode,
            model={"id": "mock", "simulated": True},
            config=dict(self.cfg),
            device="simulated",
        )

    def capabilities(self) -> Capabilities:
        streaming = str(self.cfg["streaming"])
        return Capabilities(
            batch=True,
            max_batch_size=None,
            concurrent=True,
            streaming=None if streaming == "none" else streaming,
            partials=streaming == "native",
            word_timestamps=bool(self.cfg["word_timestamps"]),
            long_form="whole",
            in_process=True,
            server_metrics=True,
            notes="simulated",
        )

    # -- inputs -----------------------------------------------------------------

    def prepare(self, audio: np.ndarray, item: WorkItem) -> MockPayload:
        self._seq += 1
        return MockPayload(
            seq=self._seq,
            duration_s=len(audio) / self.sample_rate,
            text=_words_for(item, float(self.cfg["words_per_s"])),
        )

    def prepare_chunk(self, chunk: np.ndarray) -> float:
        return len(chunk) / self.sample_rate

    # -- the simulated device -----------------------------------------------------

    def _cost_s(self, durations: Sequence[float]) -> float:
        rows = len(durations)
        padded = max(durations) * (1.0 + float(self.cfg["batch_discount"]) * (rows - 1))
        return (float(self.cfg["base_ms"]) + float(self.cfg["ms_per_audio_s"]) * padded) / 1e3

    async def _run(self) -> None:
        width = int(self.cfg["max_batch"])
        while True:
            first = await self._queue.get()
            batch = [first]
            while len(batch) < width and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            self._batch_widths.append(len(batch))
            await asyncio.sleep(self._cost_s([b[0] for b in batch]))
            t = time.perf_counter()
            for _, fut in batch:
                if not fut.done():
                    fut.set_result(t)

    def _fate(self) -> Optional[str]:
        self._served += 1
        n = self._served
        for key in ("hang_every", "reject_every", "fail_every"):
            every = int(self.cfg[key])
            if every and n % every == 0:
                return key
        return None

    def _text(self, text: str) -> str:
        every = int(self.cfg["drop_word_every"])
        if not every:
            return text
        return " ".join(w for k, w in enumerate(text.split(), 1) if k % every)

    async def _complete(self, p: MockPayload, t_submit: float) -> Transcript:
        fate = self._fate()
        if fate == "hang_every":
            await asyncio.Event().wait()  # never set: the runner's timeout fires
        if fate == "reject_every":
            raise RequestFailed("simulated back-pressure", kind="rejected", status=503)
        fut: "asyncio.Future[float]" = asyncio.get_running_loop().create_future()
        await self._queue.put((p.duration_s, fut))
        t_done = await fut
        if fate == "fail_every":
            raise RequestFailed("simulated failure", status=500)
        return Transcript(text=self._text(p.text), t_submit=t_submit, t_done=t_done)

    async def transcribe(self, payload: Any) -> Transcript:
        return await self._complete(payload, time.perf_counter())

    async def transcribe_batch(self, payloads: Sequence[Any]) -> List[Transcript]:
        t_submit = time.perf_counter()
        await asyncio.sleep(self._cost_s([p.duration_s for p in payloads]))
        t = time.perf_counter()
        self._batch_widths.append(len(payloads))
        return [Transcript(text=self._text(p.text), t_submit=t_submit, t_done=t) for p in payloads]

    async def open_stream(self, *, word_timestamps: bool = False) -> StreamSession:
        mode = str(self.cfg["streaming"])
        if mode == "none":
            raise Unsupported("mock configured with streaming=none")
        return _MockStream(self, mode, word_timestamps)

    # -- observability ------------------------------------------------------------

    def server_metrics_begin(self) -> None:
        self._batch_widths = []

    def server_metrics_end(self) -> Dict[str, Any]:
        from benchmarks.asrbench.metrics import summarize

        return summarize(self._batch_widths, "batch_size_observed") if self._batch_widths else {}


class _MockStream(StreamSession):
    def __init__(self, backend: MockBackend, mode: str, word_timestamps: bool) -> None:
        super().__init__()
        self.b = backend
        self.mode = mode
        self.word_timestamps = word_timestamps
        self.received_s = 0.0
        self.total_text: Optional[str] = None
        self._tasks: List[asyncio.Task] = []

    async def send(self, chunk: Any, is_last: bool) -> None:
        self.received_s += float(chunk)
        delay = self.b._cost_s([float(chunk)])
        loop = asyncio.get_running_loop()
        if self.mode == "native" and not is_last:
            seen = self.received_s
            self._tasks.append(loop.create_task(self._emit("partial", delay, seen)))
        if is_last:
            final_delay = delay + float(self.b.cfg["base_ms"]) / 1e3
            if self.mode == "buffered":
                final_delay = self.b._cost_s([self.received_s])
            self._tasks.append(loop.create_task(self._emit("final", final_delay, self.received_s)))

    def _words(self, upto_s: float) -> List[str]:
        n = max(1, int(self.received_s * float(self.b.cfg["words_per_s"])))
        words = [f"w{k}" for k in range(n)]
        rate = float(self.b.cfg["words_per_s"])
        return words[: int(upto_s * rate)]

    async def _emit(self, kind: str, delay: float, upto_s: float) -> None:
        await asyncio.sleep(delay)
        words = self._words(upto_s)
        timings = None
        if kind == "final" and self.word_timestamps:
            rate = float(self.b.cfg["words_per_s"])
            timings = [
                {"word": w, "start": k / rate, "end": (k + 1) / rate} for k, w in enumerate(words)
            ]
        self.push(StreamEvent(kind, time.perf_counter(), " ".join(words), timings))

    async def abort(self) -> None:
        for t in self._tasks:
            t.cancel()
