# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Streaming sessions, paced like a live microphone.

Chunk ``k`` of a session covers audio ``[k*c, (k+1)*c)`` and is *due* at
``t_audio0 + (k+1)*c`` -- the moment a microphone would have produced it.  Due
times are absolute, computed from the session start, so pacing never drifts the
way "sleep ``c`` after each send" does; a chunk sent late is recorded as late
(``chunk_lag_ms``, ``late_chunk_pct``) rather than shifting every later chunk.

What is measured, per session (see ``analysis.summarize_sessions``):

* ``first_partial_ms`` -- first transcript text minus the audio start.  It
  includes at least one chunk of algorithmic delay, which is the point: that is
  what a user waits.
* ``final_latency_ms`` -- final transcript minus the end of the audio, i.e. how
  long after the speaker stops the result is settled.
* ``emission_delay_ms`` -- per word, first stable appearance minus the word's
  end time; needs word timestamps (``word_timestamps: true``).

``pacing: unpaced`` sends chunks back to back; the latency figures then lose
their real-time meaning and the point measures streaming *throughput* (rtfx).
Sessions start staggered across one chunk period so N sessions do not hit the
server on the same tick, which no real population of clients would do.

A backend's streaming mode (native / buffered / segmented) is recorded with
every result, and ``compare`` refuses to diff two modes: a buffered backend's
"first partial" arrives after the audio ends and is a different quantity.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from typing import Any, Dict, List, Optional

import numpy as np

from benchmarks.asrbench.analysis import summarize_sessions
from benchmarks.asrbench.audio import split_chunks
from benchmarks.asrbench.backends.base import Backend, Capabilities, RequestFailed
from benchmarks.asrbench.results import (
    FAILED,
    REJECTED,
    TIMEOUT,
    BenchmarkResult,
    SessionRecord,
)
from benchmarks.asrbench.runners.base import Benchmark, RunContext


class Streaming(Benchmark):
    kind = "streaming"
    mode = "streaming"
    PARAMS = {
        "chunk_ms": [320],
        "sessions": [1],
        #: Sequential sessions per concurrent slot.
        "rounds": 1,
        "pacing": "realtime",
        #: "uniform" spreads session starts over one chunk period; "none"; or seconds.
        "stagger": "uniform",
        "late_tolerance_ms": None,
        #: Extra wait for the final result after the last chunk.
        "final_timeout_s": 30.0,
        "word_timestamps": False,
    }
    #: "round" = one concurrent round of ``sessions`` unpaced sessions: an engine
    #: that captures per concurrent-stream batch width needs the warm-up at the
    #: point's own concurrency, not one session at a time.
    COMMON = {**Benchmark.COMMON, "warmup": "round"}

    def __init__(self, name: str, params: Dict[str, Any], workload: Any) -> None:
        super().__init__(name, params, workload)
        if self.params["pacing"] not in ("realtime", "unpaced"):
            raise ValueError(f"benchmark {name!r}: pacing must be realtime or unpaced")
        self._audio: Dict[str, np.ndarray] = {}
        self._chunks: Dict[tuple, List[Any]] = {}

    def points(self) -> List[Dict[str, Any]]:
        return [
            {"chunk_ms": int(c), "sessions": int(s)}
            for c in self.params["chunk_ms"]
            for s in self.params["sessions"]
        ]

    def check(self, caps: Capabilities, point: Dict[str, Any]) -> Optional[str]:
        if not caps.streaming:
            return "backend has no streaming interface"
        if point["sessions"] > 1 and not caps.concurrent:
            return "backend does not accept concurrent sessions"
        if self.params.get("word_timestamps") and not caps.word_timestamps:
            return "word_timestamps requested but the backend cannot produce them"
        return None

    def prepare(self, backend: Backend, audio: Dict[str, np.ndarray]) -> None:
        self._audio = audio
        self._backend = backend

    def _chunks_for(self, uid: str, chunk_ms: int) -> List[Any]:
        key = (uid, chunk_ms)
        if key not in self._chunks:
            n = int(round(self._backend.sample_rate * chunk_ms / 1000.0))
            self._chunks[key] = [
                self._backend.prepare_chunk(c) for c in split_chunks(self._audio[uid], n)
            ]
        return self._chunks[key]

    def _tolerance_ms(self, chunk_ms: int) -> float:
        tol = self.params.get("late_tolerance_ms")
        return float(tol) if tol is not None else max(10.0, 0.1 * chunk_ms)

    async def _session(
        self,
        ctx: RunContext,
        item: Any,
        point: Dict[str, Any],
        result: BenchmarkResult,
        rep: int,
        phase: str,
        seq: int,
        worker: Optional[int],
        paced: bool,
    ) -> SessionRecord:
        chunk_ms = int(point["chunk_ms"])
        chunk_s = chunk_ms / 1000.0
        chunks = self._chunks_for(item.uid, chunk_ms)
        rec = SessionRecord(
            result_id=result.result_id,
            rep=rep,
            phase=phase,
            seq=seq,
            item=item.uid,
            duration_s=item.duration_s,
            chunk_ms=chunk_ms,
            paced=paced,
            worker=worker,
        )
        session = None
        try:
            session = await ctx.backend.open_stream(
                word_timestamps=bool(self.params.get("word_timestamps"))
            )
            t_audio0 = time.perf_counter()
            rec.t_audio0 = ctx.clock.rel(t_audio0)  # type: ignore[assignment]
            for k, chunk in enumerate(chunks):
                due = t_audio0 + min((k + 1) * chunk_s, item.duration_s)
                if paced:
                    delay = due - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                t_send = time.perf_counter()
                rec.chunk_due.append(ctx.clock.rel(due if paced else t_send))  # type: ignore[arg-type]
                rec.chunk_sent.append(ctx.clock.rel(t_send))  # type: ignore[arg-type]
                # Bounded: a server that stops reading would otherwise block the
                # send (transport back-pressure) and hang the benchmark.
                await asyncio.wait_for(
                    session.send(chunk, is_last=k == len(chunks) - 1),
                    float(self.params["final_timeout_s"]),
                )
            final = await session.wait_final(float(self.params["final_timeout_s"]))
            rec.text = final.text if ctx.store_text else None
            rec.words = final.words
        except asyncio.TimeoutError:
            rec.status, rec.error = TIMEOUT, "no final result in time"
            if session is not None:
                await session.abort()
        except RequestFailed as exc:
            rec.status = REJECTED if exc.kind == "rejected" else FAILED
            rec.error = str(exc)
        except Exception as exc:  # noqa: BLE001 -- one session must not end the run
            rec.status, rec.error = FAILED, f"{type(exc).__name__}: {exc}"
            if session is not None:
                await session.abort()
        if session is not None:
            # Event text is kept even without store_text: emission delay and
            # ``recompute`` both need it.
            rec.events = [
                {"kind": e.kind, "t": ctx.clock.rel(e.t), "text": e.text} for e in session.events
            ]
        ctx.served = True
        ctx.keep(rec)
        return rec

    async def measure(
        self, ctx: RunContext, point: Dict[str, Any], result: BenchmarkResult
    ) -> List[Any]:
        items = self.workload.items
        paced = self.params["pacing"] == "realtime"
        # Recorded with the result (not its params) so `compare` can refuse to
        # diff a native stream against a buffered one.
        result.server["streaming_mode"] = ctx.caps.streaming
        n_slots = int(point["sessions"])
        n_warm = (
            n_slots
            if str(self.params.get("warmup")).lower() == "round"
            else self.warmup_count(len(items))
        )
        warm_counter = itertools.count()

        async def warm_slot() -> None:
            while (k := next(warm_counter)) < n_warm:
                await self._session(
                    ctx, items[k % len(items)], point, result, 0, "warmup", k, None, False
                )

        await asyncio.gather(*(warm_slot() for _ in range(min(n_slots, n_warm))))
        await ctx.backend.synchronize()

        rounds = max(1, int(self.params.get("rounds") or 1))
        chunk_s = point["chunk_ms"] / 1000.0
        stagger = self.params.get("stagger", "uniform")
        records: List[SessionRecord] = []
        ctx.open_window()
        for rep in range(self.reps):
            counter = itertools.count()

            async def slot(w: int, rep: int = rep, counter: Any = counter) -> None:
                # Staggering models independent live clients; an unpaced run
                # measures throughput and starts everything at once.
                if paced and stagger == "uniform":
                    await asyncio.sleep(w * chunk_s / n_slots)
                elif paced and stagger not in (None, "none"):
                    await asyncio.sleep(w * float(stagger))
                for _ in range(rounds):
                    i = next(counter)
                    records.append(
                        await self._session(
                            ctx, items[i % len(items)], point, result, rep, "measure", i, w, paced
                        )
                    )

            await asyncio.gather(*(slot(w) for w in range(n_slots)))
        ctx.close_window()
        return records

    def summarize(self, records: List[Any], result: BenchmarkResult) -> Dict[str, float]:
        out = summarize_sessions(
            records, late_tolerance_ms=self._tolerance_ms(int(result.point["chunk_ms"]))
        )
        out["concurrent_sessions"] = float(result.point["sessions"])
        return out
