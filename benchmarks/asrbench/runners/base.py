# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""What every benchmark shares: the run context, request issue, and the point loop.

A benchmark is a parameter sweep.  Each *point* (one batch size, one
concurrency level, one chunk size...) becomes one
:class:`~benchmarks.asrbench.results.BenchmarkResult` with its own measured
window, so resources, server metrics and summaries are attributed to exactly
the requests that produced them.  Warm-up happens inside the point, before the
window opens, and its records are stored with ``phase="warmup"``.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import numpy as np

from benchmarks.asrbench.backends.base import (
    Backend,
    Capabilities,
    RequestFailed,
    Transcript,
    Unsupported,
)
from benchmarks.asrbench.results import (
    FAILED,
    REJECTED,
    REQUESTS,
    SESSIONS,
    TIMEOUT,
    BenchmarkResult,
    Clock,
    RequestRecord,
    SessionRecord,
    append_jsonl,
)
from benchmarks.asrbench.workloads import WorkItem, Workload, bucket_of


@dataclass
class RunContext:
    backend: Backend
    caps: Capabilities
    clock: Clock
    run_dir: Path
    timeout_s: float = 120.0
    store_text: bool = True
    #: Set when this backend instance has already served a request -- a
    #: cold-start measurement after that would not be cold.
    served: bool = False
    init_s: Optional[float] = None
    #: The measured window of the current point (run clock).
    window: List[float] = field(default_factory=list)
    _requests: List[RequestRecord] = field(default_factory=list)
    _sessions: List[SessionRecord] = field(default_factory=list)

    # -- window ---------------------------------------------------------------

    def open_window(self) -> None:
        """Start the measured phase: reset allocator peaks and server counters."""
        self.backend.reset_peak_memory()
        self.backend.server_metrics_begin()
        self.window = [self.clock.now()]

    def close_window(self) -> None:
        if len(self.window) == 1:
            self.window.append(self.clock.now())

    # -- records --------------------------------------------------------------

    def keep(self, rec: Any) -> None:
        if isinstance(rec, SessionRecord):
            self._sessions.append(rec)
        else:
            self._requests.append(rec)

    def flush(self) -> None:
        if self._requests:
            append_jsonl(self.run_dir / REQUESTS, (r.to_json() for r in self._requests))
            self._requests = []
        if self._sessions:
            append_jsonl(self.run_dir / SESSIONS, (s.to_json() for s in self._sessions))
            self._sessions = []

    # -- one request ------------------------------------------------------------

    async def issue(
        self,
        payload: Any,
        item: WorkItem,
        *,
        result_id: str,
        rep: int,
        phase: str,
        seq: int,
        t_sched: Optional[float] = None,
        worker: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> RequestRecord:
        """Run one request; never raises for a request-level outcome.

        ``t_sched`` is the perf_counter time the request was *due*; latency is
        measured from it, which is what keeps an open-loop test free of
        coordinated omission.  It defaults to the submission time.
        """
        t_call = time.perf_counter()
        rec = RequestRecord(
            result_id=result_id,
            rep=rep,
            phase=phase,
            seq=seq,
            item=item.uid,
            duration_s=item.duration_s,
            bucket=item.bucket or bucket_of(item.duration_s),
            worker=worker,
            t_sched=self.clock.rel(t_sched if t_sched is not None else t_call),
            t_submit=self.clock.rel(t_call),
        )
        timeout = self.timeout_s if timeout_s is None else timeout_s
        try:
            tr: Transcript = await asyncio.wait_for(self.backend.transcribe(payload), timeout)
            self._fill(rec, tr)
        except asyncio.TimeoutError:
            rec.status, rec.error = TIMEOUT, f"no result within {timeout:g}s"
            await self.backend.cancel(payload)
        except RequestFailed as exc:
            rec.status = REJECTED if exc.kind == "rejected" else FAILED
            rec.error = str(exc)
            rec.t_done = self.clock.now()
        except Unsupported:
            raise
        except Exception as exc:  # noqa: BLE001 -- one request must not end the run
            rec.status, rec.error = FAILED, f"{type(exc).__name__}: {exc}"
            rec.t_done = self.clock.now()
        self.served = True
        self.keep(rec)
        return rec

    def _fill(self, rec: RequestRecord, tr: Transcript) -> None:
        rec.t_submit = self.clock.rel(tr.t_submit) if tr.t_submit else rec.t_submit
        if rec.t_sched > rec.t_submit:
            rec.t_sched = rec.t_submit  # a backend that queued before "submit"
        rec.t_first = self.clock.rel(tr.t_first)
        rec.t_done = self.clock.rel(tr.t_done)
        rec.text = tr.text if self.store_text else None


def warmup_items(items: Sequence[WorkItem], n: int) -> List[WorkItem]:
    """*n* items covering as many duration buckets as possible (shapes to compile)."""
    if n <= 0 or not items:
        return []
    by_bucket: Dict[str, List[WorkItem]] = {}
    for it in items:
        by_bucket.setdefault(it.bucket or bucket_of(it.duration_s), []).append(it)
    out: List[WorkItem] = []
    for group in itertools.cycle(list(by_bucket.values())):
        if len(out) >= n:
            break
        out.append(group[len(out) // len(by_bucket) % len(group)])
    return out


def point_suffix(point: Dict[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in point.items())


class Benchmark(ABC):
    """One benchmark entry of a profile: a kind, its parameters, its workload."""

    kind: ClassVar[str] = ""
    mode: ClassVar[str] = "offline"
    #: Defaults for every parameter this kind accepts; anything else is an error.
    PARAMS: ClassVar[Dict[str, Any]] = {}
    #: Accepted by every kind.
    COMMON: ClassVar[Dict[str, Any]] = {
        "workload": None,
        "repetitions": 1,
        "warmup": 1,
        "timeout_s": None,
    }

    def __init__(self, name: str, params: Dict[str, Any], workload: Workload) -> None:
        allowed = {**self.COMMON, **self.PARAMS}
        unknown = set(params) - set(allowed) - {"kind", "name"}
        if unknown:
            raise ValueError(
                f"benchmark {name!r} ({self.kind}): unknown parameter(s) {sorted(unknown)}; "
                f"accepted: {sorted(allowed)}"
            )
        self.name = name
        self.params: Dict[str, Any] = {
            **allowed,
            **{k: v for k, v in params.items() if k not in ("kind", "name")},
        }
        self.workload = workload
        self.payloads: Dict[str, Any] = {}

    def points(self) -> List[Dict[str, Any]]:
        return [{}]

    def check(self, caps: Capabilities, point: Dict[str, Any]) -> Optional[str]:
        """Why this point cannot run on a backend with *caps*; ``None`` if it can."""
        return None

    def prepare(self, backend: Backend, audio: Dict[str, np.ndarray]) -> None:
        """Build every request payload before any clock starts."""
        for it in self.workload.items:
            if it.uid not in self.payloads:
                self.payloads[it.uid] = backend.prepare(audio[it.uid], it)

    def result_id(self, point: Dict[str, Any]) -> str:
        return f"{self.name}[{point_suffix(point)}]" if point else self.name

    def new_result(self, point: Dict[str, Any]) -> BenchmarkResult:
        return BenchmarkResult(
            result_id=self.result_id(point),
            benchmark=self.name,
            kind=self.kind,
            mode=self.mode,
            point=dict(point),
            workload=self.workload.spec.name,
            workload_id=self.workload.id,
            params=dict(self.params),
        )

    @abstractmethod
    async def measure(
        self, ctx: RunContext, point: Dict[str, Any], result: BenchmarkResult
    ) -> List[Any]:
        """Run warm-up and the measured phase; return the measured-phase records.

        Must call ``ctx.open_window()`` when the measured phase starts and
        ``ctx.close_window()`` when it ends.
        """

    @abstractmethod
    def summarize(self, records: List[Any], result: BenchmarkResult) -> Dict[str, float]:
        """Records -> summary.  Also used by ``recompute`` on stored records."""

    @property
    def reps(self) -> int:
        return max(1, int(self.params.get("repetitions") or 1))

    def warmup_count(self, total: int, key: str = "warmup") -> int:
        """Warm-up size: an int, or ``"all"`` -- one full pass over the *total* units.

        ``all`` is the default for the kinds that replay a fixed set of inputs,
        because an engine that compiles or graph-captures per input shape pays
        that cost on the first occurrence of each shape (measured: ~27 ms
        against ~1.8 ms warm for one OASR Conformer shape).  A warm-up that
        does not cover the shapes the measured pass will see puts compilation
        inside the measurement.
        """
        value = self.params.get(key)
        if isinstance(value, str):
            if value.lower() in ("all", "pass"):
                return total
            value = int(value)
        return max(0, int(value or 0))

    def timeout(self, ctx: RunContext) -> float:
        t = self.params.get("timeout_s")
        return float(t) if t is not None else ctx.timeout_s
