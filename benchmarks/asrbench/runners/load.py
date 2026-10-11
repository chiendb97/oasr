# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Concurrent load: closed-loop users and open-loop arrival processes.

**Closed loop** (``mode: closed``): N virtual users, each sending its next
request when the previous one completes (plus optional think time).  It answers
"what throughput does concurrency N sustain" -- but its latency is *service time
under N users*, not the latency an independent client would see, because a
slow response delays the next request instead of piling up behind it.

**Open loop** (``mode: open``): requests are released on a schedule (Poisson or
constant) that does not depend on completions -- MLPerf's Server scenario.
Latency is measured from each request's *scheduled* time, so a server that
stalls is charged for every request that should have been sent during the
stall (no coordinated omission).  ``send_lag_ms`` records how far the client
itself fell behind the schedule; if it is large, the client set the pace and
the run is flagged.

Rates are either absolute (``rates``, req/s) or fractions of a capacity measured
earlier in the same run (``rate_fractions`` + ``capacity_from``), which keeps a
profile portable across hardware: "50 % of what this box sustains" means the
same thing on a laptop and on an H100.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from typing import Any, Dict, List, Optional

from benchmarks.asrbench.analysis import summarize_requests
from benchmarks.asrbench.backends.base import Capabilities
from benchmarks.asrbench.results import BenchmarkResult, RequestRecord
from benchmarks.asrbench.runners.base import Benchmark, RunContext
from benchmarks.asrbench.workloads import ARRIVALS, arrival_offsets

#: Client lag (p99) beyond which a run is flagged as client-bound.
CLIENT_LAG_WARN_MS = 50.0


class Load(Benchmark):
    kind = "load"
    PARAMS = {
        "mode": "closed",
        # closed loop
        "concurrency": [1, 4, 16],
        "think_time_s": 0.0,
        # open loop
        "rates": None,
        "rate_fractions": None,
        "capacity_from": None,
        "arrival": "poisson",
        # bounds (per level, per repetition) -- whichever is hit first
        "num_requests": None,
        "duration_s": None,
        # warm-up: the first requests / seconds of each level are not measured.
        # "all" = one pass over the workload at the level's own concurrency/rate.
        "warmup_requests": "all",
        "warmup_s": 0.0,
        "seed": 0,
    }
    COMMON = {**Benchmark.COMMON, "warmup": 0}

    def __init__(self, name: str, params: Dict[str, Any], workload: Any) -> None:
        super().__init__(name, params, workload)
        if self.params["mode"] not in ("closed", "open"):
            raise ValueError(f"benchmark {name!r}: mode must be closed or open")
        if self.params["arrival"] not in ARRIVALS:
            raise ValueError(f"benchmark {name!r}: arrival must be one of {ARRIVALS}")
        if self.params["num_requests"] is None and self.params["duration_s"] is None:
            self.params["num_requests"] = len(workload.items)
        #: Filled by the runner from an earlier result when ``capacity_from`` is set.
        self.capacity_req_s: Optional[float] = None

    def points(self) -> List[Dict[str, Any]]:
        if self.params["mode"] == "closed":
            return [{"mode": "closed", "concurrency": int(c)} for c in self.params["concurrency"]]
        if self.params.get("rates"):
            return [{"mode": "open", "rate": float(r)} for r in self.params["rates"]]
        fracs = self.params.get("rate_fractions")
        if not fracs or not self.params.get("capacity_from"):
            raise ValueError(
                f"benchmark {self.name!r}: open loop needs `rates`, or `rate_fractions` "
                "with `capacity_from: <earlier benchmark name>`"
            )
        return [{"mode": "open", "rate_fraction": float(f)} for f in fracs]

    def check(self, caps: Capabilities, point: Dict[str, Any]) -> Optional[str]:
        if point.get("concurrency") == 1:
            return None
        if not caps.concurrent:
            return "backend does not accept overlapping requests"
        if "rate_fraction" in point and not self.capacity_req_s:
            return (
                f"capacity_from={self.params['capacity_from']!r} produced no throughput "
                "in this run (missing, unsupported or failed)"
            )
        return None

    def _warmup_n(self) -> int:
        return self.warmup_count(len(self.workload.items), "warmup_requests")

    async def measure(
        self, ctx: RunContext, point: Dict[str, Any], result: BenchmarkResult
    ) -> List[Any]:
        records: List[RequestRecord] = []
        ctx.open_window()
        for rep in range(self.reps):
            if point["mode"] == "closed":
                records += await self._closed(ctx, point, result, rep)
            else:
                records += await self._open(ctx, point, result, rep)
        ctx.close_window()
        lag = [(r.t_submit - r.t_sched) * 1e3 for r in records if r.phase == "measure"]
        if lag:
            lag.sort()
            p99 = lag[int(round(0.99 * (len(lag) - 1)))]
            if p99 > CLIENT_LAG_WARN_MS:
                result.warnings.append(
                    f"client fell behind its schedule (send_lag p99 {p99:.0f} ms): the "
                    "load generator, not the server, limited this point"
                )
        return [r for r in records if r.phase == "measure"]

    async def _closed(
        self, ctx: RunContext, point: Dict[str, Any], result: BenchmarkResult, rep: int
    ) -> List[RequestRecord]:  # noqa: E501
        items = self.workload.items
        total = self.params.get("num_requests")
        duration = self.params.get("duration_s")
        n_warm = self._warmup_n()
        think = float(self.params.get("think_time_s") or 0.0)
        timeout = self.timeout(ctx)
        counter = itertools.count()
        start = time.perf_counter()
        warm_until = start + float(self.params.get("warmup_s") or 0.0)
        out: List[RequestRecord] = []

        async def user(w: int) -> None:
            while True:
                i = next(counter)
                if total is not None and i >= int(total) + n_warm:
                    return
                now = time.perf_counter()
                if duration is not None and now - warm_until >= float(duration):
                    return
                it = items[i % len(items)]
                phase = "warmup" if (i < n_warm or now < warm_until) else "measure"
                out.append(
                    await ctx.issue(
                        self.payloads[it.uid],
                        it,
                        result_id=result.result_id,
                        rep=rep,
                        phase=phase,
                        seq=i,
                        worker=w,
                        timeout_s=timeout,
                    )
                )
                if think > 0:
                    await asyncio.sleep(think)

        await asyncio.gather(*(user(w) for w in range(int(point["concurrency"]))))
        return out

    def rate_of(self, point: Dict[str, Any]) -> float:
        if "rate" in point:
            return float(point["rate"])
        assert self.capacity_req_s, "check() guards this"
        return float(point["rate_fraction"]) * self.capacity_req_s

    async def _open(
        self, ctx: RunContext, point: Dict[str, Any], result: BenchmarkResult, rep: int
    ) -> List[RequestRecord]:  # noqa: E501
        items = self.workload.items
        rate = self.rate_of(point)
        result.server["offered_rate"] = rate
        warm_s = float(self.params.get("warmup_s") or 0.0)
        n_warm = self._warmup_n()
        total = self.params.get("num_requests")
        duration = self.params.get("duration_s")
        offsets = arrival_offsets(
            rate,
            process=self.params["arrival"],
            count=None if total is None else int(total) + n_warm,
            duration_s=None if duration is None else warm_s + float(duration),
            seed=int(self.params.get("seed") or 0) + rep,
        )
        timeout = self.timeout(ctx)
        start = time.perf_counter() + 0.05  # lead time: the first arrival is on schedule
        tasks: List["asyncio.Task[RequestRecord]"] = []
        for i, off in enumerate(offsets):
            due = start + off
            delay = due - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            it = items[i % len(items)]
            phase = "warmup" if (i < n_warm or off < warm_s) else "measure"
            tasks.append(
                asyncio.ensure_future(
                    ctx.issue(
                        self.payloads[it.uid],
                        it,
                        result_id=result.result_id,
                        rep=rep,
                        phase=phase,
                        seq=i,
                        t_sched=due,
                        timeout_s=timeout,
                    )
                )
            )
        return list(await asyncio.gather(*tasks))

    def summarize(self, records: List[Any], result: BenchmarkResult) -> Dict[str, float]:
        out = summarize_requests(records, wall_mode="span", send_lag=True, buckets=False)
        if "offered_rate" in result.server:
            out["offered_rate"] = float(result.server["offered_rate"])
            if out.get("throughput.req_per_s") and out["offered_rate"] > 0:
                out["achieved_ratio"] = out["throughput.req_per_s"] / out["offered_rate"]
        if result.point.get("concurrency") is not None:
            out["concurrency"] = float(result.point["concurrency"])
        return out
