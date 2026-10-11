# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Offline benchmarks: cold start, single-request latency, batch sweeps, backlog
throughput, and long-form.

The distinctions that matter, and that each kind keeps apart:

``single``      one request in flight at a time -- per-request latency and its
                dependence on duration (``bucket.*`` breakdown).  Batch 1,
                concurrency 1.  ``rtfx`` = audio / sum of latencies.
``batch``       one native batch call of exactly B inputs -- batch latency,
                per-request completion inside it, padding.  Needs a backend with
                a real batch API; concurrency is *not* a substitute.
``throughput``  everything submitted at once; the backend schedules -- the
                offline ceiling (the existing engine harness's ``offline``).
``longform``    whole recordings, one at a time; resources over time.
``cold_start``  initialisation, then the first request, then the same request warm.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any, Dict, List, Optional

from benchmarks.asrbench.analysis import summarize_requests
from benchmarks.asrbench.backends.base import Capabilities, RequestFailed
from benchmarks.asrbench.results import (
    FAILED,
    OK,
    REJECTED,
    BenchmarkResult,
    RequestRecord,
)
from benchmarks.asrbench.runners.base import Benchmark, RunContext, warmup_items
from benchmarks.asrbench.workloads import COMPOSITIONS, make_batches, padding_ratio


class ColdStart(Benchmark):
    kind = "cold_start"
    PARAMS = {"warm_repeats": 3}
    COMMON = {**Benchmark.COMMON, "warmup": 0}

    async def measure(
        self, ctx: RunContext, point: Dict[str, Any], result: BenchmarkResult
    ) -> List[Any]:
        if ctx.served:
            result.warnings.append(
                "the backend had already served requests -- first_request_ms is not cold; "
                "put cold_start first in the profile"
            )
        item = self.workload.items[0]
        payload = self.payloads[item.uid]
        ctx.open_window()
        records = []
        for k in range(1 + int(self.params["warm_repeats"])):
            rec = await ctx.issue(
                payload,
                item,
                result_id=result.result_id,
                rep=0,
                phase="measure",
                seq=k,
                timeout_s=self.timeout(ctx),
            )
            records.append(rec)
        ctx.close_window()
        result.server["init_s"] = ctx.init_s
        return records

    def summarize(self, records: List[Any], result: BenchmarkResult) -> Dict[str, float]:
        out: Dict[str, float] = {}
        init_s = result.server.get("init_s")
        if init_s is not None:
            out["init_s"] = float(init_s)
        recs = sorted((r for r in records if r.phase == "measure"), key=lambda r: r.seq)
        if recs and recs[0].status == OK:
            out["first_request_ms"] = recs[0].latency_ms
        warm = [r.latency_ms for r in recs[1:] if r.status == OK]
        if warm:
            out["warm_request_ms"] = statistics.median(warm)
        out["audio_s"] = recs[0].duration_s if recs else 0.0
        out["requests.total"] = float(len(recs))
        out["requests.ok"] = float(sum(r.status == OK for r in recs))
        return out


class Single(Benchmark):
    kind = "single"
    PARAMS: Dict[str, Any] = {}
    COMMON = {**Benchmark.COMMON, "warmup": "all"}

    async def measure(
        self, ctx: RunContext, point: Dict[str, Any], result: BenchmarkResult
    ) -> List[Any]:
        items = self.workload.items
        timeout = self.timeout(ctx)
        n_warm = self.warmup_count(len(items))
        warm = list(items) if n_warm >= len(items) else warmup_items(items, n_warm)
        for k, it in enumerate(warm):
            await ctx.issue(
                self.payloads[it.uid],
                it,
                result_id=result.result_id,
                rep=0,
                phase="warmup",
                seq=k,
                timeout_s=timeout,
            )
        await ctx.backend.synchronize()
        ctx.open_window()
        records = []
        seq = 0
        for rep in range(self.reps):
            for it in items:
                records.append(
                    await ctx.issue(
                        self.payloads[it.uid],
                        it,
                        result_id=result.result_id,
                        rep=rep,
                        phase="measure",
                        seq=seq,
                        timeout_s=timeout,
                    )
                )
                seq += 1
        ctx.close_window()
        return records

    def summarize(self, records: List[Any], result: BenchmarkResult) -> Dict[str, float]:
        return summarize_requests(records, wall_mode="sum_service")


class LongForm(Single):
    """``single`` over long recordings: same protocol, different question.

    Kept as its own kind so its rows are never pooled with utterance latency,
    and so a backend that bounds input length is reported ``unsupported`` here
    rather than failing every request.  Warm-up uses the *shortest* item -- a
    warm-up pass over hour-long audio would double the cost of the benchmark.
    """

    kind = "longform"
    PARAMS = {"allow_bounded": False}
    COMMON = {**Benchmark.COMMON, "warmup": 1}

    def check(self, caps: Capabilities, point: Dict[str, Any]) -> Optional[str]:
        if caps.long_form is None and not self.params.get("allow_bounded"):
            return (
                "backend declares no long-form support (bounded input length); "
                "set allow_bounded: true to try anyway"
            )
        return None

    async def measure(
        self, ctx: RunContext, point: Dict[str, Any], result: BenchmarkResult
    ) -> List[Any]:
        items = self.workload.items
        timeout = self.timeout(ctx)
        shortest = sorted(items, key=lambda i: i.duration_s)[: self.warmup_count(len(items))]
        for k, it in enumerate(shortest):
            await ctx.issue(
                self.payloads[it.uid],
                it,
                result_id=result.result_id,
                rep=0,
                phase="warmup",
                seq=k,
                timeout_s=timeout,
            )
        await ctx.backend.synchronize()
        ctx.open_window()
        records, seq = [], 0
        for rep in range(self.reps):
            for it in items:
                records.append(
                    await ctx.issue(
                        self.payloads[it.uid],
                        it,
                        result_id=result.result_id,
                        rep=rep,
                        phase="measure",
                        seq=seq,
                        timeout_s=timeout,
                    )
                )
                seq += 1
        ctx.close_window()
        return records


class Batch(Benchmark):
    kind = "batch"
    PARAMS = {
        "batch_sizes": [1, 2, 4, 8],
        "compositions": ["uniform", "mixed"],
        #: Batches per repetition; default: enough to cover the workload once.
        "num_batches": None,
        "seed": 0,
    }
    COMMON = {**Benchmark.COMMON, "warmup": "all"}

    def points(self) -> List[Dict[str, Any]]:
        comps = self.params["compositions"]
        if isinstance(comps, str):
            comps = [comps]
        bad = [c for c in comps if c not in COMPOSITIONS]
        if bad:
            raise ValueError(f"benchmark {self.name!r}: composition(s) {bad} not in {COMPOSITIONS}")
        return [
            {"batch_size": int(b), "composition": c}
            for b in self.params["batch_sizes"]
            for c in comps
        ]

    def check(self, caps: Capabilities, point: Dict[str, Any]) -> Optional[str]:
        if not caps.batch:
            return "backend has no native batch API (concurrency is measured by `load`, not here)"
        if caps.max_batch_size is not None and point["batch_size"] > caps.max_batch_size:
            return (
                f"batch_size {point['batch_size']} exceeds the backend's max {caps.max_batch_size}"
            )
        return None

    def _batches(self, point: Dict[str, Any]) -> List[List[Any]]:
        return make_batches(
            self.workload.items,
            point["batch_size"],
            point["composition"],
            num_batches=self.params.get("num_batches"),
            seed=int(self.params.get("seed") or 0),
        )

    async def _run_batch(
        self,
        ctx: RunContext,
        batch: List[Any],
        result: BenchmarkResult,
        rep: int,
        phase: str,
        index: int,
        seq0: int,
        timeout: float,
    ) -> List[RequestRecord]:
        payloads = [self.payloads[it.uid] for it in batch]
        t_call = time.perf_counter()
        recs = [
            RequestRecord(
                result_id=result.result_id,
                rep=rep,
                phase=phase,
                seq=seq0 + k,
                item=it.uid,
                duration_s=it.duration_s,
                bucket=it.bucket,
                batch=index,
                t_sched=ctx.clock.rel(t_call),
                t_submit=ctx.clock.rel(t_call),
            )
            for k, it in enumerate(batch)
        ]
        try:
            outs = await asyncio.wait_for(ctx.backend.transcribe_batch(payloads), timeout)
            if len(outs) != len(batch):
                raise RequestFailed(f"backend returned {len(outs)} results for {len(batch)} inputs")
            for rec, tr in zip(recs, outs):
                ctx._fill(rec, tr)
                # Every row of a batch was submitted together: latency is from the call.
                rec.t_sched = rec.t_submit = ctx.clock.rel(t_call)
        except asyncio.TimeoutError:
            for rec in recs:
                rec.status, rec.error = "timeout", f"batch exceeded {timeout:g}s"
        except RequestFailed as exc:
            for rec in recs:
                rec.status = REJECTED if exc.kind == "rejected" else FAILED
                rec.error, rec.t_done = str(exc), ctx.clock.now()
        except Exception as exc:  # noqa: BLE001 -- e.g. an out-of-memory at this width
            for rec in recs:
                rec.status, rec.error = FAILED, f"{type(exc).__name__}: {exc}"
                rec.t_done = ctx.clock.now()
        ctx.served = True
        for rec in recs:
            ctx.keep(rec)
        return recs

    async def measure(
        self, ctx: RunContext, point: Dict[str, Any], result: BenchmarkResult
    ) -> List[Any]:
        batches = self._batches(point)
        timeout = self.timeout(ctx)
        for k, b in enumerate(batches[: self.warmup_count(len(batches))]):
            await self._run_batch(ctx, b, result, 0, "warmup", k, k * len(b), timeout)
        await ctx.backend.synchronize()
        ctx.open_window()
        records: List[RequestRecord] = []
        index = 0
        for rep in range(self.reps):
            for b in batches:
                records += await self._run_batch(
                    ctx, b, result, rep, "measure", index, len(records), timeout
                )
                index += 1
        ctx.close_window()
        result.server["padding_ratio"] = padding_ratio(batches)
        return records

    def summarize(self, records: List[Any], result: BenchmarkResult) -> Dict[str, float]:
        out = summarize_requests(records, wall_mode="batches", buckets=False)
        if "padding_ratio" in result.server:
            out["padding_ratio"] = float(result.server["padding_ratio"])
        return out


class Throughput(Benchmark):
    kind = "throughput"
    PARAMS: Dict[str, Any] = {}
    COMMON = {**Benchmark.COMMON, "warmup": "all"}

    def check(self, caps: Capabilities, point: Dict[str, Any]) -> Optional[str]:
        if not caps.concurrent:
            return (
                "backend does not schedule concurrent requests; its offline ceiling is "
                "the `batch` benchmark"
            )
        return None

    async def _burst(
        self,
        ctx: RunContext,
        result: BenchmarkResult,
        rep: int,
        phase: str,
        items: List[Any],
        timeout: float,
    ) -> List[RequestRecord]:  # noqa: E501
        t0 = time.perf_counter()
        return list(
            await asyncio.gather(
                *(
                    ctx.issue(
                        self.payloads[it.uid],
                        it,
                        result_id=result.result_id,
                        rep=rep,
                        phase=phase,
                        seq=k,
                        t_sched=t0,
                        timeout_s=timeout,
                    )
                    for k, it in enumerate(items)
                )
            )
        )

    async def measure(
        self, ctx: RunContext, point: Dict[str, Any], result: BenchmarkResult
    ) -> List[Any]:
        items = self.workload.items
        timeout = self.timeout(ctx)
        n_warm = self.warmup_count(len(items))
        if n_warm:
            warm = list(items) if n_warm >= len(items) else warmup_items(items, n_warm)
            await self._burst(ctx, result, 0, "warmup", warm, timeout)
        await ctx.backend.synchronize()
        ctx.open_window()
        records: List[RequestRecord] = []
        for rep in range(self.reps):
            records += await self._burst(ctx, result, rep, "measure", items, timeout)
        ctx.close_window()
        return records

    def summarize(self, records: List[Any], result: BenchmarkResult) -> Dict[str, float]:
        return summarize_requests(records, wall_mode="span", buckets=False)
