# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Run a resolved profile against one backend and write a run directory.

Order of operations, and why:

1. Resolve every workload and write it into the run *before* loading a model, so
   a selection problem fails in seconds and the inputs are on record even if
   the backend crashes later.
2. Group benchmarks by service mode (``offline`` / ``streaming``).  Engines and
   servers are single-mode, so each group gets its own backend instance, loaded
   once and timed (``init_s``).  ``cold_start`` runs first in its group --
   anything earlier would warm it.
3. Decode each workload's audio and build every payload before any clock starts.
4. For each sweep point: capability check (``unsupported`` is a result, not a
   skip), warm-up, measured window, then summary + resources + server metrics
   for exactly that window.  ``run.json`` is rewritten after every point, so a
   crash keeps everything finished so far.
"""

from __future__ import annotations

import asyncio
import gc
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.asrbench import backends, environment
from benchmarks.asrbench.metrics import direction, low_sample_percentiles, unit
from benchmarks.asrbench.monitor import ResourceMonitor, summarize_window
from benchmarks.asrbench.profiles import Profile
from benchmarks.asrbench.results import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_UNSUPPORTED,
    BenchmarkResult,
    Clock,
    base_meta,
    default_results_root,
    new_run_dir,
    write_run_json,
)
from benchmarks.asrbench.runners import Benchmark, RunContext, get_kind
from benchmarks.asrbench.runners.load import Load
from benchmarks.asrbench.workloads import Workload, load_audio, resolve

MODES = ("offline", "streaming")


def _log(msg: str) -> None:
    print(f"[asrbench] {msg}", flush=True)


def build_benchmarks(profile: Profile, workloads: Dict[str, Workload]) -> List[Benchmark]:
    out = []
    for entry in profile.benchmarks:
        cls = get_kind(entry["kind"])
        params = dict(entry)
        out.append(cls(entry["name"], params, workloads[entry["workload"]]))
    return out


def resolve_workloads(profile: Profile, data_root: Optional[Path]) -> Dict[str, Workload]:
    used = {b["workload"] for b in profile.benchmarks}
    return {name: resolve(profile.workload_spec(name), data_root) for name in sorted(used)}


def _headline(r: BenchmarkResult) -> str:
    s = r.summary
    if r.status != STATUS_OK:
        return f"{r.status.upper()}: {r.reason}"
    parts = []
    for key, label, fmt in (
        ("first_request_ms", "first", "{:.1f} ms"),
        ("latency_ms.p50", "p50", "{:.1f} ms"),
        ("latency_ms.p95", "p95", "{:.1f} ms"),
        ("final_latency_ms.p50", "final p50", "{:.0f} ms"),
        ("first_partial_ms.p50", "first-partial p50", "{:.0f} ms"),
        ("rtfx", "RTFx", "{:.1f}"),
        ("throughput.req_per_s", "req/s", "{:.2f}"),
        ("error_rate", "errors", "{:.1%}"),
    ):
        if key in s:
            parts.append(f"{label} " + fmt.format(s[key]))
    return "  ".join(parts) or "ok"


async def _run_point(
    bench: Benchmark,
    point: Dict[str, Any],
    ctx: RunContext,
    monitor: Optional[ResourceMonitor],
) -> BenchmarkResult:
    result = bench.new_result(point)
    reason = bench.check(ctx.caps, point)
    if reason:
        result.status, result.reason = STATUS_UNSUPPORTED, reason
        return result
    ctx.window = []
    try:
        records = await bench.measure(ctx, point, result)
        await ctx.backend.synchronize()
        result.summary = bench.summarize(records, result)
    except backends.Unsupported as exc:
        result.status, result.reason = STATUS_UNSUPPORTED, str(exc)
    except Exception as exc:  # noqa: BLE001 -- record and carry on with the next point
        result.status = STATUS_FAILED
        result.reason = f"{type(exc).__name__}: {exc}"
        result.server["traceback"] = traceback.format_exc(limit=8)
    finally:
        if ctx.window:
            ctx.close_window()
        ctx.flush()
    result.window = list(ctx.window)
    if result.status != STATUS_OK:
        return result
    result.summary.update(ctx.backend.peak_memory())
    server = ctx.backend.server_metrics_end()
    result.summary.update({k: v for k, v in server.items() if isinstance(v, (int, float))})
    result.server.update({k: v for k, v in server.items() if not isinstance(v, (int, float))})
    if monitor is not None and len(result.window) == 2:
        origin = ctx.clock.origin_mono
        result.summary.update(
            summarize_window(
                monitor.samples(),
                origin + result.window[0],
                origin + result.window[1],
                audio_s=result.summary.get("audio_s"),
            )
        )
    tails = [
        k
        for k in low_sample_percentiles(result.summary)
        if k.endswith((".p95", ".p99")) and unit(k) == "ms" and direction(k) == "lower"
    ]
    if tails:
        result.warnings.append(
            f"{len(tails)} tail percentile(s) rest on too few samples to differ from the max "
            f"(e.g. {tails[0]}); add requests or repetitions before reading them"
        )
    return result


async def run_profile(
    profile: Profile,
    backend_name: str,
    backend_options: Optional[Dict[str, Any]] = None,
    *,
    data_root: Optional[Path] = None,
    results_root: Optional[Path] = None,
    tag: str = "",
    monitor_interval_s: Optional[float] = None,
    store_text: bool = True,
) -> Path:
    clock = Clock()
    run_id = uuid.uuid4().hex[:8]
    backend_options = dict(backend_options or {})
    sample_rate = int(profile.defaults.get("sample_rate", 16000))
    backend_options.setdefault("sample_rate", sample_rate)

    workloads = resolve_workloads(profile, data_root)
    benches = build_benchmarks(profile, workloads)

    run_dir = new_run_dir(
        results_root or default_results_root(), profile.name, backend_name, run_id
    )
    for name, w in workloads.items():
        w.save(run_dir / "workloads" / f"{name}.json")
        for msg in w.warnings:
            _log(f"workload {name}: {msg}")
    meta = base_meta(run_id, clock)
    simulated = backend_name == "mock"
    meta.update(
        {
            "tag": tag,
            "profile": profile.to_json(),
            "environment": environment.collect(),
            "backend": {"name": backend_name, "options": backend_options, "instances": {}},
            "workloads": {
                name: {
                    "id": w.id,
                    "dataset": f"{w.dataset.get('name')}/{w.dataset.get('subset')}",
                    "dataset_revision": w.dataset.get("revision"),
                    "manifest_sha256": w.dataset.get("manifest_sha256"),
                    "num_items": len(w.items),
                    "total_audio_s": round(w.total_audio_s, 3),
                    "synthetic": w.synthetic,
                    "warnings": w.warnings,
                }
                for name, w in workloads.items()
            },
            "simulated": {
                "backend": simulated,
                "inputs": any(w.synthetic for w in workloads.values()),
            },
        }
    )
    interval = float(
        monitor_interval_s
        if monitor_interval_s is not None
        else profile.defaults.get("monitor_interval_s", 0.2)
    )
    meta["monitor"] = {"enabled": interval > 0, "interval_s": interval}
    results: List[BenchmarkResult] = []
    write_run_json(run_dir, meta, results)
    _log(f"run {run_id} -> {run_dir}")

    capacity: Dict[str, float] = {}
    audio_cache: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        group = [b for b in benches if b.mode == mode]
        if not group:
            continue
        group.sort(key=lambda b: b.kind != "cold_start")  # stable: profile order otherwise
        cls = backends.backend_class(backend_name)
        static = cls.mode_unsupported(mode, backend_options)
        if static:
            for b in group:
                for point in b.points():
                    r = b.new_result(point)
                    r.status, r.reason = STATUS_UNSUPPORTED, static
                    results.append(r)
                    _log(f"{r.result_id:<48} UNSUPPORTED: {static}")
            meta["backend"]["instances"][mode] = {"unsupported": static}
            write_run_json(run_dir, meta, results)
            continue
        backend = backends.create(backend_name, backend_options, mode)
        _log(f"loading {backend_name} ({mode}) ...")
        t0 = time.perf_counter()
        load_error: Optional[str] = None
        try:
            await backend.load()
        except Exception as exc:  # noqa: BLE001 -- a backend that cannot load fails its group
            load_error = f"{type(exc).__name__}: {exc}"
        init_s = time.perf_counter() - t0
        if load_error is not None:
            _log(f"backend failed to load: {load_error}")
            for b in group:
                for point in b.points():
                    r = b.new_result(point)
                    r.status, r.reason = STATUS_FAILED, f"backend load failed: {load_error}"
                    results.append(r)
            meta["backend"]["instances"][mode] = {"error": load_error, "init_s": init_s}
            write_run_json(run_dir, meta, results)
            await backend.close()
            continue
        caps = backend.capabilities()
        meta["backend"]["instances"][mode] = {
            "info": backend.info().to_json(),
            "capabilities": caps.to_json(),
            "init_s": init_s,
        }
        _log(f"loaded in {init_s:.2f}s; capabilities: {caps.to_json()}")
        monitor: Optional[ResourceMonitor] = None
        pid = backend.monitored_pid()
        if interval > 0 and pid:
            monitor = ResourceMonitor(run_dir / "resources.jsonl", interval_s=interval)
            monitor.start(pid)
        ctx = RunContext(
            backend=backend,
            caps=caps,
            clock=clock,
            run_dir=run_dir,
            timeout_s=float(profile.defaults.get("timeout_s", 120.0)),
            store_text=store_text,
            init_s=init_s,
        )
        try:
            for bench in group:
                wname = bench.workload.spec.name
                if wname not in audio_cache:
                    _log(f"decoding workload {wname} ({len(bench.workload.items)} items) ...")
                    audio_cache[wname] = load_audio(bench.workload.items, sample_rate)
                bench.prepare(backend, audio_cache[wname])
                if isinstance(bench, Load) and bench.params.get("capacity_from"):
                    bench.capacity_req_s = capacity.get(str(bench.params["capacity_from"]))
                for point in bench.points():
                    result = await _run_point(bench, point, ctx, monitor)
                    results.append(result)
                    rate = result.summary.get("throughput.req_per_s")
                    if result.status == STATUS_OK and rate:
                        capacity[bench.name] = max(capacity.get(bench.name, 0.0), float(rate))
                    _log(f"{result.result_id:<48} {_headline(result)}")
                    for w in result.warnings:
                        _log(f"  warning: {w}")
                    write_run_json(run_dir, meta, results)
        finally:
            if monitor is not None:
                monitor.stop()
                meta["monitor"].setdefault("files", []).append(monitor.out_path.name)
                meta["monitor"]["probes"] = monitor.header()
            await backend.close()
            del backend
            gc.collect()

    meta["finished_utc"] = _utc_now()
    meta["wall_s"] = clock.now()
    write_run_json(run_dir, meta, results)
    _log(f"done: {len(results)} result(s) in {clock.now():.1f}s -> {run_dir}")
    return run_dir


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(
    profile: Profile, backend_name: str, backend_options: Optional[Dict[str, Any]] = None, **kw: Any
) -> Path:
    return asyncio.run(run_profile(profile, backend_name, backend_options, **kw))


# ---------------------------------------------------------------------------
# Recompute
# ---------------------------------------------------------------------------


def recompute(run_path: Path, *, tolerance: float = 1e-9) -> Dict[str, Any]:
    """Re-derive every summary from the stored records; report disagreements.

    Only keys computed from records are compared -- resources, allocator peaks
    and server metrics come from other sources and are left alone.
    """
    from benchmarks.asrbench.results import load_run

    run = load_run(run_path)
    requests = run.requests()
    sessions = run.sessions()
    workloads: Dict[str, Workload] = {}
    report: Dict[str, Any] = {"run": str(run.path), "checked": 0, "mismatches": []}
    for r in run.results:
        if r.status != STATUS_OK:
            continue
        if r.workload not in workloads:
            workloads[r.workload] = Workload.from_json(run.workload(r.workload))
        bench = get_kind(r.kind)(r.benchmark, dict(r.params), workloads[r.workload])
        records = [
            x
            for x in (sessions if r.kind == "streaming" else requests)
            if x.result_id == r.result_id
        ]
        fresh = bench.summarize(records, r)
        report["checked"] += 1
        for key, value in fresh.items():
            stored = r.summary.get(key)
            if stored is None or value is None:
                if stored != value:
                    report["mismatches"].append(
                        {"result": r.result_id, "key": key, "stored": stored, "recomputed": value}
                    )
                continue
            if abs(float(stored) - float(value)) > tolerance * max(1.0, abs(float(value))):
                report["mismatches"].append(
                    {"result": r.result_id, "key": key, "stored": stored, "recomputed": value}
                )
    return report
