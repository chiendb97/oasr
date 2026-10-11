# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark runners end to end on the mock backend: protocol, not numbers.

The mock's timings are simulated, so nothing here asserts a speed.  What is
asserted is the measurement protocol: warm-up is excluded, concurrency is
bounded, open-loop latency is charged from the schedule, failures are counted,
paced chunks follow the clock, unsupported points are reported, and every
summary can be recomputed from the stored records.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from asrbench_fixtures import run_tiny, tiny_profile

from benchmarks.asrbench.backends.mock import MockBackend
from benchmarks.asrbench.results import Clock, load_run
from benchmarks.asrbench.runner import _run_point, recompute
from benchmarks.asrbench.runners import RunContext, get_kind
from benchmarks.asrbench.workloads import WorkloadSpec, load_audio, resolve


def measure(
    data_root: Path,
    tmp_path: Path,
    kind: str,
    params: Dict[str, Any],
    workload: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
    mode: str = "offline",
    timeout_s: float = 5.0,
) -> List[Any]:
    """Run every point of one benchmark; return ``[(result, records)]``."""

    async def go():
        backend = MockBackend(options or {}, mode)
        await backend.load()
        try:
            w = resolve(
                WorkloadSpec.from_dict("w", {"dataset": "synthetic/default", **workload}), data_root
            )
            bench = get_kind(kind)("b", {"workload": "w", **params}, w)
            bench.prepare(backend, load_audio(w.items, 16000))
            ctx = RunContext(
                backend, backend.capabilities(), Clock(), tmp_path, timeout_s=timeout_s, init_s=0.0
            )
            out = []
            for point in bench.points():
                result = await _run_point(bench, point, ctx, None)
                out.append((result, [*ctx._requests, *ctx._sessions]))
            return out
        finally:
            await backend.close()

    tmp_path.mkdir(parents=True, exist_ok=True)
    out = asyncio.run(go())
    # _run_point flushes records to disk; read them back per result.
    req_file, ses_file = tmp_path / "requests.jsonl", tmp_path / "sessions.jsonl"
    rows = [
        json.loads(x)
        for f in (req_file, ses_file)
        if f.exists()
        for x in f.read_text().splitlines()
    ]
    return [(r, [x for x in rows if x["result_id"] == r.result_id]) for r, _ in out]


SHORT = {"sampling": "stratified", "num_items": 4, "duration_range": [0, 12]}


def test_single_warms_every_item_then_measures_each_rep(data_root, tmp_path):
    ((result, rows),) = measure(data_root, tmp_path, "single", {"repetitions": 2}, SHORT)
    assert result.status == "ok"
    warm = [r for r in rows if r["phase"] == "warmup"]
    meas = [r for r in rows if r["phase"] == "measure"]
    assert len(warm) == 4  # default warmup: "all"
    assert len(meas) == 8 and {r["rep"] for r in meas} == {0, 1}
    assert result.summary["requests.ok"] == 8
    assert result.window[0] >= max(r["t_done"] for r in warm)  # warm-up outside the window


def test_batch_points_and_unsupported_width(data_root, tmp_path):
    out = measure(
        data_root, tmp_path, "batch",
        {"batch_sizes": [2, 3], "compositions": ["uniform"], "num_batches": 2}, SHORT,
    )  # fmt: skip
    (r2, rows2), (r3, rows3) = out
    meas = [x for x in rows2 if x["phase"] == "measure"]
    assert len(meas) == 4 and {x["batch"] for x in meas} == {0, 1}
    assert "batch_latency_ms.p50" in r2.summary and "padding_ratio" in r2.summary
    # every row of one batch completes together and shares its submission time
    for b in (0, 1):
        rows = [x for x in meas if x["batch"] == b]
        assert len({x["t_submit"] for x in rows}) == 1

    class Narrow(MockBackend):
        def capabilities(self):
            caps = super().capabilities()
            caps.max_batch_size = 2
            return caps

    bench = get_kind("batch")("b", {"workload": "w", "batch_sizes": [3]}, None)  # type: ignore[arg-type]
    assert "exceeds" in bench.check(
        Narrow().capabilities(), {"batch_size": 3, "composition": "mixed"}
    )


def test_closed_loop_bounds_concurrency(data_root, tmp_path):
    out = measure(
        data_root, tmp_path, "load",
        {"mode": "closed", "concurrency": [2], "num_requests": 8, "warmup_requests": 0},
        SHORT, options={"base_ms": 20},
    )  # fmt: skip
    ((result, rows),) = out
    meas = [x for x in rows if x["phase"] == "measure"]
    assert len(meas) == 8
    events = sorted([(x["t_submit"], 1) for x in meas] + [(x["t_done"], -1) for x in meas])
    inflight = peak = 0
    for _, d in events:
        inflight += d
        peak = max(peak, inflight)
    assert peak == 2
    assert {x["worker"] for x in meas} == {0, 1}


def test_open_loop_charges_latency_from_the_schedule(data_root, tmp_path):
    # One-row "server" that takes ~25 ms per request, offered 200 req/s: requests
    # queue, so latency grows well past the service time -- and is measured from
    # the scheduled send, not from when the request got through.
    out = measure(
        data_root, tmp_path, "load",
        {"mode": "open", "rates": [200.0], "arrival": "constant", "num_requests": 10, "warmup_requests": 0},
        SHORT, options={"base_ms": 25, "ms_per_audio_s": 0, "max_batch": 1},
    )  # fmt: skip
    ((result, rows),) = out
    meas = sorted((x for x in rows if x["phase"] == "measure"), key=lambda x: x["seq"])
    gaps = [b["t_sched"] - a["t_sched"] for a, b in zip(meas, meas[1:])]
    assert gaps == pytest.approx([0.005] * 9, abs=1e-6)  # the schedule, not the send times
    lat = [(x["t_done"] - x["t_sched"]) * 1e3 for x in meas]
    assert lat[-1] > 150  # the 10th request waited behind 9 others
    assert result.summary["offered_rate"] == 200.0
    assert result.summary["achieved_ratio"] < 0.5  # saturated: one row, 40 req/s max


def test_timeouts_failures_and_rejections_are_classified(data_root, tmp_path):
    out = measure(
        data_root, tmp_path, "single", {"warmup": 0},
        {"sampling": "all", "duration_range": [0, 12]},
        options={"hang_every": 4, "fail_every": 5, "reject_every": 6}, timeout_s=0.2,
    )  # fmt: skip
    ((result, rows),) = out
    s = result.summary
    statuses = [x["status"] for x in rows]
    assert statuses.count("timeout") == 1 and s["requests.timeout"] == 1
    assert statuses.count("failed") == 1 and statuses.count("rejected") == 1
    assert s["error_rate"] == pytest.approx(3 / len(rows))
    assert s["latency_ms.n"] == len(rows) - 3


def test_paced_streaming_follows_the_clock(data_root, tmp_path):
    out = measure(
        data_root, tmp_path, "streaming",
        {"chunk_ms": [250], "sessions": [2], "pacing": "realtime", "word_timestamps": True},
        {"sampling": "first", "num_items": 2, "duration_range": [0, 2]},
        mode="streaming",
    )  # fmt: skip
    ((result, rows),) = out
    meas = [x for x in rows if x["phase"] == "measure"]
    assert len(meas) == 2 and result.status == "ok"
    for s in meas:
        due = [d - s["t_audio0"] for d in s["chunk_due"]]
        expect = [min((k + 1) * 0.25, s["duration_s"]) for k in range(len(due))]
        assert due == pytest.approx(expect, abs=1e-6)
        assert all(sent >= d - 1e-6 for sent, d in zip(s["chunk_sent"], s["chunk_due"]))
        assert s["events"][-1]["kind"] == "final"
    # Staggered: the two sessions do not start on the same tick.
    starts = sorted(s["t_audio0"] for s in meas)
    assert starts[1] - starts[0] == pytest.approx(0.125, abs=0.03)
    assert result.summary["late_chunk_pct"] == 0.0
    assert result.summary["final_latency_ms.p50"] > 0
    assert result.summary["emission_delay_ms.n"] > 0
    assert result.server["streaming_mode"] == "native"


def test_streaming_unsupported_is_reported(data_root, tmp_path):
    (result, rows), = measure(
        data_root, tmp_path, "streaming", {"chunk_ms": [500]},
        {"sampling": "first", "num_items": 1}, options={"streaming": "none"}, mode="streaming",
    )  # fmt: skip
    assert result.status == "unsupported" and "streaming" in result.reason and not rows


def test_unknown_parameter_is_an_error():
    with pytest.raises(ValueError, match="unknown parameter"):
        get_kind("single")("b", {"batch_sizes": [1]}, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Whole runs
# ---------------------------------------------------------------------------


def test_full_run_records_everything_and_recomputes_exactly(mock_run):
    run = load_run(mock_run)
    by_name = {}
    for r in run.results:
        by_name.setdefault(r.benchmark, []).append(r)
    assert set(by_name) == {b["name"] for b in tiny_profile()["benchmarks"]}
    assert all(r.status == "ok" for r in run.results), [
        (r.result_id, r.reason) for r in run.results
    ]
    # capacity_from: the open-loop rate is a fraction of the measured throughput.
    (thr,) = by_name["throughput"]
    (opn,) = by_name["load_open"]
    assert opn.summary["offered_rate"] == pytest.approx(0.5 * thr.summary["throughput.req_per_s"])
    assert run.meta["simulated"] == {"backend": True, "inputs": True}
    assert set(run.meta["backend"]["instances"]) == {"offline", "streaming"}
    assert run.meta["environment"]["git"]["commit"]
    assert (run.path / "workloads" / "utts.json").is_file()
    rec = recompute(mock_run)
    assert rec["checked"] == len(run.results) and rec["mismatches"] == []


def test_cold_start_runs_first_in_its_group(mock_run):
    run = load_run(mock_run)
    offline = [r for r in run.results if r.mode == "offline"]
    assert offline[0].kind == "cold_start"
    assert not offline[0].warnings  # it really was cold
    assert "init_s" in offline[0].summary and "first_request_ms" in offline[0].summary


def test_resource_samples_are_attributed_to_windows(mock_run):
    run = load_run(mock_run)
    if not run.meta["monitor"].get("probes", {}).get("psutil"):
        pytest.skip("psutil not installed: no process samples")
    (single,) = [r for r in run.results if r.benchmark == "single"]
    assert single.summary.get("rss_mb.max", 0) > 0


def test_missing_capacity_source_is_unsupported_not_guessed(data_root, tmp_path):
    prof = tiny_profile()
    prof["benchmarks"] = [b for b in prof["benchmarks"] if b["name"] == "load_open"]
    run = load_run(run_tiny(data_root, tmp_path, profile=prof, monitor_interval_s=0))
    (r,) = run.results
    assert r.status == "unsupported" and "capacity_from" in r.reason


def test_statically_unsupported_mode_skips_loading(data_root, tmp_path, monkeypatch):
    loads = []

    class NoStreaming(MockBackend):
        @classmethod
        def mode_unsupported(cls, mode, options):
            return "no streams here" if mode == "streaming" else None

        async def load(self):
            loads.append(self.mode)
            await super().load()

    from benchmarks.asrbench import backends

    monkeypatch.setitem(backends.BACKENDS, "nostream", f"{__name__}:_placeholder")
    monkeypatch.setattr(backends, "backend_class", lambda name: NoStreaming)
    run = load_run(run_tiny(data_root, tmp_path, backend="nostream", monitor_interval_s=0))
    assert loads == ["offline"]
    stream = [r for r in run.results if r.mode == "streaming"]
    assert stream and all(r.status == "unsupported" for r in stream)
