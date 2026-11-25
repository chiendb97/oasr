# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``-o client_procs=N``: requests sent by worker processes, recorded as before.

Against in-process fake servers (HTTP and the realtime WebSocket), so the real
process boundary is crossed: spawned workers, pickled payloads, timestamps
taken in another process.  What is asserted is that nothing about a record
changes meaning on the way -- every payload arrives intact, every outcome keeps
its kind, every timestamp is on the run clock -- and that a worker that dies
costs only its own requests.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import numpy as np
import pytest
from asrbench_fixtures import run_tiny

from benchmarks.asrbench.backends import (
    Backend,
    BackendInfo,
    Capabilities,
    RequestFailed,
    Transcript,
    create,
)
from benchmarks.asrbench.backends.client_pool import (
    ClientPool,
    check_clock,
    requested_procs,
)
from benchmarks.asrbench.results import FAILED, OK, REJECTED, TIMEOUT, Clock, load_run
from benchmarks.asrbench.runner import recompute
from benchmarks.asrbench.runners import RunContext
from benchmarks.asrbench.workloads import WorkItem

pytest.importorskip("httpx")

SR = 16000
#: Frame counts the fake server treats specially.
REJECT, BREAK, HANG = 1600, 3200, 4800


def _frames(body: bytes) -> int:
    """Frames of the WAV inside a multipart body."""
    at = body.find(b"RIFF")
    with wave.open(io.BytesIO(body[at:])) as w:
        return w.getnframes()


class _Handler(BaseHTTPRequestHandler):
    seen: List[int] = []

    def log_message(self, *args: Any) -> None:  # quiet
        pass

    def do_POST(self) -> None:  # noqa: N802 -- http.server's name
        body = self.rfile.read(int(self.headers["Content-Length"]))
        n = _frames(body)
        type(self).seen.append(n)
        if n == REJECT:
            return self._reply(503, b"busy", "text/plain")
        if n == BREAK:
            return self._reply(500, b"kaput", "text/plain")
        if n == HANG:
            time.sleep(1.5)
        self._reply(200, json.dumps({"text": f"frames={n}"}).encode(), "application/json")

    def _reply(self, code: int, data: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def http_server():
    handler = type("Handler", (_Handler,), {"seen": []})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", handler
    srv.shutdown()
    srv.server_close()


def _item(n: int) -> WorkItem:
    return WorkItem(uid=f"u{n}", source_id=f"u{n}", audio_path="", duration_s=n / SR)


async def _pool(url: str, procs: int = 2, **options: Any) -> ClientPool:
    inner = create("openai_http", {"base_url": url, "client_procs": procs, **options})
    await inner.load()
    pool = ClientPool(inner, "openai_http", procs)
    await pool.start()
    return pool


def test_requested_procs():
    assert requested_procs({}) == 1
    assert requested_procs({"client_procs": "4"}) == 4
    for bad in (0, -1, "two"):
        with pytest.raises(ValueError):
            requested_procs({"client_procs": bad})


def test_check_clock_refuses_a_clock_that_is_not_ours():
    check_clock(10.0, 10.5, 11.0, 0)
    with pytest.raises(RuntimeError, match="clock"):
        check_clock(10.0, 3.0, 11.0, 0)


def test_an_in_process_backend_cannot_spread_its_client():
    # Refused as an unknown option, before anything loads...
    with pytest.raises(ValueError, match="client_procs"):
        create("mock", {"client_procs": 2})
    # ...and by the pool itself for any backend that names no server.
    with pytest.raises(ValueError, match="in this process"):
        ClientPool(create("mock", {}), "mock", 2)


def test_payloads_and_outcomes_cross_the_process_boundary(http_server):
    url, handler = http_server
    lengths = [800, 12345, REJECT, BREAK, HANG, 4000]

    async def go():
        pool = await _pool(url)
        try:
            audio = {n: np.zeros(n, np.float32) for n in lengths}
            tokens = [pool.prepare(audio[n], _item(n)) for n in lengths]
            # A second benchmark over the same audio reuses what was shipped;
            # different audio under the same id is a new payload.
            same = pool.prepare(audio[800], _item(800))
            again = pool.prepare(np.zeros(1, np.float32), _item(800))
            await pool.prepared()
            # Every worker holds every payload before any clock starts.
            assert pool.describe()["payloads"] == [len(lengths) + 1] * 2
            ctx = RunContext(pool, pool.capabilities(), Clock(), run_dir=None, timeout_s=0.5)
            t_before = ctx.clock.now()
            recs = await asyncio.gather(
                *(
                    ctx.issue(tok, _item(n), result_id="r", rep=0, phase="measure", seq=i)
                    for i, (tok, n) in enumerate(zip(tokens, lengths))
                )
            )
            t_after = ctx.clock.now()
            # The hung request was abandoned here; the pool is still usable and
            # its in-flight counts went back to zero.
            ok = await ctx.issue(tokens[0], _item(800), result_id="r", rep=0, phase="m", seq=9)
            inflight = [h.inflight for h in pool.handles]
            served = pool.describe()["served"]
            return recs, t_before, t_after, ok, inflight, served, (same, again), tokens
        finally:
            await pool.close()

    recs, t0, t1, ok, inflight, served, (same, again), tokens = asyncio.run(go())
    by_len = dict(zip(lengths, recs))
    for n in (800, 12345, 4000):
        r = by_len[n]
        assert r.status == OK and r.text == f"frames={n}"
        # Stamped in a worker, on the run clock.
        assert t0 <= r.t_sched <= r.t_submit <= r.t_done <= t1
    assert by_len[REJECT].status == REJECTED
    assert by_len[BREAK].status == FAILED and "500" in by_len[BREAK].error
    assert by_len[HANG].status == TIMEOUT
    assert ok.status == OK and ok.text == "frames=800"
    assert inflight == [0, 0]
    assert all(s > 0 for s in served), served  # both workers carried requests
    assert same.key == tokens[0].key and again.key != tokens[0].key


def test_a_worker_that_dies_costs_only_its_own_requests(http_server):
    url, _ = http_server

    async def go():
        pool = await _pool(url)
        try:
            tok = pool.prepare(np.zeros(HANG, np.float32), _item(HANG))
            fast = pool.prepare(np.zeros(800, np.float32), _item(800))
            await pool.prepared()
            slow = [asyncio.ensure_future(pool.transcribe(tok)) for _ in range(2)]
            await asyncio.sleep(0.3)  # one on each worker
            victim = pool.handles[0]
            victim.proc.kill()
            # Bounded: a pool that lost track of the dead worker's requests
            # would otherwise leave this waiting forever.
            outcomes = await asyncio.wait_for(asyncio.gather(*slow, return_exceptions=True), 10)
            after = [await asyncio.wait_for(pool.transcribe(fast), 10) for _ in range(4)]
            return outcomes, after, pool.describe()
        finally:
            await pool.close()

    outcomes, after, described = asyncio.run(go())
    failed = [o for o in outcomes if isinstance(o, RequestFailed)]
    assert len(failed) == 1 and "exited" in str(failed[0])
    assert sum(1 for o in outcomes if not isinstance(o, Exception)) == 1
    assert [t.text for t in after] == ["frames=800"] * 4
    assert described["exited"] == [0]


def test_streams_cross_the_process_boundary():
    websockets = pytest.importorskip("websockets")

    async def server(ws):
        json.loads(await ws.recv())  # session.update
        got = 0
        async for msg in ws:
            if isinstance(msg, bytes):
                got += len(msg)
                if got > 3 * 3200:
                    await ws.send(json.dumps({"type": "error", "error": {"message": "too long"}}))
                    return
                await ws.send(json.dumps({
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "a", "text": "a" * (got // 3200),
                }))  # fmt: skip
            elif json.loads(msg).get("type") == "input_audio_buffer.commit":
                await ws.send(json.dumps({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": f"bytes={got}",
                }))  # fmt: skip
                return

    async def session(pool: ClientPool, chunks: int) -> Any:
        s = await pool.open_stream()
        for k in range(chunks):
            await s.send(pool.prepare_chunk(np.zeros(1600, np.float32)), is_last=k == chunks - 1)
        try:
            return s, await s.wait_final(5.0)
        except RequestFailed as exc:
            return s, exc

    async def go():
        async with websockets.serve(server, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            pool = await _pool(f"http://127.0.0.1:{port}", realtime=True)
            try:
                t0 = time.perf_counter()
                out = await asyncio.wait_for(
                    asyncio.gather(session(pool, 3), session(pool, 3), session(pool, 5)), 20
                )
                t1 = time.perf_counter()
                inflight = [h.inflight for h in pool.handles]
                return out, t0, t1, inflight
            finally:
                await pool.close()

    out, t0, t1, inflight = asyncio.run(go())
    for s, final in out[:2]:
        assert final.text == "bytes=9600"
        kinds = [e.kind for e in s.events]
        assert kinds == ["partial"] * 3 + ["final"]
        ts = [e.t for e in s.events]
        assert ts == sorted(ts) and t0 <= ts[0] and ts[-1] <= t1  # worker stamps, our clock
    s, err = out[2]
    assert isinstance(err, RequestFailed) and "too long" in str(err)
    assert inflight == [0, 0]


def test_a_profile_runs_through_client_processes(http_server, data_root, tmp_path):
    url, handler = http_server
    profile: Dict[str, Any] = {
        "name": "pool",
        "defaults": {"primary": "synthetic/default", "seed": 0, "timeout_s": 10},
        "workloads": {
            "utts": {
                "dataset": "${primary}",
                "sampling": "stratified",
                "num_items": 6,
                "duration_range": [0, 12],
            }
        },
        "benchmarks": [
            {"name": "single", "kind": "single", "workload": "utts"},
            {"name": "throughput", "kind": "throughput", "workload": "utts"},
            {
                "name": "load_closed",
                "kind": "load",
                "workload": "utts",
                "mode": "closed",
                "concurrency": [1, 6],
                "num_requests": 12,
            },
            {
                "name": "load_open",
                "kind": "load",
                "workload": "utts",
                "mode": "open",
                "rates": [40],
                "num_requests": 12,
            },
        ],
    }
    path = run_tiny(
        data_root,
        tmp_path,
        {"base_url": url, "client_procs": 3},
        profile=profile,
        backend="openai_http",
    )
    run = load_run(path)
    assert {r.status for r in run.results} == {"ok"}, [(r.result_id, r.reason) for r in run.results]
    client = run.meta["backend"]["instances"]["offline"]["client"]
    assert client["procs"] == 3 and len(set(client["pids"])) == 3 and client["exited"] == []
    assert sum(client["served"]) == len(handler.seen)
    assert run.meta["backend"]["options"]["client_procs"] == 3
    requests = run.requests()
    assert requests and all(r.status == OK for r in requests)
    for r in requests:
        assert r.text == f"frames={round(r.duration_s * SR)}"
        assert r.t_sched <= r.t_submit <= r.t_done
    assert recompute(path)["mismatches"] == []


def test_workers_attach_to_the_main_processs_server():
    # A spawned server's address is only known after load; workers must attach
    # to it (never spawn their own) and get only options their class accepts:
    # an instance adds keys to its own options as it runs, which a fresh one
    # would refuse as unknown.
    inner = create(
        "oasr_server", {"ckpt_dir": "/ckpt", "transport": "grpc", "client_procs": 4}, "offline"
    )
    inner.base_url, inner.grpc_addr = "http://127.0.0.1:1234", "127.0.0.1:1235"
    inner.options.update({"metrics_url": "http://127.0.0.1:1234/metrics", "server_pid": 42})
    inner.options["realtime"] = True
    opts = ClientPool(inner, "oasr_server", 4)._worker_options
    assert opts["base_url"] == "http://127.0.0.1:1234" and opts["grpc_addr"] == "127.0.0.1:1235"
    assert "client_procs" not in opts and "metrics_url" not in opts and "realtime" not in opts
    create("oasr_server", opts, "offline")  # a worker's instance accepts every one of them


class _BusyClient(Backend):
    """A "network" backend whose client work is all CPU: the load generator saturates."""

    name = "busy"
    description = "test double"

    def __init__(self, in_process: bool) -> None:
        super().__init__({}, "offline")
        self.in_process = in_process

    async def load(self) -> None:
        return None

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name)

    def capabilities(self) -> Capabilities:
        return Capabilities(concurrent=True, in_process=self.in_process)

    async def transcribe(self, payload: Any) -> Transcript:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.01:  # 10 ms of client CPU, no await
            pass
        await asyncio.sleep(0)
        return Transcript("x", t0, time.perf_counter())


@pytest.mark.parametrize("in_process", [False, True])
def test_a_saturated_client_is_flagged(in_process, data_root, tmp_path):
    # A closed loop hides a saturated client from send_lag_ms (each user
    # schedules its next request when it gets to it), so a point measured
    # through one busy Python process used to read as the server's number.
    from benchmarks.asrbench.runner import _run_point
    from benchmarks.asrbench.runners import get_kind
    from benchmarks.asrbench.workloads import WorkloadSpec, load_audio, resolve

    async def go():
        backend = _BusyClient(in_process)
        w = resolve(
            WorkloadSpec.from_dict(
                "w", {"dataset": "synthetic/default", "num_items": 4, "duration_range": [0, 5]}
            ),
            data_root,
        )
        bench = get_kind("load")("b", {"workload": "w", "concurrency": [8], "num_requests": 40}, w)
        bench.prepare(backend, load_audio(w.items, SR))
        ctx = RunContext(backend, backend.capabilities(), Clock(), tmp_path, timeout_s=5.0)
        return await _run_point(bench, bench.points()[0], ctx, None)

    result = asyncio.run(go())
    assert result.status == "ok"
    if in_process:
        # The engine is the client there: its CPU is the system under test.
        assert not any(k.startswith("client_") for k in result.summary)
        assert not any("load generator" in w for w in result.warnings)
    else:
        assert result.summary["client_cpu_pct.max"] >= 90
        # 10 ms of CPU per request without an await: the loop runs timers late.
        assert result.summary["client_loop_lag_ms.p99"] >= 5
        assert any("client_procs" in w for w in result.warnings), result.warnings


def test_the_client_check_reads_both_signals():
    # Either signal alone flags a point: a client can hold a closed loop back
    # without being saturated (one loop serialising a batch of responses).
    from types import SimpleNamespace

    from benchmarks.asrbench.runner import _client_limits

    def check(cpu, p99):
        summary = {}
        lag = None if p99 is None else {"n": 10.0, "p50": 0.1, "p99": p99, "max": p99}
        why = _client_limits(SimpleNamespace(client_cpu_pct=cpu, client_loop_lag_ms=lag), summary)
        return why, summary

    assert check(40.0, 1.2) == ("", {
        "client_cpu_pct.max": 40.0, "client_loop_lag_ms.n": 10.0, "client_loop_lag_ms.p50": 0.1,
        "client_loop_lag_ms.p99": 1.2, "client_loop_lag_ms.max": 1.2,
    })  # fmt: skip
    assert "event loop" in check(64.0, 10.3)[0]
    assert "98 % busy" in check(98.0, 1.0)[0]
    assert check(None, None) == ("", {})
