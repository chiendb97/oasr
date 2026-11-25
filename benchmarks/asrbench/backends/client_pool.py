# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""A load generator spread over processes: ``-o client_procs=N``.

One Python process is not a load generator for a fast server.  Building each
multipart upload, parsing each response, and the event loop that drives them
cost the client about a millisecond of CPU per request (33 % of a core at
264 req/s), all of it on one interpreter.  Against one U2++ ``oasr-server`` it
delivered 411 req/s at 16 in flight and 145-190 at 256, where 8 worker
processes delivered 1,030-1,169 and 1,474-1,507 and a separate 8-process load
generator 1,039 and 1,550: every served point above a few in flight measured
the client.

:class:`ClientPool` keeps the runner and the records exactly as they are and
moves the wire work into ``N`` worker processes.  The main process keeps its own
instance of the backend for everything that is not a request -- the spawned
server, its PID for the resource monitor, the ``/metrics`` scrapes, ``info`` and
capabilities -- and each worker runs a second instance attached to the same
server (:meth:`Backend.client_options`).  A request is handed to the worker with
the fewest in flight, by key: every payload is shipped to every worker before
any clock starts (:meth:`prepared`), so a request message is a few bytes.

Timestamps stay honest across processes because ``perf_counter`` is the
system-wide monotonic clock on Linux, macOS and Windows; :meth:`start` checks
that each worker's clock reads between two of the main process's, and refuses
to run when it does not.  A worker stamps ``t_submit``/``t_done`` and stream
events itself.  The main process still stamps the schedule, so the hand-off to
a worker (about 0.1 ms) is client time, and is reported as such in
``send_lag_ms``.
"""

from __future__ import annotations

import asyncio
import itertools
import multiprocessing as mp
import os
import pickle
import signal
import socket
import struct
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

#: The backend option that asks for a pool.
OPTION = "client_procs"

_HEADER = struct.Struct("!Q")
#: How long a worker has to import, attach and answer.
START_TIMEOUT_S = 120.0
#: A worker's clock may read this far outside the main process's two reads
#: (the reads straddle a round trip, so this only absorbs timer granularity).
CLOCK_SLACK_S = 1e-3


def requested_procs(options: Dict[str, Any]) -> int:
    """``client_procs`` as an int >= 1 (1 when unset)."""
    raw = options.get(OPTION)
    if raw in (None, ""):
        return 1
    try:
        n = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{OPTION} must be an integer >= 1, got {raw!r}") from None
    if n < 1:
        raise ValueError(f"{OPTION} must be an integer >= 1, got {raw!r}")
    return n


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------


def _send(writer: asyncio.StreamWriter, msg: Tuple[Any, ...]) -> None:
    data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    writer.writelines((_HEADER.pack(len(data)), data))


async def _recv(reader: asyncio.StreamReader) -> Tuple[Any, ...]:
    (n,) = _HEADER.unpack(await reader.readexactly(_HEADER.size))
    return pickle.loads(await reader.readexactly(n))


# ---------------------------------------------------------------------------
# worker process
# ---------------------------------------------------------------------------


@dataclass
class WorkerSpec:
    backend: str
    options: Dict[str, Any]
    mode: str
    index: int


def _worker_entry(sock: socket.socket, spec: WorkerSpec) -> None:
    """Process entry point: serve requests until the main process goes away."""
    # Ctrl-C belongs to the main process, which stops the workers itself; a
    # worker that also caught it would print a traceback per process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    asyncio.run(_Worker(sock, spec).serve())


class _Worker:
    def __init__(self, sock: socket.socket, spec: WorkerSpec) -> None:
        self.sock = sock
        self.spec = spec
        self.payloads: Dict[int, Any] = {}
        self.tasks: Dict[int, "asyncio.Task[None]"] = {}
        self.sessions: Dict[int, "_WorkerSession"] = {}
        #: Sessions the main process closed while they were still opening.
        self.closed_early: set = set()
        self.writer: Optional[asyncio.StreamWriter] = None
        self.backend: Optional[Backend] = None

    def send(self, msg: Tuple[Any, ...]) -> None:
        if self.writer is not None and not self.writer.is_closing():
            _send(self.writer, msg)

    async def serve(self) -> None:
        from benchmarks.asrbench import backends

        reader, self.writer = await asyncio.open_unix_connection(sock=self.sock)
        try:
            self.backend = backends.create(self.spec.backend, self.spec.options, self.spec.mode)
            await self.backend.load()
        except Exception as exc:  # noqa: BLE001 -- reported to the main process
            self.send(("failed", f"{type(exc).__name__}: {exc}", traceback.format_exc()))
            await self.writer.drain()
            return
        self.send(("ready", os.getpid()))
        try:
            while True:
                try:
                    msg = await _recv(reader)
                except (asyncio.IncompleteReadError, ConnectionError):
                    break  # the main process is gone
                if not self.dispatch(msg):
                    break
        finally:
            for task in list(self.tasks.values()):
                task.cancel()
            for sess in list(self.sessions.values()):
                await sess.close()
            await self.backend.close()
            self.writer.close()

    def dispatch(self, msg: Tuple[Any, ...]) -> bool:
        op = msg[0]
        if op == "payload":
            self.payloads[msg[1]] = msg[2]
        elif op == "sync":
            self.send(("synced", msg[1], len(self.payloads)))
        elif op == "ping":
            self.send(("pong", msg[1], time.perf_counter()))
        elif op == "req":
            rid, key = msg[1], msg[2]
            self.tasks[rid] = asyncio.ensure_future(self._request(rid, key))
        elif op == "abort":
            task = self.tasks.pop(msg[1], None)
            if task is not None:
                task.cancel()
        elif op == "open":
            asyncio.ensure_future(self._open(msg[1], msg[2]))
        elif op == "chunk":
            sess = self.sessions.get(msg[1])
            if sess is not None:
                sess.chunks.put_nowait((msg[2], msg[3]))
        elif op == "close":
            sess = self.sessions.pop(msg[1], None)
            if sess is not None:
                asyncio.ensure_future(sess.close())
            else:
                self.closed_early.add(msg[1])
        elif op == "stop":
            return False
        return True

    async def _request(self, rid: int, key: int) -> None:
        assert self.backend is not None
        try:
            tr = await self.backend.transcribe(self.payloads[key])
        except asyncio.CancelledError:
            return  # the main process gave up on it; it answers nothing
        except asyncio.TimeoutError as exc:
            self.send(("timeout", rid, str(exc)))
        except Unsupported as exc:
            self.send(("unsupported", rid, str(exc)))
        except RequestFailed as exc:
            self.send(("fail", rid, str(exc), exc.kind, exc.status))
        except Exception as exc:  # noqa: BLE001 -- one request must not end the worker
            self.send(("fail", rid, f"{type(exc).__name__}: {exc}", "failed", None))
        else:
            self.send(("done", rid, tr.text, tr.t_submit, tr.t_done, tr.t_first, tr.words, tr.meta))
        finally:
            self.tasks.pop(rid, None)

    async def _open(self, sid: int, word_timestamps: bool) -> None:
        assert self.backend is not None
        try:
            stream = await self.backend.open_stream(word_timestamps=word_timestamps)
        except Unsupported as exc:
            self.send(("open_fail", sid, "unsupported", str(exc)))
            return
        except RequestFailed as exc:
            self.send(("open_fail", sid, exc.kind, str(exc)))
            return
        except Exception as exc:  # noqa: BLE001
            self.send(("open_fail", sid, "failed", f"{type(exc).__name__}: {exc}"))
            return
        if sid in self.closed_early:
            # Given up on while it opened: nobody will send to it or close it.
            self.closed_early.discard(sid)
            await stream.abort()
            return
        self.sessions[sid] = _WorkerSession(self, sid, stream)
        self.send(("opened", sid))


class _WorkerSession:
    """One real stream in a worker, mirrored to the main process event by event."""

    def __init__(self, worker: _Worker, sid: int, stream: StreamSession) -> None:
        self.worker = worker
        self.sid = sid
        self.stream = stream
        self.chunks: "asyncio.Queue[Tuple[Any, bool]]" = asyncio.Queue()
        # Every event the stream records is forwarded as it is recorded, with
        # the worker's timestamp.  Hooked before any await, so none is missed.
        sent = len(stream.events)
        push = stream.push

        def forward(event: StreamEvent) -> None:
            push(event)
            worker.send(("event", sid, event.kind, event.t, event.text, event.words))

        stream.push = forward  # type: ignore[method-assign]
        for event in stream.events[:sent]:
            worker.send(("event", sid, event.kind, event.t, event.text, event.words))
        self.pump = asyncio.ensure_future(self._pump())
        self.outcome = asyncio.ensure_future(self._outcome())

    async def _pump(self) -> None:
        """Send chunks in arrival order; a send that fails ends the session."""
        while True:
            chunk, last = await self.chunks.get()
            try:
                await self.stream.send(chunk, is_last=last)
            except Exception as exc:  # noqa: BLE001
                self.stream.push(
                    StreamEvent("error", time.perf_counter(), f"{type(exc).__name__}: {exc}")
                )
                return
            if last:
                return

    async def _outcome(self) -> None:
        """Report how the session ended, with the kind its own ``wait_final`` gives."""
        try:
            await self.stream.wait_final(None)
        except asyncio.CancelledError:
            return
        except RequestFailed as exc:
            self.worker.send(("end", self.sid, exc.kind, str(exc)))
            return
        except Exception as exc:  # noqa: BLE001
            self.worker.send(("end", self.sid, "failed", f"{type(exc).__name__}: {exc}"))
            return
        self.worker.send(("end", self.sid, None, None))

    async def close(self) -> None:
        self.pump.cancel()
        self.outcome.cancel()
        try:
            await self.stream.abort()
        except Exception:  # noqa: BLE001 -- best effort; the session is over
            pass


# ---------------------------------------------------------------------------
# main process
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemotePayload:
    """A payload held by every worker, named by its key."""

    key: int
    item: str


@dataclass
class _Handle:
    index: int
    proc: Any
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    pid: int = 0
    alive: bool = True
    inflight: int = 0
    served: int = 0
    pending: Dict[int, "asyncio.Future[Any]"] = field(default_factory=dict)
    streams: Dict[int, "_RemoteStream"] = field(default_factory=dict)
    reader_task: Optional["asyncio.Task[None]"] = None

    def send(self, msg: Tuple[Any, ...]) -> None:
        if not self.alive:
            raise RequestFailed(f"client process {self.index} has exited")
        _send(self.writer, msg)


class _RemoteStream(StreamSession):
    """The main process's view of a stream that lives in a worker."""

    def __init__(self, pool: "ClientPool", handle: _Handle, sid: int) -> None:
        super().__init__()
        self.pool = pool
        self.handle = handle
        self.sid = sid
        self.kind = "failed"
        self._closed = False

    def record(self, event: StreamEvent) -> None:
        # Not ``push``: an error event must not end the wait before the worker
        # says which kind of error it was (``end``).
        self.events.append(event)

    def end(self, kind: Optional[str], message: Optional[str]) -> None:
        if kind is not None:
            self.error, self.kind = message or "stream failed", kind
        self._final.set()

    async def send(self, chunk: Any, is_last: bool) -> None:
        if self._final.is_set():
            return  # already ended; the worker recorded why
        self.handle.send(("chunk", self.sid, chunk, is_last))

    async def wait_final(self, timeout: Optional[float]) -> StreamEvent:
        try:
            await asyncio.wait_for(self._final.wait(), timeout)
            if self.error is not None:
                raise RequestFailed(self.error, kind=self.kind)
            final = next((e for e in reversed(self.events) if e.kind == "final"), None)
            if final is None:
                raise RequestFailed("the stream ended without a final result")
            return final
        finally:
            self._close()

    async def abort(self) -> None:
        self._close()

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.handle.streams.pop(self.sid, None)
        self.handle.inflight -= 1
        if self.handle.alive:
            self.handle.send(("close", self.sid))


class ClientPool(Backend):
    """``inner``'s requests and streams, run by ``procs`` worker processes."""

    name = "client_pool"
    description = "a network backend's client spread over worker processes"

    def __init__(self, inner: Backend, backend_name: str, procs: int) -> None:
        super().__init__({}, inner.mode)
        options = inner.client_options()
        if options is None:
            raise ValueError(
                f"{OPTION}: backend {backend_name!r} runs inference in this process; "
                "only a backend that talks to a server can spread its client"
            )
        self.inner = inner
        self.backend_name = backend_name
        self.procs = int(procs)
        self.options = dict(inner.options)
        # Only what the class accepts: an instance adds keys to its own options
        # as it loads (oasr_server's ``metrics_url``), which a fresh one refuses.
        accepted = set(type(inner).OPTIONS) | {"sample_rate"}
        self._worker_options = {k: v for k, v in options.items() if k in accepted and k != OPTION}
        self.handles: List[_Handle] = []
        self._ids = itertools.count()
        #: uid -> (key, the audio it was built from)
        self._keys: Dict[str, Tuple[int, np.ndarray]] = {}
        self._unshipped: List[Tuple[int, Any]] = []
        self._syncs: Dict[Tuple[int, int], "asyncio.Future[Any]"] = {}
        self._pongs: Dict[int, "asyncio.Future[float]"] = {}
        self.start_s: Optional[float] = None
        #: Payloads each worker reported holding at the last :meth:`prepared`.
        self.held: List[int] = []

    @property
    def sample_rate(self) -> int:
        return self.inner.sample_rate

    # -- lifecycle ----------------------------------------------------------------

    async def load(self) -> None:
        raise RuntimeError("ClientPool wraps a loaded backend: call start()")

    async def start(self) -> None:
        """Spawn the workers, wait until each is attached, and check their clocks."""
        t0 = time.perf_counter()
        ctx = mp.get_context("spawn")
        try:
            for i in range(self.procs):
                ours, theirs = socket.socketpair()
                spec = WorkerSpec(self.backend_name, self._worker_options, self.mode, i)
                proc = ctx.Process(
                    target=_worker_entry,
                    args=(theirs, spec),
                    name=f"asrbench-client-{i}",
                    daemon=True,
                )
                proc.start()
                theirs.close()
                reader, writer = await asyncio.open_unix_connection(sock=ours)
                self.handles.append(_Handle(i, proc, reader, writer))
            await asyncio.wait_for(
                asyncio.gather(*(self._ready(h) for h in self.handles)), START_TIMEOUT_S
            )
            for h in self.handles:
                h.reader_task = asyncio.ensure_future(self._read(h))
            for h in self.handles:
                await self._check_clock(h)
        except BaseException:
            await self._stop_workers()
            raise
        self.start_s = time.perf_counter() - t0

    async def _ready(self, h: _Handle) -> None:
        try:
            msg = await _recv(h.reader)
        except (asyncio.IncompleteReadError, ConnectionError):
            raise RuntimeError(
                f"client process {h.index} exited during start-up (code {h.proc.exitcode})"
            ) from None
        if msg[0] == "failed":
            raise RuntimeError(f"client process {h.index} could not attach: {msg[1]}\n{msg[2]}")
        h.pid = int(msg[1])

    async def _check_clock(self, h: _Handle) -> None:
        """Refuse a worker whose ``perf_counter`` is not this process's clock."""
        token = next(self._ids)
        fut: "asyncio.Future[float]" = asyncio.get_running_loop().create_future()
        self._pongs[token] = fut
        t0 = time.perf_counter()
        h.send(("ping", token))
        theirs = await asyncio.wait_for(fut, 10.0)
        t1 = time.perf_counter()
        check_clock(t0, theirs, t1, h.index)

    async def close(self) -> None:
        """Stop the workers, then the main process's own backend (and its server)."""
        await self._stop_workers()
        await self.inner.close()

    async def _stop_workers(self) -> None:
        for h in self.handles:
            if h.alive:
                try:
                    _send(h.writer, ("stop",))
                    await asyncio.wait_for(h.writer.drain(), 5.0)
                except Exception:  # noqa: BLE001 -- it may already be gone
                    pass
        for h in self.handles:
            await asyncio.get_running_loop().run_in_executor(None, h.proc.join, 10.0)
            if h.proc.is_alive():
                h.proc.terminate()
                h.proc.join(5.0)
            if h.reader_task is not None:
                h.reader_task.cancel()
            h.writer.close()
            self._fail(h, "client pool closed")
        self.handles = []

    # -- the main process's reader -----------------------------------------------

    async def _read(self, h: _Handle) -> None:
        try:
            while True:
                msg = await _recv(h.reader)
                self._on(h, msg)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- EOF or a broken frame: the worker is gone
            pass
        code = h.proc.exitcode
        self._fail(h, f"client process {h.index} exited" + (f" (code {code})" if code else ""))

    def _fail(self, h: _Handle, why: str) -> None:
        h.alive = False
        for fut in h.pending.values():
            if not fut.done():
                fut.set_exception(RequestFailed(why))
        h.pending.clear()
        for stream in list(h.streams.values()):
            stream.end("failed", why)
        for fut in self._syncs.values():
            if not fut.done():
                fut.set_exception(RuntimeError(why))

    def _on(self, h: _Handle, msg: Tuple[Any, ...]) -> None:
        op = msg[0]
        if op in ("done", "fail", "timeout", "unsupported", "opened", "open_fail"):
            fut = h.pending.pop(msg[1], None)
            if fut is None or fut.done():
                return  # abandoned: the request was cancelled here
            if op == "done":
                _, _, text, t_submit, t_done, t_first, words, meta = msg
                fut.set_result(Transcript(text, t_submit, t_done, t_first, words, meta))
            elif op == "fail":
                fut.set_exception(RequestFailed(msg[2], kind=msg[3], status=msg[4]))
            elif op == "timeout":
                fut.set_exception(asyncio.TimeoutError(msg[2]))
            elif op == "unsupported":
                fut.set_exception(Unsupported(msg[2]))
            elif op == "opened":
                fut.set_result(None)
            else:
                kind, why = msg[2], msg[3]
                fut.set_exception(
                    Unsupported(why) if kind == "unsupported" else RequestFailed(why, kind=kind)
                )
        elif op == "event":
            stream = h.streams.get(msg[1])
            if stream is not None:
                stream.record(StreamEvent(msg[2], msg[3], msg[4], msg[5]))
        elif op == "end":
            stream = h.streams.get(msg[1])
            if stream is not None:
                stream.end(msg[2], msg[3])
        elif op == "synced":
            fut = self._syncs.get(msg[1])
            if fut is not None and not fut.done():
                fut.set_result(msg[2])
        elif op == "pong":
            fut = self._pongs.pop(msg[1], None)
            if fut is not None and not fut.done():
                fut.set_result(msg[2])

    # -- routing ------------------------------------------------------------------

    def _pick(self) -> _Handle:
        """The live worker with the fewest in flight; ties go to the lowest index.

        Not round-robin on ties: at one request in flight that sent each request
        to a worker idle for ``procs`` cycles, whose interpreter and core had
        gone cold.  Measured on U2++ with 8 workers on 8 cores, the request
        itself took 4.0 ms that way, 3.85-3.90 ms lowest-first, and 3.55-3.68 ms
        in one process (the rest is nine processes sharing eight cores).
        Lowest-first still spreads a high load: every request raises its
        worker's count.
        """
        live = [h for h in self.handles if h.alive]
        if not live:
            raise RequestFailed("every client process has exited")
        return min(live, key=lambda h: h.inflight)

    # -- inputs -------------------------------------------------------------------

    def prepare(self, audio: np.ndarray, item: WorkItem) -> RemotePayload:
        """Build the payload here; it reaches the workers in :meth:`prepared`."""
        known = self._keys.get(item.uid)
        if known is not None and known[1] is audio:
            return RemotePayload(known[0], item.uid)
        key = next(self._ids)
        self._keys[item.uid] = (key, audio)
        self._unshipped.append((key, self.inner.prepare(audio, item)))
        return RemotePayload(key, item.uid)

    def prepare_chunk(self, chunk: np.ndarray) -> Any:
        return self.inner.prepare_chunk(chunk)

    async def prepared(self) -> None:
        """Ship every new payload to every worker and wait until each holds them."""
        batch, self._unshipped = self._unshipped, []
        live = [h for h in self.handles if h.alive]
        for key, payload in batch:
            for h in live:
                h.send(("payload", key, payload))
                await h.writer.drain()
        token = next(self._ids)
        loop = asyncio.get_running_loop()
        futs = []
        for h in live:
            fut = loop.create_future()
            self._syncs[(token, h.index)] = fut
            h.send(("sync", (token, h.index)))
            futs.append(fut)
        try:
            self.held = list(await asyncio.wait_for(asyncio.gather(*futs), START_TIMEOUT_S))
        finally:
            for h in live:
                self._syncs.pop((token, h.index), None)
        short = [h.index for h, n in zip(live, self.held) if n < len(self._keys)]
        if short:
            raise RuntimeError(
                f"client process(es) {short} hold {self.held} payloads of {len(self._keys)}"
            )

    # -- inference ----------------------------------------------------------------

    async def transcribe(self, payload: Any) -> Transcript:
        if not isinstance(payload, RemotePayload):
            raise TypeError("ClientPool.transcribe takes a payload from its own prepare()")
        h = self._pick()
        rid = next(self._ids)
        fut: "asyncio.Future[Transcript]" = asyncio.get_running_loop().create_future()
        h.pending[rid] = fut
        h.inflight += 1
        try:
            h.send(("req", rid, payload.key))
            tr = await fut
            h.served += 1
            return tr
        except asyncio.CancelledError:
            # Timed out here: the worker stops waiting too, and answers nothing.
            if h.alive and h.pending.pop(rid, None) is not None:
                h.send(("abort", rid))
            raise
        finally:
            h.pending.pop(rid, None)
            h.inflight -= 1

    async def transcribe_batch(self, payloads: Sequence[Any]) -> List[Transcript]:
        raise Unsupported(f"{self.backend_name} has no native batch call through {OPTION}")

    async def open_stream(self, *, word_timestamps: bool = False) -> StreamSession:
        h = self._pick()
        sid = next(self._ids)
        fut: "asyncio.Future[None]" = asyncio.get_running_loop().create_future()
        h.pending[sid] = fut
        stream = _RemoteStream(self, h, sid)
        h.streams[sid] = stream
        h.inflight += 1
        try:
            h.send(("open", sid, bool(word_timestamps)))
            await fut
        except BaseException:
            h.streams.pop(sid, None)
            h.pending.pop(sid, None)
            h.inflight -= 1
            if h.alive:
                h.send(("close", sid))
            raise
        h.served += 1
        return stream

    # -- everything else is the main process's own backend ---------------------------

    def info(self) -> BackendInfo:
        info = self.inner.info()
        info.config["client"] = self.describe()
        return info

    def describe(self) -> Dict[str, Any]:
        return {
            "procs": self.procs,
            "pids": [h.pid for h in self.handles],
            "start_s": self.start_s,
            "served": [h.served for h in self.handles],
            "payloads": list(self.held),
            "exited": [h.index for h in self.handles if not h.alive],
        }

    def client_pids(self) -> List[int]:
        return [os.getpid(), *(h.pid for h in self.handles if h.alive)]

    def capabilities(self) -> Capabilities:
        caps = self.inner.capabilities()
        caps.batch = False
        return caps

    async def synchronize(self) -> None:
        await self.inner.synchronize()

    def monitored_pid(self) -> Optional[int]:
        return self.inner.monitored_pid()

    def reset_peak_memory(self) -> None:
        self.inner.reset_peak_memory()

    def peak_memory(self) -> Dict[str, float]:
        return self.inner.peak_memory()

    def server_metrics_begin(self) -> None:
        self.inner.server_metrics_begin()

    def server_metrics_end(self) -> Dict[str, Any]:
        return self.inner.server_metrics_end()


def check_clock(before: float, theirs: float, after: float, index: int) -> None:
    """Raise unless a worker's ``perf_counter`` read lies between two of ours."""
    if not before - CLOCK_SLACK_S <= theirs <= after + CLOCK_SLACK_S:
        raise RuntimeError(
            f"client process {index}'s clock is not this process's: it read {theirs:.6f} "
            f"between {before:.6f} and {after:.6f}; timestamps from it would not line up"
        )
