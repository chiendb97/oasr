# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ZeroMQ worker that drives an ``ASREngine`` on behalf of the Rust frontend.

Single-thread mode (default) — one Python thread polls the ZMQ ROUTER socket,
dispatches commands into the engine, then runs ``engine.step()`` and
publishes outputs.

Two-thread mode (``--worker-threads 2``) — an I/O thread owns the socket and
dispatches commands; a step thread runs ``engine.step()`` in a tight loop and
hands events back to the I/O thread via a ``queue.Queue``.  Both modes are
safe because ``ASREngine`` is now lock-protected.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import ipc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class WorkerOptions:
    zmq_endpoint: str
    engine_config_json: str  # path to a JSON file or inline JSON
    cuda_device: Optional[int] = None
    worker_threads: int = 1
    max_concurrent_requests: int = 256
    overload_emit_interval_s: float = 1.0
    poll_block_ms: int = 5
    # Drain a whole burst per tick so that ``engine.step()`` runs on a full
    # batch.  With 128 concurrent streams sending ~10 chunks each, the worker
    # sees ~1400 messages per "wave"; the old 64 cap caused the engine to
    # process small slices (B=1 forward dominated) and dropped throughput
    # ~3-4× vs the in-process engine baseline.
    max_inbound_per_tick: int = 4096
    log_level: str = "info"

    @classmethod
    def from_argv(cls, argv: List[str]) -> "WorkerOptions":
        p = argparse.ArgumentParser(prog="oasr.serving")
        p.add_argument("--zmq-endpoint", required=True,
                       help="e.g. ipc:///tmp/oasr-12345-0.sock or tcp://127.0.0.1:5555")
        p.add_argument("--engine-config-json", required=True,
                       help="path to JSON file with EngineConfig fields, or inline JSON")
        p.add_argument("--cuda-device", type=int, default=None,
                       help="set CUDA_VISIBLE_DEVICES before importing torch")
        p.add_argument("--worker-threads", type=int, default=1, choices=(1, 2))
        p.add_argument("--max-concurrent-requests", type=int, default=256)
        p.add_argument("--overload-emit-interval-s", type=float, default=1.0)
        p.add_argument("--poll-block-ms", type=int, default=5)
        p.add_argument("--max-inbound-per-tick", type=int, default=64)
        p.add_argument("--log-level", default="info")
        args = p.parse_args(argv)
        return cls(
            zmq_endpoint=args.zmq_endpoint,
            engine_config_json=args.engine_config_json,
            cuda_device=args.cuda_device,
            worker_threads=args.worker_threads,
            max_concurrent_requests=args.max_concurrent_requests,
            overload_emit_interval_s=args.overload_emit_interval_s,
            poll_block_ms=args.poll_block_ms,
            max_inbound_per_tick=args.max_inbound_per_tick,
            log_level=args.log_level,
        )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_engine_config(json_arg: str):
    """Build an ``EngineConfig`` from JSON (file path or inline string)."""
    from oasr.engine import EngineConfig

    if Path(json_arg).is_file():
        with open(json_arg, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        obj = json.loads(json_arg)

    # ``dtype`` arrives as a string; map to torch.dtype.
    if isinstance(obj.get("dtype"), str):
        import torch
        dtype_str = obj["dtype"].lower().replace("torch.", "")
        mapping = {
            "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
        }
        if dtype_str not in mapping:
            raise ValueError(f"unsupported dtype string: {obj['dtype']!r}")
        obj["dtype"] = mapping[dtype_str]

    return EngineConfig(**obj)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class EngineWorker:
    """Owns an :class:`~oasr.engine.ASREngine` and serves it over ZMQ ROUTER."""

    def __init__(self, options: WorkerOptions, engine=None):
        self.opt = options

        # ``engine`` is injectable for unit tests; production path loads it
        # from the JSON config so we don't pay the model-load cost in tests.
        if engine is None:
            engine = self._build_engine()
        self.engine = engine

        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.ROUTER)
        # Drop late messages instead of buffering forever when peers are gone.
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(self.opt.zmq_endpoint)

        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)

        # request_id → identity for routing outputs back to the right DEALER.
        self._rid_to_identity: Dict[str, bytes] = {}
        # Most recently seen identity, used for connectionless events
        # (Pong, Overloaded) when we need to fan out.
        self._last_identity: Optional[bytes] = None

        # Two-thread mode plumbing.
        self._outbound: "queue.Queue[List[bytes]]" = queue.Queue(maxsize=4096)
        self._shutdown = threading.Event()
        self._last_overload_emit = 0.0

        # Cache static model_info for Pong replies.
        self._model_info = self._collect_model_info()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_engine(self):
        if self.opt.cuda_device is not None:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.opt.cuda_device)
        cfg = _load_engine_config(self.opt.engine_config_json)
        from oasr.engine import ASREngine
        return ASREngine(cfg)

    def _collect_model_info(self) -> Dict[str, Any]:
        cfg = getattr(self.engine, "_config", None)
        info: Dict[str, Any] = {}
        if cfg is not None:
            info["ckpt_dir"] = getattr(cfg, "ckpt_dir", None)
            info["device"] = getattr(cfg, "device", None)
            info["dtype"] = str(getattr(cfg, "dtype", None))
            info["chunk_size"] = getattr(cfg, "chunk_size", None)
            info["max_batch_size"] = getattr(cfg, "max_batch_size", None)
            info["decoder_type"] = getattr(cfg, "decoder_type", None)
            mc = getattr(cfg, "_model_config", None)
            if mc is not None:
                info["vocab_size"] = getattr(mc, "vocab_size", None)
        return info

    def announce_ready(self) -> None:
        """Tell the Rust supervisor we're listening.  Must be called after
        ``__init__`` and before :meth:`run`."""
        print("READY", flush=True)

    # ------------------------------------------------------------------
    # Command dispatch (always called from a thread that holds no engine lock)
    # ------------------------------------------------------------------

    def handle_cmd(self, msg: ipc.IncomingMessage) -> None:
        """Dispatch one inbound command, queueing any resulting events."""
        self._last_identity = msg.identity
        header = msg.header
        cmd = header.get("type")
        if cmd == ipc.CMD_PING:
            seq = header.get("seq", 0)
            self._emit(msg.identity, ipc.make_pong(
                seq=seq,
                model_info=self._model_info,
                num_running=self.engine.num_running,
                num_waiting=self.engine.num_waiting,
            ))
            return

        # All other commands carry a request_id.
        rid = header.get("request_id")
        if not isinstance(rid, str) or not rid:
            self._emit(msg.identity, ipc.make_error(
                request_id="", code=ipc.ERR_INVALID_CMD,
                message=f"missing request_id in {cmd!r}",
            ))
            return

        try:
            if cmd == ipc.CMD_CREATE_OFFLINE:
                self._cmd_create_offline(msg.identity, header, msg.payload)
            elif cmd == ipc.CMD_CREATE_STREAMING:
                self._cmd_create_streaming(msg.identity, header)
            elif cmd == ipc.CMD_FEED_CHUNK:
                self._cmd_feed_chunk(msg.identity, header, msg.payload)
            elif cmd == ipc.CMD_CANCEL:
                self._cmd_cancel(msg.identity, rid)
            else:
                self._emit(msg.identity, ipc.make_error(
                    request_id=rid, code=ipc.ERR_INVALID_CMD,
                    message=f"unknown command type {cmd!r}",
                ))
        except KeyError as e:
            self._emit(msg.identity, ipc.make_error(
                request_id=rid, code=ipc.ERR_UNKNOWN_REQUEST, message=str(e),
            ))
        except Exception as e:  # noqa: BLE001 — surface as Error to caller
            logger.exception("handle_cmd error for %s", cmd)
            self._emit(msg.identity, ipc.make_error(
                request_id=rid, code=ipc.ERR_INTERNAL, message=repr(e),
            ))

    # ----- per-command handlers ---------------------------------------

    def _admission_blocked(self) -> Optional[str]:
        cap = self.opt.max_concurrent_requests
        load = self.engine.num_running + self.engine.num_waiting
        if load >= cap:
            return f"in-flight {load} >= cap {cap}"
        return None

    def _cmd_create_offline(self, identity: bytes, header: Dict[str, Any],
                            payload: Optional[bytes]) -> None:
        rid, sample_rate = ipc.require_fields(header, "request_id", "sample_rate")
        priority = int(header.get("priority", 0))
        if payload is None or len(payload) == 0:
            self._emit(identity, ipc.make_error(
                request_id=rid, code=ipc.ERR_INVALID_CMD,
                message="CreateOffline requires audio payload",
            ))
            return
        blocked = self._admission_blocked()
        if blocked is not None:
            self._emit(identity, ipc.make_error(
                request_id=rid, code=ipc.ERR_BUSY, message=blocked,
            ))
            return
        # `np.frombuffer` returns a read-only view; torch.from_numpy warns.
        audio = np.frombuffer(payload, dtype=np.float32)
        if not audio.flags.writeable:
            audio = audio.copy()
        self._rid_to_identity[rid] = identity
        self.engine.add_request(
            audio=audio, request_id=rid, sample_rate=int(sample_rate),
            streaming=False, priority=priority,
        )
        self._emit(identity, ipc.make_accepted(rid))

    def _cmd_create_streaming(self, identity: bytes, header: Dict[str, Any]) -> None:
        rid, sample_rate = ipc.require_fields(header, "request_id", "sample_rate")
        priority = int(header.get("priority", 0))
        blocked = self._admission_blocked()
        if blocked is not None:
            self._emit(identity, ipc.make_error(
                request_id=rid, code=ipc.ERR_BUSY, message=blocked,
            ))
            return
        self._rid_to_identity[rid] = identity
        self.engine.add_streaming_request(
            request_id=rid, sample_rate=int(sample_rate), priority=priority,
        )
        self._emit(identity, ipc.make_accepted(rid))

    def _cmd_feed_chunk(self, identity: bytes, header: Dict[str, Any],
                        payload: Optional[bytes]) -> None:
        rid, is_last = ipc.require_fields(header, "request_id", "is_last")
        if payload is None:
            payload = b""
        chunk = np.frombuffer(payload, dtype=np.float32)
        # ``np.frombuffer`` returns a read-only view; engine's append wants a
        # writable buffer when it later concatenates with ``audio_tail``.
        if not chunk.flags.writeable:
            chunk = chunk.copy()
        self.engine.feed_chunk(rid, chunk, is_last=bool(is_last))
        # No explicit ack; partials/finals will arrive via step().

    def _cmd_cancel(self, identity: bytes, rid: str) -> None:
        self.engine.abort_request(rid)
        self._rid_to_identity.pop(rid, None)
        # No explicit ack; the WS handler on the Rust side already dropped.

    # ------------------------------------------------------------------
    # Output handling — called from whichever thread runs engine.step()
    # ------------------------------------------------------------------

    def _publish_outputs(self, outputs: List[Any]) -> None:
        for out in outputs:
            rid = out.request_id
            identity = self._rid_to_identity.get(rid)
            if identity is None:
                # Output for a request that has no known identity — probably
                # an old cancellation race.  Drop quietly.
                continue
            text = out.text
            tokens = out.tokens
            scores = out.scores
            if out.finished:
                event = ipc.make_final(rid, text, tokens, scores)
                self._rid_to_identity.pop(rid, None)
            else:
                event = ipc.make_partial(rid, text, tokens, scores)
            self._emit(identity, event)

    def _maybe_emit_overloaded(self) -> None:
        now = time.monotonic()
        if now - self._last_overload_emit < self.opt.overload_emit_interval_s:
            return
        load = self.engine.num_running + self.engine.num_waiting
        cap = self.opt.max_concurrent_requests
        if load < cap or self._last_identity is None:
            return
        self._emit(self._last_identity, ipc.make_overloaded(
            reason=f"in-flight {load} >= cap {cap}", queue_depth=load,
        ))
        self._last_overload_emit = now

    # ------------------------------------------------------------------
    # Emit helpers — uniform across single- and two-thread modes
    # ------------------------------------------------------------------

    def _emit(self, identity: bytes, event: Dict[str, Any]) -> None:
        frames = ipc.build_outgoing(identity, event)
        if self.opt.worker_threads == 1:
            self._sock.send_multipart(frames)
        else:
            try:
                self._outbound.put_nowait(frames)
            except queue.Full:
                logger.warning("outbound queue full; dropping event %s", event.get("type"))

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Run the worker until SIGTERM/SIGINT.  Returns the exit code."""
        self._install_signal_handlers()
        try:
            if self.opt.worker_threads == 2:
                return self._run_two_thread()
            return self._run_single_thread()
        finally:
            self._cleanup()

    def _install_signal_handlers(self) -> None:
        # signal.signal can only be called from the main thread.  Tests that
        # drive run() from a worker thread skip the install; the test still
        # controls shutdown via ``_shutdown.set()``.
        if threading.current_thread() is not threading.main_thread():
            return
        def _stop(_sig, _frm):
            self._shutdown.set()
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)

    # ----- single-thread mode -----------------------------------------

    def _run_single_thread(self) -> int:
        import zmq
        while not self._shutdown.is_set():
            # Drain inbound until the queue is empty or the per-tick budget
            # is exhausted, then step.  Skipping ``poll`` between recvs (the
            # NOBLOCK recv itself signals empty via ``zmq.Again``) and using
            # a large per-tick budget is what keeps the forward batch size
            # near ``max_batch_size`` under bursty arrival — otherwise the
            # engine processes small slices (B=1 paths dominate) and
            # throughput drops several-fold vs the in-process engine.
            if not self._engine_has_work():
                # Idle — block briefly waiting for the first command.
                events = self._poller.poll(timeout=self.opt.poll_block_ms)
                if not events:
                    self._maybe_emit_overloaded()
                    continue
            for _ in range(self.opt.max_inbound_per_tick):
                try:
                    frames = self._sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                self._dispatch_frames(frames)

            try:
                outputs = self.engine.step()
            except Exception:  # noqa: BLE001 — fatal
                logger.exception("fatal error in engine.step(); aborting in-flight")
                self._fail_all_in_flight("engine.step crashed")
                return 1
            if outputs:
                self._publish_outputs(outputs)
            self._maybe_emit_overloaded()
        return 0

    # ----- two-thread mode --------------------------------------------

    def _run_two_thread(self) -> int:
        import zmq

        # I/O thread drains the outbound queue + handles inbound.  Step thread
        # drives engine.step().
        rc: Dict[str, int] = {"value": 0}

        def step_thread():
            while not self._shutdown.is_set():
                if not self._engine_has_work():
                    time.sleep(0.0005)
                    continue
                try:
                    outputs = self.engine.step()
                except Exception:
                    logger.exception("fatal error in engine.step(); aborting in-flight")
                    self._fail_all_in_flight("engine.step crashed")
                    rc["value"] = 1
                    self._shutdown.set()
                    return
                if outputs:
                    self._publish_outputs(outputs)
                self._maybe_emit_overloaded()

        st = threading.Thread(target=step_thread, name="oasr-step", daemon=True)
        st.start()

        try:
            while not self._shutdown.is_set():
                events = self._poller.poll(timeout=self.opt.poll_block_ms)
                # Drain inbound.
                if events:
                    for _ in range(self.opt.max_inbound_per_tick):
                        try:
                            frames = self._sock.recv_multipart(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        self._dispatch_frames(frames)
                # Drain outbound.
                while True:
                    try:
                        out = self._outbound.get_nowait()
                    except queue.Empty:
                        break
                    self._sock.send_multipart(out)
        finally:
            self._shutdown.set()
            st.join(timeout=5.0)

        return rc["value"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch_frames(self, frames: List[bytes]) -> None:
        try:
            msg = ipc.parse_incoming(frames)
        except Exception as e:  # noqa: BLE001
            logger.warning("malformed inbound frames: %r", e)
            return
        self.handle_cmd(msg)

    def _engine_has_work(self) -> bool:
        return (self.engine.num_running + self.engine.num_waiting) > 0

    def _fail_all_in_flight(self, message: str) -> None:
        for rid, identity in list(self._rid_to_identity.items()):
            try:
                self._sock.send_multipart(ipc.build_outgoing(
                    identity, ipc.make_error(rid, ipc.ERR_INTERNAL, message),
                ))
            except Exception:  # noqa: BLE001 — best effort
                pass
            self._rid_to_identity.pop(rid, None)

    def _cleanup(self) -> None:
        # Abort any in-flight requests gracefully.
        for rid in list(self._rid_to_identity.keys()):
            try:
                self.engine.abort_request(rid)
            except Exception:  # noqa: BLE001
                pass
            try:
                identity = self._rid_to_identity.pop(rid, None)
                if identity is not None:
                    self._sock.send_multipart(ipc.build_outgoing(
                        identity, ipc.make_error(rid, ipc.ERR_SHUTDOWN, "worker shutting down"),
                    ))
            except Exception:  # noqa: BLE001
                pass
        try:
            self._sock.close(linger=0)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Module entrypoint — `python -m oasr.serving`
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    opts = WorkerOptions.from_argv(argv)
    logging.basicConfig(
        level=opts.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    worker = EngineWorker(opts)
    worker.announce_ready()
    return worker.run()
