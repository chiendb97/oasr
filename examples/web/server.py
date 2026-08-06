#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Same-origin web bridge for a browser ASR demo against the OASR server.

A browser cannot talk to the OASR server directly: the server speaks gRPC
(``oasr.speech.v1.Speech``) and a no-CORS HTTP endpoint, neither of which a page
can reach cross-origin.  This FastAPI app serves the static demo page *and*
relays to OASR over gRPC, so everything is same-origin.

Two transcription modes are exposed to the page:

* ``POST /api/recognize`` — offline.  Body is raw little-endian f32 mono PCM,
  ``?sample_rate=`` query param.  If ``--offline-addr`` equals
  ``--streaming-addr`` (one streaming server runs everything) the whole clip is
  pushed through ``StreamingRecognize`` and the final transcript returned;
  otherwise the unary ``Recognize`` RPC is used against the offline server.
* ``WS /api/stream`` — streaming.  The page sends a JSON config frame, then
  binary f32 PCM chunks, then ``{"type":"eof"}``; the bridge drives
  ``StreamingRecognize`` and relays ``partial`` / ``final`` results back.

Run::

    pip install fastapi "uvicorn[standard]" grpcio grpcio-tools
    python examples/web/server.py            # http://127.0.0.1:8000

See ``examples/web/README.md`` for the full setup.

Logging
-------

Every hop is traced: browser request in → gRPC call out → each audio chunk →
each interim/final result → response out.  All lines belonging to one request
carry the same correlation id (``[http-1a2b3c4d]`` / ``[ws-…]``), which is also
returned to the page as the ``x-oasr-trace`` response header, and every upstream
line carries OASR's own ``rid=`` so bridge logs join against server logs.

``--log-level info`` (default) gives one line per lifecycle event; ``debug``
adds per-chunk and per-partial detail plus uvicorn's access log::

    python examples/web/server.py --log-level debug --log-file /tmp/bridge.log
"""

import argparse
import asyncio
import importlib
import json
import logging
import sys
import tempfile
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import grpc

# ---------------------------------------------------------------------------
# Configuration (populated by main(), read by the request handlers)
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
DEFAULT_PROTO = HERE.parents[1] / "rust" / "proto" / "oasr_speech_v1.proto"
# Serve the repo's canonical logo rather than duplicating the PNG under static/.
LOGO_PATH = HERE.parents[1] / "docs" / "assets" / "logos" / "oasr-logo-text.png"
LOGO_ONLY_PATH = HERE.parents[1] / "docs" / "assets" / "logos" / "oasr-logo-only.png"

# Bypass any ``http_proxy``/``HTTPS_PROXY`` so we reach the loopback server
# directly, mirroring examples/recognize/grpc_client.py.
GRPC_OPTS = [("grpc.enable_http_proxy", 0)]


@dataclass
class BridgeConfig:
    streaming_addr: str = "127.0.0.1:50051"
    offline_addr: str = "127.0.0.1:50051"
    sample_rate: int = 16000
    chunk_ms: int = 320
    proto: Path = DEFAULT_PROTO

    @property
    def offline_via_stream(self) -> bool:
        """Offline goes through StreamingRecognize when both addrs match."""
        return self.offline_addr == self.streaming_addr


CFG = BridgeConfig()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG = logging.getLogger("oasr.webdemo")
LOG_HTTP = LOG.getChild("http")  # browser <-> bridge, offline path
LOG_WS = LOG.getChild("ws")  # browser <-> bridge, streaming path
LOG_GRPC = LOG.getChild("grpc")  # bridge <-> oasr-server
LOG_PROTO = LOG.getChild("proto")  # stub compilation

#: Correlation id of the in-flight request, stamped onto every log record by
#: ``_TraceFilter``.  A ContextVar rather than a parameter so the nested request
#: generator and the relay task inherit it without threading it through calls.
_TRACE: ContextVar[str] = ContextVar("oasr_webdemo_trace", default="-")

_LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(trace)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


class _TraceFilter(logging.Filter):
    """Stamp each record with the current correlation id (``-`` when idle)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace = _TRACE.get()
        return True


def setup_logging(level: str, log_file: Optional[str] = None) -> None:
    """Install our handlers on the root logger; uvicorn/grpc propagate into them."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    trace = _TraceFilter()

    root = logging.getLogger()
    for stale in root.handlers[:]:
        root.removeHandler(stale)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(fmt)
        handler.addFilter(trace)  # on the handler: covers uvicorn's records too
        root.addHandler(handler)
    root.setLevel(lvl)

    # uvicorn is started with ``log_config=None`` so it installs no handlers of
    # its own and its records land here instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi"):
        uvi = logging.getLogger(name)
        uvi.handlers.clear()
        uvi.propagate = True
    # grpc's transport chatter is noise unless something is badly wrong.
    logging.getLogger("grpc").setLevel(max(lvl, logging.WARNING))


def _trace_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _audio_seconds(nbytes: int, sample_rate: int) -> float:
    """Wall-clock duration of ``nbytes`` of f32 mono PCM."""
    return (nbytes // 4) / sample_rate if sample_rate > 0 else 0.0


def _audio_desc(nbytes: int, sample_rate: int) -> str:
    return f"{nbytes} B / {nbytes // 4} frames / {_audio_seconds(nbytes, sample_rate):.2f}s"


def _speedup(nbytes: int, sample_rate: int, elapsed_ms: float) -> float:
    """Audio seconds per wall second — the demo's end-to-end RTFx."""
    return _audio_seconds(nbytes, sample_rate) / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0


def _preview(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _rpc_error(exc: grpc.aio.AioRpcError) -> str:
    code = exc.code().name if exc.code() is not None else "UNKNOWN"
    return f"{code}: {exc.details()}"


@dataclass
class _StreamStats:
    """Counters for one ``WS /api/stream`` session, shared by its three tasks."""

    chunks_in: int = 0  # browser -> bridge
    bytes_in: int = 0
    chunks_sent: int = 0  # bridge -> oasr-server
    bytes_sent: int = 0
    partials: int = 0
    finals: int = 0
    first_response_ms: Optional[float] = None
    request_id: str = ""


# ---------------------------------------------------------------------------
# gRPC stub generation (compiled from the proto at runtime, like grpc_client.py)
# ---------------------------------------------------------------------------

_PB = None  # (pb, pb_grpc), memoized


def get_pb():
    """Compile ``oasr_speech_v1.proto`` and import the generated modules once."""
    global _PB
    if _PB is not None:
        LOG_PROTO.debug("stub cache hit")
        return _PB
    try:
        import grpc_tools
        from grpc_tools import protoc
    except ImportError as exc:  # pragma: no cover - dependency hint
        LOG_PROTO.error("grpcio-tools is not importable: %s", exc)
        raise SystemExit(
            "missing grpcio-tools (install with `pip install grpcio grpcio-tools`)"
        ) from exc

    proto = CFG.proto
    if not proto.is_file():
        LOG_PROTO.error("proto file not found: %s", proto)
        raise SystemExit(f"proto file not found: {proto}")

    # grpcio-tools bundles the well-known protos (google/protobuf/*.proto) under
    # its _proto dir; add it to the import path so duration.proto resolves.
    well_known = Path(grpc_tools.__file__).parent / "_proto"

    out = Path(tempfile.mkdtemp(prefix="oasr-webdemo-grpc-"))
    LOG_PROTO.info("compiling %s -> %s", proto, out)
    t0 = time.perf_counter()
    rc = protoc.main([
        "protoc",
        f"--proto_path={proto.parent}",
        f"--proto_path={well_known}",
        f"--python_out={out}",
        f"--grpc_python_out={out}",
        str(proto),
    ])
    if rc != 0:
        LOG_PROTO.error("protoc failed with rc=%d for %s", rc, proto)
        raise SystemExit(f"protoc failed with rc={rc}")
    sys.path.insert(0, str(out))
    pb = importlib.import_module("oasr_speech_v1_pb2")
    pb_grpc = importlib.import_module("oasr_speech_v1_pb2_grpc")
    _PB = (pb, pb_grpc)
    LOG_PROTO.info("stubs ready in %.0f ms", _ms(t0))
    return _PB


def _recognition_config(pb, sample_rate: int):
    return pb.RecognitionConfig(
        encoding=pb.RecognitionConfig.LINEAR32F,  # raw little-endian f32 mono
        sample_rate_hertz=sample_rate,
        language_code="en-US",
        max_alternatives=1,
    )


def _top_transcript(result) -> str:
    return result.alternatives[0].transcript if result.alternatives else ""


# ---------------------------------------------------------------------------
# Offline transcription
# ---------------------------------------------------------------------------


async def _offline_unary(raw: bytes, sample_rate: int) -> Tuple[str, str]:
    """Unary Recognize against a dedicated offline-mode server."""
    pb, pb_grpc = get_pb()
    LOG_GRPC.info(
        "offline/unary -> %s | %s @ %d Hz",
        CFG.offline_addr, _audio_desc(len(raw), sample_rate), sample_rate,
    )
    t0 = time.perf_counter()
    async with grpc.aio.insecure_channel(CFG.offline_addr, options=GRPC_OPTS) as ch:
        LOG_GRPC.debug("channel up to %s (%.1f ms)", CFG.offline_addr, _ms(t0))
        stub = pb_grpc.SpeechStub(ch)
        req = pb.RecognizeRequest(
            config=_recognition_config(pb, sample_rate),
            audio=pb.RecognitionAudio(content=raw),
        )
        LOG_GRPC.debug("Recognize sent (timeout=120s)")
        resp = await stub.Recognize(req, timeout=120.0)
    elapsed = _ms(t0)
    for i, result in enumerate(resp.results):
        # NB: the unary SpeechRecognitionResult has no ``is_final`` — that field
        # only exists on the streaming variant.
        LOG_GRPC.debug(
            "result[%d] alts=%d finish=%r: %r",
            i, len(result.alternatives), result.finish_reason,
            _preview(_top_transcript(result)),
        )
    text = " ".join(_top_transcript(r) for r in resp.results).strip()
    LOG_GRPC.info(
        "offline/unary <- rid=%s %d result(s) in %.1f ms (%.1fx RT), %d chars: %r",
        resp.request_id or "?", len(resp.results), elapsed,
        _speedup(len(raw), sample_rate, elapsed), len(text), _preview(text),
    )
    return text, resp.request_id


async def _offline_via_stream(raw: bytes, sample_rate: int) -> Tuple[str, str]:
    """Push the whole clip through StreamingRecognize, return the final text.

    Partials/finals from the OASR streaming pipeline are cumulative for a single
    utterance, so the last final (or the last partial if none closed) carries the
    complete transcript.
    """
    pb, pb_grpc = get_pb()
    chunk_bytes = max(4, int(sample_rate * CFG.chunk_ms / 1000) * 4)
    n_chunks = max(1, -(-len(raw) // chunk_bytes))
    trace = _TRACE.get()
    LOG_GRPC.info(
        "offline/stream -> %s | %s @ %d Hz as %d chunk(s) of %d B",
        CFG.streaming_addr, _audio_desc(len(raw), sample_rate), sample_rate,
        n_chunks, chunk_bytes,
    )

    async def requests():
        # The generator is driven by grpc's own task; re-stamp the trace id so
        # its lines stay attributable to this request.
        _TRACE.set(trace)
        LOG_GRPC.debug("config frame sent (interim_results=False)")
        yield pb.StreamingRecognizeRequest(
            streaming_config=pb.StreamingRecognitionConfig(
                config=_recognition_config(pb, sample_rate),
                interim_results=False,
            )
        )
        sent = 0
        for i in range(0, len(raw), chunk_bytes):
            piece = raw[i:i + chunk_bytes]
            sent += len(piece)
            LOG_GRPC.debug(
                "-> chunk %d/%d: %d B (%d/%d B sent)",
                i // chunk_bytes + 1, n_chunks, len(piece), sent, len(raw),
            )
            yield pb.StreamingRecognizeRequest(audio_content=piece)
        LOG_GRPC.debug("half-close after %d B", sent)

    final_text, partial_text, rid = "", "", ""
    n_partial = n_final = 0
    first_ms: Optional[float] = None
    t0 = time.perf_counter()
    async with grpc.aio.insecure_channel(CFG.streaming_addr, options=GRPC_OPTS) as ch:
        LOG_GRPC.debug("channel up to %s (%.1f ms)", CFG.streaming_addr, _ms(t0))
        stub = pb_grpc.SpeechStub(ch)
        async for resp in stub.StreamingRecognize(requests()):
            if first_ms is None:
                first_ms = _ms(t0)
                LOG_GRPC.debug("first response after %.1f ms", first_ms)
            rid = resp.request_id or rid
            for result in resp.results:
                text = _top_transcript(result)
                partial_text = text
                if result.is_final:
                    final_text = text
                    n_final += 1
                    LOG_GRPC.debug("<- final #%d (%d chars): %r", n_final, len(text),
                                   _preview(text))
                else:
                    n_partial += 1
                    LOG_GRPC.debug("<- partial #%d (%d chars): %r", n_partial, len(text),
                                   _preview(text))
    elapsed = _ms(t0)
    out = (final_text or partial_text).strip()
    LOG_GRPC.info(
        "offline/stream <- rid=%s %d partial / %d final in %.1f ms "
        "(first %s, %.1fx RT), %d chars: %r",
        rid or "?", n_partial, n_final, elapsed,
        f"{first_ms:.1f} ms" if first_ms is not None else "n/a",
        _speedup(len(raw), sample_rate, elapsed), len(out), _preview(out),
    )
    if not final_text and partial_text:
        LOG_GRPC.warning("stream closed with no final result; using last partial")
    return out, rid


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def build_app():
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="OASR web demo bridge")
    # Same-origin in practice; allow-all keeps the demo flexible if the page is
    # served from elsewhere.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def trace_http(request: Request, call_next):
        """Open a correlation id for every HTTP request and time it end to end."""
        trace = _trace_id("http")
        token = _TRACE.set(trace)
        t0 = time.perf_counter()
        path = request.url.path
        # /api/* is the interesting traffic; static assets stay at DEBUG so INFO
        # reads as one line per real request.
        api = path.startswith("/api/")
        client = f"{request.client.host}:{request.client.port}" if request.client else "?"
        LOG_HTTP.log(logging.INFO if api else logging.DEBUG,
                     "-> %s %s from %s", request.method, path, client)
        try:
            response = await call_next(request)
        except Exception:
            LOG_HTTP.exception("xx %s %s raised after %.1f ms", request.method, path, _ms(t0))
            raise
        else:
            level = logging.INFO if (api or response.status_code >= 400) else logging.DEBUG
            LOG_HTTP.log(level, "<- %s %s %d in %.1f ms",
                         request.method, path, response.status_code, _ms(t0))
            response.headers["x-oasr-trace"] = trace  # join browser devtools to these logs
            return response
        finally:
            _TRACE.reset(token)

    @app.get("/api/info")
    async def info():
        payload = {
            "streaming_addr": CFG.streaming_addr,
            "offline_addr": CFG.offline_addr,
            "offline_via_stream": CFG.offline_via_stream,
            "sample_rate": CFG.sample_rate,
            "chunk_ms": CFG.chunk_ms,
        }
        LOG_HTTP.debug("info: %s", payload)
        return payload

    @app.post("/api/recognize")
    async def recognize(request: Request):
        t0 = time.perf_counter()
        raw_rate = request.query_params.get("sample_rate", CFG.sample_rate)
        try:
            sample_rate = int(raw_rate)
        except ValueError:
            LOG_HTTP.warning("rejected: bad sample_rate=%r", raw_rate)
            return JSONResponse(status_code=400, content={"error": "bad sample_rate"})
        raw = await request.body()
        LOG_HTTP.info("offline request: %s @ %d Hz (body read in %.1f ms), route=%s",
                      _audio_desc(len(raw), sample_rate), sample_rate, _ms(t0),
                      "stream(shared)" if CFG.offline_via_stream else "unary")
        if not raw:
            LOG_HTTP.warning("rejected: empty audio body")
            return JSONResponse(status_code=400, content={"error": "empty audio body"})
        try:
            if CFG.offline_via_stream:
                text, rid = await _offline_via_stream(raw, sample_rate)
            else:
                text, rid = await _offline_unary(raw, sample_rate)
        except grpc.aio.AioRpcError as exc:
            LOG_HTTP.error("upstream failed after %.1f ms: %s | debug=%s",
                           _ms(t0), _rpc_error(exc), exc.debug_error_string())
            return JSONResponse(
                status_code=502,
                content={"error": f"{exc.code().name}: {exc.details()}"},
            )
        elapsed = _ms(t0)
        LOG_HTTP.info("offline done rid=%s in %.1f ms (%.1fx RT), %d chars: %r",
                      rid or "?", elapsed, _speedup(len(raw), sample_rate, elapsed),
                      len(text), _preview(text))
        return {"transcript": text, "request_id": rid}

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket):
        trace = _trace_id("ws")
        token = _TRACE.set(trace)
        t_open = time.perf_counter()
        client = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
        stats = _StreamStats()
        sample_rate = CFG.sample_rate
        try:
            await ws.accept()
            LOG_WS.info("open from %s", client)
            pb, pb_grpc = get_pb()

            # First frame: JSON config.
            try:
                cfg = json.loads(await ws.receive_text())
            except (WebSocketDisconnect, json.JSONDecodeError) as exc:
                LOG_WS.warning("no usable config frame (%s: %s); closing",
                               type(exc).__name__, exc)
                await ws.close()
                return
            sample_rate = int(cfg.get("sample_rate", CFG.sample_rate))
            interim = bool(cfg.get("interim", True))
            LOG_WS.info("config: sample_rate=%d interim=%s -> %s",
                        sample_rate, interim, CFG.streaming_addr)

            audio_q: asyncio.Queue = asyncio.Queue()

            async def requests():
                # Driven by grpc's task; re-stamp the trace id (see above).
                _TRACE.set(trace)
                LOG_GRPC.debug("config frame sent (interim_results=%s)", interim)
                yield pb.StreamingRecognizeRequest(
                    streaming_config=pb.StreamingRecognitionConfig(
                        config=_recognition_config(pb, sample_rate),
                        interim_results=interim,
                    )
                )
                while True:
                    item = await audio_q.get()
                    if item is None:  # sentinel -> half-close the request stream
                        LOG_GRPC.info("half-close: %d chunk(s) / %s forwarded",
                                      stats.chunks_sent,
                                      _audio_desc(stats.bytes_sent, sample_rate))
                        return
                    stats.chunks_sent += 1
                    stats.bytes_sent += len(item)
                    LOG_GRPC.debug("-> chunk %d: %d B (total %s, queue=%d)",
                                   stats.chunks_sent, len(item),
                                   _audio_desc(stats.bytes_sent, sample_rate),
                                   audio_q.qsize())
                    yield pb.StreamingRecognizeRequest(audio_content=item)

            t_rpc = time.perf_counter()
            async with grpc.aio.insecure_channel(CFG.streaming_addr, options=GRPC_OPTS) as ch:
                LOG_GRPC.debug("channel up to %s (%.1f ms)", CFG.streaming_addr, _ms(t_rpc))
                stub = pb_grpc.SpeechStub(ch)
                call = stub.StreamingRecognize(requests())
                LOG_GRPC.info("StreamingRecognize opened")

                async def relay():
                    try:
                        async for resp in call:
                            if stats.first_response_ms is None:
                                stats.first_response_ms = _ms(t_rpc)
                                LOG_GRPC.info("first response after %.1f ms (rid=%s)",
                                              stats.first_response_ms, resp.request_id or "?")
                            stats.request_id = resp.request_id or stats.request_id
                            for result in resp.results:
                                text = _top_transcript(result)
                                if result.is_final:
                                    stats.finals += 1
                                    LOG_WS.info("<- final #%d (%d chars): %r",
                                                stats.finals, len(text), _preview(text))
                                else:
                                    stats.partials += 1
                                    LOG_WS.debug("<- partial #%d (%d chars): %r",
                                                 stats.partials, len(text), _preview(text))
                                await ws.send_json({
                                    "type": "final" if result.is_final else "partial",
                                    "transcript": _top_transcript(result),
                                    "request_id": resp.request_id,
                                })
                    except grpc.aio.AioRpcError as exc:
                        LOG_GRPC.error("stream RPC failed after %.1f ms: %s | debug=%s",
                                       _ms(t_rpc), _rpc_error(exc), exc.debug_error_string())
                        await _safe_send(ws, {
                            "type": "error",
                            "error": f"{exc.code().name}: {exc.details()}",
                        })
                    finally:
                        LOG_GRPC.info("relay closed: %d partial / %d final, rid=%s",
                                      stats.partials, stats.finals, stats.request_id or "?")

                relay_task = asyncio.create_task(relay())
                try:
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            LOG_WS.info("client disconnected (code=%s) after %d chunk(s)",
                                        msg.get("code"), stats.chunks_in)
                            break
                        data = msg.get("bytes")
                        if data is not None:
                            stats.chunks_in += 1
                            stats.bytes_in += len(data)
                            LOG_WS.debug("-> audio chunk %d: %d B (total %s)",
                                         stats.chunks_in, len(data),
                                         _audio_desc(stats.bytes_in, sample_rate))
                            await audio_q.put(bytes(data))
                            continue
                        text = msg.get("text")
                        if text is not None:
                            try:
                                payload = json.loads(text)
                            except json.JSONDecodeError:
                                LOG_WS.warning("ignoring non-JSON text frame (%d chars)",
                                               len(text))
                                continue
                            LOG_WS.debug("control frame: %s", payload)
                            if payload.get("type") == "eof":
                                LOG_WS.info("eof after %d chunk(s) / %s",
                                            stats.chunks_in,
                                            _audio_desc(stats.bytes_in, sample_rate))
                                break
                except WebSocketDisconnect:
                    LOG_WS.info("client vanished mid-stream after %d chunk(s)",
                                stats.chunks_in)
                finally:
                    LOG_WS.debug("flushing: sentinel queued, awaiting relay")
                    await audio_q.put(None)  # flush / half-close
                    await relay_task

                await _safe_send(ws, {"type": "done"})
            await _safe_close(ws)
        finally:
            wall = _ms(t_open)
            LOG_WS.info(
                "closed: rid=%s in %d chunk(s) / %s, out %d partial / %d final, "
                "first %s, session %.1f ms (%.1fx RT)",
                stats.request_id or "?", stats.chunks_in,
                _audio_desc(stats.bytes_in, sample_rate), stats.partials, stats.finals,
                f"{stats.first_response_ms:.1f} ms" if stats.first_response_ms is not None
                else "n/a",
                wall, _speedup(stats.bytes_in, sample_rate, wall),
            )
            _TRACE.reset(token)

    @app.get("/oasr-logo-text.png")
    async def logo():
        return FileResponse(str(LOGO_PATH), media_type="image/png")

    @app.get("/oasr-logo-only.png")
    async def logo_only():
        return FileResponse(str(LOGO_ONLY_PATH), media_type="image/png")

    # Static files last so /api/* and the logo route match first.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    LOG.debug("routes mounted (static=%s)", STATIC_DIR)
    return app


async def _safe_send(ws, payload) -> None:
    try:
        await ws.send_json(payload)
    except Exception as exc:
        kind = payload.get("type", "?") if isinstance(payload, dict) else "?"
        LOG_WS.debug("send(%s) dropped, socket already gone: %s: %s",
                     kind, type(exc).__name__, exc)


async def _safe_close(ws) -> None:
    try:
        await ws.close()
    except Exception as exc:
        LOG_WS.debug("close() ignored: %s: %s", type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0", help="bridge bind host")
    p.add_argument("--port", type=int, default=8000, help="bridge bind port")
    p.add_argument("--streaming-addr", default="127.0.0.1:50051",
                   help="gRPC addr of the streaming-mode oasr-server")
    p.add_argument("--offline-addr", default=None,
                   help="gRPC addr of the offline-mode oasr-server "
                        "(default: same as --streaming-addr; offline then runs "
                        "through StreamingRecognize)")
    p.add_argument("--sample-rate", type=int, default=16000,
                   help="default PCM sample rate the page sends (Hz)")
    p.add_argument("--chunk-ms", type=int, default=320,
                   help="chunk size used when streaming offline clips (ms)")
    p.add_argument("--proto", type=Path, default=DEFAULT_PROTO,
                   help="path to oasr_speech_v1.proto")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"],
                   help="bridge log verbosity; debug adds per-chunk / per-partial "
                        "lines and uvicorn's access log")
    p.add_argument("--log-file", default=None,
                   help="also append logs to this file")
    p.add_argument("--ssl-certfile", default=None,
                   help="TLS cert (PEM) to serve over HTTPS. Required for "
                        "microphone access when the page is opened from a "
                        "non-localhost host (browsers gate getUserMedia to "
                        "secure contexts).")
    p.add_argument("--ssl-keyfile", default=None,
                   help="TLS private key (PEM) paired with --ssl-certfile")
    args = p.parse_args(argv)

    setup_logging(args.log_level, args.log_file)

    CFG.streaming_addr = args.streaming_addr
    CFG.offline_addr = args.offline_addr or args.streaming_addr
    CFG.sample_rate = args.sample_rate
    CFG.chunk_ms = args.chunk_ms
    CFG.proto = args.proto
    LOG.info("config: streaming=%s offline=%s (%s) sample_rate=%d Hz chunk_ms=%d proto=%s",
             CFG.streaming_addr, CFG.offline_addr,
             "via StreamingRecognize" if CFG.offline_via_stream else "via unary Recognize",
             CFG.sample_rate, CFG.chunk_ms, CFG.proto)

    if not STATIC_DIR.is_dir():
        LOG.error("static dir not found: %s", STATIC_DIR)
        return 1

    get_pb()  # fail fast if protoc / proto is unavailable

    try:
        import uvicorn
    except ImportError:
        LOG.error('missing fastapi/uvicorn (install with `pip install fastapi '
                  '"uvicorn[standard]"`)')
        return 1

    mode = ("offline=stream(shared)" if CFG.offline_via_stream
            else f"offline=unary({CFG.offline_addr})")
    scheme = "https" if args.ssl_certfile else "http"
    LOG.info("OASR web demo on %s://%s:%d  [streaming=%s %s]",
             scheme, args.host, args.port, CFG.streaming_addr, mode)
    if scheme == "http" and args.host not in ("127.0.0.1", "localhost"):
        LOG.warning("microphone needs a secure context. Open via http://localhost "
                    "(e.g. `ssh -L %d:localhost:%d <host>`), or pass "
                    "--ssl-certfile/--ssl-keyfile to serve HTTPS. File upload "
                    "works either way.", args.port, args.port)
    # log_config=None: keep the handlers setup_logging() installed; uvicorn's own
    # loggers propagate into them.  Its access log only adds noise next to our
    # richer per-request pair, so it is reserved for --log-level debug.
    uvicorn.run(build_app(), host=args.host, port=args.port,
                log_level=args.log_level, log_config=None,
                access_log=(args.log_level == "debug"),
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
