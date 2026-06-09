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
"""

import argparse
import asyncio
import importlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import grpc

# ---------------------------------------------------------------------------
# Configuration (populated by main(), read by the request handlers)
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
DEFAULT_PROTO = HERE.parents[1] / "rust" / "proto" / "oasr_speech_v1.proto"
# Serve the repo's canonical logo rather than duplicating the PNG under static/.
LOGO_PATH = HERE.parents[1] / "docs" / "assets" / "logos" / "oasr-logo-text.png"

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
# gRPC stub generation (compiled from the proto at runtime, like grpc_client.py)
# ---------------------------------------------------------------------------

_PB = None  # (pb, pb_grpc), memoized


def get_pb():
    """Compile ``oasr_speech_v1.proto`` and import the generated modules once."""
    global _PB
    if _PB is not None:
        return _PB
    try:
        import grpc_tools
        from grpc_tools import protoc
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "missing grpcio-tools (install with `pip install grpcio grpcio-tools`)"
        ) from exc

    proto = CFG.proto
    if not proto.is_file():
        raise SystemExit(f"proto file not found: {proto}")

    # grpcio-tools bundles the well-known protos (google/protobuf/*.proto) under
    # its _proto dir; add it to the import path so duration.proto resolves.
    well_known = Path(grpc_tools.__file__).parent / "_proto"

    out = Path(tempfile.mkdtemp(prefix="oasr-webdemo-grpc-"))
    rc = protoc.main([
        "protoc",
        f"--proto_path={proto.parent}",
        f"--proto_path={well_known}",
        f"--python_out={out}",
        f"--grpc_python_out={out}",
        str(proto),
    ])
    if rc != 0:
        raise SystemExit(f"protoc failed with rc={rc}")
    sys.path.insert(0, str(out))
    pb = importlib.import_module("oasr_speech_v1_pb2")
    pb_grpc = importlib.import_module("oasr_speech_v1_pb2_grpc")
    _PB = (pb, pb_grpc)
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
    async with grpc.aio.insecure_channel(CFG.offline_addr, options=GRPC_OPTS) as ch:
        stub = pb_grpc.SpeechStub(ch)
        req = pb.RecognizeRequest(
            config=_recognition_config(pb, sample_rate),
            audio=pb.RecognitionAudio(content=raw),
        )
        resp = await stub.Recognize(req, timeout=120.0)
    text = " ".join(_top_transcript(r) for r in resp.results).strip()
    return text, resp.request_id


async def _offline_via_stream(raw: bytes, sample_rate: int) -> Tuple[str, str]:
    """Push the whole clip through StreamingRecognize, return the final text.

    Partials/finals from the OASR streaming pipeline are cumulative for a single
    utterance, so the last final (or the last partial if none closed) carries the
    complete transcript.
    """
    pb, pb_grpc = get_pb()
    chunk_bytes = max(4, int(sample_rate * CFG.chunk_ms / 1000) * 4)

    async def requests():
        yield pb.StreamingRecognizeRequest(
            streaming_config=pb.StreamingRecognitionConfig(
                config=_recognition_config(pb, sample_rate),
                interim_results=False,
            )
        )
        for i in range(0, len(raw), chunk_bytes):
            yield pb.StreamingRecognizeRequest(audio_content=raw[i:i + chunk_bytes])

    final_text, partial_text, rid = "", "", ""
    async with grpc.aio.insecure_channel(CFG.streaming_addr, options=GRPC_OPTS) as ch:
        stub = pb_grpc.SpeechStub(ch)
        async for resp in stub.StreamingRecognize(requests()):
            rid = resp.request_id or rid
            for result in resp.results:
                text = _top_transcript(result)
                partial_text = text
                if result.is_final:
                    final_text = text
    return (final_text or partial_text).strip(), rid


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

    @app.get("/api/info")
    async def info():
        return {
            "streaming_addr": CFG.streaming_addr,
            "offline_addr": CFG.offline_addr,
            "offline_via_stream": CFG.offline_via_stream,
            "sample_rate": CFG.sample_rate,
            "chunk_ms": CFG.chunk_ms,
        }

    @app.post("/api/recognize")
    async def recognize(request: Request):
        try:
            sample_rate = int(request.query_params.get("sample_rate", CFG.sample_rate))
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "bad sample_rate"})
        raw = await request.body()
        if not raw:
            return JSONResponse(status_code=400, content={"error": "empty audio body"})
        try:
            if CFG.offline_via_stream:
                text, rid = await _offline_via_stream(raw, sample_rate)
            else:
                text, rid = await _offline_unary(raw, sample_rate)
        except grpc.aio.AioRpcError as exc:
            return JSONResponse(
                status_code=502,
                content={"error": f"{exc.code().name}: {exc.details()}"},
            )
        return {"transcript": text, "request_id": rid}

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        pb, pb_grpc = get_pb()

        # First frame: JSON config.
        try:
            cfg = json.loads(await ws.receive_text())
        except (WebSocketDisconnect, json.JSONDecodeError):
            await ws.close()
            return
        sample_rate = int(cfg.get("sample_rate", CFG.sample_rate))
        interim = bool(cfg.get("interim", True))

        audio_q: asyncio.Queue = asyncio.Queue()

        async def requests():
            yield pb.StreamingRecognizeRequest(
                streaming_config=pb.StreamingRecognitionConfig(
                    config=_recognition_config(pb, sample_rate),
                    interim_results=interim,
                )
            )
            while True:
                item = await audio_q.get()
                if item is None:  # sentinel -> half-close the request stream
                    return
                yield pb.StreamingRecognizeRequest(audio_content=item)

        async with grpc.aio.insecure_channel(CFG.streaming_addr, options=GRPC_OPTS) as ch:
            stub = pb_grpc.SpeechStub(ch)
            call = stub.StreamingRecognize(requests())

            async def relay():
                try:
                    async for resp in call:
                        for result in resp.results:
                            await ws.send_json({
                                "type": "final" if result.is_final else "partial",
                                "transcript": _top_transcript(result),
                                "request_id": resp.request_id,
                            })
                except grpc.aio.AioRpcError as exc:
                    await _safe_send(ws, {
                        "type": "error",
                        "error": f"{exc.code().name}: {exc.details()}",
                    })

            relay_task = asyncio.create_task(relay())
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    data = msg.get("bytes")
                    if data is not None:
                        await audio_q.put(bytes(data))
                        continue
                    text = msg.get("text")
                    if text is not None:
                        try:
                            payload = json.loads(text)
                        except json.JSONDecodeError:
                            continue
                        if payload.get("type") == "eof":
                            break
            except WebSocketDisconnect:
                pass
            finally:
                await audio_q.put(None)  # flush / half-close
                await relay_task

            await _safe_send(ws, {"type": "done"})
        await _safe_close(ws)

    @app.get("/oasr-logo-text.png")
    async def logo():
        return FileResponse(str(LOGO_PATH), media_type="image/png")

    # Static files last so /api/* and the logo route match first.
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


async def _safe_send(ws, payload) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        pass


async def _safe_close(ws) -> None:
    try:
        await ws.close()
    except Exception:
        pass


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
    p.add_argument("--ssl-certfile", default=None,
                   help="TLS cert (PEM) to serve over HTTPS. Required for "
                        "microphone access when the page is opened from a "
                        "non-localhost host (browsers gate getUserMedia to "
                        "secure contexts).")
    p.add_argument("--ssl-keyfile", default=None,
                   help="TLS private key (PEM) paired with --ssl-certfile")
    args = p.parse_args(argv)

    CFG.streaming_addr = args.streaming_addr
    CFG.offline_addr = args.offline_addr or args.streaming_addr
    CFG.sample_rate = args.sample_rate
    CFG.chunk_ms = args.chunk_ms
    CFG.proto = args.proto

    if not STATIC_DIR.is_dir():
        print(f"static dir not found: {STATIC_DIR}", file=sys.stderr)
        return 1

    get_pb()  # fail fast if protoc / proto is unavailable

    try:
        import uvicorn
    except ImportError:
        print('missing fastapi/uvicorn (install with `pip install fastapi '
              '"uvicorn[standard]"`)', file=sys.stderr)
        return 1

    mode = ("offline=stream(shared)" if CFG.offline_via_stream
            else f"offline=unary({CFG.offline_addr})")
    scheme = "https" if args.ssl_certfile else "http"
    print(f"OASR web demo on {scheme}://{args.host}:{args.port}  "
          f"[streaming={CFG.streaming_addr} {mode}]")
    if scheme == "http" and args.host not in ("127.0.0.1", "localhost"):
        print("note: microphone needs a secure context. Open via "
              "http://localhost (e.g. `ssh -L {p}:localhost:{p} <host>`), or "
              "pass --ssl-certfile/--ssl-keyfile to serve HTTPS. File upload "
              "works either way.".format(p=args.port), file=sys.stderr)
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="info",
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
