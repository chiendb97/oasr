# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Servers over the network: the OpenAI-compatible surface, and ``oasr-server``.

``openai_http`` measures any server implementing ``POST /v1/audio/transcriptions``
(multipart upload) -- the de-facto common API for ASR serving (OASR, vLLM's
Whisper/Qwen2-Audio serving, speaches/faster-whisper-server, and others).  It is
the fairest cross-framework seam available: the same request bytes, the same
transport, measured by the same client.  Streaming uses the OpenAI *realtime
transcription* WebSocket protocol when a server speaks it (``realtime=true``);
whether that stream is incremental or buffered is a property of the server the
client cannot observe, so it is declared (``streaming_mode``) and recorded.

``oasr_server`` is the same client plus process management: it spawns
``oasr-server`` in the requested service mode on free ports, waits for
``/readyz``, monitors the *server's* PID for resources, and scrapes its
Prometheus ``/metrics`` around each window for server-side queue wait and batch
width.  Requests carry a WAV body encoded before the clock starts.

A server's own batching is driven by concurrency, so neither backend offers the
``batch`` benchmark: a multipart upload is one request.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.asrbench.audio import encode_wav, to_pcm16
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

#: The dispatcher drains engine metrics into the exporter every 250 ms.
METRICS_SETTLE_S = 0.3


def parse_prometheus(text: str) -> Dict[str, float]:
    """Sum every series of each ``*_sum`` / ``*_count`` / counter, across labels."""
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_part, _, value = line.rpartition(" ")
        name = name_part.split("{", 1)[0].strip()
        try:
            v = float(value)
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + v
    return out


class _WsStream(StreamSession):
    """One OpenAI-realtime transcription session over a WebSocket."""

    def __init__(self, ws: Any) -> None:
        super().__init__()
        self.ws = ws
        self._reader: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._reader = asyncio.get_running_loop().create_task(self._read())

    async def _read(self) -> None:
        try:
            async for message in self.ws:
                t = time.perf_counter()
                if isinstance(message, bytes):
                    continue
                ev = json.loads(message)
                kind = ev.get("type", "")
                if kind.endswith("input_audio_transcription.delta"):
                    # ``text`` is OASR's extension (full transcript so far); a
                    # server without it gets the deltas accumulated.
                    prev = next((e.text for e in reversed(self.events) if e.kind == "partial"), "")
                    self.push(
                        StreamEvent("partial", t, ev.get("text") or prev + ev.get("delta", ""))
                    )
                elif kind.endswith("input_audio_transcription.completed"):
                    self.push(StreamEvent("final", t, ev.get("transcript", ""), ev.get("words")))
                    return
                elif kind == "error":
                    err = ev.get("error") or {}
                    self.push(StreamEvent("error", t, str(err.get("message", err))))
                    return
        except Exception as exc:  # noqa: BLE001 -- connection dropped
            self.push(StreamEvent("error", time.perf_counter(), f"{type(exc).__name__}: {exc}"))

    async def send(self, chunk: Any, is_last: bool) -> None:
        await self.ws.send(chunk)
        if is_last:
            await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    async def wait_final(self, timeout: Optional[float]) -> StreamEvent:
        try:
            return await super().wait_final(timeout)
        finally:
            await self._close()

    async def abort(self) -> None:
        await self._close()

    async def _close(self) -> None:
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
        try:
            await self.ws.close()
        except Exception:  # noqa: BLE001
            pass


class OpenAIHttpBackend(Backend):
    name = "openai_http"
    description = "Any server with an OpenAI-compatible /v1/audio/transcriptions (+ realtime WS)"
    OPTIONS = {
        "base_url": "server URL, e.g. http://127.0.0.1:8000",
        "model": "model field of the request (server-specific)",
        "language": "language field (only for servers/models that accept one)",
        "response_format": "json | text | verbose_json (json)",
        "api_key": "bearer token; or set api_key_env",
        "api_key_env": "environment variable holding the token (OPENAI_API_KEY)",
        "timeout_s": "per-request HTTP timeout (600)",
        "realtime": "use the /v1/realtime WebSocket for streaming (false)",
        "realtime_path": "realtime endpoint path (/v1/realtime)",
        "streaming_mode": "what the server's stream is: native | buffered (native)",
        "metrics_url": "Prometheus endpoint for server-side metrics (none)",
        "max_connections": "HTTP connection pool size (1024)",
        "trust_env": "honour HTTP(S)_PROXY for the server URL (false)",
        "server_pid": "PID of the server process, for resource monitoring",
    }

    def __init__(self, options: Optional[Dict[str, Any]] = None, mode: str = "offline") -> None:
        super().__init__(options, mode)
        self.base_url = str(self.options.get("base_url") or "").rstrip("/")
        self._client: Any = None
        self._metrics0: Dict[str, float] = {}

    @classmethod
    def available(cls) -> Tuple[bool, str]:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False, "needs httpx (pip install 'oasr[serving]')"
        return True, ""

    @classmethod
    def mode_unsupported(cls, mode: str, options: Dict[str, Any]) -> Optional[str]:
        if mode == "streaming" and not options.get("realtime"):
            return "realtime streaming not enabled (-o realtime=true)"
        return None

    # -- lifecycle --------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        key = self.options.get("api_key") or os.environ.get(
            str(self.options.get("api_key_env") or "OPENAI_API_KEY"), ""
        )
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def load(self) -> None:
        import httpx

        if not self.base_url:
            raise ValueError(f"{self.name} needs -o base_url=http://host:port")
        n = int(self.options.get("max_connections", 1024))
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(float(self.options.get("timeout_s", 600.0))),
            limits=httpx.Limits(max_connections=n, max_keepalive_connections=n),
            headers=self._headers(),
            # The box may export a proxy that does not route to localhost.
            trust_env=bool(self.options.get("trust_env", False)),
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def info(self) -> BackendInfo:
        from benchmarks.asrbench.environment import packages

        return BackendInfo(
            name=self.name,
            version=self._server_version(),
            mode=self.mode,
            model={"id": self.options.get("model") or self._server_model()},
            config={
                "base_url": self.base_url,
                "response_format": self.options.get("response_format", "json"),
                "wire": "multipart WAV PCM16",
                "client": {
                    "httpx": packages().get("httpx", ""),
                    "websockets": packages().get("websockets", ""),
                },
            },
            device="remote",
        )

    def _server_version(self) -> str:
        return ""

    def _server_model(self) -> str:
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/models", headers=self._headers())
            with _no_proxy_opener().open(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            models = [m.get("id") for m in data.get("data", [])]
            return ",".join(str(m) for m in models if m)
        except Exception:  # noqa: BLE001 -- informational only
            return ""

    def capabilities(self) -> Capabilities:
        realtime = bool(self.options.get("realtime", False))
        mode = str(self.options.get("streaming_mode", "native"))
        return Capabilities(
            offline=True,
            batch=False,
            concurrent=True,
            streaming=mode if realtime else None,
            partials=realtime and mode == "native",
            word_timestamps=False,
            long_form="whole",
            in_process=False,
            server_metrics=bool(self.options.get("metrics_url")),
            notes="long_form is assumed; a server that bounds input reports failures",
        )

    # -- inputs ---------------------------------------------------------------------

    def prepare(self, audio: np.ndarray, item: WorkItem) -> bytes:
        return encode_wav(audio, self.sample_rate)

    def prepare_chunk(self, chunk: np.ndarray) -> bytes:
        return to_pcm16(chunk).tobytes()

    # -- inference ------------------------------------------------------------------

    def _form(self) -> Dict[str, str]:
        form = {"response_format": str(self.options.get("response_format", "json"))}
        for key in ("model", "language"):
            if self.options.get(key):
                form[key] = str(self.options[key])
        return form

    async def transcribe(self, payload: Any) -> Transcript:
        import httpx

        t_submit = time.perf_counter()
        try:
            resp = await self._client.post(
                f"{self.base_url}/v1/audio/transcriptions",
                data=self._form(),
                files={"file": ("audio.wav", payload, "audio/wav")},
            )
        except httpx.TimeoutException as exc:
            raise asyncio.TimeoutError(str(exc)) from None
        except httpx.HTTPError as exc:
            raise RequestFailed(f"transport: {type(exc).__name__}: {exc}") from None
        t_done = time.perf_counter()
        if resp.status_code in (429, 503):
            raise RequestFailed(
                f"HTTP {resp.status_code}", kind="rejected", status=resp.status_code
            )
        if resp.status_code != 200:
            raise RequestFailed(
                f"HTTP {resp.status_code}: {resp.text[:200]}", status=resp.status_code
            )
        if "json" in resp.headers.get("content-type", ""):
            body = resp.json()
            text = body.get("text", "")
            words = body.get("words")
        else:
            text, words = resp.text.strip(), None
        return Transcript(text=text, t_submit=t_submit, t_done=t_done, words=words)

    async def open_stream(self, *, word_timestamps: bool = False) -> StreamSession:
        if not self.options.get("realtime"):
            raise Unsupported(f"{self.name}: realtime streaming not enabled (-o realtime=true)")
        if word_timestamps:
            raise Unsupported("word timestamps are not part of the realtime protocol")
        import websockets

        url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        url += str(self.options.get("realtime_path", "/v1/realtime"))
        from oasr.client import _ws_header_kwarg

        ws = await websockets.connect(url, max_size=None, **_ws_header_kwarg(self._headers()))
        session = {"sample_rate": self.sample_rate, "encoding": "LINEAR16", "interim_results": True}
        for key in ("model", "language"):
            if self.options.get(key):
                session[key] = self.options[key]
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        stream = _WsStream(ws)
        stream.start()
        return stream

    # -- observability ---------------------------------------------------------------

    def monitored_pid(self) -> Optional[int]:
        pid = self.options.get("server_pid")
        return int(pid) if pid else None

    def _scrape(self) -> Dict[str, float]:
        url = self.options.get("metrics_url")
        if not url:
            return {}
        try:
            with _no_proxy_opener().open(str(url), timeout=5) as resp:
                return parse_prometheus(resp.read().decode())
        except Exception:  # noqa: BLE001
            return {}

    def server_metrics_begin(self) -> None:
        self._metrics0 = self._scrape()

    def server_metrics_end(self) -> Dict[str, Any]:
        if not self.options.get("metrics_url"):
            return {}
        time.sleep(METRICS_SETTLE_S)
        m1, m0 = self._scrape(), self._metrics0
        out: Dict[str, Any] = {}

        def delta(name: str) -> float:
            return m1.get(name, 0.0) - m0.get(name, 0.0)

        n = delta("oasr_engine_queue_wait_seconds_count")
        if n > 0:
            out["queue_wait_ms.mean"] = 1e3 * delta("oasr_engine_queue_wait_seconds_sum") / n
            out["queue_wait_ms.n"] = n
        b = delta("oasr_engine_batch_size_count")
        if b > 0:
            out["batch_size_observed.mean"] = delta("oasr_engine_batch_size_sum") / b
            out["batch_size_observed.n"] = b
        return out


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


#: ``oasr-server --service-mode`` values.
SERVICE_MODES = ("offline", "streaming")
#: What carries an ``oasr_server`` request: the HTTP/WebSocket front-end or gRPC.
TRANSPORTS = ("http", "grpc")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class OasrServerBackend(OpenAIHttpBackend):
    name = "oasr_server"
    description = "oasr-server, spawned or attached; requests over HTTP/WebSocket or gRPC"
    OPTIONS = {
        **{
            k: v
            for k, v in OpenAIHttpBackend.OPTIONS.items()
            if k not in ("base_url", "realtime", "streaming_mode", "metrics_url", "server_pid")
        },
        "base_url": "attach to an already-running oasr-server instead of spawning one",
        "server_pid": "PID of the attached server, for resource monitoring",
        "transport": "http (/v1/audio/transcriptions + /v1/realtime) | grpc (Recognize + "
        "StreamingRecognize) (http)",
        "grpc_addr": "host:port of an attached server's gRPC endpoint (transport=grpc)",
        "grpc_encoding": "unary gRPC payload: wav (the HTTP upload's bytes) | linear16 | "
        "linear32f (wav)",
        "ckpt_dir": "checkpoint directory (default: $CKPT_DIR)",
        "architecture": "force a registered architecture",
        "dtype": "float16 | bfloat16 | float32 (bfloat16)",
        "max_batch_size": "engine batch width",
        "chunk_size": "streaming encoder chunk, frames",
        "decode_method": "decode family",
        "long_form": "window over-length audio (fixed-window frontends)",
        "server_args": "extra oasr-server flags, YAML list",
        "server_bin": "oasr-server executable ($OASR_RS_BIN, then PATH)",
        "ready_timeout_s": "startup timeout (300)",
        "log_level": "server log level (warn)",
    }

    def __init__(self, options: Optional[Dict[str, Any]] = None, mode: str = "offline") -> None:
        super().__init__(options, mode)
        self.proc: Optional[subprocess.Popen] = None
        self._log: Optional[Any] = None
        #: What the server serves: the requested mode for a spawned server, the
        #: probed ``--service-mode`` for an attached one.  One running server is
        #: one mode, and both instances of a run may share it.
        self.server_mode = mode
        self._attached_model: Dict[str, Any] = {}
        self.transport = str(self.options.get("transport", "http")).lower()
        if self.transport not in TRANSPORTS:
            raise ValueError(f"oasr_server: transport must be one of {TRANSPORTS}")
        self.grpc_encoding = str(self.options.get("grpc_encoding", "wav")).lower()
        from benchmarks.asrbench.backends.grpc_speech import UNARY_ENCODINGS

        if self.grpc_encoding not in UNARY_ENCODINGS:
            raise ValueError(f"oasr_server: grpc_encoding must be one of {list(UNARY_ENCODINGS)}")
        self.grpc_addr = str(self.options.get("grpc_addr") or "")
        self._grpc: Any = None  # (channel, stub, pb)

    @classmethod
    def available(cls) -> Tuple[bool, str]:
        ok, why = OpenAIHttpBackend.available()
        if not ok:
            return ok, why
        try:
            import websockets  # noqa: F401
        except ImportError:
            return False, "needs websockets (pip install 'oasr[serving]')"
        return True, ""

    @classmethod
    def mode_unsupported(cls, mode: str, options: Dict[str, Any]) -> Optional[str]:
        return None

    def _bin(self) -> str:
        cand = (
            self.options.get("server_bin")
            or os.environ.get("OASR_RS_BIN")
            or shutil.which("oasr-server")
        )
        if not cand:
            raise RuntimeError("oasr-server not found: pip install -e . or set OASR_RS_BIN")
        return str(cand)

    def _command(self, http: str, grpc: str) -> List[str]:
        o = self.options
        ckpt = o.get("ckpt_dir") or os.environ.get("CKPT_DIR")
        if not ckpt:
            raise ValueError("oasr_server needs -o ckpt_dir=... (or $CKPT_DIR)")
        cmd = [
            self._bin(),
            "--ckpt-dir",
            str(ckpt),
            "--service-mode",
            self.mode,
            "--dtype",
            str(o.get("dtype", "bfloat16")),
            "--http-bind",
            http,
            "--grpc-bind",
            grpc,
            "--log-level",
            str(o.get("log_level", "warn")),
            # Admission must not be what a load test measures.
            "--max-concurrent-requests",
            "100000",
            "--listen-backlog",
            "4096",
        ]
        for key, flag in (
            ("max_batch_size", "--max-batch-size"),
            ("chunk_size", "--chunk-size"),
            ("decode_method", "--decode-method"),
        ):
            if o.get(key) is not None:
                cmd += [flag, str(o[key])]
        if o.get("long_form"):
            cmd.append("--long-form")
        if o.get("architecture"):
            cmd += ["--engine-config", self._engine_config({"architecture": o["architecture"]})]
        cmd += [str(a) for a in (o.get("server_args") or [])]
        return cmd

    def _engine_config(self, doc: Dict[str, Any]) -> str:
        import tempfile

        fd, path = tempfile.mkstemp(prefix="asrbench_engine_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f)
        return path

    async def load(self) -> None:
        if self.transport == "grpc":
            from benchmarks.asrbench.backends import grpc_speech

            ok, why = grpc_speech.available()
            if not ok:
                raise RuntimeError(f"transport=grpc {why}")
            if self.options.get("base_url") and not self.grpc_addr:
                raise ValueError(
                    "transport=grpc on an attached server needs -o grpc_addr=host:port"
                )
        if self.options.get("base_url"):
            self.base_url = str(self.options["base_url"]).rstrip("/")
            self._attached_model = self._served_model()
            mode = self._attached_model.get("service_mode")
            try:
                # /v1/models carries it when the engine reports it; the realtime
                # handshake always does.
                self.server_mode = mode if mode in SERVICE_MODES else await self._probe_mode()
            except Exception as exc:  # noqa: BLE001 -- name the server, not the socket
                raise RuntimeError(
                    f"cannot attach to oasr-server at {self.base_url}: {exc}"
                ) from exc
        else:
            http = f"127.0.0.1:{_free_port()}"
            grpc = f"127.0.0.1:{_free_port()}"
            self.base_url = f"http://{http}"
            self.grpc_addr = grpc
            cmd = self._command(http, grpc)
            self._log = open(
                Path(os.environ.get("TMPDIR", "/tmp"))
                / f"asrbench-oasr-server-{os.getpid()}-{self.mode}.log",
                "w",
            )
            self.proc = subprocess.Popen(
                cmd, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
            )
            self.options["server_pid"] = self.proc.pid
            deadline = time.monotonic() + float(self.options.get("ready_timeout_s", 300))
            while True:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"oasr-server exited with {self.proc.returncode} during startup "
                        f"(log: {self._log.name})"
                    )
                try:
                    with _no_proxy_opener().open(f"{self.base_url}/readyz", timeout=2) as r:
                        if r.status == 200:
                            break
                except (urllib.error.URLError, ConnectionError, OSError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"oasr-server not ready after the timeout (log: {self._log.name})"
                    )
                await asyncio.sleep(0.5)
        self.options.setdefault("metrics_url", f"{self.base_url}/metrics")
        await super().load()
        if self.transport == "grpc":
            from benchmarks.asrbench.backends import grpc_speech

            pb, pb_grpc = grpc_speech.stubs()
            chan = grpc_speech.channel(self.grpc_addr)
            try:
                await asyncio.wait_for(chan.channel_ready(), 10)
            except asyncio.TimeoutError:
                await chan.close()
                raise RuntimeError(f"no gRPC endpoint at {self.grpc_addr}") from None
            self._grpc = (chan, pb_grpc.SpeechStub(chan), pb)

    async def close(self) -> None:
        if self._grpc is not None:
            await self._grpc[0].close()
            self._grpc = None
        await super().close()
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if self._log is not None:
            self._log.close()

    async def _probe_mode(self) -> str:
        """The attached server's ``--service-mode``, from the realtime handshake.

        The session is configured and closed without audio; this happens in
        ``load()``, outside every timed window.
        """
        import websockets

        from oasr.client import _ws_header_kwarg

        url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        url += str(self.options.get("realtime_path", "/v1/realtime"))
        ws = await asyncio.wait_for(
            websockets.connect(url, max_size=None, **_ws_header_kwarg(self._headers())), 10
        )
        try:
            session = {"sample_rate": self.sample_rate, "encoding": "LINEAR16"}
            await ws.send(json.dumps({"type": "session.update", "session": session}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if msg.get("type") == "transcription_session.created":
                    mode = (msg.get("session") or {}).get("mode")
                    if mode not in SERVICE_MODES:
                        raise RuntimeError(f"realtime handshake reports no service mode: {msg}")
                    return str(mode)
                if msg.get("type") == "error":
                    raise RuntimeError(f"realtime handshake refused: {msg.get('error')}")
        finally:
            await ws.close()

    def _served_model(self) -> Dict[str, Any]:
        """``/v1/models``'s first row (id + the engine's model info), for provenance."""
        try:
            with _no_proxy_opener().open(f"{self.base_url}/v1/models", timeout=5) as resp:
                rows = json.loads(resp.read().decode()).get("data") or []
        except (urllib.error.URLError, OSError, ValueError):
            return {}
        if not rows:
            return {}
        return {"id": rows[0].get("id"), **(rows[0].get("info") or {})}

    def _server_version(self) -> str:
        from benchmarks.asrbench.environment import packages

        return packages().get("oasr", "")

    def info(self) -> BackendInfo:
        from benchmarks.asrbench.backends.oasr_engine import checkpoint_fingerprint

        info = super().info()
        ckpt = self.options.get("ckpt_dir") or os.environ.get("CKPT_DIR")
        attached = bool(self.options.get("base_url"))
        if ckpt and not attached:
            model = checkpoint_fingerprint(str(ckpt))
            model["id"] = Path(str(ckpt)).name
            info.model = model
        elif self._attached_model:
            info.model = dict(self._attached_model)
        if attached:  # whatever it was started with, as the server reports it
            info.dtype = str(self._attached_model.get("dtype") or "").replace("torch.", "")
        else:
            info.dtype = str(self.options.get("dtype", "bfloat16"))
        if self.transport == "grpc":
            from benchmarks.asrbench.backends.grpc_speech import UNARY_ENCODINGS
            from benchmarks.asrbench.environment import packages

            info.config["wire"] = (
                f"gRPC Recognize {UNARY_ENCODINGS[self.grpc_encoding]}; "
                "StreamingRecognize LINEAR16"
            )
            info.config["grpc_addr"] = self.grpc_addr
            info.config["client"] = {"grpcio": packages().get("grpcio", "")}
        info.config.update(
            {
                "transport": self.transport,
                "service_mode": self.server_mode,
                "spawned": self.proc is not None,
                "command": self.proc.args if self.proc is not None else None,
            }
        )
        return info

    def capabilities(self) -> Capabilities:
        streaming = self.server_mode == "streaming"
        return Capabilities(
            # A streaming-mode server answers the upload route with 400 by design.
            offline=not streaming,
            batch=False,
            concurrent=True,
            streaming="native" if streaming else "buffered",
            partials=streaming,
            word_timestamps=False,
            long_form="whole" if not streaming else None,
            in_process=False,
            server_metrics=True,
            notes=(
                f"{self.transport}; "
                + (
                    "streaming-mode server: streaming only"
                    if streaming
                    else "offline-mode server: a stream is buffered until it ends"
                )
            ),
        )

    def prepare(self, audio: np.ndarray, item: WorkItem) -> Any:
        if self.transport == "grpc":
            from benchmarks.asrbench.backends.grpc_speech import unary_request

            language = str(self.options.get("language") or "")
            return unary_request(
                self._grpc[2], audio, self.sample_rate, self.grpc_encoding, language
            )
        return super().prepare(audio, item)

    async def open_stream(self, *, word_timestamps: bool = False) -> StreamSession:
        if self.transport == "grpc":
            from benchmarks.asrbench.backends.grpc_speech import GrpcStream

            if word_timestamps:
                raise Unsupported("word timestamps are not requested over the streaming RPC")
            _, stub, pb = self._grpc
            return await GrpcStream.open(
                stub, pb, self.sample_rate, str(self.options.get("language") or "")
            )
        self.options["realtime"] = True
        return await super().open_stream(word_timestamps=word_timestamps)

    async def transcribe(self, payload: Any) -> Transcript:
        if self.server_mode == "streaming":
            raise Unsupported("a streaming-mode oasr-server serves streaming requests only")
        if self.transport == "grpc":
            from benchmarks.asrbench.backends.grpc_speech import recognize

            return await recognize(
                self._grpc[1], payload, float(self.options.get("timeout_s", 600))
            )
        return await super().transcribe(payload)
