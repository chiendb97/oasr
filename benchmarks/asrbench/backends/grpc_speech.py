# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""OASR's gRPC Speech API as an asrbench transport (``oasr_server -o transport=grpc``).

The schema is ``rust/proto/oasr_speech_v1.proto``; stubs are compiled from it
once per process with ``grpcio-tools`` (the repo ships no generated code), the
same way ``benchmarks/service`` does.

What is sent, chosen so that only the transport differs from the HTTP path:

- **Unary** ``Recognize``: by default the *same WAV bytes* the HTTP backend
  uploads (``encoding=WAV``).  ``-o grpc_encoding=linear16`` sends the headerless
  PCM16 samples instead, ``linear32f`` float32 (twice the bytes).
- **Streaming** ``StreamingRecognize``: PCM16 chunks (``LINEAR16``), the same
  samples the realtime WebSocket carries; a container cannot be streamed.

Status mapping follows the HTTP backend: ``RESOURCE_EXHAUSTED`` (the server's
admission back-pressure) is a *rejection*; ``DEADLINE_EXCEEDED`` a timeout;
anything else, ``UNAVAILABLE`` included, a failure.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from benchmarks.asrbench.audio import encode_wav, to_pcm16
from benchmarks.asrbench.backends.base import (
    RequestFailed,
    StreamEvent,
    StreamSession,
    Transcript,
)

PROTO = Path(__file__).resolve().parents[3] / "rust" / "proto" / "oasr_speech_v1.proto"

#: ``-o grpc_encoding`` values -> RecognitionConfig.AudioEncoding names.
UNARY_ENCODINGS = {"wav": "WAV", "linear16": "LINEAR16", "linear32f": "LINEAR32F"}

#: Messages are sized by the server (``--max-audio-mib``), not by gRPC's 4 MiB default.
CHANNEL_OPTIONS = [
    # The box may export a proxy that does not route to the server.
    ("grpc.enable_http_proxy", 0),
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
]

_lock = threading.Lock()
_stubs: Optional[Tuple[Any, Any]] = None


def available() -> Tuple[bool, str]:
    try:
        import grpc  # noqa: F401
        import grpc_tools  # noqa: F401
    except ImportError:
        return False, "needs grpcio and grpcio-tools (pip install grpcio grpcio-tools)"
    if not PROTO.is_file():
        return False, f"{PROTO} not found (run from a source checkout)"
    return True, ""


def stubs() -> Tuple[Any, Any]:
    """``(pb2, pb2_grpc)`` for the Speech service, compiled on first use."""
    global _stubs
    with _lock:
        if _stubs is None:
            if "oasr_speech_v1_pb2" in sys.modules and "oasr_speech_v1_pb2_grpc" in sys.modules:
                # Already compiled in this process (benchmarks/service): a second
                # copy would collide in protobuf's descriptor pool.
                _stubs = sys.modules["oasr_speech_v1_pb2"], sys.modules["oasr_speech_v1_pb2_grpc"]
            else:
                _stubs = _compile()
    return _stubs


def _compile() -> Tuple[Any, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # grpc_tools imports the deprecated pkg_resources
        import grpc_tools
        from grpc_tools import protoc
    out = Path(tempfile.mkdtemp(prefix="asrbench-grpc-"))
    # ``protoc.main`` does not add the bundled well-known types the way
    # ``python -m grpc_tools.protoc`` does, and the schema imports duration.proto.
    well_known = Path(grpc_tools.__file__).resolve().parent / "_proto"
    rc = protoc.main(
        [
            "protoc",
            f"--proto_path={PROTO.parent}",
            f"--proto_path={well_known}",
            f"--python_out={out}",
            f"--grpc_python_out={out}",
            str(PROTO),
        ]
    )
    if rc != 0:
        raise RuntimeError(f"protoc failed on {PROTO} (rc={rc})")
    modules = []
    for name in ("oasr_speech_v1_pb2", "oasr_speech_v1_pb2_grpc"):
        # The _grpc module imports the _pb2 one by name: register before loading it.
        spec = importlib.util.spec_from_file_location(name, out / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules[0], modules[1]


def channel(addr: str) -> Any:
    import grpc

    return grpc.aio.insecure_channel(addr, options=CHANNEL_OPTIONS)


def recognition_config(pb: Any, encoding: str, sample_rate: int, language: str = "") -> Any:
    return pb.RecognitionConfig(
        encoding=getattr(pb.RecognitionConfig, encoding),
        sample_rate_hertz=int(sample_rate),
        language_code=language,
    )


def unary_request(
    pb: Any, audio: np.ndarray, sample_rate: int, encoding: str, language: str = ""
) -> Any:
    """A complete ``RecognizeRequest``, built outside the timed window."""
    name = UNARY_ENCODINGS[encoding]
    if name == "WAV":
        content = encode_wav(audio, sample_rate)
    elif name == "LINEAR16":
        content = to_pcm16(audio).tobytes()
    else:
        content = np.ascontiguousarray(audio, dtype="<f4").tobytes()
    return pb.RecognizeRequest(
        config=recognition_config(pb, name, sample_rate, language),
        audio=pb.RecognitionAudio(content=content),
    )


def _failure(exc: Any) -> Exception:
    import grpc

    code = exc.code()
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return asyncio.TimeoutError(exc.details() or "deadline exceeded")
    kind = "rejected" if code == grpc.StatusCode.RESOURCE_EXHAUSTED else "failed"
    return RequestFailed(f"gRPC {code.name}: {exc.details()}", kind=kind)


async def recognize(stub: Any, request: Any, timeout_s: float) -> Transcript:
    import grpc

    t_submit = time.perf_counter()
    try:
        resp = await stub.Recognize(request, timeout=timeout_s)
    except grpc.aio.AioRpcError as exc:
        raise _failure(exc) from None
    t_done = time.perf_counter()
    # One result per recognized span (Google STT semantics): concatenated.
    text = " ".join(r.alternatives[0].transcript for r in resp.results if r.alternatives)
    return Transcript(text=text.strip(), t_submit=t_submit, t_done=t_done)


class GrpcStream(StreamSession):
    """One ``StreamingRecognize`` call: config first, then PCM16 chunks, then half-close.

    The first ``is_final`` result ends the session, as the realtime WebSocket's
    first ``completed`` event does.
    """

    def __init__(self, call: Any, pb: Any) -> None:
        super().__init__()
        self.call = call
        self.pb = pb
        self._kind = "failed"
        self._reader: Optional[asyncio.Task] = None

    @classmethod
    async def open(cls, stub: Any, pb: Any, sample_rate: int, language: str = "") -> "GrpcStream":
        call = stub.StreamingRecognize()
        stream = cls(call, pb)
        cfg = pb.StreamingRecognitionConfig(
            config=recognition_config(pb, "LINEAR16", sample_rate, language),
            interim_results=True,
        )
        await call.write(pb.StreamingRecognizeRequest(streaming_config=cfg))
        stream._reader = asyncio.get_running_loop().create_task(stream._read())
        return stream

    async def _read(self) -> None:
        import grpc

        try:
            async for resp in self.call:
                t = time.perf_counter()
                for r in resp.results:
                    text = r.alternatives[0].transcript if r.alternatives else ""
                    if r.is_final:
                        self.push(StreamEvent("final", t, text))
                        return
                    self.push(StreamEvent("partial", t, text))
            self.push(StreamEvent("error", time.perf_counter(), "stream ended without a final"))
        except grpc.aio.AioRpcError as exc:
            if exc.code() == grpc.StatusCode.CANCELLED:
                return
            if exc.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                self._kind = "rejected"
            self.push(
                StreamEvent(
                    "error", time.perf_counter(), f"gRPC {exc.code().name}: {exc.details()}"
                )
            )
        except Exception as exc:  # noqa: BLE001 -- connection dropped
            self.push(StreamEvent("error", time.perf_counter(), f"{type(exc).__name__}: {exc}"))

    async def send(self, chunk: Any, is_last: bool) -> None:
        if self._final.is_set():
            return  # the call already ended (an error); the reader recorded why
        await self.call.write(self.pb.StreamingRecognizeRequest(audio_content=chunk))
        if is_last:
            await self.call.done_writing()

    async def wait_final(self, timeout: Optional[float]) -> StreamEvent:
        try:
            return await super().wait_final(timeout)
        except RequestFailed as exc:
            raise RequestFailed(str(exc), kind=self._kind) from None
        finally:
            await self._close()

    async def abort(self) -> None:
        await self._close()

    async def _close(self) -> None:
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
        self.call.cancel()
