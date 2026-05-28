#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Minimal gRPC client for the OASR `oasr.speech.v1.Speech` service.

Demonstrates both RPCs.  Generates the Python stubs at runtime from
``rust/proto/oasr_speech_v1.proto`` so this example stays self-contained.

Dependencies::

    pip install grpcio grpcio-tools soundfile numpy

Usage::

    # Unary Recognize (server must run with --service-mode offline)
    python examples/recognize/grpc_client.py \\
        --addr 127.0.0.1:50051 --rpc recognize \\
        --wav tests/fixtures/hello.wav

    # Bidi StreamingRecognize (server must run with --service-mode streaming)
    python examples/recognize/grpc_client.py \\
        --addr 127.0.0.1:50051 --rpc streaming_recognize \\
        --wav tests/fixtures/hello.wav --chunk-ms 640
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Stub generation
# ---------------------------------------------------------------------------


DEFAULT_PROTO = (
    Path(__file__).resolve().parents[2] / "rust" / "proto" / "oasr_speech_v1.proto"
)


def load_stubs(proto_path: Path):
    """Compile ``oasr_speech_v1.proto`` and import the generated modules."""
    try:
        from grpc_tools import protoc
    except ImportError as exc:
        raise SystemExit(
            "missing grpcio-tools (install with `pip install grpcio grpcio-tools`)"
        ) from exc

    out = Path(tempfile.mkdtemp(prefix="oasr-example-grpc-"))
    rc = protoc.main([
        "protoc",
        f"--proto_path={proto_path.parent}",
        f"--python_out={out}",
        f"--grpc_python_out={out}",
        str(proto_path),
    ])
    if rc != 0:
        raise SystemExit(f"protoc failed with rc={rc}")
    sys.path.insert(0, str(out))
    pb = importlib.import_module("oasr_speech_v1_pb2")
    pb_grpc = importlib.import_module("oasr_speech_v1_pb2_grpc")
    return pb, pb_grpc


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as f32 mono samples."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1).astype("float32")
    return data, sr


# ---------------------------------------------------------------------------
# Unary Recognize
# ---------------------------------------------------------------------------


async def run_recognize(addr: str, wav: Path, proto_path: Path) -> int:
    import grpc

    pb, pb_grpc = load_stubs(proto_path)
    samples, sr = read_wav(wav)

    # Bypass any ``http_proxy``/``HTTPS_PROXY`` in the environment so the
    # client talks to the loopback server directly.
    channel_options = [("grpc.enable_http_proxy", 0)]
    async with grpc.aio.insecure_channel(addr, options=channel_options) as channel:
        stub = pb_grpc.SpeechStub(channel)
        req = pb.RecognizeRequest(
            config=pb.RecognitionConfig(
                encoding=pb.RecognitionConfig.LINEAR32F,
                sample_rate_hertz=sr,
                language_code="en-US",
                max_alternatives=1,
            ),
            audio=pb.RecognitionAudio(
                content=samples.astype("<f4").tobytes(),
            ),
        )
        resp = await stub.Recognize(req, timeout=120.0)

    print(f"request_id: {resp.request_id}")
    for i, result in enumerate(resp.results):
        for j, alt in enumerate(result.alternatives):
            print(f"results[{i}].alternatives[{j}]: "
                  f"transcript={alt.transcript!r} confidence={alt.confidence:.3f}")
    return 0


# ---------------------------------------------------------------------------
# Bidi StreamingRecognize
# ---------------------------------------------------------------------------


async def run_streaming(addr: str, wav: Path, chunk_ms: int, proto_path: Path) -> int:
    import grpc

    pb, pb_grpc = load_stubs(proto_path)
    samples, sr = read_wav(wav)
    chunk_samples = int(sr * chunk_ms / 1000)

    # Bypass any ``http_proxy``/``HTTPS_PROXY`` in the environment so the
    # client talks to the loopback server directly.
    channel_options = [("grpc.enable_http_proxy", 0)]
    async with grpc.aio.insecure_channel(addr, options=channel_options) as channel:
        stub = pb_grpc.SpeechStub(channel)

        async def requests():
            # First message MUST carry streaming_config.
            yield pb.StreamingRecognizeRequest(
                streaming_config=pb.StreamingRecognitionConfig(
                    config=pb.RecognitionConfig(
                        encoding=pb.RecognitionConfig.LINEAR32F,
                        sample_rate_hertz=sr,
                        language_code="en-US",
                        max_alternatives=1,
                    ),
                    interim_results=True,
                )
            )
            # Subsequent messages: raw f32 PCM chunks.
            for i in range(0, len(samples), chunk_samples):
                chunk = samples[i:i + chunk_samples]
                if chunk.size == 0:
                    break
                yield pb.StreamingRecognizeRequest(
                    audio_content=chunk.astype("<f4").tobytes(),
                )

        responses = stub.StreamingRecognize(requests())
        async for resp in responses:
            for result in resp.results:
                tag = "FINAL  " if result.is_final else "PARTIAL"
                transcript = (
                    result.alternatives[0].transcript if result.alternatives else ""
                )
                print(f"[{tag}] rid={resp.request_id} text={transcript!r}")
                if result.is_final:
                    return 0
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--addr", default="127.0.0.1:50051",
                   help="gRPC endpoint of the oasr-server (default 127.0.0.1:50051)")
    p.add_argument("--wav", required=True, type=Path,
                   help="Path to the audio file to transcribe")
    p.add_argument("--rpc", default="recognize",
                   choices=("recognize", "streaming_recognize"),
                   help="Which RPC to call (default: recognize). "
                        "`recognize` requires --service-mode offline; "
                        "`streaming_recognize` requires --service-mode streaming.")
    p.add_argument("--chunk-ms", type=int, default=640,
                   help="Streaming chunk size in milliseconds (default 640)")
    p.add_argument("--proto", type=Path, default=DEFAULT_PROTO,
                   help="Path to oasr_speech_v1.proto")
    args = p.parse_args(argv)

    if not args.wav.is_file():
        print(f"audio file not found: {args.wav}", file=sys.stderr)
        return 1
    if not args.proto.is_file():
        print(f"proto file not found: {args.proto}", file=sys.stderr)
        return 1

    if args.rpc == "recognize":
        return asyncio.run(run_recognize(args.addr, args.wav, args.proto))
    return asyncio.run(
        run_streaming(args.addr, args.wav, args.chunk_ms, args.proto)
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
