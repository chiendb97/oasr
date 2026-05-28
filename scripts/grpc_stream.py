#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""gRPC bidi streaming client for the OASR Speech service.

Requires ``grpcio`` and ``grpcio-tools``::

    pip install grpcio grpcio-tools

Generates the stubs at runtime from ``rust/proto/oasr_speech_v1.proto`` so
the script stays self-contained.

Usage::

    python scripts/grpc_stream.py --addr 127.0.0.1:50051 \
        --wav tests/fixtures/hello.wav --chunk-ms 640
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1).astype("float32")
    return data, sr


def _gen_stubs(proto_path: Path):
    """Compile the proto into a temporary dir and import the result."""
    try:
        from grpc_tools import protoc
    except ImportError as e:
        sys.stderr.write(
            "missing grpcio-tools (install with `pip install grpcio grpcio-tools`)\n"
        )
        raise SystemExit(1) from e
    if shutil.which("protoc") is None and not (
        Path(sys.prefix) / "lib" / "site-packages" / "grpc_tools"
    ).exists():
        # grpc_tools bundles its own protoc; just use the entry point.
        pass

    out = Path(tempfile.mkdtemp(prefix="oasr-grpc-"))
    args = [
        "protoc",
        f"--proto_path={proto_path.parent}",
        f"--python_out={out}",
        f"--grpc_python_out={out}",
        str(proto_path),
    ]
    rc = protoc.main(args)
    if rc != 0:
        raise RuntimeError(f"protoc failed with rc={rc}")

    sys.path.insert(0, str(out))
    pb = importlib.import_module("oasr_speech_v1_pb2")
    pb_grpc = importlib.import_module("oasr_speech_v1_pb2_grpc")
    return pb, pb_grpc


async def run(addr: str, wav: Path, chunk_ms: int, proto: Path) -> int:
    import grpc

    pb, pb_grpc = _gen_stubs(proto)

    samples, sr = read_wav(wav)
    chunk_samples = int(sr * chunk_ms / 1000)

    async with grpc.aio.insecure_channel(addr) as channel:
        stub = pb_grpc.SpeechStub(channel)

        async def requests():
            cfg = pb.RecognitionConfig(
                encoding=pb.RecognitionConfig.LINEAR32F,
                sample_rate_hertz=sr,
                language_code="en-US",
                priority=0,
            )
            yield pb.StreamingRecognizeRequest(
                streaming_config=pb.StreamingRecognitionConfig(
                    config=cfg, interim_results=True,
                )
            )
            for i in range(0, len(samples), chunk_samples):
                chunk = samples[i:i + chunk_samples]
                if chunk.size == 0:
                    break
                yield pb.StreamingRecognizeRequest(
                    audio_content=chunk.astype("<f4").tobytes()
                )

        responses = stub.StreamingRecognize(requests())
        async for resp in responses:
            for r in resp.results:
                tag = "FINAL" if r.is_final else "PARTIAL"
                text = r.alternatives[0].transcript if r.alternatives else ""
                print(f"[{tag}] rid={resp.request_id} text={text!r}")
                if r.is_final:
                    return 0
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--addr", required=True)
    p.add_argument("--wav", required=True, type=Path)
    p.add_argument("--chunk-ms", type=int, default=640)
    p.add_argument(
        "--proto",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "rust" / "proto" / "oasr_speech_v1.proto",
    )
    args = p.parse_args(argv)
    return asyncio.run(run(args.addr, args.wav, args.chunk_ms, args.proto))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
