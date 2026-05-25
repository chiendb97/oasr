#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tiny WebSocket client for the OASR streaming endpoint.

Usage::

    python scripts/ws_stream.py --url ws://127.0.0.1:8080/v1/stream \
        --wav tests/fixtures/hello.wav --chunk-ms 640

Reads the WAV at the given chunk size, sends a ``start`` JSON frame followed
by binary PCM f32 LE chunks, then ``{"type":"end"}``.  Prints each partial /
final / error event as it arrives.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (mono f32 samples in [-1, 1], sample_rate)."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1).astype("float32")
    return data, sr


async def run(url: str, wav: Path, chunk_ms: int, sample_rate: int | None) -> int:
    import websockets

    samples, file_sr = read_wav(wav)
    sr = sample_rate or file_sr

    chunk_samples = int(sr * chunk_ms / 1000)
    print(f"[client] connecting to {url} (sr={sr}, chunk={chunk_samples} samples)",
          file=sys.stderr, flush=True)

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "start", "sample_rate": sr, "format": "pcm_f32le",
        }))

        async def producer():
            for i in range(0, len(samples), chunk_samples):
                chunk = samples[i:i + chunk_samples]
                if chunk.size == 0:
                    break
                await ws.send(chunk.astype("<f4").tobytes())
            await ws.send(json.dumps({"type": "end"}))

        async def consumer():
            async for msg in ws:
                if isinstance(msg, bytes):
                    print(f"[bin {len(msg)} bytes]", flush=True)
                    continue
                try:
                    ev = json.loads(msg)
                except json.JSONDecodeError:
                    print(f"[non-json] {msg}", flush=True)
                    continue
                print(json.dumps(ev), flush=True)
                if ev.get("type") in {"final", "error"}:
                    break

        await asyncio.gather(producer(), consumer())
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--wav", required=True, type=Path)
    p.add_argument("--chunk-ms", type=int, default=640)
    p.add_argument("--sample-rate", type=int, default=None,
                   help="override the WAV's declared sample rate")
    args = p.parse_args(argv)
    return asyncio.run(run(args.url, args.wav, args.chunk_ms, args.sample_rate))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
