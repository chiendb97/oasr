#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""OASR Serving Benchmark — HTTP + WebSocket against the Rust frontend.

Measures end-to-end serving throughput and latency for the OASR ``oasr-server``
binary.  Mirrors the CLI shape of ``benchmarks/bench_engine.py`` so the same
``--ckpt-dir`` / ``--audio-dir`` / ``--num-utterances`` arguments work.

Subroutines (one or more)
-------------------------
``offline``  ``POST /v1/transcriptions`` with raw f32 PCM bodies, N concurrent
             clients.  Reports RTF, total throughput, p50 / p95 / p99 latency.
``whisper``  ``POST /v1/audio/transcriptions`` (OpenAI Whisper-compat) with
             multipart WAV uploads, N concurrent clients.
``streaming`` ``GET /v1/stream`` (WebSocket).  Each client opens one stream
             at a time and feeds the audio in fixed-duration chunks; reports
             first-partial latency, total stream wall time, and RTF.

Examples
--------
# Quick smoke (auto-spawns oasr-server)
python benchmarks/bench_service.py \\
    --ckpt-dir /data01/kilm/users/chiendb/models/asr/am/20210610_u2pp_conformer_exp_librispeech \\
    --audio-dir /data01/kilm/users/chiendb/data/asr/ljspeech-sr16k-dataset/wavs \\
    --subroutines streaming \\
    --max-batch-size 64 --num-utterances 2000

# Use a running server (skip spawn)
python benchmarks/bench_service.py \\
    --audio-dir /path/to/wavs --subroutines offline streaming \\
    --server-url http://127.0.0.1:8080 --num-utterances 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1).astype("float32")
    return data, sr


@dataclass
class Sample:
    path: Path
    samples: np.ndarray
    sample_rate: int
    duration_s: float
    wav_bytes: bytes  # the original on-disk WAV (used by whisper-compat path)


def _load_dataset(audio_dir: Path, n: int) -> List[Sample]:
    wavs = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() == ".wav")
    if not wavs:
        raise SystemExit(f"no .wav files found under {audio_dir}")
    if n > len(wavs):
        # Repeat the list so the harness can run more requests than files.
        reps = (n + len(wavs) - 1) // len(wavs)
        wavs = (wavs * reps)[:n]
    else:
        wavs = wavs[:n]

    samples: List[Sample] = []
    print(f"[bench] loading {len(wavs)} wavs from {audio_dir}", flush=True)
    for p in wavs:
        data, sr = _read_wav(p)
        with open(p, "rb") as f:
            wav_bytes = f.read()
        samples.append(
            Sample(
                path=p,
                samples=data,
                sample_rate=sr,
                duration_s=len(data) / float(sr),
                wav_bytes=wav_bytes,
            )
        )
    total_audio = sum(s.duration_s for s in samples)
    print(f"[bench] loaded {len(samples)} samples — total audio {total_audio:.1f} s",
          flush=True)
    return samples


# ---------------------------------------------------------------------------
# Server spawn / health
# ---------------------------------------------------------------------------


def _resolve_server_bin() -> Path:
    env = os.environ.get("OASR_RS_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    # Default location for an editable repo.
    here = Path(__file__).resolve().parents[1]
    candidate = here / "rust" / "target" / "release" / "oasr-server"
    if candidate.is_file():
        return candidate
    raise SystemExit(
        "oasr-server binary not found; build via `cd rust && cargo build --release` "
        "or set OASR_RS_BIN=/path/to/oasr-server"
    )


@dataclass
class ServerHandle:
    proc: Optional[subprocess.Popen]
    http_url: str
    grpc_addr: str
    spawned: bool

    def stop(self) -> None:
        if self.proc is None or not self.spawned:
            return
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        except Exception:
            pass


def _spawn_server(args: argparse.Namespace) -> ServerHandle:
    bin_path = _resolve_server_bin()
    http_bind = args.http_bind
    grpc_bind = args.grpc_bind
    cmd = [
        str(bin_path),
        "--ckpt-dir", str(args.ckpt_dir),
        "--num-workers", str(args.num_workers),
        "--worker-threads", str(args.worker_threads),
        "--http-bind", http_bind,
        "--grpc-bind", grpc_bind,
        # Engine in-flight cap should comfortably exceed the bench
        # concurrency so admission doesn't reject load with BUSY.
        "--max-concurrent-requests", str(max(args.concurrency * 8, 256)),
        "--log-level", args.server_log_level,
        "--dtype", args.dtype,
    ]
    if args.max_batch_size is not None:
        cmd.extend(["--max-batch-size", str(args.max_batch_size)])
    if args.chunk_size is not None:
        cmd.extend(["--chunk-size", str(args.chunk_size)])
    if args.cuda_devices:
        cmd.extend(["--cuda-devices", args.cuda_devices])
    if args.worker_script is not None:
        cmd.extend(["--worker-script", str(args.worker_script)])

    print(f"[bench] spawning: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    proc = subprocess.Popen(cmd, env=env)

    # Wait for /readyz.
    import urllib.error
    import urllib.request
    http_url = f"http://{http_bind}"
    deadline = time.time() + args.ready_timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{http_url}/readyz", timeout=2) as resp:
                if resp.status == 200:
                    print("[bench] server ready", flush=True)
                    return ServerHandle(proc=proc, http_url=http_url,
                                        grpc_addr=grpc_bind, spawned=True)
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if proc.poll() is not None:
            raise SystemExit(
                f"oasr-server exited prematurely (rc={proc.returncode})"
            )
        time.sleep(1.0)
    proc.terminate()
    raise SystemExit(f"server did not become ready within {args.ready_timeout_s}s")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    name: str
    n_ok: int = 0
    n_rejected: int = 0  # backpressure (503 / Error{BUSY}) — server says retry later
    n_fail: int = 0  # hard transport failures (TCP reset, decode, etc.)
    wall_s: float = 0.0
    total_audio_s: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)
    first_partial_ms: List[float] = field(default_factory=list)
    partial_counts: List[int] = field(default_factory=list)

    def pretty(self) -> str:
        if not self.latencies_ms:
            return (
                f"{self.name}: no successful requests "
                f"(rejected={self.n_rejected}, fail={self.n_fail})"
            )
        rtf = self.total_audio_s / self.wall_s if self.wall_s > 0 else float("inf")
        rps = self.n_ok / self.wall_s if self.wall_s > 0 else 0.0
        lat = sorted(self.latencies_ms)
        p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]
        lines = [
            f"{self.name}:",
            f"  requests   ok={self.n_ok}  rejected={self.n_rejected}  fail={self.n_fail}",
            f"  wall       {self.wall_s:.2f} s",
            f"  audio      {self.total_audio_s:.2f} s",
            f"  RTF        {rtf:.2f}x   (audio_seconds / wall_seconds — higher is faster)",
            f"  throughput {rps:.2f} req/s",
            f"  latency    mean={statistics.fmean(lat):.0f} ms"
            f"  p50={p(0.5):.0f}  p90={p(0.90):.0f}  p95={p(0.95):.0f}"
            f"  p99={p(0.99):.0f}  max={lat[-1]:.0f}",
        ]
        if self.first_partial_ms:
            fp = sorted(self.first_partial_ms)
            lines.append(
                f"  first-partial   mean={statistics.fmean(fp):.0f} ms"
                f"  p50={fp[len(fp)//2]:.0f}"
                f"  p95={fp[min(len(fp)-1, int(0.95*len(fp)))]:.0f}"
            )
        if self.partial_counts:
            lines.append(
                f"  partials/req    mean={statistics.fmean(self.partial_counts):.1f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Offline benchmark — POST /v1/transcriptions (raw PCM)
# ---------------------------------------------------------------------------


async def _bench_offline(http_url: str, samples: List[Sample], concurrency: int) -> RunStats:
    import httpx

    stats = RunStats(name="offline (POST /v1/transcriptions, raw f32 PCM)")
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:

        async def one(sample: Sample) -> None:
            async with sem:
                body = sample.samples.astype("<f4").tobytes()
                url = f"{http_url}/v1/transcriptions?sample_rate={sample.sample_rate}&encoding=f32_le"
                start = time.perf_counter()
                try:
                    resp = await client.post(
                        url,
                        content=body,
                        headers={"content-type": "application/octet-stream"},
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    if resp.status_code == 503:
                        stats.n_rejected += 1
                        return
                    if resp.status_code != 200:
                        stats.n_fail += 1
                        return
                    _ = resp.json()  # validate JSON shape
                    stats.n_ok += 1
                    stats.latencies_ms.append(elapsed_ms)
                    stats.total_audio_s += sample.duration_s
                except Exception:
                    stats.n_fail += 1

        t0 = time.perf_counter()
        await asyncio.gather(*(one(s) for s in samples))
        stats.wall_s = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# Whisper-compat benchmark — POST /v1/audio/transcriptions (multipart WAV)
# ---------------------------------------------------------------------------


async def _bench_whisper(http_url: str, samples: List[Sample], concurrency: int) -> RunStats:
    import httpx

    stats = RunStats(name="whisper-compat (POST /v1/audio/transcriptions)")
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:

        async def one(sample: Sample) -> None:
            async with sem:
                start = time.perf_counter()
                try:
                    files = {
                        "file": (sample.path.name, sample.wav_bytes, "audio/wav"),
                    }
                    resp = await client.post(
                        f"{http_url}/v1/audio/transcriptions",
                        files=files,
                        data={"response_format": "json"},
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    if resp.status_code == 503:
                        stats.n_rejected += 1
                        return
                    if resp.status_code != 200:
                        stats.n_fail += 1
                        return
                    _ = resp.json()
                    stats.n_ok += 1
                    stats.latencies_ms.append(elapsed_ms)
                    stats.total_audio_s += sample.duration_s
                except Exception:
                    stats.n_fail += 1

        t0 = time.perf_counter()
        await asyncio.gather(*(one(s) for s in samples))
        stats.wall_s = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# Streaming benchmark — WebSocket
# ---------------------------------------------------------------------------


async def _bench_streaming(
    http_url: str,
    samples: List[Sample],
    concurrency: int,
    chunk_ms: int,
) -> RunStats:
    import websockets

    ws_url = http_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/v1/stream"

    stats = RunStats(name=f"streaming (WS /v1/stream, chunk={chunk_ms}ms)")
    sem = asyncio.Semaphore(concurrency)

    async def one(sample: Sample) -> None:
        async with sem:
            chunk_samples = int(sample.sample_rate * chunk_ms / 1000)
            start = time.perf_counter()
            first_partial_at: Optional[float] = None
            partials = 0
            try:
                async with websockets.connect(
                    ws_url,
                    max_size=None,
                    ping_interval=None,
                    open_timeout=30,
                    close_timeout=10,
                ) as ws:
                    await ws.send(json.dumps({
                        "type": "start",
                        "sample_rate": sample.sample_rate,
                        "format": "pcm_f32le",
                    }))

                    async def producer():
                        # Pace chunks at real-time — matches a live mic.  Set
                        # `--realtime 0` to disable pacing (max-rate test).
                        per_chunk = chunk_ms / 1000.0 if pace_realtime else 0.0
                        last = time.perf_counter()
                        try:
                            for i in range(0, len(sample.samples), chunk_samples):
                                chunk = sample.samples[i:i + chunk_samples]
                                if chunk.size == 0:
                                    break
                                await ws.send(chunk.astype("<f4").tobytes())
                                if per_chunk > 0:
                                    elapsed = time.perf_counter() - last
                                    if elapsed < per_chunk:
                                        await asyncio.sleep(per_chunk - elapsed)
                                    last = time.perf_counter()
                            await ws.send(json.dumps({"type": "end"}))
                        except websockets.exceptions.ConnectionClosed:
                            # Server closed (e.g. BUSY rejection mid-stream);
                            # let the consumer collect the error event.
                            return

                    async def consumer():
                        nonlocal first_partial_at, partials
                        async for msg in ws:
                            if isinstance(msg, bytes):
                                continue
                            try:
                                ev = json.loads(msg)
                            except json.JSONDecodeError:
                                continue
                            t = ev.get("type")
                            if t == "partial":
                                partials += 1
                                if first_partial_at is None:
                                    first_partial_at = time.perf_counter()
                            elif t in ("final", "error"):
                                return t

                    prod_task = asyncio.create_task(producer())
                    final_kind = await consumer()
                    # Drain the producer if it's still going (e.g. on error).
                    if not prod_task.done():
                        prod_task.cancel()
                        try:
                            await prod_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    if final_kind == "final":
                        stats.n_ok += 1
                        stats.latencies_ms.append(elapsed_ms)
                        stats.total_audio_s += sample.duration_s
                        stats.partial_counts.append(partials)
                        if first_partial_at is not None:
                            stats.first_partial_ms.append(
                                (first_partial_at - start) * 1000.0
                            )
                    elif final_kind == "error":
                        # Worker emitted an Error event (typically BUSY).
                        stats.n_rejected += 1
                    else:
                        stats.n_fail += 1
            except Exception:
                # Transport-level failure (TCP reset on overload, etc.) — count
                # alongside rejections, since under heavy backpressure the
                # server may close the WebSocket before the Error frame lands.
                stats.n_rejected += 1

    pace_realtime = False  # set by caller via closure (see entry point)
    # Bind via attribute capture: rewrap with the value
    pace_realtime = bool(getattr(_bench_streaming, "_pace", False))

    t0 = time.perf_counter()
    await asyncio.gather(*(one(s) for s in samples))
    stats.wall_s = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OASR Serving Benchmark — HTTP + WebSocket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--ckpt-dir", default=None, type=Path,
                   help="WeNet checkpoint dir (required only when this script "
                        "spawns the server; ignored if --server-url is set)")
    p.add_argument("--audio-dir", required=True, type=Path,
                   help="Directory containing .wav files")
    p.add_argument(
        "--subroutines",
        nargs="+",
        default=["offline"],
        choices=["offline", "whisper", "streaming"],
        help="Which serving paths to benchmark (default: offline)",
    )
    p.add_argument("--num-utterances", type=int, default=200)
    p.add_argument("--concurrency", "-c", type=int, default=16,
                   help="Concurrent in-flight requests per subroutine (default 16)")

    # Streaming-specific
    p.add_argument("--chunk-ms", type=int, default=640,
                   help="WebSocket chunk size in milliseconds (default 640)")
    p.add_argument(
        "--realtime",
        type=int, default=0, choices=(0, 1),
        help="If 1, pace WS chunks at real-time (sleep chunk_ms between sends). "
             "Default 0 = send as fast as the network allows.",
    )

    # Server spawn
    p.add_argument("--server-url", default=None,
                   help="If set, benchmark against an already-running server (e.g. http://127.0.0.1:8080). "
                        "Otherwise this script spawns oasr-server itself.")
    p.add_argument("--http-bind", default="127.0.0.1:8080")
    p.add_argument("--grpc-bind", default="127.0.0.1:50051")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--cuda-devices", default=None)
    p.add_argument("--worker-threads", type=int, default=1, choices=(1, 2))
    p.add_argument("--worker-script", default=None, type=Path,
                   help="Override what the Rust server spawns (e.g. "
                        "scripts/fake_engine_worker.py for stub testing).")
    p.add_argument("--max-batch-size", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Encoder chunk size (frames) for the engine (default: engine config)")
    p.add_argument("--dtype", default="float16",
                   choices=("float16", "bfloat16", "float32"))
    p.add_argument("--ready-timeout-s", type=int, default=180)
    p.add_argument("--server-log-level", default="warn")

    p.add_argument("--output-path", default=None, type=Path,
                   help="Optional JSON file path for the result summary")

    return p


def main() -> int:
    args = _build_parser().parse_args()

    samples = _load_dataset(args.audio_dir, args.num_utterances)

    handle: Optional[ServerHandle] = None
    if args.server_url is None:
        if args.ckpt_dir is None and args.worker_script is None:
            raise SystemExit(
                "--ckpt-dir is required when this script spawns the server"
                " (or pass --worker-script to use a stub worker)"
            )
        if args.ckpt_dir is None:
            # Stub worker won't load any model; supply a dummy ckpt path so the
            # supervisor's argument validation passes.
            args.ckpt_dir = Path("/tmp/fake-ckpt")
        handle = _spawn_server(args)
        http_url = handle.http_url
    else:
        http_url = args.server_url.rstrip("/")

    # Bind the streaming pacing flag onto the function so the inner closure can
    # read it without changing the signature.
    _bench_streaming._pace = bool(args.realtime)  # type: ignore[attr-defined]

    all_stats: List[RunStats] = []
    try:
        for sub in args.subroutines:
            print(f"\n=== {sub} ===  (concurrency={args.concurrency}, "
                  f"n={len(samples)})", flush=True)
            if sub == "offline":
                s = asyncio.run(_bench_offline(http_url, samples, args.concurrency))
            elif sub == "whisper":
                s = asyncio.run(_bench_whisper(http_url, samples, args.concurrency))
            elif sub == "streaming":
                s = asyncio.run(
                    _bench_streaming(http_url, samples, args.concurrency, args.chunk_ms)
                )
            else:
                raise SystemExit(f"unknown subroutine: {sub}")
            print(s.pretty(), flush=True)
            all_stats.append(s)
    finally:
        if handle is not None:
            handle.stop()

    if args.output_path:
        out = []
        for s in all_stats:
            out.append({
                "name": s.name,
                "n_ok": s.n_ok,
                "n_rejected": s.n_rejected,
                "n_fail": s.n_fail,
                "wall_s": s.wall_s, "total_audio_s": s.total_audio_s,
                "rtf": (s.total_audio_s / s.wall_s) if s.wall_s > 0 else None,
                "latencies_ms": s.latencies_ms,
                "first_partial_ms": s.first_partial_ms,
                "partial_counts": s.partial_counts,
            })
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(out, indent=2))
        print(f"[bench] wrote {args.output_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
