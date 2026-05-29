#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""OASR Serving Benchmark — HTTP + gRPC against the Rust frontend.

Measures end-to-end serving throughput and latency for the OASR ``oasr-server``
binary.  Mirrors the CLI shape of ``benchmarks/bench_engine.py`` so the same
``--ckpt-dir`` / ``--audio-dir`` / ``--num-utterances`` arguments work.

Subroutines (one or more)
-------------------------
``offline``         ``POST /v1/speech:recognize`` (Google STT v1-shaped JSON,
                    audio as base64), N concurrent clients.  Reports RTF,
                    total throughput, p50 / p95 / p99 latency.
``grpc_offline``    Unary ``oasr.speech.v1.Speech/Recognize``.
``grpc_streaming``  Bidi ``oasr.speech.v1.Speech/StreamingRecognize``.  Each
                    client opens one stream, feeds fixed-duration chunks,
                    and consumes interim + final results.

The pre-refactor ``streaming`` (WebSocket) and ``whisper`` (multipart)
subroutines are intentionally removed: ``oasr-server`` now exposes only the
HTTP + gRPC surfaces shaped after Google Cloud Speech-to-Text v1.

Examples
--------
# Quick smoke (auto-spawns oasr-server)
python benchmarks/bench_service.py \\
    --ckpt-dir /data01/kilm/users/chiendb/models/asr/am/20210610_u2pp_conformer_exp_librispeech \\
    --audio-dir /data01/kilm/users/chiendb/data/asr/ljspeech-sr16k-dataset/wavs \\
    --subroutines grpc_streaming \\
    --max-batch-size 64 --num-utterances 2000

# Use a running server (skip spawn)
python benchmarks/bench_service.py \\
    --audio-dir /path/to/wavs --subroutines offline grpc_streaming \\
    --server-url http://127.0.0.1:8080 --num-utterances 500
"""

from __future__ import annotations

import argparse
import asyncio
import base64
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


def _envint(name: str, default):
    """Read an int from ``$name`` if set, else return ``default``.

    Lets ``.env`` (``MAX_BATCH_SIZE``, ``NUM_UTTERANCES``, ``CONCURRENCY``,
    ``CHUNK_MS``) set the default for an argparse int flag — the CLI flag
    still wins because argparse only consults the default when the flag is
    absent.
    """
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        raise SystemExit(f"env var {name}={v!r} is not an int")


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
    wav_bytes: bytes  # on-disk WAV (used by the WAV-encoded HTTP path)


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
    import shutil

    env = os.environ.get("OASR_RS_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    # `pip install` ships the front-end as the `oasr-server` console script
    # (oasr._server_cli:main → oasr._core), which lands on PATH.
    on_path = shutil.which("oasr-server")
    if on_path:
        return Path(on_path)
    # Default location for an editable repo built with `cargo build --release`.
    here = Path(__file__).resolve().parents[1]
    candidate = here / "rust" / "target" / "release" / "oasr-server"
    if candidate.is_file():
        return candidate
    raise SystemExit(
        "oasr-server not found; install via `pip install -e .` (builds oasr._core), "
        "build via `cd rust && cargo build --release`, or set OASR_RS_BIN=/path/to/oasr-server"
    )


_OFFLINE_SUBROUTINES = frozenset({"offline", "grpc_offline"})
_STREAMING_SUBROUTINES = frozenset({"grpc_streaming"})
_REMOVED_SUBROUTINES = {
    "whisper": (
        "the Whisper-compat endpoint /v1/audio/transcriptions is no longer "
        "served — oasr-server now exposes only the Google STT v1-shaped "
        "HTTP + gRPC surface.  Use the `offline` subroutine for /v1/speech:recognize."
    ),
    "streaming": (
        "the WebSocket endpoint /v1/stream is no longer served — switch to "
        "the gRPC bidi `grpc_streaming` subroutine."
    ),
}


def _derive_service_mode(subroutines: List[str]) -> str:
    """Pick the engine ``service_mode`` from the chosen subroutines.

    The Rust frontend wraps a single-mode engine, so all chosen
    subroutines must agree.  ``offline`` / ``grpc_offline`` require
    ``service_mode=offline``; ``grpc_streaming`` requires
    ``service_mode=streaming``.  Mixed sets are rejected.
    """
    for sub in subroutines:
        if sub in _REMOVED_SUBROUTINES:
            raise SystemExit(
                f"subroutine {sub!r} has been removed: {_REMOVED_SUBROUTINES[sub]}"
            )
    has_offline = any(s in _OFFLINE_SUBROUTINES for s in subroutines)
    has_streaming = any(s in _STREAMING_SUBROUTINES for s in subroutines)
    if has_offline and has_streaming:
        raise SystemExit(
            "cannot mix offline and streaming subroutines in one bench "
            "run — the engine runs in one mode per lifecycle.  Run them "
            "in separate invocations."
        )
    return "streaming" if has_streaming else "offline"


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
    if args.num_workers != 1:
        raise SystemExit(
            "--num-workers > 1 is no longer supported: oasr-server now runs one "
            "in-process engine per process.  For multi-GPU, launch N oasr-server "
            "processes manually with distinct --http-bind/--grpc-bind and "
            "CUDA_VISIBLE_DEVICES set per process."
        )
    if args.worker_script is not None:
        raise SystemExit(
            "--worker-script is no longer supported: the Rust binary no longer "
            "spawns a Python subprocess (engine is in-process via PyO3)."
        )
    bin_path = _resolve_server_bin()
    http_bind = args.http_bind
    grpc_bind = args.grpc_bind
    cmd = [
        str(bin_path),
        "--ckpt-dir", str(args.ckpt_dir),
        "--http-bind", http_bind,
        "--grpc-bind", grpc_bind,
        # Engine in-flight cap should comfortably exceed the bench
        # concurrency so admission doesn't reject load with BUSY.
        "--max-concurrent-requests", str(max(args.concurrency * 8, 256)),
        "--log-level", args.server_log_level,
        "--dtype", args.dtype,
        "--service-mode", args.service_mode,
    ]
    if args.max_batch_size is not None:
        cmd.extend(["--max-batch-size", str(args.max_batch_size)])
    if args.chunk_size is not None:
        cmd.extend(["--chunk-size", str(args.chunk_size)])

    print(f"[bench] spawning: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    # The binary picks the active GPU via CUDA_VISIBLE_DEVICES; map the
    # legacy --cuda-devices flag onto it (first device only, since one
    # process == one engine == one GPU).
    if args.cuda_devices:
        first = args.cuda_devices.split(",")[0].strip()
        if first:
            env["CUDA_VISIBLE_DEVICES"] = first
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
    n_rejected: int = 0  # backpressure (503 / RESOURCE_EXHAUSTED) — server says retry later
    n_fail: int = 0  # hard transport failures (TCP reset, decode, etc.)
    wall_s: float = 0.0
    total_audio_s: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)
    first_partial_ms: List[float] = field(default_factory=list)
    partial_counts: List[int] = field(default_factory=list)
    # rejection-cause histogram (engine error code | "transport:<ExcType>")
    error_codes: dict = field(default_factory=dict)

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
        if self.error_codes:
            lines.append("  rejection breakdown:")
            for code, n in sorted(self.error_codes.items(),
                                  key=lambda kv: -kv[1]):
                lines.append(f"    {code}: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Offline benchmark — POST /v1/speech:recognize (JSON + base64 audio)
# ---------------------------------------------------------------------------


_WIRE_TO_ENCODING = {
    "f32_le": "LINEAR32F",
    "i16_le": "LINEAR16",
    "wav": "WAV",
}


def _to_wire_bytes(samples: np.ndarray, wire_encoding: str, wav_bytes: bytes) -> bytes:
    """Encode an f32 mono sample buffer to the chosen wire format.

    ``f32_le`` is a passthrough cast.  ``i16_le`` clips to [-1, 1] then
    scales by 32767 — matches ``oasr-asr::decode_raw_pcm``'s widening (i16
    samples are divided by 32768 on the server side; we scale by 32767
    so a sample of 1.0 lands one count short of saturation).  ``wav``
    forwards the on-disk WAV container so the server picks up the
    embedded sample rate.
    """
    if wire_encoding == "f32_le":
        return samples.astype("<f4", copy=False).tobytes()
    if wire_encoding == "i16_le":
        clipped = np.clip(samples, -1.0, 1.0)
        return (clipped * 32767.0).astype("<i2").tobytes()
    if wire_encoding == "wav":
        return wav_bytes
    raise ValueError(f"unsupported wire encoding: {wire_encoding!r}")


async def _bench_offline(
    http_url: str,
    samples: List[Sample],
    concurrency: int,
    wire_encoding: str,
) -> RunStats:
    import httpx

    enc_name = _WIRE_TO_ENCODING[wire_encoding]
    stats = RunStats(
        name=f"offline (POST /v1/speech:recognize, encoding={enc_name})"
    )
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:

        async def one(sample: Sample) -> None:
            async with sem:
                raw = _to_wire_bytes(sample.samples, wire_encoding, sample.wav_bytes)
                body = {
                    "config": {
                        "encoding": enc_name,
                        "sampleRateHertz": sample.sample_rate,
                        "languageCode": "en-US",
                    },
                    "audio": {
                        "content": base64.b64encode(raw).decode("ascii"),
                    },
                }
                start = time.perf_counter()
                try:
                    resp = await client.post(
                        f"{http_url}/v1/speech:recognize",
                        json=body,
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
# gRPC stub generation
# ---------------------------------------------------------------------------


def _load_grpc_stubs(proto_path: Path):
    """Compile ``rust/proto/oasr_speech_v1.proto`` to a temp dir and import it.

    Keeps the bench self-contained — we don't ship pre-generated stubs in
    the repo.
    """
    import importlib
    import shutil
    import tempfile
    try:
        from grpc_tools import protoc
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "grpc subroutines require grpcio-tools "
            "(install with `pip install grpcio grpcio-tools`)"
        ) from e
    _ = shutil  # silence linter — kept for parity with grpc_stream.py
    out = Path(tempfile.mkdtemp(prefix="oasr-bench-grpc-"))
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
# gRPC offline benchmark — unary Recognize
# ---------------------------------------------------------------------------


async def _bench_grpc_offline(
    grpc_addr: str, samples: List[Sample], concurrency: int, proto_path: Path,
) -> RunStats:
    import grpc

    pb, pb_grpc = _load_grpc_stubs(proto_path)
    stats = RunStats(name="grpc-offline (Recognize unary, LINEAR32F PCM)")
    sem = asyncio.Semaphore(concurrency)

    # Disable gRPC's HTTP proxy honoring — the env may carry an
    # ``http_proxy`` that's irrelevant for the localhost frontend.
    channel_options = [("grpc.enable_http_proxy", 0)]
    async with grpc.aio.insecure_channel(grpc_addr, options=channel_options) as channel:
        stub = pb_grpc.SpeechStub(channel)

        async def one(sample: Sample) -> None:
            async with sem:
                cfg = pb.RecognitionConfig(
                    encoding=pb.RecognitionConfig.LINEAR32F,
                    sample_rate_hertz=sample.sample_rate,
                    language_code="en-US",
                    priority=0,
                )
                audio = pb.RecognitionAudio(
                    content=sample.samples.astype("<f4").tobytes(),
                )
                req = pb.RecognizeRequest(config=cfg, audio=audio)
                start = time.perf_counter()
                try:
                    await stub.Recognize(req, timeout=600.0)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    stats.n_ok += 1
                    stats.latencies_ms.append(elapsed_ms)
                    stats.total_audio_s += sample.duration_s
                except grpc.aio.AioRpcError as exc:
                    code = exc.code()
                    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                        stats.n_rejected += 1
                    else:
                        stats.n_fail += 1
                    key = f"grpc:{code.name}"
                    stats.error_codes[key] = stats.error_codes.get(key, 0) + 1
                except Exception:
                    stats.n_fail += 1

        t0 = time.perf_counter()
        await asyncio.gather(*(one(s) for s in samples))
        stats.wall_s = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# gRPC streaming benchmark — bidi StreamingRecognize
# ---------------------------------------------------------------------------


async def _bench_grpc_streaming(
    grpc_addr: str,
    samples: List[Sample],
    concurrency: int,
    chunk_ms: int,
    proto_path: Path,
    pace_realtime: bool,
) -> RunStats:
    import grpc

    pb, pb_grpc = _load_grpc_stubs(proto_path)
    stats = RunStats(
        name=f"grpc-streaming (StreamingRecognize, chunk={chunk_ms}ms)"
    )
    sem = asyncio.Semaphore(concurrency)

    # Disable gRPC's HTTP proxy honoring — the env may carry an
    # ``http_proxy`` that's irrelevant for the localhost frontend.
    channel_options = [("grpc.enable_http_proxy", 0)]
    async with grpc.aio.insecure_channel(grpc_addr, options=channel_options) as channel:
        stub = pb_grpc.SpeechStub(channel)

        async def one(sample: Sample) -> None:
            async with sem:
                chunk_samples = int(sample.sample_rate * chunk_ms / 1000)
                per_chunk = chunk_ms / 1000.0 if pace_realtime else 0.0
                first_partial_at: Optional[float] = None
                partials = 0
                start = time.perf_counter()

                async def requests():
                    cfg = pb.RecognitionConfig(
                        encoding=pb.RecognitionConfig.LINEAR32F,
                        sample_rate_hertz=sample.sample_rate,
                        language_code="en-US",
                        priority=0,
                    )
                    yield pb.StreamingRecognizeRequest(
                        streaming_config=pb.StreamingRecognitionConfig(
                            config=cfg, interim_results=True,
                        )
                    )
                    last = time.perf_counter()
                    for i in range(0, len(sample.samples), chunk_samples):
                        chunk = sample.samples[i:i + chunk_samples]
                        if chunk.size == 0:
                            break
                        yield pb.StreamingRecognizeRequest(
                            audio_content=chunk.astype("<f4").tobytes()
                        )
                        if per_chunk > 0:
                            elapsed = time.perf_counter() - last
                            if elapsed < per_chunk:
                                await asyncio.sleep(per_chunk - elapsed)
                            last = time.perf_counter()

                try:
                    final_seen = False
                    async for resp in stub.StreamingRecognize(requests()):
                        is_final = any(r.is_final for r in resp.results)
                        if is_final:
                            final_seen = True
                            break
                        if resp.results:
                            partials += 1
                            if first_partial_at is None:
                                first_partial_at = time.perf_counter()
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    if final_seen:
                        stats.n_ok += 1
                        stats.latencies_ms.append(elapsed_ms)
                        stats.total_audio_s += sample.duration_s
                        stats.partial_counts.append(partials)
                        if first_partial_at is not None:
                            stats.first_partial_ms.append(
                                (first_partial_at - start) * 1000.0
                            )
                    else:
                        stats.n_fail += 1
                except grpc.aio.AioRpcError as exc:
                    code = exc.code()
                    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                        stats.n_rejected += 1
                    else:
                        stats.n_fail += 1
                    key = f"grpc:{code.name}"
                    stats.error_codes[key] = stats.error_codes.get(key, 0) + 1
                except Exception:
                    stats.n_fail += 1

        t0 = time.perf_counter()
        await asyncio.gather(*(one(s) for s in samples))
        stats.wall_s = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OASR Serving Benchmark — HTTP + gRPC",
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
        choices=[
            "offline",
            "grpc_offline", "grpc_streaming",
            # Removed surfaces — accepted at the parser level so the
            # error message at runtime is targeted (rather than argparse's
            # generic "invalid choice").
            "whisper", "streaming",
        ],
        help=(
            "Which serving paths to benchmark (default: offline).  All "
            "chosen subroutines must agree on engine mode — "
            "offline/grpc_offline → service_mode=offline; "
            "grpc_streaming → service_mode=streaming."
        ),
    )
    p.add_argument(
        "--proto",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "rust" / "proto" / "oasr_speech_v1.proto",
        help="Path to the gRPC schema (only used by grpc_* subroutines).",
    )
    p.add_argument("--num-utterances", type=int,
                   default=_envint("NUM_UTTERANCES", 200),
                   help="Number of .wav files per subroutine "
                        "(reads $NUM_UTTERANCES if set; default 200)")
    p.add_argument("--concurrency", "-c", type=int,
                   default=_envint("CONCURRENCY", 16),
                   help="Concurrent in-flight requests per subroutine "
                        "(reads $CONCURRENCY if set; default 16)")

    # Streaming-specific
    p.add_argument("--chunk-ms", type=int,
                   default=_envint("CHUNK_MS", 640),
                   help="gRPC bidi chunk size in milliseconds "
                        "(reads $CHUNK_MS if set; default 640)")
    p.add_argument(
        "--realtime",
        type=int, default=0, choices=(0, 1),
        help="If 1, pace streaming chunks at real-time (sleep chunk_ms between sends). "
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
                   help="Removed after the PyO3 in-process migration — accepted "
                        "for back-compat; the script now rejects non-None values.")
    p.add_argument("--max-batch-size", type=int,
                   default=_envint("MAX_BATCH_SIZE", None),
                   help="Engine encoder forward batch size "
                        "(reads $MAX_BATCH_SIZE if set; default: engine config)")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Encoder chunk size (frames) for the engine (default: engine config)")
    p.add_argument("--dtype", default="float16",
                   choices=("float16", "bfloat16", "float32"))
    p.add_argument("--ready-timeout-s", type=int, default=180)
    p.add_argument("--server-log-level", default="warn")
    p.add_argument(
        "--wire-encoding",
        default="i16_le",
        choices=("f32_le", "i16_le", "wav"),
        help="PCM/container wire format for the HTTP offline path. "
             "Default i16_le halves wire bytes vs f32_le; server-side widens "
             "to f32 (see oasr-asr::decode_raw_pcm).  `wav` posts the original "
             "WAV container.",
    )

    p.add_argument("--output-path", default=None, type=Path,
                   help="Optional JSON file path for the result summary")

    return p


def main() -> int:
    args = _build_parser().parse_args()
    args.service_mode = _derive_service_mode(args.subroutines)

    samples = _load_dataset(args.audio_dir, args.num_utterances)

    handle: Optional[ServerHandle] = None
    grpc_addr: str
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
        grpc_addr = handle.grpc_addr
    else:
        http_url = args.server_url.rstrip("/")
        grpc_addr = args.grpc_bind

    all_stats: List[RunStats] = []
    try:
        for sub in args.subroutines:
            print(f"\n=== {sub} ===  (concurrency={args.concurrency}, "
                  f"n={len(samples)}, mode={args.service_mode})", flush=True)
            if sub == "offline":
                s = asyncio.run(
                    _bench_offline(http_url, samples, args.concurrency, args.wire_encoding)
                )
            elif sub == "grpc_offline":
                s = asyncio.run(_bench_grpc_offline(
                    grpc_addr, samples, args.concurrency, args.proto,
                ))
            elif sub == "grpc_streaming":
                s = asyncio.run(_bench_grpc_streaming(
                    grpc_addr, samples, args.concurrency, args.chunk_ms,
                    args.proto, bool(args.realtime),
                ))
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
