# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""ASR Engine benchmark routines (offline and streaming transcription).

Metrics
-------
* ``median_ms`` — median wall-clock time (ms) to process *N* utterances.
* ``rtf``       — Real-Time Factor = process_time / total_audio_duration.
                  RTF < 1 means faster-than-real-time.
* ``throughput_utts_per_sec`` — utterances processed per second.

Subroutines
-----------
* ``offline``         — OfflineEngine (batch forward, ctc_prefix_beam).
* ``streaming``       — ASREngine (chunk-by-chunk, ctc_prefix_beam).
* ``offline_wfst``    — OfflineEngine with WFST decoder (requires --wfst-path).
* ``streaming_wfst``  — ASREngine with WFST decoder (requires --wfst-path).
"""

from __future__ import annotations
from benchmarks.routines.bench_utils import BenchResult, OutputWriter
from oasr.engine import ASREngine, EngineConfig, OfflineEngine

import argparse
import glob
import os
import statistics
import time
from typing import Any, List, Optional, Tuple

import torch


SUBROUTINES = ["offline", "streaming", "offline_wfst", "streaming_wfst"]

# ---------------------------------------------------------------------------
# Default sweep configs
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "offline": [
        {"num_utterances": 10, "batch_size": 1},
        {"num_utterances": 10, "batch_size": 4},
        {"num_utterances": 10, "batch_size": 8},
    ],
    "streaming": [
        {"num_utterances": 10, "chunk_size": 8},
        {"num_utterances": 10, "chunk_size": 16},
        {"num_utterances": 10, "chunk_size": 32},
    ],
    "offline_wfst": [
        {"num_utterances": 10, "batch_size": 4},
    ],
    "streaming_wfst": [
        {"num_utterances": 10, "chunk_size": 16},
    ],
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _collect_wav_paths(audio_dir: str, num_utterances: int) -> List[str]:
    """Return up to *num_utterances* sorted .wav paths from *audio_dir*."""
    paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not paths:
        raise RuntimeError(f"No .wav files found in {audio_dir!r}")
    return paths[:num_utterances]


def _get_audio_durations(wav_paths: List[str]) -> List[float]:
    """Return audio durations in seconds (fast header-only read via soundfile)."""
    try:
        import soundfile as sf

        return [sf.info(p).duration for p in wav_paths]
    except ImportError:
        pass

    # Fallback: load waveform (slower but always works)
    import torchaudio

    durations = []
    for p in wav_paths:
        waveform, sr = torchaudio.load(p)
        durations.append(waveform.shape[-1] / sr)
    return durations


def _load_waveforms(
    wav_paths: List[str], sample_rate: int = 16000
) -> Tuple[List[torch.Tensor], List[float]]:
    """Load audio files into CPU float32 waveform tensors (unscaled).

    Parameters
    ----------
    wav_paths : list of str
        Paths to ``.wav`` files.
    sample_rate : int
        Target sample rate; files at a different rate are resampled.

    Returns
    -------
    waveforms : List[Tensor]
        1-D float32 CPU tensors of shape ``(T,)``.
    durations : List[float]
        Audio durations in seconds derived from the loaded waveforms.
    """
    import torchaudio

    waveforms: List[torch.Tensor] = []
    durations: List[float] = []
    for p in wav_paths:
        waveform, sr = torchaudio.load(p)  # (C, T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        wav = waveform.squeeze(0).float()
        waveforms.append(wav)
        durations.append(wav.shape[-1] / sample_rate)
    return waveforms, durations


def _resolve_fst_file(wfst_path: Optional[str]) -> Optional[str]:
    """Resolve WFST directory or file to the HLG.pt path."""
    if wfst_path is None:
        return None
    if os.path.isdir(wfst_path):
        candidate = os.path.join(wfst_path, "HLG.pt")
        return candidate if os.path.exists(candidate) else None
    return wfst_path if os.path.exists(wfst_path) else None


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_offline(
    engine: OfflineEngine,
    waveforms: List[torch.Tensor],
    durations: List[float],
    batch_size: int,
    num_iters: int,
) -> tuple[float, float, float]:
    """Time OfflineEngine over pre-loaded *waveforms*.

    The full waveform list is handed to ``engine.transcribe`` in one call so
    the engine's offline pipeline can overlap CPU feature prep for later
    micro-batches with GPU forward+decode for earlier ones.  ``batch_size`` is
    interpreted as the GPU forward micro-batch size and is forwarded to the
    engine via ``offline_micro_batch_size``; processing the full list in one
    call instead of chunking at the benchmark layer is what lets the
    producer thread keep the GPU continuously fed.

    Audio must be loaded before calling this function so that file I/O is
    excluded from the timed region.

    Returns
    -------
    (median_ms, std_ms, rtf)
    """
    total_duration = sum(durations)

    def _run_all():
        engine.transcribe(waveforms)

    # Warmup
    _run_all()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        _run_all()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    rtf = (median_ms / 1000.0) / total_duration
    return median_ms, std_ms, rtf


def _time_streaming(
    engine: ASREngine,
    waveforms: List[torch.Tensor],
    durations: List[float],
    num_iters: int,
) -> tuple[float, float, float]:
    """Time ASREngine over pre-loaded *waveforms* (one request per file).

    What this measures
    ------------------
    *Backlog-throughput streaming.*  ``add_request(full_waveform)`` does
    **not** run feature extraction — it just splits the waveform into
    per-step audio-sample chunks.  ``engine.run()`` then loops ``step()``
    and each step processes one chunk per active stream:

    1. Batched GPU FBANK across all streams' next-pending chunks (one
       kernel call for the whole pool, never sees audio beyond what's
       already enqueued for each stream).
    2. Per-stream ``forward_chunk_paged`` on the freshly-produced features.
    3. Per-stream streaming CTC decode + cache commit.

    So the timed compute is identical to what a server would do when it
    receives chunks from ``max_batch_size`` concurrent real-time clients
    back-to-back.  Audio is pre-loaded into memory so the number reflects
    compute only, not disk I/O.  RTF = wall_clock / total_audio_seconds —
    lower is better; RTF < 1 means the system keeps up with real time.

    For an RTF number that reflects *single-stream* interactive latency
    (one client sending one chunk every 640 ms), set ``max_batch_size=1``.

    Returns
    -------
    (median_ms, std_ms, rtf)
    """
    total_duration = sum(durations)

    def _run_all():
        for wav in waveforms:
            engine.add_request(wav)
        engine.run()

    # Warmup
    _run_all()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        _run_all()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    rtf = (median_ms / 1000.0) / total_duration
    return median_ms, std_ms, rtf


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


def _run_config(
    *,
    subroutine: str,
    ckpt_dir: str,
    audio_dir: str,
    fst_file: Optional[str],
    num_utterances: int,
    batch_size: int,
    chunk_size: int,
    num_left_chunks: int,
    max_batch_size: int = 32,
    dtype: "torch.dtype" = None,
    num_iters: int,
    output: OutputWriter,
) -> None:
    """Run one benchmark configuration and write results to *output*."""

    if dtype is None:
        dtype = torch.float16

    is_streaming = subroutine.startswith("streaming")
    is_wfst = subroutine.endswith("wfst")

    if is_wfst and fst_file is None:
        print(f"  [SKIP] {subroutine}: --wfst-path required but not provided")
        return

    # Collect audio paths, then pre-load waveforms before starting the engine.
    # This ensures file I/O is excluded from the timed benchmark region.
    try:
        wav_paths = _collect_wav_paths(audio_dir, num_utterances)
    except RuntimeError as exc:
        print(f"  [ERROR] {subroutine}: {exc}")
        return

    try:
        waveforms, durations = _load_waveforms(wav_paths)
    except Exception as exc:
        print(f"  [ERROR] {subroutine}: failed to load audio — {exc}")
        return

    n = len(waveforms)
    avg_dur = sum(durations) / n

    decoder_type = "ctc_wfst" if is_wfst else "ctc_gpu"
    dtype_str = {torch.float16: "float16", torch.bfloat16: "bfloat16",
                 torch.float32: "float32"}.get(dtype, "float16")

    try:
        if is_streaming:
            cfg = EngineConfig(
                ckpt_dir=ckpt_dir,
                device="cuda",
                dtype=dtype,
                decoder_type=decoder_type,
                chunk_size=chunk_size,
                num_left_chunks=num_left_chunks,
                max_batch_size=max_batch_size,
                use_paged_cache=True,
                fst_path=fst_file,
            )
            engine: Any = ASREngine(cfg)
            shape_str = (
                f"N={n}, chunk={chunk_size}, max_bs={max_batch_size}, "
                f"avg_dur={avg_dur:.1f}s"
            )
            median_ms, std_ms, rtf = _time_streaming(
                engine, waveforms, durations, num_iters)
        else:
            cfg = EngineConfig(
                ckpt_dir=ckpt_dir,
                device="cuda",
                dtype=dtype,
                decoder_type=decoder_type,
                fst_path=fst_file,
                offline_micro_batch_size=batch_size,
            )
            engine = OfflineEngine(cfg)
            shape_str = f"N={n}, batch={batch_size}, avg_dur={avg_dur:.1f}s"
            median_ms, std_ms, rtf = _time_offline(
                engine, waveforms, durations, batch_size, num_iters)
    except Exception as exc:
        print(f"  [ERROR] {subroutine}: {exc}")
        import traceback
        traceback.print_exc()
        return

    throughput = n / (median_ms / 1000.0)

    result = BenchResult(
        routine="engine",
        subroutine=subroutine,
        backend=subroutine,
        shape=shape_str,
        dtype=dtype_str,
        median_ms=median_ms,
        std_ms=std_ms,
        extra={
            "rtf": round(rtf, 4),
            "throughput_utts_per_sec": round(throughput, 2),
            "total_audio_s": round(sum(durations), 2),
        },
    )
    output.write_result(result)
    print(f"         RTF={rtf:.4f}  throughput={throughput:.2f} utts/s  "
          f"total_audio={sum(durations):.1f}s")


# ---------------------------------------------------------------------------
# Routine module interface
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    """Add engine-specific CLI arguments to *parser*."""
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Path to WeNet checkpoint directory (required for engine benchmarks)",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory containing .wav files (required for engine benchmarks)",
    )
    parser.add_argument(
        "--wfst-path",
        type=str,
        default=None,
        help="WFST directory (containing HLG.pt) or path to HLG.pt "
             "(required for *_wfst subroutines)",
    )
    parser.add_argument(
        "--num-utterances",
        type=int,
        default=10,
        help="Number of .wav files to include per benchmark run (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Utterances per OfflineEngine.transcribe() call (default: sweep 1/4/8)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Max concurrent streaming requests in the ASREngine scheduler (default: 32)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Encoder chunk size for streaming transcription (default: sweep 8/16/32)",
    )
    parser.add_argument(
        "--num-left-chunks",
        type=int,
        default=-1,
        help="Left context chunks for streaming (-1 = unlimited; default: -1)",
    )


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    """Entry point called by oasr_benchmark.py."""
    subroutine = getattr(args, "subroutine", "offline") or "offline"
    ckpt_dir = getattr(args, "ckpt_dir", None)
    audio_dir = getattr(args, "audio_dir", None)
    wfst_path = getattr(args, "wfst_path", None)
    num_utterances = getattr(args, "num_utterances", 10) or 10
    batch_size = getattr(args, "batch_size", None)
    max_batch_size = getattr(args, "max_batch_size", 32) or 32
    chunk_size = getattr(args, "chunk_size", None)
    num_left_chunks = getattr(args, "num_left_chunks", -1)
    dtype_str = getattr(args, "dtype", "float16") or "float16"
    # E2E benchmarks are slower — default to fewer iterations
    num_iters = min(getattr(args, "num_iters", 5), 20)

    if ckpt_dir is None:
        raise ValueError("--ckpt-dir is required for engine benchmarks")
    if audio_dir is None:
        raise ValueError("--audio-dir is required for engine benchmarks")

    fst_file = _resolve_fst_file(wfst_path)

    from benchmarks.routines.bench_utils import parse_dtype
    dtype = parse_dtype(dtype_str)

    for cfg in _resolve_configs(args, subroutine):
        _run_config(
            subroutine=subroutine,
            ckpt_dir=ckpt_dir,
            audio_dir=audio_dir,
            fst_file=fst_file,
            num_utterances=cfg.get("num_utterances", num_utterances),
            batch_size=cfg.get("batch_size", batch_size or 4),
            chunk_size=cfg.get("chunk_size", chunk_size or 16),
            num_left_chunks=num_left_chunks,
            max_batch_size=max_batch_size,
            dtype=dtype,
            num_iters=num_iters,
            output=output,
        )


def _resolve_configs(
    args: argparse.Namespace, subroutine: str
) -> list[dict[str, Any]]:
    """Return a list of config dicts for the sweep.

    If the user passed explicit shape args, run a single config; otherwise
    fall back to DEFAULT_CONFIGS.
    """
    num_utterances = getattr(args, "num_utterances", None)
    batch_size = getattr(args, "batch_size", None)
    chunk_size = getattr(args, "chunk_size", None)

    if subroutine in ("offline", "offline_wfst") and batch_size is not None:
        return [{"num_utterances": num_utterances or 10, "batch_size": batch_size}]
    if subroutine in ("streaming", "streaming_wfst") and chunk_size is not None:
        return [{"num_utterances": num_utterances or 10, "chunk_size": chunk_size}]

    # Default sweep
    return DEFAULT_CONFIGS.get(subroutine, DEFAULT_CONFIGS["offline"])


# ---------------------------------------------------------------------------
# Standalone entry (backwards-compat / direct invocation)
# ---------------------------------------------------------------------------


def run_standalone() -> None:
    """Run the engine benchmark as a standalone script."""
    parser = argparse.ArgumentParser(
        description="OASR Engine Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline benchmark
  python benchmarks/bench_engine.py \\
      --ckpt-dir /path/to/checkpoint \\
      --audio-dir /path/to/wavs \\
      --subroutines offline

  # Streaming benchmark with specific chunk size
  python benchmarks/bench_engine.py \\
      --ckpt-dir /path/to/checkpoint \\
      --audio-dir /path/to/wavs \\
      --subroutines streaming --chunk-size 16

  # All subroutines including WFST
  python benchmarks/bench_engine.py \\
      --ckpt-dir /path/to/checkpoint \\
      --audio-dir /path/to/wavs \\
      --wfst-path /path/to/lang_bpe \\
      --subroutines offline streaming offline_wfst streaming_wfst
""",
    )
    parser.add_argument("--ckpt-dir", required=True,
                        help="WeNet checkpoint directory")
    parser.add_argument("--audio-dir", required=True,
                        help="Directory with .wav files")
    parser.add_argument(
        "--wfst-path", default=None,
        help="WFST directory (HLG.pt) or path to HLG.pt",
    )
    parser.add_argument("--num-utterances", type=int, default=10,
                        help="Number of .wav files per run (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Utterances per OfflineEngine.transcribe() call (default: 4)")
    parser.add_argument("--max-batch-size", type=int, default=32,
                        help="Max concurrent streaming requests in the ASREngine scheduler "
                             "(default: 32)")
    parser.add_argument("--chunk-size", type=int, default=16,
                        help="Encoder chunk size for streaming (default: 16)")
    parser.add_argument("--num-left-chunks", type=int, default=-1,
                        help="Left-context chunks for streaming; -1 = unlimited (default: -1)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                        default="float16", help="Model precision (default: float16)")
    parser.add_argument("--num-iters", type=int, default=5,
                        help="Number of timed iterations (default: 5)")
    parser.add_argument("--output-path", default=None, help="CSV output path")
    parser.add_argument(
        "--subroutines",
        nargs="+",
        default=["offline", "streaming"],
        choices=SUBROUTINES,
        help="Subroutines to run (default: offline streaming)",
    )
    args = parser.parse_args()

    from benchmarks.routines.bench_utils import parse_dtype
    dtype = parse_dtype(args.dtype)

    fst_file = _resolve_fst_file(args.wfst_path)
    output = OutputWriter(output_path=args.output_path)
    output.write_header("OASR ASR Engine Benchmark")

    for sub in args.subroutines:
        output.write_header(f"--- {sub} ---")
        _run_config(
            subroutine=sub,
            ckpt_dir=args.ckpt_dir,
            audio_dir=args.audio_dir,
            fst_file=fst_file,
            num_utterances=args.num_utterances,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            chunk_size=args.chunk_size,
            num_left_chunks=args.num_left_chunks,
            dtype=dtype,
            num_iters=args.num_iters,
            output=output,
        )

    output.finalize()
