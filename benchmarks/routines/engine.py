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
* ``offline``         — ``ASREngine.transcribe_offline`` (batch forward, ctc_cuda).
* ``streaming``       — ``ASREngine.transcribe`` (chunk-by-chunk, ctc_cuda).
* ``offline_wfst``    — offline path with WFST decoder (requires --wfst-path).
* ``streaming_wfst``  — streaming path with WFST decoder (requires --wfst-path).

Per-decode-family subroutines (see :data:`FAMILY_SUBROUTINES`) cover the
non-CTC paradigms: ``transducer_offline`` / ``transducer_streaming``,
``paraformer_offline``, ``rescoring_offline``, ``aed_offline``, ``llm_offline``.
Each requires a ``--ckpt-dir`` whose checkpoint advertises the matching
capability and **skips with a message** otherwise, so a family subroutine can
never silently report CTC numbers under a non-CTC name.

They are timed with an explicit ``engine.step()`` loop rather than ``run()``, and
report two extra metrics the AR families live or die by:

* ``tick_p50_ms`` / ``tick_p99_ms`` — per-``step()`` wall time. One tick is what
  the serving dispatcher holds the GIL for, so its p99 bounds cancel latency,
  admission latency, and the streaming-partial cadence. For an incremental
  strategy a tick is up to ``decode_steps_per_tick`` batched decoder steps, which
  is why the number is model-dependent and has to be measured.
* ``tokens_per_sec`` / ``tokens`` — generated (or emitted) tokens, the unit of
  work for label-synchronous decoding, where ``utts/s`` hides transcript length.
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import time
from typing import Any, List, Optional, Tuple

import torch

from benchmarks.routines.bench_utils import BenchResult, OutputWriter
from oasr.engine import ASREngine, EngineConfig

#: Per-decode-family subroutines: ``name → (service_mode, capability,
#: decode_method)``.  ``capability`` is the entry in ``model.capabilities`` the
#: checkpoint must advertise; ``decode_method`` is ``None`` when the model's
#: default already selects the family (only rescoring must be opted into, since
#: a U2++ hybrid defaults to plain CTC).
FAMILY_SUBROUTINES: dict[str, tuple[str, str, Optional[str]]] = {
    "transducer_offline": ("offline", "transducer", None),
    "transducer_streaming": ("streaming", "transducer", None),
    "paraformer_offline": ("offline", "paraformer", None),
    "rescoring_offline": ("offline", "ctc_aed_rescoring", "ctc_aed_rescoring"),
    "aed_offline": ("offline", "aed", None),
    "llm_offline": ("offline", "llm", None),
}

SUBROUTINES = [
    "offline",
    "streaming",
    "offline_wfst",
    "streaming_wfst",
    "offline_packing",
    "offline_length_batch",
    *FAMILY_SUBROUTINES,
]

# ---------------------------------------------------------------------------
# Default sweep configs
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "offline": [
        {"num_utterances": 10, "max_batch_size": 1},
        {"num_utterances": 10, "max_batch_size": 4},
        {"num_utterances": 10, "max_batch_size": 8},
    ],
    "streaming": [
        {"num_utterances": 10, "chunk_size": 8},
        {"num_utterances": 10, "chunk_size": 16},
        {"num_utterances": 10, "chunk_size": 32},
    ],
    "offline_wfst": [
        {"num_utterances": 10, "max_batch_size": 4},
    ],
    "streaming_wfst": [
        {"num_utterances": 10, "chunk_size": 16},
    ],
    "offline_packing": [
        {"num_utterances": 10, "max_batch_size": 32},
    ],
    "offline_length_batch": [
        {"num_utterances": 10, "max_batch_size": 32},
    ],
    # Family sweeps: batch width is the axis that matters for the AR families
    # (a decoder step's cost is dominated by the weight read, so it amortises
    # across the batch — the whole point of continuous batching).
    "transducer_offline": [
        {"num_utterances": 10, "max_batch_size": 4},
        {"num_utterances": 10, "max_batch_size": 16},
    ],
    "transducer_streaming": [
        {"num_utterances": 10, "chunk_size": 16},
    ],
    "paraformer_offline": [
        {"num_utterances": 10, "max_batch_size": 8},
        {"num_utterances": 10, "max_batch_size": 32},
    ],
    "rescoring_offline": [
        {"num_utterances": 10, "max_batch_size": 8},
    ],
    "aed_offline": [
        {"num_utterances": 10, "max_batch_size": 4},
        {"num_utterances": 10, "max_batch_size": 8},
    ],
    "llm_offline": [
        {"num_utterances": 8, "max_batch_size": 4},
        {"num_utterances": 8, "max_batch_size": 8},
    ],
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


def _is_streaming_subroutine(subroutine: str) -> bool:
    """Whether *subroutine* runs the engine in streaming mode.

    Family subroutines declare their mode in :data:`FAMILY_SUBROUTINES` (e.g.
    ``transducer_streaming``), so the name-prefix rule alone is not enough.
    """
    family = FAMILY_SUBROUTINES.get(subroutine)
    if family is not None:
        return family[0] == "streaming"
    return subroutine.startswith("streaming")


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
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
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
    engine: ASREngine,
    waveforms: List[torch.Tensor],
    durations: List[float],
    num_iters: int,
) -> tuple[float, float, float]:
    """Time the offline path on ``engine`` over pre-loaded *waveforms*.

    The full waveform list is handed to ``engine.transcribe_offline`` in one
    call so the engine's offline pipeline can overlap CPU feature prep for
    later micro-batches with GPU forward+decode for earlier ones.  The GPU
    forward width comes from ``EngineConfig.max_batch_size`` (configured by
    the caller before ``engine`` is built); processing the full list in one
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
        engine.transcribe_offline(waveforms)

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


def _time_offline_stepwise(
    engine: ASREngine,
    waveforms: List[torch.Tensor],
    durations: List[float],
    num_iters: int,
    admit_mode: str = "burst",
) -> tuple[float, float, float, dict[str, Any]]:
    """Offline timing with an explicit ``step()`` loop, for the decode families.

    ``transcribe_offline`` hides the tick structure behind ``run()``, but for the
    incremental families the tick *is* the unit that matters: the serving
    dispatcher holds the GIL for exactly one ``step()``, so per-tick p99 bounds
    how long a cancel, an admission, or a partial can be delayed.  This drives
    the same public API the dispatcher does — ``add_request`` then ``step()``
    until the engine drains — and times each tick individually.

    ``admit_mode`` controls *when* requests arrive, which matters a great deal for
    the incremental families:

    * ``"burst"`` — everything up front, so the scheduler forms one wide batch.
      The flattering case, and what a batch-transcription job looks like.
    * ``"trickle"`` — one request per tick, reproducing an interactive service
      where arrivals are independent. Each arrival is prefilled separately, so
      this is the case that exposes how well the strategy batches *across*
      independently-admitted requests (keystone C2).

    Comparing the two isolates batching efficiency from raw decoder speed: the
    same total work, the same tokens, only the arrival pattern differs.

    Each tick is followed by ``cuda.synchronize()`` so the GPU work it queued is
    attributed to it rather than to a later tick.  That makes the per-tick numbers
    slightly conservative and the total directly comparable to
    :func:`_time_offline`.

    Returns
    -------
    (median_ms, std_ms, rtf, extra)
        ``extra`` carries ``tick_p50_ms`` / ``tick_p99_ms`` / ``ticks`` /
        ``tokens`` / ``tokens_per_sec`` / ``admit_mode``.
    """
    total_duration = sum(durations)
    cuda = torch.cuda.is_available()
    trickle = admit_mode == "trickle"
    # Generous cap: even a fully serialised AR run finishes well inside this, and
    # it turns a stuck engine into a clear error instead of a hung benchmark.
    max_ticks = 100_000

    def _run_all() -> tuple[list[float], int]:
        pending = list(waveforms)
        if not trickle:
            for w in pending:
                engine.add_request(w, streaming=False)
            pending = []
        ticks: list[float] = []
        tokens = 0
        for _ in range(max_ticks):
            if pending:
                # One arrival per tick — the interactive-service pattern.
                engine.add_request(pending.pop(0), streaming=False)
            elif engine.num_running + engine.num_waiting == 0:
                break
            t0 = time.perf_counter()
            outputs = engine.step()
            if cuda:
                torch.cuda.synchronize()
            ticks.append((time.perf_counter() - t0) * 1000.0)
            for out in outputs:
                if out.finished and out.tokens:
                    tokens += len(out.tokens[0])
        else:
            raise RuntimeError(
                f"engine did not drain within {max_ticks} steps "
                f"(running={engine.num_running}, waiting={engine.num_waiting})"
            )
        return ticks, tokens

    # Warmup (also absorbs any lazy JIT / graph capture on this path).
    _run_all()
    if cuda:
        torch.cuda.synchronize()

    times_ms: list[float] = []
    all_ticks: list[float] = []
    tokens_total = 0
    for _ in range(num_iters):
        t0 = time.perf_counter()
        ticks, tokens = _run_all()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        all_ticks.extend(ticks)
        tokens_total += tokens

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    rtf = (median_ms / 1000.0) / total_duration

    all_ticks.sort()
    extra = {
        "admit_mode": admit_mode,
        "ticks": len(all_ticks) // max(1, num_iters),
        "tick_p50_ms": round(_percentile(all_ticks, 50), 3),
        "tick_p99_ms": round(_percentile(all_ticks, 99), 3),
        "tokens": tokens_total // max(1, num_iters),
        "tokens_per_sec": round((tokens_total / max(1, num_iters)) / (median_ms / 1000.0), 2),
    }
    return median_ms, std_ms, rtf, extra


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list (0.0 when empty)."""
    if not sorted_values:
        return 0.0
    idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def _warmup_engine(
    engine: Any,
    waveforms: List[torch.Tensor],
    *,
    is_streaming: bool,
) -> None:
    """Warm up FlexAttention's ``torch.compile`` cache.

    The first encoder forward through FlexAttention triggers a Triton
    kernel compile (and a separate compile for the BlockMask
    constructor). Both are wrapped with ``dynamic=True`` so a single
    compile covers a wide range of subsequent shapes, but the first
    invocation still pays the compile cost — billing it to the timed
    region distorts measurements badly.

    This helper runs the engine on a small, **representative** subset
    of the workload — short and long waveforms, single and batched —
    so the steady-state shapes have been compiled before timing starts.
    The engine is reset between phases so internal state doesn't carry
    over.
    """
    if not waveforms:
        return

    # Pick a short and a long sample so both small-T_kv and large-T_kv
    # FlexAttention shapes get traced.
    sorted_by_len = sorted(waveforms, key=lambda w: w.numel())
    short = sorted_by_len[0]
    long = sorted_by_len[-1]
    samples = [short, long]

    if is_streaming:
        chunk_samples = engine._input_processor.streaming_audio_chunk_samples  # type: ignore[attr-defined]
        # Single-stream phase: B=1 path.
        rid = engine.add_streaming_request(sample_rate=16000)
        chunks = _split_waveform_into_chunks(short, chunk_samples)
        for j, c in enumerate(chunks):
            engine.feed_chunk(rid, c, is_last=(j == len(chunks) - 1))
        engine.run()

        # Multi-stream phase: hits the batched paged path.
        rids = [engine.add_streaming_request(sample_rate=16000) for _ in samples]
        for rid_, wav in zip(rids, samples):
            chs = _split_waveform_into_chunks(wav, chunk_samples)
            for j, c in enumerate(chs):
                engine.feed_chunk(rid_, c, is_last=(j == len(chs) - 1))
        engine.run()
    else:
        # Offline path: run a tiny batch through transcribe_offline(...).
        engine.transcribe_offline(samples)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _split_waveform_into_chunks(wav: torch.Tensor, chunk_samples: int) -> List[torch.Tensor]:
    """Split a 1-D waveform into per-call chunks of ``chunk_samples`` samples.

    The final chunk may be shorter.  Chunks are contiguous CPU float32 views
    that the engine can hand directly to ``feed_chunk`` without further copies.
    """
    n = wav.numel()
    chunks: List[torch.Tensor] = []
    for start in range(0, n, chunk_samples):
        chunks.append(wav[start : start + chunk_samples].contiguous())
    return chunks


def _time_streaming(
    engine: ASREngine,
    waveforms: List[torch.Tensor],
    durations: List[float],
    num_iters: int,
) -> tuple[float, float, float]:
    """Time ASREngine over pre-loaded *waveforms* fed chunk-by-chunk.

    What this measures
    ------------------
    *Backlog-throughput streaming.*  Each waveform is pre-split (outside the
    timed region) into per-call audio chunks of size ``stride * frame_shift``
    samples.  Inside the timed region we register a streaming request and
    push chunks via :meth:`engine.feed_chunk`, exactly mirroring how a
    real-time client would deliver audio.  ``engine.run()`` then drains the
    backlog by looping ``step()``; each step processes one chunk per active
    stream:

    1. Batched GPU FBANK across all streams' next-pending chunks (one
       kernel call for the whole pool, never sees audio beyond what's
       already enqueued for each stream).
    2. Per-stream ``forward_chunk_paged`` on the freshly-produced features.
    3. Per-stream streaming CTC decode + cache commit.

    Audio is pre-loaded into memory so the number reflects compute only,
    not disk I/O.  RTF = wall_clock / total_audio_seconds — lower is better;
    RTF < 1 means the system keeps up with real time.

    For an RTF number that reflects *single-stream* interactive latency
    (one client sending one chunk every 640 ms), set ``max_batch_size=1``.

    Returns
    -------
    (median_ms, std_ms, rtf)
    """
    total_duration = sum(durations)

    # Pre-split waveforms once so chunking cost is excluded from the timed
    # region.  Chunk size matches the engine's ``stride * frame_shift``
    # (one encoder chunk worth of new audio).
    chunk_samples = engine._input_processor.streaming_audio_chunk_samples  # type: ignore[attr-defined]
    chunks_per_utt: List[List[torch.Tensor]] = [
        _split_waveform_into_chunks(wav, chunk_samples) for wav in waveforms
    ]

    def _run_all():
        for chunks in chunks_per_utt:
            if not chunks:
                continue
            rid = engine.add_streaming_request(sample_rate=16000)
            last = len(chunks) - 1
            for j, c in enumerate(chunks):
                engine.feed_chunk(rid, c, is_last=(j == last))
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
    chunk_size: int,
    num_left_chunks: int,
    max_batch_size: int = 32,
    dtype: "torch.dtype" = None,
    num_iters: int,
    output: OutputWriter,
    use_cuda_graphs: Optional[bool] = None,
    max_packed_frames: int = 6000,
    max_batch_frames: Optional[int] = None,
    admit_mode: str = "burst",
    decode_admit_window_ms: float = 0.0,
) -> None:
    """Run one benchmark configuration and write results to *output*."""

    if dtype is None:
        dtype = torch.bfloat16

    family = FAMILY_SUBROUTINES.get(subroutine)
    if family is not None:
        family_mode, capability, decode_method = family
        is_streaming = family_mode == "streaming"
    else:
        capability = decode_method = None
        is_streaming = subroutine.startswith("streaming")
    is_wfst = subroutine.endswith("wfst")
    is_packing = subroutine == "offline_packing"
    is_length_batch = subroutine == "offline_length_batch"

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

    decoder_type = "ctc_wfst" if is_wfst else "ctc_cuda"
    dtype_str = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }.get(dtype, "float16")

    extra_metrics: dict[str, Any] = {}
    try:
        if is_streaming:
            cfg_kwargs = {
                "ckpt_dir": ckpt_dir,
                "device": "cuda",
                "dtype": dtype,
                "service_mode": "streaming",
                "decoder_type": decoder_type,
                "chunk_size": chunk_size,
                "num_left_chunks": num_left_chunks,
                "max_batch_size": max_batch_size,
                "fst_path": fst_file,
            }
            if decode_method is not None:
                cfg_kwargs["decode_method"] = decode_method
            if use_cuda_graphs is not None:
                cfg_kwargs["use_cuda_graphs"] = use_cuda_graphs
            cfg = EngineConfig(**cfg_kwargs)
            engine: Any = ASREngine(cfg)
            if not _family_matches(engine, subroutine, capability):
                return
            shape_str = (
                f"N={n}, chunk={chunk_size}, max_bs={max_batch_size}, " f"avg_dur={avg_dur:.1f}s"
            )
            _warmup_engine(engine, waveforms, is_streaming=True)
            median_ms, std_ms, rtf = _time_streaming(engine, waveforms, durations, num_iters)
        else:
            cfg = EngineConfig(
                ckpt_dir=ckpt_dir,
                device="cuda",
                dtype=dtype,
                service_mode="offline",
                decoder_type=decoder_type,
                fst_path=fst_file,
                max_batch_size=max_batch_size,
                enable_sequence_packing=is_packing,
                max_packed_frames=max_packed_frames,
                max_batch_frames=max_batch_frames if is_length_batch else None,
                decode_admit_window_ms=decode_admit_window_ms,
                **({"decode_method": decode_method} if decode_method else {}),
            )
            engine = ASREngine(cfg)
            if not _family_matches(engine, subroutine, capability):
                return
            if is_packing:
                shape_str = f"N={n}, pack_frames={max_packed_frames}, " f"avg_dur={avg_dur:.1f}s"
            elif is_length_batch:
                shape_str = (
                    f"N={n}, batch_frames={max_batch_frames}, "
                    f"max_bs={max_batch_size}, avg_dur={avg_dur:.1f}s"
                )
            else:
                shape_str = f"N={n}, batch={max_batch_size}, avg_dur={avg_dur:.1f}s"
                if family is not None:
                    shape_str += f", admit={admit_mode}"
                    if decode_admit_window_ms:
                        shape_str += f"+{decode_admit_window_ms:.0f}ms"
            _warmup_engine(engine, waveforms, is_streaming=False)
            if family is not None:
                # Family subroutines report tick latency + tokens/s on top of
                # the shared metrics; the CTC gates keep the original timing
                # path untouched so their numbers stay comparable over time.
                median_ms, std_ms, rtf, extra_metrics = _time_offline_stepwise(
                    engine, waveforms, durations, num_iters, admit_mode=admit_mode
                )
            else:
                median_ms, std_ms, rtf = _time_offline(engine, waveforms, durations, num_iters)
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
            "rtf": round(rtf, 8),
            "throughput_utts_per_sec": round(throughput, 2),
            "total_audio_s": round(sum(durations), 2),
            **extra_metrics,
        },
    )
    output.write_result(result)
    print(
        f"         RTF={rtf:.8f}  throughput={throughput:.2f} utts/s  "
        f"total_audio={sum(durations):.1f}s"
    )
    if extra_metrics:
        print(
            f"         tokens/s={extra_metrics['tokens_per_sec']:.2f}  "
            f"tokens={extra_metrics['tokens']}  "
            f"ticks={extra_metrics['ticks']}  "
            f"tick_p50={extra_metrics['tick_p50_ms']:.2f}ms  "
            f"tick_p99={extra_metrics['tick_p99_ms']:.2f}ms"
        )


def _family_matches(engine: Any, subroutine: str, capability: Optional[str]) -> bool:
    """Verify the loaded checkpoint actually runs the family this subroutine names.

    Without this a ``--ckpt-dir`` pointing at a CTC Conformer would happily run
    ``aed_offline`` — the engine falls back to the model's default family — and the
    result would be filed under a name it has nothing to do with.  Skips loudly
    instead, naming what the checkpoint does support.
    """
    if capability is None:
        return True
    caps = list(getattr(engine, "capabilities", []))
    resolved = getattr(engine, "decode_method", None)
    if capability not in caps:
        print(
            f"  [SKIP] {subroutine}: this checkpoint advertises {caps} — "
            f"{capability!r} is not among them. Point --ckpt-dir at a "
            f"{capability} checkpoint."
        )
        return False
    if resolved != capability:
        print(
            f"  [SKIP] {subroutine}: engine resolved decode_method={resolved!r}, "
            f"not {capability!r}; refusing to report it under this name."
        )
        return False
    return True


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
        "--max-batch-size",
        type=int,
        default=None,
        help="Encoder forward batch size — streaming concurrent pool cap "
        "and offline pipeline micro-batch width. "
        "Default: 32 for streaming; sweep 1/4/8 for offline.",
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
    parser.add_argument(
        "--max-packed-frames",
        type=int,
        default=6000,
        help="Post-subsampling token budget per packed row for the "
        "offline_packing subroutine (default: 6000).",
    )
    parser.add_argument(
        "--admit-mode",
        type=str,
        default="burst",
        choices=("burst", "trickle"),
        help="Arrival pattern for the per-family subroutines: 'burst' adds every "
        "request up front (one wide scheduler batch); 'trickle' adds one per engine "
        "tick, reproducing independent interactive arrivals. Same work either way — "
        "the delta isolates how well a decode family batches across separately "
        "admitted requests (default: burst).",
    )
    parser.add_argument(
        "--max-batch-frames",
        type=int,
        default=None,
        help="Padded-frame budget (max_len * batch) for the "
        "offline_length_batch subroutine (input feature frames).",
    )


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    """Entry point called by oasr_benchmark.py."""
    subroutine = getattr(args, "subroutine", "offline") or "offline"
    ckpt_dir = getattr(args, "ckpt_dir", None)
    audio_dir = getattr(args, "audio_dir", None)
    wfst_path = getattr(args, "wfst_path", None)
    num_utterances = getattr(args, "num_utterances", 10) or 10
    max_batch_size = getattr(args, "max_batch_size", None)
    chunk_size = getattr(args, "chunk_size", None)
    num_left_chunks = getattr(args, "num_left_chunks", -1)
    max_packed_frames = getattr(args, "max_packed_frames", 6000) or 6000
    max_batch_frames = getattr(args, "max_batch_frames", None)
    dtype_str = getattr(args, "dtype", "bfloat16") or "bfloat16"
    # E2E benchmarks are slower — default to fewer iterations
    num_iters = min(getattr(args, "num_iters", 5), 20)

    if ckpt_dir is None:
        raise ValueError("--ckpt-dir is required for engine benchmarks")
    if audio_dir is None:
        raise ValueError("--audio-dir is required for engine benchmarks")

    fst_file = _resolve_fst_file(wfst_path)

    from benchmarks.routines.bench_utils import parse_dtype

    dtype = parse_dtype(dtype_str)

    is_streaming = _is_streaming_subroutine(subroutine)
    default_mbs = 32 if is_streaming else 4

    for cfg in _resolve_configs(args, subroutine):
        _run_config(
            subroutine=subroutine,
            ckpt_dir=ckpt_dir,
            audio_dir=audio_dir,
            fst_file=fst_file,
            num_utterances=cfg.get("num_utterances", num_utterances),
            chunk_size=cfg.get("chunk_size", chunk_size or 16),
            num_left_chunks=num_left_chunks,
            max_batch_size=cfg.get(
                "max_batch_size", max_batch_size if max_batch_size else default_mbs
            ),
            dtype=dtype,
            num_iters=num_iters,
            output=output,
            max_packed_frames=max_packed_frames,
            max_batch_frames=max_batch_frames,
            admit_mode=getattr(args, "admit_mode", "burst") or "burst",
        )


def _resolve_configs(args: argparse.Namespace, subroutine: str) -> list[dict[str, Any]]:
    """Return a list of config dicts for the sweep.

    If the user passed explicit shape args, run a single config; otherwise
    fall back to DEFAULT_CONFIGS.
    """
    num_utterances = getattr(args, "num_utterances", None)
    max_batch_size = getattr(args, "max_batch_size", None)
    chunk_size = getattr(args, "chunk_size", None)

    if (
        subroutine
        in (
            "offline",
            "offline_wfst",
            "offline_packing",
            "offline_length_batch",
        )
        and max_batch_size is not None
    ):
        return [
            {
                "num_utterances": num_utterances or 10,
                "max_batch_size": max_batch_size,
            }
        ]
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
    parser.add_argument("--ckpt-dir", required=True, help="WeNet checkpoint directory")
    parser.add_argument("--audio-dir", required=True, help="Directory with .wav files")
    parser.add_argument(
        "--wfst-path",
        default=None,
        help="WFST directory (HLG.pt) or path to HLG.pt",
    )
    parser.add_argument(
        "--num-utterances", type=int, default=10, help="Number of .wav files per run (default: 10)"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Encoder forward batch size — streaming concurrent pool "
        "cap and offline pipeline micro-batch width. "
        "Default: 32 for streaming, 4 for offline.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=16, help="Encoder chunk size for streaming (default: 16)"
    )
    parser.add_argument(
        "--num-left-chunks",
        type=int,
        default=-1,
        help="Left-context chunks for streaming; -1 = unlimited (default: -1)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Model precision (default: bfloat16)",
    )
    parser.add_argument(
        "--num-iters", type=int, default=5, help="Number of timed iterations (default: 5)"
    )
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

    explicit_mbs = args.max_batch_size is not None

    for sub in args.subroutines:
        output.write_header(f"--- {sub} ---")
        is_streaming = _is_streaming_subroutine(sub)
        default_mbs = 32 if is_streaming else 4
        max_batch_size = args.max_batch_size if explicit_mbs else default_mbs
        _run_config(
            subroutine=sub,
            ckpt_dir=args.ckpt_dir,
            audio_dir=args.audio_dir,
            fst_file=fst_file,
            num_utterances=args.num_utterances,
            max_batch_size=max_batch_size,
            chunk_size=args.chunk_size,
            num_left_chunks=args.num_left_chunks,
            dtype=dtype,
            num_iters=args.num_iters,
            output=output,
        )

    output.finalize()
