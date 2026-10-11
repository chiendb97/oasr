# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Statistics and the metric vocabulary -- every reported number is defined here.

Summaries are flat ``{"name.stat": value}`` dicts.  Flat keys make a summary a
row (CSV), a diff target (``compare``) and a plot series without a schema walk,
and :data:`METRICS` gives each family its unit, the direction that counts as
better, and one sentence of definition, so ``report`` and ``compare`` never have
to guess whether a bigger number is good.

Percentiles use the repository's one rule -- :func:`benchmarks.core.metrics.percentile`,
nearest rank over ``n - 1`` positions (numpy's ``method="nearest"``), so a p99 here
and a p99 in the engine or service harness mean the same thing.  It is restated
rather than imported because that module imports ``torch``, and ``report`` /
``compare`` must run on a laptop with neither a GPU nor torch;
``tests/benchmarks/test_asrbench_metrics.py`` pins the two to each other.
"""

from __future__ import annotations

import fnmatch
import math
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

#: Percentiles every latency summary carries.
PERCENTILES = (50, 90, 95, 99)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile, ``pct`` in ``[0, 100]`` (``0`` = min, ``100`` = max)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def min_samples_for(pct: float) -> int:
    """Samples needed before ``pct`` is distinguishable from the maximum.

    A p99 over 20 requests *is* the max; reporting it as a p99 invites reading a
    tail that was never observed.  Summaries flag such percentiles.
    """
    tail = 100.0 - pct
    return 1 if tail <= 0 else int(math.ceil(100.0 / tail))


def summarize(values: Iterable[float], prefix: str) -> Dict[str, float]:
    """``n, mean, std, min, p50, p90, p95, p99, max`` of *values* under *prefix*."""
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    out: Dict[str, float] = {f"{prefix}.n": float(len(xs))}
    if not xs:
        return out
    out[f"{prefix}.mean"] = statistics.fmean(xs)
    out[f"{prefix}.std"] = statistics.stdev(xs) if len(xs) > 1 else 0.0
    out[f"{prefix}.min"] = min(xs)
    for p in PERCENTILES:
        out[f"{prefix}.p{p}"] = percentile(xs, p)
    out[f"{prefix}.max"] = max(xs)
    return out


def summarize_reps(values: Sequence[float], prefix: str) -> Dict[str, float]:
    """Across-repetition dispersion of a per-repetition aggregate (e.g. RTFx)."""
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {}
    return {
        f"{prefix}.rep_mean": statistics.fmean(xs),
        f"{prefix}.rep_std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        f"{prefix}.rep_min": min(xs),
        f"{prefix}.rep_max": max(xs),
        f"{prefix}.reps": float(len(xs)),
    }


def rtfx(audio_s: float, wall_s: float) -> Optional[float]:
    """Audio seconds processed per wall second -- higher is faster."""
    return audio_s / wall_s if wall_s > 0 else None


def rtf(wall_s: float, audio_s: float) -> Optional[float]:
    """Wall seconds per audio second (``1 / rtfx``) -- lower is faster."""
    return wall_s / audio_s if audio_s > 0 else None


def low_sample_percentiles(summary: Mapping[str, float]) -> List[str]:
    """Percentile keys whose sample count cannot resolve them (see :func:`min_samples_for`)."""
    flagged = []
    for key, value in summary.items():
        if not key.endswith(".n"):
            continue
        prefix, n = key[:-2], int(value)
        for p in PERCENTILES:
            if f"{prefix}.p{p}" in summary and n < min_samples_for(p):
                flagged.append(f"{prefix}.p{p}")
    return sorted(flagged)


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

LOWER = "lower"
HIGHER = "higher"
INFO = "info"  # descriptive -- never a regression in either direction


@dataclass(frozen=True)
class MetricDef:
    pattern: str  # fnmatch pattern over flat summary keys
    unit: str
    better: str
    description: str


METRICS: List[MetricDef] = [
    # -- request latency -----------------------------------------------------
    MetricDef(
        "latency_ms.*",
        "ms",
        LOWER,
        "End-to-end request latency: completion minus *scheduled* start (open-loop) "
        "or submission (all other benchmarks).",
    ),
    MetricDef(
        "service_ms.*",
        "ms",
        LOWER,
        "Completion minus actual submission; equals latency_ms unless the client "
        "fell behind its arrival schedule.",
    ),
    MetricDef(
        "batch_latency_ms.*",
        "ms",
        LOWER,
        "Wall time of one batch call, first submit to last result.",
    ),
    MetricDef("rtf.*", "s/s", LOWER, "Per-request real-time factor: latency / audio duration."),
    MetricDef(
        "rtfx*",
        "x",
        HIGHER,
        "Aggregate audio seconds processed per wall second over the benchmark.",
    ),
    MetricDef("throughput.req_per_s*", "req/s", HIGHER, "Completed requests per wall second."),
    MetricDef(
        "throughput.audio_s_per_s*", "s/s", HIGHER, "Audio seconds completed per wall second."
    ),
    MetricDef("init_s", "s", LOWER, "Backend initialisation (model load) wall time."),
    MetricDef(
        "first_request_ms", "ms", LOWER, "Latency of the first request after initialisation (cold)."
    ),
    MetricDef(
        "warm_request_ms", "ms", LOWER, "Latency of the same request, repeated warm, for contrast."
    ),
    # -- reliability -----------------------------------------------------------
    MetricDef("requests.*", "count", INFO, "Request counts by outcome."),
    MetricDef("sessions.*", "count", INFO, "Streaming session counts by outcome."),
    MetricDef("error_rate", "ratio", LOWER, "(failed + timeout + rejected) / total."),
    MetricDef("timeout_rate", "ratio", LOWER, "Timed-out requests / total."),
    MetricDef(
        "empty_rate",
        "ratio",
        LOWER,
        "Successful requests that returned an empty transcript -- a fast run that "
        "transcribes nothing is not fast.",
    ),
    # -- load generator health ------------------------------------------------
    MetricDef(
        "send_lag_ms.*",
        "ms",
        INFO,
        "Actual submission minus scheduled submission; large values mean the client, "
        "not the server, set the pace.",
    ),
    MetricDef("offered_rate", "req/s", INFO, "Configured open-loop arrival rate."),
    MetricDef(
        "achieved_ratio",
        "ratio",
        HIGHER,
        "Open loop: completed req/s over offered req/s.  Well below 1 means the "
        "backend could not keep up with the arrival rate (saturation).",
    ),
    MetricDef(
        "queue_wait_ms.*", "ms", LOWER, "Server-side queue wait, when the backend exposes it."
    ),
    MetricDef(
        "batch_size_observed.*", "rows", INFO, "Batch widths the backend actually formed (engine)."
    ),
    MetricDef("padding_ratio", "ratio", INFO, "Input property: padded frames / real frames - 1."),
    MetricDef(
        "engine_pad_fraction.*",
        "ratio",
        INFO,
        "OASR engine: mean padded fraction of the batches it actually formed.",
    ),
    MetricDef(
        "server_samples_dropped",
        "count",
        INFO,
        "Server-side histogram samples dropped (buffer cap); server percentiles undercount.",
    ),
    # -- streaming --------------------------------------------------------------
    MetricDef(
        "first_partial_ms.*",
        "ms",
        LOWER,
        "First non-empty partial minus the session's audio start (includes one chunk "
        "of algorithmic delay).",
    ),
    MetricDef(
        "final_latency_ms.*",
        "ms",
        LOWER,
        "Final transcript minus end of audio (paced) or last chunk sent (unpaced): "
        "how long a user waits after they stop speaking.",
    ),
    MetricDef(
        "emission_delay_ms.*",
        "ms",
        LOWER,
        "Per word: first partial showing it minus the word's end time in the audio "
        "(needs word timestamps).",
    ),
    MetricDef(
        "final_after_last_chunk_ms.*",
        "ms",
        LOWER,
        "Final transcript minus the moment the last chunk was sent: the server's "
        "share of final_latency_ms once the client has delivered everything.",
    ),
    MetricDef("partial_interval_ms.*", "ms", INFO, "Time between consecutive partial updates."),
    MetricDef("concurrent_sessions", "sessions", INFO, "Concurrent streaming sessions."),
    MetricDef("concurrency", "requests", INFO, "Closed-loop concurrent users."),
    MetricDef("partials_per_session.*", "count", INFO, "Partial updates per session."),
    MetricDef("late_chunk_pct", "%", LOWER, "Chunks sent later than the pacing tolerance allows."),
    MetricDef("chunk_lag_ms.*", "ms", INFO, "Chunk send time minus its real-time schedule."),
    # -- resources --------------------------------------------------------------
    MetricDef("cpu_pct.*", "%", INFO, "Process CPU utilisation (100 = one core)."),
    MetricDef("rss_mb.*", "MiB", LOWER, "Process resident memory."),
    MetricDef("sys_mem_mb.*", "MiB", INFO, "System memory in use."),
    MetricDef("gpu_util_pct.*", "%", INFO, "Device utilisation (NVML)."),
    MetricDef("gpu_mem_mb.*", "MiB", LOWER, "Device memory in use (NVML, all processes)."),
    MetricDef(
        "gpu_alloc_mb.*", "MiB", LOWER, "Peak torch allocator memory (in-process backends only)."
    ),
    MetricDef("gpu_reserved_mb.*", "MiB", LOWER, "Peak torch allocator reservation (in-process)."),
    MetricDef("gpu_power_w.*", "W", INFO, "Device power draw (NVML)."),
    MetricDef("energy_j", "J", LOWER, "Device energy over the measured window (power integral)."),
    MetricDef("energy_j_per_audio_h", "J/h", LOWER, "Device energy per hour of audio processed."),
    # -- correctness guardrail --------------------------------------------------
    MetricDef("wer_pct", "%", LOWER, "Corpus WER over the fixed labelled subset."),
    MetricDef(
        "changed_pct", "%", LOWER, "Utterances whose hypothesis differs from the reference run."
    ),
    MetricDef("drift_wer_pct", "%", LOWER, "WER of the hypotheses against the reference run's."),
    # -- inputs ---------------------------------------------------------------------
    MetricDef("audio_s*", "s", INFO, "Audio seconds in the measured phase."),
    MetricDef("wall_s*", "s", INFO, "Wall time of the measured phase."),
]


def lookup(key: str) -> Optional[MetricDef]:
    """The first :class:`MetricDef` whose pattern matches *key*.

    A per-bucket breakdown (``bucket.<name>.latency_ms.p50``) shares the
    definition of the metric it breaks down.
    """
    if key.startswith("bucket."):
        key = key.split(".", 2)[-1]
    for m in METRICS:
        if fnmatch.fnmatchcase(key, m.pattern):
            return m
    return None


def direction(key: str) -> str:
    """``lower`` / ``higher`` / ``info`` for *key*.  Counts and ``.n`` are always info."""
    if key.endswith((".n", ".reps", ".std", ".rep_std")):
        return INFO
    m = lookup(key)
    return m.better if m else INFO


def unit(key: str) -> str:
    m = lookup(key)
    return m.unit if m else ""
