# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The two result schemas every OASR benchmark emits.

Declared once, because the column list used to live in three places -- the CSV
writer, the table in ``benchmarks/README.md``, and the ``/benchmark-kernel``
skill -- and the two copies had already drifted: the README's ``shape`` example
(``B=256_M=200_N=200_K=64``) matched no routine in the tree.

Two categories, not four:

``kernel``
    One op, several backends, a fixed shape.  The question is "how fast is this
    kernel, and how does it compare against torch".

``workload``
    A whole ASR pipeline -- the engine, the server in front of it, a decoder, or
    an accuracy sweep.  The question is "how much audio per second, at what
    latency, at what error rate".  One table for all four so that the
    engine-vs-service comparison ``docs/benchmarks.md`` asks for is a query
    rather than a CSV-to-JSON join by hand.

Both open with the same :data:`ENVELOPE_COLUMNS`, so rows from different
categories can sit in one directory and be joined on ``run_id``.

Compatibility: every column name the previous 12-column kernel schema used
survives here with the same meaning and the same number formatting, so the CSVs
recorded under ``.artifacts/`` stay readable and comparable.  ``bandwidth_tb_s``
keeps its decimal-terabyte definition for the same reason -- renaming it to GB/s
would orphan that corpus for cosmetics.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

#: Bumped when a column is removed or its meaning changes.  Adding a column does
#: not require a bump: a reader keyed on names tolerates new ones, and every
#: consumer in this repo is keyed on names.
SCHEMA_VERSION = 2

CATEGORY_KERNEL = "kernel"
CATEGORY_WORKLOAD = "workload"

#: Harness discriminator for :data:`WORKLOAD_COLUMNS`.
HARNESSES = ("engine", "service", "accuracy", "decoder")

# ---------------------------------------------------------------------------
# Envelope -- identical in both schemas
# ---------------------------------------------------------------------------

ENVELOPE_COLUMNS = [
    "schema_version",
    "category",
    "run_id",
    "timestamp",
    "case_tag",
    "git_commit",
    "device",
    "sm",
]

# ---------------------------------------------------------------------------
# Kernel schema
# ---------------------------------------------------------------------------

KERNEL_COLUMNS = ENVELOPE_COLUMNS + [
    # identity
    "routine",
    "subroutine",
    "backend",
    "ref_backend",
    # problem
    "shape",
    "params",
    "dtype",
    # measurement
    "median_ms",
    "mean_ms",
    "std_ms",
    "min_ms",
    "p99_ms",
    "iters",
    "warmup_iters",
    "timer",
    # derived
    "flops",
    "bytes",
    "tflops",
    "bandwidth_tb_s",
    "arith_intensity",
    "speedup_vs_ref",
    # correctness + repro
    "refcheck",
    "max_abs_diff",
    "repro_command",
]

# ---------------------------------------------------------------------------
# Workload schema
# ---------------------------------------------------------------------------

WORKLOAD_COLUMNS = ENVELOPE_COLUMNS + [
    # harness + identity
    "harness",
    "subroutine",
    # A decoder harness compares implementations the way a kernel benchmark
    # does, so the workload row carries the comparison columns too; the engine
    # and service harnesses leave them empty.
    "backend",
    "ref_backend",
    "speedup_vs_ref",
    "shape",
    "params",
    "architecture",
    "ckpt",
    "manifest",
    "decode_method",
    "decode_options",
    # configuration
    "dtype",
    "max_batch_size",
    "chunk_size",
    "service_mode",
    "cuda_graphs",
    "admit_mode",
    "vad_mode",
    "vad_backend",
    "vad_options",
    "concurrency",
    "transport",
    "wire_encoding",
    "realtime",
    # volume
    "requests_ok",
    "requests_rejected",
    "requests_failed",
    "utterances",
    "frames",
    "audio_s",
    "wall_s",
    # measurement
    "median_ms",
    "std_ms",
    "iters",
    # speed
    "rtfx",
    "throughput_utts_per_s",
    "throughput_req_per_s",
    "throughput_frames_per_s",
    "tokens",
    "tokens_per_s",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "latency_max_ms",
    "first_partial_p50_ms",
    "first_partial_p95_ms",
    "partials_per_req",
    "ticks",
    "tick_p50_ms",
    "tick_p99_ms",
    # quality
    "metric",
    "normalizer",
    "error_rate_pct",
    "substitutions",
    "deletions",
    "insertions",
    "ref_units",
    # repro
    "repro_command",
]

# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

#: Per-column format specs.  The four shared with the previous schema keep their
#: old precision so a diff against an ``.artifacts/`` CSV is about the numbers,
#: not about trailing digits.
_FORMATS: dict[str, str] = {
    "median_ms": ".4f",
    "mean_ms": ".4f",
    "std_ms": ".4f",
    "min_ms": ".4f",
    "p99_ms": ".4f",
    "tflops": ".2f",
    "bandwidth_tb_s": ".4f",
    "arith_intensity": ".2f",
    "speedup_vs_ref": ".4f",
    "max_abs_diff": ".6g",
    "flops": ".6g",
    "bytes": ".0f",
    "rtfx": ".2f",
    "throughput_utts_per_s": ".2f",
    "throughput_req_per_s": ".2f",
    "throughput_frames_per_s": ".2f",
    "tokens_per_s": ".2f",
    "latency_mean_ms": ".1f",
    "latency_p50_ms": ".1f",
    "latency_p90_ms": ".1f",
    "latency_p95_ms": ".1f",
    "latency_p99_ms": ".1f",
    "latency_max_ms": ".1f",
    "first_partial_p50_ms": ".1f",
    "first_partial_p95_ms": ".1f",
    "partials_per_req": ".2f",
    "tick_p50_ms": ".2f",
    "tick_p99_ms": ".2f",
    "audio_s": ".2f",
    "wall_s": ".2f",
    "error_rate_pct": ".3f",
}


def format_cell(column: str, value: Any) -> str:
    """Render one cell.

    ``None`` becomes the empty string rather than ``0`` -- the distinction
    matters.  A depthwise conv has no meaningful FLOP count, and writing
    ``tflops=0.0`` there (as the previous harness did) reads as "measured zero"
    rather than "not applicable".
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    spec = _FORMATS.get(column)
    if spec is not None and isinstance(value, (int, float)):
        return format(float(value), spec)
    return str(value)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class KernelRow:
    """One (config x backend) measurement of a single kernel.

    ``flops`` and ``bytes`` are the inputs, not the outputs: a family declares
    how much work its op does and how many bytes it touches, and ``tflops`` /
    ``bandwidth_tb_s`` / ``arith_intensity`` derive from them here.  That is why
    the six copies of the bandwidth formula and the six copies of the GEMM FLOP
    formula could go -- what stayed per-family is the *model*, which genuinely
    differs, not the division.
    """

    routine: str
    subroutine: str
    backend: str
    shape: str
    params: str
    dtype: str
    median_ms: float
    std_ms: float = 0.0
    mean_ms: Optional[float] = None
    min_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    iters: int = 0
    warmup_iters: int = 0
    timer: str = ""
    flops: Optional[float] = None
    bytes: Optional[int] = None
    tflops: Optional[float] = None
    bandwidth_tb_s: Optional[float] = None
    arith_intensity: Optional[float] = None
    ref_backend: str = ""
    speedup_vs_ref: Optional[float] = None
    refcheck: str = "skip"
    max_abs_diff: Optional[float] = None
    repro_command: str = ""

    def __post_init__(self) -> None:
        secs = self.median_ms * 1e-3
        if secs > 0:
            if self.flops is not None and self.tflops is None:
                self.tflops = self.flops / secs / 1e12
            if self.bytes is not None and self.bandwidth_tb_s is None:
                self.bandwidth_tb_s = self.bytes / secs / 1e12
        if self.arith_intensity is None and self.flops is not None and self.bytes:
            self.arith_intensity = self.flops / self.bytes

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class WorkloadRow:
    """One measurement of an ASR pipeline.

    ``harness`` says which of the four produced it; cells the harness has no
    opinion about stay ``None`` and render empty.  There is exactly one
    real-time figure, :attr:`rtfx` = ``audio_s / wall_s`` -- the previous suite
    had a column named ``rtf`` that meant ``wall/audio`` in the engine and
    ``audio/wall`` in the service, i.e. its own reciprocal depending on which
    file wrote it.
    """

    harness: str
    subroutine: str
    # comparison (decoder harnesses)
    backend: str = ""
    ref_backend: str = ""
    speedup_vs_ref: Optional[float] = None
    shape: str = ""
    params: str = ""
    # identity
    architecture: str = ""
    ckpt: str = ""
    manifest: str = ""
    decode_method: str = ""
    decode_options: str = ""
    # configuration
    dtype: str = ""
    max_batch_size: Optional[int] = None
    chunk_size: Optional[int] = None
    service_mode: str = ""
    cuda_graphs: str = ""
    admit_mode: str = ""
    vad_mode: str = ""
    vad_backend: str = ""
    vad_options: str = ""
    concurrency: Optional[int] = None
    transport: str = ""
    wire_encoding: str = ""
    realtime: Optional[int] = None
    # volume
    requests_ok: Optional[int] = None
    requests_rejected: Optional[int] = None
    requests_failed: Optional[int] = None
    utterances: Optional[int] = None
    frames: Optional[int] = None
    audio_s: Optional[float] = None
    wall_s: Optional[float] = None
    # measurement
    median_ms: Optional[float] = None
    std_ms: Optional[float] = None
    iters: Optional[int] = None
    # speed
    rtfx: Optional[float] = None
    throughput_utts_per_s: Optional[float] = None
    throughput_req_per_s: Optional[float] = None
    throughput_frames_per_s: Optional[float] = None
    tokens: Optional[int] = None
    tokens_per_s: Optional[float] = None
    latency_mean_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p90_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    latency_max_ms: Optional[float] = None
    first_partial_p50_ms: Optional[float] = None
    first_partial_p95_ms: Optional[float] = None
    partials_per_req: Optional[float] = None
    ticks: Optional[int] = None
    tick_p50_ms: Optional[float] = None
    tick_p99_ms: Optional[float] = None
    # quality
    metric: str = ""
    normalizer: str = ""
    error_rate_pct: Optional[float] = None
    substitutions: Optional[int] = None
    deletions: Optional[int] = None
    insertions: Optional[int] = None
    ref_units: Optional[int] = None
    # repro
    repro_command: str = ""

    def __post_init__(self) -> None:
        if self.rtfx is None and self.audio_s and self.wall_s:
            self.rtfx = self.audio_s / self.wall_s
        if self.throughput_utts_per_s is None and self.utterances and self.wall_s:
            self.throughput_utts_per_s = self.utterances / self.wall_s
        if self.throughput_frames_per_s is None and self.frames and self.wall_s:
            self.throughput_frames_per_s = self.frames / self.wall_s

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


COLUMNS_FOR_CATEGORY = {
    CATEGORY_KERNEL: KERNEL_COLUMNS,
    CATEGORY_WORKLOAD: WORKLOAD_COLUMNS,
}
