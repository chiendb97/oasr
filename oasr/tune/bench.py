# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The one benchmark protocol every OASR tuner measures with.

``scripts/tune_asr_gemm.py``, ``oasr.autotune()``, the calibration sweep and
``oasr tune build`` all rank candidates through :func:`measure`, so a number in
a tuning file means the same thing whoever produced it.  Before this there were
two protocols, and the runtime autotuner's -- eager ``triton.testing.do_bench``
with an L2 flush -- misranked ASR-sized GEMMs: every arm is faster than the
~9.6 us it costs to *issue* one, so it read 10-20 us for all of them (2.00x
reported where the truth was 4.6x).

What the protocol does, and why each step is there:

1. **Refuses to run inside a CUDA-graph capture.**  Tuning belongs before
   capture (``AGENTS.md`` rule 11); a timing taken inside one is meaningless.
2. **Times what production runs.**  Each arm is a CUDA graph of *N*
   back-to-back calls (N chosen so a replay is >= ~200 us, clamped to
   [8, 64]); the replay time / N is the per-call GPU time.  Inductor measured
   that one call per replay ranks *worse* than eager timing (Spearman 0.69 vs
   0.77) while >= 5 calls reach 0.94-0.95, which is why the floor is 8.
   Each call is followed by a tiny unrelated kernel whose own cost is measured
   and subtracted (``separator``): a served GEMM sits between norms and
   activations, and a library kernel launched with programmatic dependent
   launch overlaps only a cooperating predecessor -- back-to-back copies of
   itself, which is what an unseparated loop measures.
3. **Rotates operand copies when L2 would lie** (:func:`l2_copies`): the caller
   passes a ``make_call(i)`` bound to copy ``i``, and the graph cycles copies so
   the weights are cold when the model's working set exceeds L2 -- but stay warm
   when it does not, as they would in the served model.
4. **Interleaves arms in rounds** (shuffled per round), so drift -- clocks,
   power cap, a neighbour -- lands on every arm instead of on whichever ran
   last.  The estimator is the median over rounds; sigma is 1.4826 x MAD.
5. **Successive halving**: every arm gets a few rounds, the slower half is
   dropped, the rest get more, until three remain and their relative sigma is
   under ``stop_stdrel``.  *Forced* arms (the default, the incumbent) are never
   dropped, because the speedup gate needs them measured to the end.
6. **Single-replay gate** (``solo_ms``): back-to-back independent launches can
   overlap on the GPU and flatter a low-occupancy tile.  A win that does not
   survive a one-call replay won by self-overlap.
7. **Records the conditions** -- SM clock at start and end, other compute
   processes on the device -- so a contended or throttled measurement is
   visible in the evidence and kept out of a shipped table.
"""

from __future__ import annotations

import contextlib
import gc
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "BenchPolicy",
    "Arm",
    "Measurement",
    "Conditions",
    "measure",
    "measure_issue_us",
    "l2_copies",
    "numerics_error",
    "numerics_ok",
    "pick_winner",
    "StickyDeviceError",
    "conditions",
]

#: Bumped whenever a change to this module could move a measured number.
#: Recorded in every tuning file's provenance (``bench_protocol``).
#: 3: a separator kernel between the calls of every timed graph (``separator``).
PROTOCOL_VERSION = 3


class StickyDeviceError(RuntimeError):
    """A candidate left the CUDA context unusable; later arms cannot be trusted."""


@dataclass(frozen=True)
class BenchPolicy:
    """Knobs of :func:`measure`.  Defaults are the protocol; override only to A/B it."""

    #: Calls per captured graph.  ``None`` sizes it from a quick estimate.
    calls_per_graph: Optional[int] = None
    target_graph_us: float = 200.0
    min_calls: int = 8
    max_calls: int = 64
    #: Rounds per halving stage.  After stage 0 the best half survives, after
    #: stage 1 the best ``finalists``; the last stage runs until the stop rule.
    stage_rounds: Tuple[int, ...] = (3, 6, 15)
    finalists: int = 3
    #: Stop the last stage once every finalist's sigma/median is below this.
    stop_stdrel: float = 0.01
    #: Measure the single-replay (overlap-free) time of the finalists.
    solo: bool = True
    solo_reps: int = 7
    #: Warm-up calls per arm before capture.
    warmup_calls: int = 3
    #: With rotated operand copies, a graph must visit every copy each replay
    #: or the copies it skips stay L2-resident between replays; the graph is
    #: lengthened to the copy count, up to this many calls.
    max_rotation_calls: int = 256
    #: Fixed shuffle seed, so a re-run interleaves identically.
    seed: int = 0
    #: Put a tiny unrelated kernel after every call in a timed graph and subtract
    #: its measured cost.  In a served graph a GEMM's neighbours are norms,
    #: activations and adds; back-to-back copies of one library kernel are not.
    #: cuBLAS's Blackwell kernels (``nvjet_sm100_*``) launch with programmatic
    #: dependent launch and overlap a cooperating predecessor: back-to-back they
    #: measured 1.7-2.3 us on B200, behind one ordinary kernel 0.62-0.66 us
    #: more, while OASR's kernels moved by 0.02 us -- so the back-to-back loop
    #: credited cuBLAS with ~0.65 us per call it cannot have in a model.
    separator: bool = True
    separator_reps: int = 7


@dataclass
class Arm:
    """One candidate: a factory of calls, one per operand copy."""

    name: str
    #: ``make_call(i)`` returns a zero-argument callable bound to operand copy
    #: ``i``.  Most callers ignore ``i`` (one copy); L2-rotating callers index
    #: their copies with it.
    make_call: Callable[[int], Callable[[], Any]]
    #: Never eliminated by halving (the default, the incumbent).
    forced: bool = False
    #: Opaque payload the caller wants back (a tactic, a config).
    payload: object = None


@dataclass
class Measurement:
    name: str
    #: Per-call GPU time, median over rounds, in ms.  ``inf`` on failure.
    median_ms: float = math.inf
    #: 1.4826 x MAD over rounds, in ms.
    sigma_ms: float = 0.0
    #: Rounds measured.
    n: int = 0
    #: Single-replay time (one call per graph, launch constant included).
    solo_ms: Optional[float] = None
    #: Host time to issue one call eagerly, in us (see :func:`measure_issue_us`).
    issue_us: Optional[float] = None
    calls_per_graph: int = 0
    status: str = "ok"
    error: str = ""
    #: The halving stage the arm was eliminated in (``None``: survived).
    eliminated_at: Optional[int] = None
    samples: List[float] = field(default_factory=list)
    payload: object = None

    @property
    def stdrel(self) -> float:
        return (
            self.sigma_ms / self.median_ms
            if self.median_ms and math.isfinite(self.median_ms)
            else math.inf
        )


@dataclass
class Conditions:
    """What the device was doing while a measurement ran."""

    sm_clock_mhz_start: Optional[int] = None
    sm_clock_mhz_end: Optional[int] = None
    other_processes: int = 0
    utilization_pct: Optional[int] = None
    contended: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "sm_clock_mhz": [self.sm_clock_mhz_start, self.sm_clock_mhz_end],
            "other_processes": self.other_processes,
            "contended": self.contended,
        }


# =============================================================================
# Device conditions (NVML, best effort)
# =============================================================================


def _nvml_handle():
    try:
        import pynvml

        pynvml.nvmlInit()
        idx = torch.cuda.current_device()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            entry = visible.split(",")[idx].strip()
            if entry.startswith("GPU-"):
                return pynvml, pynvml.nvmlDeviceGetHandleByUUID(entry)
            idx = int(entry)
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(idx)
    except Exception:  # noqa: BLE001 -- diagnostics must never break a measurement
        return None, None


def _other_processes(procs: Sequence[Any], pid: Optional[int] = None) -> int:
    """How many of NVML's compute processes on the device are not this one.

    NVML reports PIDs in the *host* namespace.  Inside a container this
    process is listed under a PID it cannot see, so comparing against
    ``os.getpid()`` counted it as a neighbour -- every row of the first Modal
    tuning matrix (2026-10-01) said ``contended`` on a dedicated GPU, with the
    noise of an idle one.  When no listed PID is ours, one of them still is: it
    holds a context on this device.
    """
    pids = {int(p.pid) for p in procs}
    me = os.getpid() if pid is None else pid
    return len(pids - {me}) if me in pids else max(0, len(pids) - 1)


def conditions() -> Conditions:
    """A snapshot of clock and contention for this device (fields ``None`` without NVML)."""
    c = Conditions()
    nvml, h = _nvml_handle()
    if h is None:
        return c
    with contextlib.suppress(Exception):
        c.sm_clock_mhz_start = int(nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM))
    with contextlib.suppress(Exception):
        c.other_processes = _other_processes(nvml.nvmlDeviceGetComputeRunningProcesses(h))
    with contextlib.suppress(Exception):
        c.utilization_pct = int(nvml.nvmlDeviceGetUtilizationRates(h).gpu)
    c.contended = c.other_processes > 0
    return c


def _finish_conditions(c: Conditions) -> None:
    nvml, h = _nvml_handle()
    if h is None:
        return
    with contextlib.suppress(Exception):
        c.sm_clock_mhz_end = int(nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM))
    with contextlib.suppress(Exception):
        procs = nvml.nvmlDeviceGetComputeRunningProcesses(h)
        c.other_processes = max(c.other_processes, _other_processes(procs))
    c.contended = c.other_processes > 0


# =============================================================================
# L2 policy and numerics
# =============================================================================


def l2_copies(
    weight_bytes: int,
    *,
    working_set_bytes: Optional[int] = None,
    l2_bytes: Optional[int] = None,
    max_copies: int = 64,
    max_total_bytes: int = 1 << 30,
) -> int:
    """How many copies of an op's weights to rotate through, for honest L2 state.

    The weights of one layer are L2-resident in the served model only if the
    model's whole per-forward weight working set fits: then every other layer's
    weights pass through L2 between two calls of this one and do not evict it.
    If ``working_set_bytes`` (the census knows it) is at most half of L2, the
    weights stay warm and one copy is right.  Otherwise rotate enough copies to
    exceed 3x L2 (the cutlass_profiler rule), bounded by memory.
    """
    if weight_bytes <= 0:
        return 1
    if l2_bytes is None:
        l2_bytes = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 0)
    if l2_bytes <= 0:
        return 1
    ws = working_set_bytes if working_set_bytes is not None else 3 * l2_bytes
    if ws <= l2_bytes // 2:
        return 1
    need = math.ceil(3 * l2_bytes / weight_bytes)
    return int(max(1, min(need, max_copies, max_total_bytes // max(weight_bytes, 1))))


#: Absolute error floor by dtype, below which a difference is rounding.
_ATOL = {torch.float16: 1e-3, torch.bfloat16: 8e-3, torch.float32: 1e-5}


def numerics_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    """``max |out - ref|`` in fp32, ``inf`` when ``out`` holds a NaN or an Inf."""
    diff = out.float() - ref.float()
    if not torch.isfinite(out.float()).all():
        return math.inf
    return float(diff.abs().max().item()) if diff.numel() else 0.0


def numerics_ok(err: float, reference_err: Optional[float], dtype: torch.dtype) -> bool:
    """The numerical gate: within 4x the library's own low-precision error.

    A tile that compiles, launches and writes every row can still be wrong --
    ``block_n=16`` was ~100x off and emptied a transcript -- so no candidate is
    ranked until it passes this.
    """
    if not math.isfinite(err):
        return False
    floor = _ATOL.get(dtype, 1e-3)
    bound = max(4.0 * reference_err, floor) if reference_err is not None else 64 * floor
    return err <= bound


# =============================================================================
# Measurement
# =============================================================================


def _check_not_capturing() -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "oasr.tune.bench.measure() called inside a CUDA-graph capture. Tune before "
            "capture (prewarm), never inside it -- AGENTS.md rule 11."
        )


def _sync_or_sticky(where: str) -> None:
    try:
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        raise StickyDeviceError(f"device unusable after {where}: {exc}") from exc


_SIDE_STREAM: Optional[torch.cuda.Stream] = None


def _side_stream() -> torch.cuda.Stream:
    """One warm-up stream for every capture.

    Not one per capture: the split-K / Stream-K workspace cache is keyed on the
    stream, and spreading warm-ups over the pool of 32 stream handles once
    stranded 30 GiB of workspaces in a sweep.
    """
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    return _SIDE_STREAM


def _capture(calls: Sequence[Callable[[], Any]], warmup: int) -> torch.cuda.CUDAGraph:
    side = _side_stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            for c in calls[: min(len(calls), 4)]:
                c()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _replay_ms(graph: torch.cuda.CUDAGraph, calls: int) -> float:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    graph.replay()
    e.record()
    e.synchronize()
    return float(s.elapsed_time(e)) / calls


_SEPARATOR_BUF: Dict[int, torch.Tensor] = {}


def _separator() -> Callable[[], Any]:
    """One tiny kernel with no programmatic-launch attribute (a 1-element add)."""
    dev = torch.cuda.current_device()
    buf = _SEPARATOR_BUF.get(dev)
    if buf is None:
        buf = _SEPARATOR_BUF[dev] = torch.zeros(1, device="cuda")
    return lambda: buf.add_(1.0)


def _then(call: Callable[[], Any], after: Callable[[], Any]) -> Callable[[], Any]:
    def both():
        call()
        after()

    return both


def _robust(samples: Sequence[float]) -> Tuple[float, float]:
    med = statistics.median(samples)
    mad = statistics.median(abs(x - med) for x in samples) if len(samples) > 1 else 0.0
    return med, 1.4826 * mad


def _auto_calls(arm: Arm, policy: BenchPolicy) -> int:
    """Calls per graph from a quick 8-call capture of *arm*."""
    probe = [arm.make_call(i) for i in range(8)]
    g = _capture(probe, policy.warmup_calls)
    est_ms = min(_replay_ms(g, len(probe)) for _ in range(3))
    del g
    est_us = max(est_ms * 1000.0, 0.1)
    return int(
        min(policy.max_calls, max(policy.min_calls, math.ceil(policy.target_graph_us / est_us)))
    )


def measure(
    arms: Sequence[Arm],
    policy: BenchPolicy = BenchPolicy(),
    *,
    copies: int = 1,
    record_conditions: bool = True,
) -> Tuple[Dict[str, Measurement], Conditions]:
    """Rank *arms* with the protocol described in the module docstring.

    ``copies`` is how many operand copies the arms' ``make_call(i)`` cycle
    through (see :func:`l2_copies`); every captured graph then visits all of
    them, so no copy stays cached between replays.

    Returns ``(measurements by name, conditions)``.  An arm that raises during
    warm-up or capture is ``status="error"`` with ``median_ms=inf``; one that
    leaves the context unusable raises :class:`StickyDeviceError`, because no
    later number in this process can be trusted.
    """
    _check_not_capturing()
    cond = conditions() if record_conditions else Conditions()
    results: Dict[str, Measurement] = {a.name: Measurement(a.name, payload=a.payload) for a in arms}
    alive = list(arms)
    if not alive:
        return results, cond

    # Calls per graph: one number for every arm of this shape, so the launch
    # constant each carries is the same.
    calls = policy.calls_per_graph
    if calls is None:
        calls = policy.min_calls
        for a in alive:
            try:
                calls = _auto_calls(a, policy)
                break
            except Exception:  # noqa: BLE001 -- try the next arm
                _sync_or_sticky(f"sizing probe of {a.name}")
                continue
    if copies > 1:
        calls = max(calls, min(int(copies), policy.max_rotation_calls))

    sep = _separator() if policy.separator else None
    sep_ms = 0.0
    if sep is not None:
        g_sep = _capture([sep] * calls, policy.warmup_calls)
        sep_ms = min(_replay_ms(g_sep, calls) for _ in range(policy.separator_reps))
        del g_sep

    graphs: Dict[str, torch.cuda.CUDAGraph] = {}
    for a in list(alive):
        try:
            seq = [a.make_call(i) for i in range(calls)]
            if sep is not None:
                seq = [_then(c, sep) for c in seq]
            graphs[a.name] = _capture(seq, policy.warmup_calls)
            results[a.name].calls_per_graph = calls
        except StickyDeviceError:
            raise
        except Exception as exc:  # noqa: BLE001
            _sync_or_sticky(f"capturing {a.name}")
            results[a.name].status = "error"
            results[a.name].error = f"{type(exc).__name__}: {exc}"[:500]
            alive.remove(a)

    rng = random.Random(policy.seed)
    stages = list(policy.stage_rounds)
    for stage, rounds in enumerate(stages):
        last = stage == len(stages) - 1
        for r in range(rounds):
            order = list(alive)
            rng.shuffle(order)
            for a in order:
                results[a.name].samples.append(_replay_ms(graphs[a.name], calls) - sep_ms)
            if last and r >= 2:
                finite = [_robust(results[a.name].samples) for a in alive]
                if all(m > 0 and s / m <= policy.stop_stdrel for m, s in finite):
                    break
        for a in alive:
            m, s = _robust(results[a.name].samples)
            results[a.name].median_ms, results[a.name].sigma_ms = m, s
            results[a.name].n = len(results[a.name].samples)
        if last:
            break
        keep_n = (
            max(policy.finalists, math.ceil(len(alive) / 2)) if stage == 0 else policy.finalists
        )
        ranked = sorted(alive, key=lambda a: results[a.name].median_ms)
        survivors = ranked[:keep_n] + [a for a in ranked[keep_n:] if a.forced]
        for a in alive:
            if a not in survivors:
                results[a.name].eliminated_at = stage
                graphs.pop(a.name, None)
        alive = survivors

    del graphs
    gc.collect()

    if policy.solo:
        for a in alive:
            try:
                g = _capture([a.make_call(0)], 1)
                results[a.name].solo_ms = min(_replay_ms(g, 1) for _ in range(policy.solo_reps))
                del g
            except Exception:  # noqa: BLE001 -- solo is a gate, not a requirement
                _sync_or_sticky(f"solo replay of {a.name}")
    if record_conditions:
        _finish_conditions(cond)
    return results, cond


def measure_issue_us(call: Callable[[], Any], reps: int = 64) -> float:
    """Host time to *issue* one eager call, in us: the cost CUDA graphs remove.

    Timed with the GPU kept busy by the calls themselves, so the number is the
    Python + launcher overhead and not a wait.  This is what made a faster
    cuBLAS kernel a 0.90x regression on a CPU-issue-bound encoder.
    """
    _check_not_capturing()
    for _ in range(4):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        call()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) * 1e6 / reps


# =============================================================================
# Winner selection
# =============================================================================


def _objective(m: Measurement, alpha: float) -> float:
    issue_ms = (m.issue_us or 0.0) / 1000.0
    return m.median_ms + alpha * issue_ms


def pick_winner(
    results: Dict[str, Measurement],
    *,
    alpha: float = 0.0,
    tie: float = 0.05,
    prefer: Optional[Callable[[Measurement], Tuple]] = None,
) -> Optional[Measurement]:
    """The winner of *results* under ``J = GPU time + alpha * issue time``.

    Among arms within ``max(tie, 2 * sigma_rel)`` of the best J, *prefer*'s key
    decides (default: the lowest J) -- callers pass a structural preference
    (fewer launches, no split-K, less workspace, smaller tile).  ``alpha`` is
    the probability the call runs eagerly: 0 for a graph-captured shape, where
    issue cost is free, and 1 for an eager one.
    """
    ok = [
        m
        for m in results.values()
        if m.status == "ok" and math.isfinite(m.median_ms) and m.eliminated_at is None
    ]
    if not ok:
        ok = [m for m in results.values() if m.status == "ok" and math.isfinite(m.median_ms)]
    if not ok:
        return None
    best = min(ok, key=lambda m: _objective(m, alpha))
    jbest = _objective(best, alpha)
    band = max(tie, 2.0 * best.stdrel if math.isfinite(best.stdrel) else tie)
    near = [m for m in ok if _objective(m, alpha) <= jbest * (1.0 + band)]
    if prefer is None:
        return best
    return min(near, key=lambda m: (prefer(m), _objective(m, alpha)))


def speedup(winner: Measurement, baseline: Measurement, alpha: float = 0.0) -> float:
    """``J(baseline) / J(winner)``."""
    jw = _objective(winner, alpha)
    return _objective(baseline, alpha) / jw if jw > 0 else 1.0


def why_not_replaced(
    winner: Measurement,
    incumbent: Measurement,
    *,
    alpha: float = 0.0,
    min_speedup: float = 1.05,
) -> str:
    """Which of :func:`replaces`'s tests refused *winner* -- for the build's notes.

    A loop-graph win the single replay does not confirm reads very differently
    from a win under the gate, and one message for all three hid which it was.
    """
    noise = 1.0 + 2.0 * max(winner.stdrel, incumbent.stdrel)
    if not math.isfinite(noise):
        noise = 1.0
    sp = speedup(winner, incumbent, alpha)
    if sp < min_speedup:
        return f"under the gate {min_speedup}"
    if sp < noise:
        return f"within the noise ({noise:.2f}x)"
    if winner.solo_ms is not None and incumbent.solo_ms is not None:
        if winner.solo_ms >= incumbent.solo_ms:
            return (
                f"single replay disagrees ({winner.solo_ms * 1e3:.1f} vs "
                f"{incumbent.solo_ms * 1e3:.1f} us: self-overlap)"
            )
    return "accepted"


def replaces(
    winner: Measurement,
    incumbent: Measurement,
    *,
    alpha: float = 0.0,
    min_speedup: float = 1.05,
) -> bool:
    """Whether *winner* may replace *incumbent*: by more than the gate and the noise.

    Also refuses a win that only holds back-to-back: if the single-replay
    measurement says the incumbent is at least as fast, the loop win was
    self-overlap.
    """
    noise = 1.0 + 2.0 * max(winner.stdrel, incumbent.stdrel)
    if not math.isfinite(noise):
        noise = 1.0
    if speedup(winner, incumbent, alpha) < max(min_speedup, noise):
        return False
    if winner.solo_ms is not None and incumbent.solo_ms is not None:
        if winner.solo_ms >= incumbent.solo_ms:
            return False
    return True
