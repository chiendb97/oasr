# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""How a measurement is taken, in one place.

Three things here that used to be scattered or wrong:

**One sample vector.**  :func:`bench_fn` returns a :class:`Sample` whose median,
mean, sigma, min and p99 all come from the *same* timed loop.  The previous
harness took the median from ``triton.testing.do_bench`` and then ran a second,
shorter CUDA-events pass purely to get a sigma -- so the dispersion it reported
was not the dispersion of the number beside it.

**Iterations mean iterations.**  ``do_bench(warmup=, rep=)`` reads both as
*milliseconds*; the old call passed iteration counts, so ``--iters 30`` quietly
asked for a 30 ms window.  The loop below takes counts and the L2 flush that
makes ``do_bench`` honest is preserved, so medians stay comparable.

**Interleaving is available to everyone.**  :func:`interleaved_rounds` is the
round-rotation protocol ``docs/benchmarks.md`` makes a rule and four routines
had each reimplemented under a different name.  A single-order A/B lets the
second arm run on a warm allocator; rotating the order between rounds is what
stops that showing up as a speedup.
"""

from __future__ import annotations

import dataclasses
import statistics
from typing import Callable, Dict, List, Optional, Sequence

import torch

try:  # Triton supplies the L2-sized scratch buffer and the driver-level flush.
    from triton import runtime as _triton_runtime

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on boxes without triton
    _HAS_TRITON = False

TIMER_CUDA_EVENTS = "cuda_events"
TIMER_TRITON = "triton"

#: Wall-time floor for one measurement.  Below roughly this, a GPU has not left
#: its idle clocks and the number reported is the ramp, not the kernel.
MIN_MEASURE_MS = 20.0

#: Never blow up a sweep because one kernel is microscopic.
MAX_ITERS = 20000

_L2_CACHE: Optional[torch.Tensor] = None


@dataclasses.dataclass
class Sample:
    """Every statistic of one timed loop, from one vector of observations."""

    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    p99_ms: float
    iters: int
    warmup_iters: int
    timer: str

    @classmethod
    def from_times(cls, times_ms: Sequence[float], warmup_iters: int, timer: str) -> "Sample":
        from benchmarks.core.metrics import percentile

        ordered = sorted(times_ms)
        n = len(ordered)
        return cls(
            median_ms=statistics.median(ordered) if n else 0.0,
            mean_ms=statistics.fmean(ordered) if n else 0.0,
            std_ms=statistics.stdev(ordered) if n > 1 else 0.0,
            min_ms=ordered[0] if n else 0.0,
            p99_ms=percentile(ordered, 99.0),
            iters=n,
            warmup_iters=warmup_iters,
            timer=timer,
        )


def _l2_scratch() -> Optional[torch.Tensor]:
    """A buffer large enough that writing it evicts L2.  Allocated once."""
    global _L2_CACHE
    if _L2_CACHE is None and torch.cuda.is_available():
        if _HAS_TRITON:
            _L2_CACHE = _triton_runtime.driver.active.get_empty_cache_for_benchmark()
        else:
            _L2_CACHE = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device="cuda")
    return _L2_CACHE


def flush_l2() -> None:
    """Evict L2 so a re-read of a large tensor costs what it costs cold.

    Without this a kernel that re-reads its input measures an L2 hit on every
    iteration after the first, which flatters anything whose working set fits.
    """
    cache = _l2_scratch()
    if cache is None:
        return
    if _HAS_TRITON:
        _triton_runtime.driver.active.clear_cache(cache)
    else:
        cache.zero_()


def bench_times(
    fn: Callable[[], object],
    warmup_iters: int = 5,
    iters: int = 30,
    timer: str = TIMER_CUDA_EVENTS,
    flush: bool = True,
    min_measure_ms: float = MIN_MEASURE_MS,
) -> List[float]:
    """The primitive: per-iteration times in ms, one entry per measured call.

    Everything else in this module summarises this list.  Keeping the raw
    observations as the primitive is what lets the interleaver pool across
    rounds and still report a real sigma.

    ``iters`` is a floor, not a cap.  A 30 us kernel measured for exactly 30
    iterations is measured over 1 ms of wall time, which is not long enough for
    the part to leave its idle clocks -- the same kernel came out 1.4x slower
    than under a 30 ms budget purely from clock ramp.  ``min_measure_ms`` raises
    the count so the measurement spans enough wall time to be stable; the row
    records how many iterations actually ran.
    """
    if timer == TIMER_TRITON:
        if not _HAS_TRITON:
            raise RuntimeError("--timer triton requested but triton is not installed")
        import triton.testing as triton_testing

        # ``warmup`` and ``rep`` are milliseconds in this API -- passing counts
        # is the bug this replaces, so name the budget for what it is.
        return list(triton_testing.do_bench(fn, warmup=warmup_iters, rep=iters, return_mode="all"))

    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    n = iters
    if min_measure_ms > 0:
        n = max(iters, _iters_for_budget(fn, min_measure_ms, iters))

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        if flush:
            flush_l2()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _iters_for_budget(fn: Callable[[], object], budget_ms: float, floor: int) -> int:
    """How many iterations of *fn* fit in *budget_ms*, estimated from five calls."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        fn()
    end.record()
    torch.cuda.synchronize()
    per_call_ms = start.elapsed_time(end) / 5
    if per_call_ms <= 0:
        return floor
    return min(MAX_ITERS, max(floor, int(budget_ms / per_call_ms)))


def bench_fn(
    fn: Callable[[], object],
    warmup_iters: int = 5,
    iters: int = 30,
    timer: str = TIMER_CUDA_EVENTS,
    flush: bool = True,
    min_measure_ms: float = MIN_MEASURE_MS,
) -> Sample:
    """Time *fn* and return every statistic from one loop."""
    times = bench_times(
        fn,
        warmup_iters=warmup_iters,
        iters=iters,
        timer=timer,
        flush=flush,
        min_measure_ms=min_measure_ms,
    )
    return Sample.from_times(times, warmup_iters, timer)


def interleaved_rounds(
    fns: Dict[str, Callable[[], object]],
    warmup_iters: int = 5,
    iters: int = 30,
    rounds: Optional[int] = None,
    timer: str = TIMER_CUDA_EVENTS,
    flush: bool = True,
    min_measure_ms: float = MIN_MEASURE_MS,
) -> Dict[str, Sample]:
    """Measure several arms with their order rotated between rounds.

    Returns one :class:`Sample` per arm over the pooled per-iteration times, so
    the sigma spans rounds -- which is the point: it covers allocator and clock
    drift that one contiguous block of iterations cannot see.
    """
    names = list(fns)
    if rounds is None:
        rounds = min(5, max(1, iters // 4))
    per_round = max(1, iters // rounds)

    for name in names:
        for _ in range(warmup_iters):
            fns[name]()
    torch.cuda.synchronize()

    pooled: Dict[str, List[float]] = {name: [] for name in names}
    for r in range(rounds):
        shift = r % len(names)
        for name in names[shift:] + names[:shift]:
            pooled[name].extend(
                bench_times(
                    fns[name],
                    warmup_iters=0,
                    iters=per_round,
                    timer=timer,
                    flush=flush,
                    # Split across rounds: the pooled sample, not each round,
                    # is what has to span the wall-time floor.
                    min_measure_ms=min_measure_ms / rounds,
                )
            )
    return {name: Sample.from_times(times, warmup_iters, timer) for name, times in pooled.items()}


def profile_kernel(name: str, fn: Callable[[], object], warmup: int = 3, iters: int = 1) -> None:
    """Run *fn* inside an NVTX range so ``ncu`` / ``nsys`` can isolate it."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(name)
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
