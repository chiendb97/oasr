# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Testing and benchmarking utilities for OASR kernels."""

import statistics
from typing import Any, Callable, Optional, Sequence, Tuple

import torch


def bench_gpu_time(
    fn: Callable,
    args: Sequence[Any] = (),
    kwargs: Optional[dict] = None,
    enable_cupti: bool = False,
    dry_run_iters: int = 10,
    repeat_iters: int = 100,
) -> Tuple[float, float]:
    """Benchmark a GPU kernel and return ``(median_seconds, std_seconds)``.

    Uses CUPTI-based timing via ``triton.testing.do_bench`` when available
    and ``enable_cupti=True``, otherwise falls back to CUDA events.

    Parameters
    ----------
    fn : callable
        Function to benchmark. Called as ``fn(*args, **kwargs)``.
    args : sequence
        Positional arguments for *fn*.
    kwargs : dict, optional
        Keyword arguments for *fn*.
    enable_cupti : bool
        If True, prefer Triton's ``do_bench`` (CUPTI-based) when available.
    dry_run_iters : int
        Number of warmup iterations.
    repeat_iters : int
        Number of timed iterations.

    Returns
    -------
    median_s : float
        Median execution time in seconds.
    std_s : float
        Standard deviation in seconds.
    """
    if kwargs is None:
        kwargs = {}

    def _run():
        return fn(*args, **kwargs)

    # Try Triton's do_bench first
    if enable_cupti:
        try:
            import triton.testing as triton_testing

            median_ms = triton_testing.do_bench(
                _run, warmup=dry_run_iters, rep=repeat_iters
            )
            # Quick std estimate via CUDA events
            _, std_s = _cuda_events_bench(
                _run, dry_run_iters=2, num_iters=min(repeat_iters, 10)
            )
            return median_ms * 1e-3, std_s
        except ImportError:
            pass

    # Fallback: CUDA events
    return _cuda_events_bench(
        _run, dry_run_iters=dry_run_iters, num_iters=repeat_iters
    )


def _cuda_events_bench(
    fn: Callable,
    dry_run_iters: int = 10,
    num_iters: int = 100,
) -> Tuple[float, float]:
    """Time *fn* with CUDA events. Returns (median_seconds, std_seconds)."""
    for _ in range(dry_run_iters):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e-3)  # ms -> s

    times.sort()
    median = times[len(times) // 2]
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return median, std
