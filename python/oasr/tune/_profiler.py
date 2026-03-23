# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Kernel profiling for autotuning."""

import logging
from typing import Callable, List, Optional, Tuple

from ._registry import BackendEntry
from ._types import Tactic, TuneResult

logger = logging.getLogger("oasr.tune")


def _bench_cuda_events(
    fn: Callable, args: tuple, warmup: int, rep: int
) -> Tuple[float, float, float]:
    """Benchmark *fn* using CUDA events. Returns (median, min, max) in ms."""
    import torch

    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    timings: List[float] = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    timings.sort()
    median = timings[len(timings) // 2]
    return median, timings[0], timings[-1]


class Profiler:
    """Benchmarks kernel candidates to select the fastest tactic.

    Uses ``triton.testing.do_bench`` when available, falling back to
    CUDA event timing.
    """

    def __init__(self, warmup: int = 25, rep: int = 100) -> None:
        self.warmup = warmup
        self.rep = rep

    def profile_one(
        self, tactic: Tactic, runner: Callable, args: tuple
    ) -> TuneResult:
        """Benchmark a single tactic by timing its execution."""
        try:
            # Try triton.testing.do_bench first (more accurate)
            try:
                from triton.testing import do_bench

                ms = do_bench(
                    lambda: runner(*args),
                    warmup=self.warmup,
                    rep=self.rep,
                    return_mode="median",
                )
                # do_bench doesn't give us min/max directly; re-run for those
                # would be expensive. Use median as approximation.
                return TuneResult(
                    tactic=tactic,
                    median_ms=ms,
                    min_ms=ms,
                    max_ms=ms,
                    status="ok",
                )
            except ImportError:
                pass

            # Fallback: CUDA events
            median, min_ms, max_ms = _bench_cuda_events(
                runner, args, self.warmup, self.rep
            )
            return TuneResult(
                tactic=tactic,
                median_ms=median,
                min_ms=min_ms,
                max_ms=max_ms,
                status="ok",
            )
        except Exception as exc:
            logger.debug("Tactic %s failed: %s", tactic, exc)
            return TuneResult(
                tactic=tactic,
                median_ms=float("inf"),
                min_ms=float("inf"),
                max_ms=float("inf"),
                status="error",
                error_msg=str(exc),
            )

    def profile_candidates(
        self,
        candidates: List[BackendEntry],
        args: tuple,
    ) -> List[TuneResult]:
        """Profile all candidates and return results sorted by median_ms."""
        results: List[TuneResult] = []
        for entry in candidates:
            try:
                runner = entry.get_runner()
            except Exception as exc:
                logger.debug(
                    "Failed to get runner for %s: %s", entry.tactic, exc
                )
                results.append(
                    TuneResult(
                        tactic=entry.tactic,
                        median_ms=float("inf"),
                        min_ms=float("inf"),
                        max_ms=float("inf"),
                        status="error",
                        error_msg=str(exc),
                    )
                )
                continue
            result = self.profile_one(entry.tactic, runner, args)
            results.append(result)
            logger.info(
                "  %s: %.4f ms (status=%s)",
                entry.tactic.backend,
                result.median_ms,
                result.status,
            )
        results.sort(key=lambda r: r.median_ms)
        return results

    @staticmethod
    def select_best(results: List[TuneResult]) -> Optional[TuneResult]:
        """Return the fastest result with ``status='ok'``, or ``None``."""
        for r in results:
            if r.status == "ok":
                return r
        return None
