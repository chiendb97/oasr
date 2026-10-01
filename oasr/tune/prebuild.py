# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Compile the JIT modules a tuning build loads, ahead of time and without a GPU.

A build compiles its kernels on first *call*, so on a cold cache the first
minutes of a tuning run are nvcc -- on the GPU box, where they are the most
expensive CPU time there is, and where a CUTLASS template that does not compile
for the architecture fails the run only after the GPU was paid for.  Nothing
about compiling needs the device: ``JitSpec``'s cache key covers the sources,
the project headers, the flags (the ``-gencode`` target among them) and the
nvcc identity, so a CPU box that names the target through
``OASR_CUDA_ARCH_LIST`` produces the very libraries the GPU run then finds.

What is compiled is what :func:`oasr.tune.build.build_gemm` loads for the ops it
measures (:data:`~oasr.tune.build.TUNED_OPS`): the production GEMM module, the
tuning module (the space minus production), and for the CTC head the composed
GEMM + online log_softmax pair.  Modules build **concurrently**, the job budget
split by translation-unit count: built one after another, every module's tail
leaves reserved cores idle, and a reserved core bills whether or not it works.
"""

from __future__ import annotations

import contextlib
import functools
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = ["Prebuilt", "modules_for", "prebuild"]


@dataclass
class Prebuilt:
    """One module's outcome."""

    name: str
    lib: str
    sources: int
    jobs: int
    #: "cached" (already in the JIT cache), "built", or "failed".
    status: str
    seconds: float = 0.0
    error: str = ""

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def modules_for(ops: Iterable[str]) -> List[Tuple[str, Any]]:
    """``(name, JitSpec)`` for every module a build over *ops* loads."""
    import oasr.jit.gemm as jg
    from oasr.tune.build import TUNED_OPS

    wanted = set(ops) & set(TUNED_OPS)
    specs: List[Tuple[str, Any]] = []
    if wanted:
        specs.append(("gemm", jg.gen_gemm_module("production")))
        if jg.has_tuning_module():
            specs.append(("gemm_tune", jg.gen_gemm_module("tune")))
    if "gemm_log_softmax" in wanted:
        from oasr.jit.softmax import gen_softmax_module

        specs.append(("gemm_log_softmax", jg.gen_gemm_log_softmax_module()))
        specs.append(("softmax", gen_softmax_module()))
    return specs


def _split_jobs(sizes: List[int], jobs: int) -> List[int]:
    """Each module's share of *jobs*, proportional to its TU count (at least 1)."""
    total = sum(sizes) or 1
    return [max(1, min(n, round(jobs * n / total))) for n in sizes]


def prebuild(ops: Iterable[str], jobs: int, progress=None) -> List[Prebuilt]:
    """Compile (never load) every module a build over *ops* needs.

    Raises nothing on a compile failure: each module reports its own status, so
    one architecture's broken template is a line in the report rather than the
    end of the others.
    """
    from oasr.jit.core import require_known_cuda_arch
    from oasr.jit.cubin_loader import locked_compile

    require_known_cuda_arch("the tuning modules")
    specs = modules_for(ops)
    pending, results = [], []
    for name, spec in specs:
        lib = spec._get_lib_path()
        if lib.exists():
            results.append(Prebuilt(name, str(lib), len(spec.sources), 0, "cached"))
        else:
            pending.append((name, spec, lib))
    shares = _split_jobs([len(s.sources) for _, s, _ in pending], jobs)

    def _one(item) -> Prebuilt:
        (name, spec, lib), j = item
        t0 = time.time()
        rec = Prebuilt(name, str(lib), len(spec.sources), j, "built")
        try:
            locked_compile(str(lib), functools.partial(spec._compile, jobs=j))
        except Exception as exc:  # noqa: BLE001 -- reported per module
            rec.status = "failed"
            rec.error = str(exc)[-4000:]
        rec.seconds = round(time.time() - t0, 1)
        if progress is not None:
            progress(f"{name}: {rec.status} in {rec.seconds:.0f}s ({rec.sources} TUs, -j{j})")
        return rec

    if pending:
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            results.extend(pool.map(_one, zip(pending, shares)))
    return results


def peak_child_rss_mib() -> Optional[float]:
    """Peak RSS of the largest compiler process this process waited on, in MiB.

    What sizes a compile box's memory: the budget is jobs x this, not the sum of
    every process that ever ran.
    """
    with contextlib.suppress(Exception):
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    return None
