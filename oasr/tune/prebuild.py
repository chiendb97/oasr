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
GEMM + online log_softmax pair -- plus, for a quantized model's census, the int8
``scaled_mm`` module :func:`oasr.tune.build_scaled_mm.build_scaled_mm` measures.  Modules build **concurrently**, the job budget
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

__all__ = ["Prebuilt", "all_modules", "modules_for", "prebuild"]


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
    from oasr.tune.build_scaled_mm import SCALED_MM_TUNED_OPS

    ops = set(ops)
    wanted = ops & set(TUNED_OPS)
    specs: List[Tuple[str, Any]] = []
    if wanted:
        specs.append(("gemm", jg.gen_gemm_module("production")))
        if jg.has_tuning_module():
            specs.append(("gemm_tune", jg.gen_gemm_module("tune")))
    if "gemm_log_softmax" in wanted:
        from oasr.jit.softmax import gen_softmax_module

        specs.append(("gemm_log_softmax", jg.gen_gemm_log_softmax_module()))
        specs.append(("softmax", gen_softmax_module()))
    if ops & set(SCALED_MM_TUNED_OPS):
        # One module per format (on this device's lane) holds every variant, fp- and
        # 8-bit-output launchers alike.
        from oasr.jit.scaled_mm import FORMAT_OPS, gen_scaled_mm_module, module_name

        for fmt, fmt_ops in FORMAT_OPS.items():
            if ops & set(fmt_ops):
                specs.append((module_name(fmt), gen_scaled_mm_module(fmt)))
    return specs


def all_modules() -> List[Tuple[str, Any]]:
    """Every module :func:`oasr.aot.gen_all_modules` names, plus the tuning modules.

    What a cold test run or a server's first requests load, rather than what one
    tuning build does: ``oasr tune prebuild --all``, which is how CI compiles a
    rented GPU's kernels on a CPU container first (``ci/modal_app.py``).  A
    module a test generates for itself (an arm it compares against, a cell
    another census names) is not here, and the GPU run compiles it as usual.
    Deduplicated by library path, since the production GEMM module is in both.
    """
    from oasr.aot import gen_all_modules
    from oasr.tune.build import TUNED_OPS
    from oasr.tune.build_scaled_mm import SCALED_MM_TUNED_OPS

    named = [(spec.name, spec) for spec in gen_all_modules()]
    named += modules_for(set(TUNED_OPS) | set(SCALED_MM_TUNED_OPS))
    named += _mma_sync_arms()
    import oasr.jit.gemm as jg

    if jg.has_tuning_module():
        # The tuning halves of BMM and grouped GEMM, beside the GEMM one above:
        # what the kernels tests load to check every compiled variant.
        named += [
            (s.name, s) for s in (jg.gen_bmm_module("tune"), jg.gen_group_gemm_module("tune"))
        ]
    seen: set = set()
    out: List[Tuple[str, Any]] = []
    for name, spec in named:
        lib = str(spec._get_lib_path())
        if lib not in seen:
            seen.add(lib)
            out.append((name, spec))
    return out


def _mma_sync_arms() -> List[Tuple[str, Any]]:
    """The 8-bit ``mma.sync`` modules where a native lane is the default (sm_90, sm_100).

    ``oasr.aot`` names the lane a device runs; this is the one beside it that
    ``OASR_SCALED_MM_BACKEND=mma_sync`` selects and the native lanes' tests
    compare against.
    """
    from oasr.jit.core import _get_target_sm
    from oasr.jit.scaled_mm import (
        FORMATS,
        LANE_ID,
        default_lane,
        gen_scaled_mm_module,
        lane_available,
        module_name,
    )

    sm = _get_target_sm()
    return [
        (module_name(fmt, LANE_ID[fmt]), gen_scaled_mm_module(fmt, LANE_ID[fmt]))
        for fmt in FORMATS
        if lane_available(sm, fmt) and default_lane(sm, fmt) != LANE_ID[fmt]
    ]


def _split_jobs(sizes: List[int], jobs: int) -> List[int]:
    """Each module's share of *jobs*, proportional to its TU count (at least 1)."""
    total = sum(sizes) or 1
    return [max(1, min(n, round(jobs * n / total))) for n in sizes]


def prebuild(
    ops: Iterable[str],
    jobs: int,
    progress=None,
    specs: Optional[List[Tuple[str, Any]]] = None,
) -> List[Prebuilt]:
    """Compile (never load) every module a build over *ops* needs, or *specs*.

    Raises nothing on a compile failure: each module reports its own status, so
    one architecture's broken template is a line in the report rather than the
    end of the others.
    """
    from oasr.jit.core import require_known_cuda_arch
    from oasr.jit.cubin_loader import is_complete_shared_object, locked_compile

    require_known_cuda_arch("the tuning modules")
    if specs is None:
        specs = modules_for(ops)
    pending, results = [], []
    for name, spec in specs:
        lib = spec._get_lib_path()
        if is_complete_shared_object(lib):
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
