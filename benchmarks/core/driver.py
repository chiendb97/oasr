# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The one backend sweep.

Sixteen routine modules each carried a 47-68 line ``run_test`` that did the same
five things in the same order: pull settings off ``args``, resolve configs, build
the callables, refcheck against torch, then time each backend and write a row.
Only two of those steps differ per family, so only those two stayed in the
family modules.

A family module declares::

    SUBROUTINES: list[str]
    DEFAULT_CONFIGS: dict[str, list[dict]]
    INTERLEAVE: bool                      # optional, default False
    CAUSAL_SUBROUTINES: frozenset[str]    # optional

    def parse_args(parser) -> None
    def resolve_configs(args, subroutine) -> list[dict]
    def build_fns(subroutine, cfg, dtype, args) -> dict[str, Callable]
    def describe(subroutine, cfg, dtype) -> Work

``build_fns`` and ``describe`` are the enforced contract.  Under the previous
design profile mode reached for private ``_resolve_configs`` / ``_setup_for_config``
/ ``get_fn_map`` that eight of the sixteen modules never defined, so ``--profile``
raised ``AttributeError`` on half the suite.  Here profile mode is the same sweep
with one iteration and an NVTX range, so a family that benchmarks can profile.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
from typing import Any, Callable, Dict, List, Optional

import torch

from benchmarks.core import timing
from benchmarks.core.metrics import (
    REFCHECK_FAIL,
    REFCHECK_PASS,
    REFCHECK_SKIP,
    check_close,
)
from benchmarks.core.report import Reporter
from benchmarks.core.schema import KernelRow


@dataclasses.dataclass
class Work:
    """What one config costs, as declared by the family.

    ``flops`` and ``bytes`` are optional and independent: a GEMM declares FLOPs,
    a LayerNorm declares bytes, a fused MLP declares both, and a depthwise conv
    declares only bytes because it has no meaningful FLOP count.  Leaving one
    ``None`` writes an empty cell rather than a misleading ``0.0``.
    """

    shape: str
    params: str
    flops: Optional[float] = None
    bytes: Optional[int] = None
    #: Per-backend byte counts, when the arms genuinely move different amounts.
    #: A fused gated MLP writes one intermediate where the unfused path writes
    #: two, reads both back and writes the product -- reporting one number for
    #: both would credit the slower arm with traffic it did not avoid.
    bytes_by_backend: Optional[Dict[str, int]] = None

    def bytes_for(self, backend: str) -> Optional[int]:
        if self.bytes_by_backend and backend in self.bytes_by_backend:
            return self.bytes_by_backend[backend]
        return self.bytes


def params_of(cfg: Dict[str, Any]) -> str:
    """``k=v;k=v`` with keys sorted -- the structured half of the shape.

    The free-form ``shape`` column carried four mutually unparseable grammars
    across routines (``(16000, 256, 2048)``, ``B=64_S=250_H=512``,
    ``[1, 1, 512]``, ``N=10, chunk=16, avg_dur=7.3s``).  ``shape`` stays as the
    human label; this is the one a consumer can split.
    """
    return ";".join(f"{k}={v}" for k, v in sorted(cfg.items()))


DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def resolve_dtype_str(args: argparse.Namespace, module: Any) -> str:
    """The dtype for this run: an explicit --dtype, else the family's default.

    Kernel families measure fp16; the engine, service and accuracy harnesses
    serve bf16, which is what their standalone scripts defaulted to.  Folding
    those scripts into one CLI must not silently change the precision the
    documented command measures.
    """
    if getattr(args, "dtype_explicit", False) or not hasattr(module, "DEFAULT_DTYPE"):
        return args.dtype
    return module.DEFAULT_DTYPE


def parse_dtype(name: str) -> torch.dtype:
    key = name.strip().lower()
    if key not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{name}'. Choose from: {sorted(DTYPE_MAP)}")
    return DTYPE_MAP[key]


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    """The comparable part of whatever a backend returned.

    Kernels return a tensor, a ``(values, indices)`` pair, or a tuple whose
    first element is the output.  Only the first tensor is compared -- an index
    tensor legitimately differs between implementations on ties.
    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return _first_tensor(value[0])
    return None


def _refcheck(
    fns: Dict[str, Callable[[], Any]],
    backends: List[str],
    ref_backend: str,
    reporter: Reporter,
    shape: str,
    tolerance: tuple = (1e-2, 1e-2),
    non_gating: frozenset = frozenset(),
) -> Dict[str, tuple]:
    """Compare every measured backend against *ref_backend*.

    Returns ``{backend: (verdict, max_abs_diff)}``.  Each arm is invoked once
    here, separately from the timing loop, so an in-place kernel cannot have its
    result overwritten before it is read.
    """
    out: Dict[str, tuple] = dict.fromkeys(backends, (REFCHECK_SKIP, None))
    if ref_backend not in fns:
        return out
    reference = _first_tensor(fns[ref_backend]())
    if reference is None:
        return out
    reference = reference.clone()
    for backend in backends:
        if backend == ref_backend or backend not in fns:
            continue
        actual = _first_tensor(fns[backend]())
        if actual is None:
            continue
        atol, rtol = tolerance
        passed, diff = check_close(actual, reference.to(actual.dtype), atol=atol, rtol=rtol)
        out[backend] = (REFCHECK_PASS if passed else REFCHECK_FAIL, diff)
        if not passed:
            # A non-gating arm is compared for information, not agreement --
            # torchaudio computes the same feature through a different FFT
            # order, so a moderate difference there is expected, not a defect.
            if backend in non_gating:
                reporter.warn(
                    f"{shape}: {backend} vs {ref_backend} "
                    f"max_abs_diff={diff:.6g} (informational)"
                )
            else:
                reporter.error(
                    f"Output mismatch for {shape}: {backend} vs {ref_backend} "
                    f"(max_abs_diff={diff:.6g})"
                )
    return out


@contextlib.contextmanager
def _autotuning(args: argparse.Namespace):
    """Wrap a sweep in OASR's autotuner when ``--autotune`` is given.

    Autotuning is a context manager around a call, so it belongs here rather
    than inside a family: the previous suite had two private implementations of
    this (one in the gemm routine, one in conv), and the gemm one had been dead
    for some time -- it imported ``oasr.tune._types`` and
    ``oasr.tune.kernel_configs``, neither of which exists, so ``--autotune``
    raised ``ModuleNotFoundError``.  The gemm routine also registered the flag
    without ever reading it, so the unified CLI accepted ``--autotune`` and
    silently ran an ordinary sweep.
    """
    if not getattr(args, "autotune", False):
        yield
        return
    import oasr

    with oasr.autotune(not getattr(args, "no_tune", False), cache=getattr(args, "cache", None)):
        yield


def run_sweep(
    family: str,
    module: Any,
    args: argparse.Namespace,
    reporter: Reporter,
    repro_command: str = "",
) -> None:
    """Benchmark one subroutine of *module* across its backends and configs."""
    subroutine = args.subroutine or module.SUBROUTINES[0]
    dtype_str = resolve_dtype_str(args, module)
    # A family whose kernel exists in one precision only says so, rather than
    # letting a --dtype it cannot honour reach the CSV as if it had.
    forced = getattr(module, "FORCE_DTYPE", None)
    if forced and forced != dtype_str:
        reporter.warn(f"{family} supports {forced} only; ignoring --dtype {dtype_str}")
        dtype_str = forced
    dtype = parse_dtype(dtype_str)
    # A family may name its own baseline: torch.nn.LSTM is cuDNN on CUDA, and
    # calling that arm "torch" would put a misleading name in the CSV.
    ref_backend = getattr(args, "ref_backend", None) or getattr(module, "REF_BACKEND", "torch")
    interleave = getattr(args, "interleave", None)
    if interleave is None:
        interleave = subroutine in getattr(module, "INTERLEAVE_SUBROUTINES", frozenset())
        interleave = interleave or getattr(module, "INTERLEAVE", False)

    with _autotuning(args):
        for cfg in module.resolve_configs(args, subroutine):
            try:
                fns = module.build_fns(subroutine, cfg, dtype, args)
            except Exception as exc:  # a backend that cannot be built is not a crash
                reporter.error(f"{family}/{subroutine} setup failed for {cfg}: {exc}")
                continue
            work = module.describe(subroutine, cfg, dtype)

            requested = getattr(args, "backends", None) or list(fns)
            backends = []
            for backend in requested:
                if backend not in fns:
                    reporter.warn(
                        f"Unknown backend '{backend}' for {family}/{subroutine}, skipping"
                    )
                    continue
                backends.append(backend)
            if not backends:
                continue

            verdicts: Dict[str, tuple] = dict.fromkeys(backends, (REFCHECK_SKIP, None))
            if getattr(args, "refcheck", False):
                non_gating = frozenset(getattr(module, "NON_GATING_BACKENDS", frozenset()))
                tolerance = getattr(module, "TOLERANCES", {}).get(subroutine, (1e-2, 1e-2))
                verdicts = _refcheck(
                    fns, backends, ref_backend, reporter, work.shape, tolerance, non_gating
                )
                gating_failed = any(
                    verdict == REFCHECK_FAIL and backend not in non_gating
                    for backend, (verdict, _) in verdicts.items()
                )
                if gating_failed and not getattr(args, "allow_output_mismatch", False):
                    continue

            if getattr(args, "profile", False):
                for backend in backends:
                    timing.profile_kernel(
                        f"{backend}_{subroutine}", fns[backend], warmup=args.warmup_iters
                    )
                continue

            measured: Dict[str, timing.Sample] = {}
            if interleave:
                measured = timing.interleaved_rounds(
                    {b: fns[b] for b in backends},
                    warmup_iters=args.warmup_iters,
                    iters=args.iters,
                    timer=args.timer,
                    flush=not args.no_flush,
                    min_measure_ms=getattr(args, "min_measure_ms", 20.0),
                )
            else:
                for backend in backends:
                    measured[backend] = timing.bench_fn(
                        fns[backend],
                        warmup_iters=args.warmup_iters,
                        iters=args.iters,
                        timer=args.timer,
                        flush=not args.no_flush,
                    )

            ref_ms = measured[ref_backend].median_ms if ref_backend in measured else None
            for backend in backends:
                sample = measured[backend]
                verdict, diff = verdicts.get(backend, (REFCHECK_SKIP, None))
                reporter.kernel_row(
                    KernelRow(
                        routine=family,
                        subroutine=subroutine,
                        backend=backend,
                        shape=work.shape,
                        params=work.params,
                        dtype=dtype_str,
                        median_ms=sample.median_ms,
                        mean_ms=sample.mean_ms,
                        std_ms=sample.std_ms,
                        min_ms=sample.min_ms,
                        p99_ms=sample.p99_ms,
                        iters=sample.iters,
                        warmup_iters=sample.warmup_iters,
                        timer=sample.timer,
                        flops=work.flops,
                        bytes=work.bytes_for(backend),
                        ref_backend=ref_backend if ref_ms else "",
                        speedup_vs_ref=(
                            ref_ms / sample.median_ms if ref_ms and sample.median_ms else None
                        ),
                        refcheck=verdict,
                        max_abs_diff=diff,
                        repro_command=repro_command,
                    )
                )
