# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GEMM-family candidates, measured with :mod:`oasr.tune.bench`.

The bridge between the autotuner's backend registry (every compiled GEMM
variant plus the cuBLAS arm) and the benchmark protocol: it allocates one
shape's operands, gates every candidate on numerics, builds the arms --
rotating weight copies when the served model would not keep them in L2 -- and
returns ranked measurements whose names are tuning-DB config ids.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch

from oasr.tune import bench

__all__ = [
    "GemmCase",
    "CaseResult",
    "candidates",
    "choice_of",
    "choice_name",
    "structural_key",
    "benchmark_case",
    "GEMM_OPS",
]

GEMM_OPS = ("gemm", "gemm_activation", "gemm_log_softmax", "bmm")

#: The activation id the tuner measures ``gemm_activation`` with (SWISH): the
#: epilogue's cost is activation-independent, so one id stands for all.
_ACTIVATION_SWISH = 2


@dataclass(frozen=True)
class GemmCase:
    """One GEMM-family problem to measure."""

    op: str
    M: int
    N: int
    K: int
    dtype: str = "bfloat16"
    #: Batch count, for ``bmm``.
    batch: int = 1

    @property
    def torch_dtype(self) -> torch.dtype:
        dt: torch.dtype = getattr(torch, self.dtype)
        return dt

    @property
    def weight_bytes(self) -> int:
        return self.N * self.K * self.torch_dtype.itemsize * (self.batch if self.op == "bmm" else 1)


@dataclass
class CaseResult:
    case: GemmCase
    results: Dict[str, bench.Measurement]
    conditions: bench.Conditions
    #: ``name -> max |err|`` against the fp32 reference, every candidate tried.
    errors: Dict[str, float] = field(default_factory=dict)
    #: ``name -> why it was not measured`` (numerics gate, launch failure).
    rejected: Dict[str, str] = field(default_factory=dict)
    l2_copies: int = 1
    #: ``name -> registry BackendEntry`` for every candidate considered.
    entries: Dict[str, Any] = field(default_factory=dict)

    @property
    def l2_state(self) -> str:
        """``warm``, ``cold``, or ``cold-partial`` (too few copies to fill 3x L2)."""
        if self.l2_copies <= 1:
            return "warm"
        l2 = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 0)
        return "cold" if self.l2_copies * self.case.weight_bytes >= 3 * l2 else "cold-partial"

    def winner(self, alpha: float = 0.0, tie: float = 0.05) -> Optional[bench.Measurement]:
        return bench.pick_winner(self.results, alpha=alpha, tie=tie, prefer=structural_key)

    @property
    def default(self) -> Optional[bench.Measurement]:
        return self.results.get("default") or self.results.get("fused")


# =============================================================================
# Candidates and their names
# =============================================================================


def candidates(op: str):
    """Every registered candidate for GEMM-family *op* on this device."""
    from oasr.tune.autotuner import OpKey, _ensure_backends_registered, _global_registry

    _ensure_backends_registered()
    return _global_registry.get_candidates(OpKey("gemm", op))


def choice_of(tactic, sm: Optional[int] = None):
    """The ``select_default_config``-style choice a registry tactic stands for."""
    import oasr.jit.gemm as jg

    if tactic.backend == "torch":
        return "torch"
    if tactic.backend == "cutlass_fused":
        return "fused"
    if sm is None:
        from oasr.jit.core import _get_target_sm

        sm = _get_target_sm()
    d = dict(tactic.config)
    cfg: "jg.CutlassGemmConfig | jg.CutlassGemmConfigSm90"
    if "block_m" in d:
        cfg = jg.CutlassGemmConfig(
            block_m=d["block_m"],
            block_n=d["block_n"],
            block_k=d["block_k"],
            warp_m=d["warp_m"],
            warp_n=d["warp_n"],
            warp_k=d["warp_k"],
            kStages=d["kStages"],
            kSmVersion=sm,
            split_k=d.get("split_k", 1),
            stream_k=bool(d.get("stream_k", 0)),
            parallel_split_k=bool(d.get("parallel_split_k", 0)),
        )
    else:
        cfg = jg.CutlassGemmConfigSm90(
            tile_m=d["tile_m"],
            tile_n=d["tile_n"],
            tile_k=d["tile_k"],
            cluster_m=d["cluster_m"],
            cluster_n=d["cluster_n"],
            pingpong=bool(d["pingpong"]),
            is_dynamic_persistent=bool(d.get("is_dynamic_persistent", 0)),
            swap_ab=bool(d.get("swap_ab", 0)),
            max_swizzle_size=8,
            use_tma_gather=False,
            kSMs=d.get("kSMs", 1),
            kStages=d["kStages"],
            kSmVersion=sm,
        )
    default = jg.GEMM_DEFAULT
    if cfg == default:
        return default
    return cfg


def choice_name(choice) -> str:
    import oasr.jit.gemm as jg

    return jg.gemm_config_id(choice)


def structural_key(m: bench.Measurement) -> Tuple:
    """Tie-break preference among near-equal arms: cheapest structure first.

    Plain CUTLASS (or the fused launcher), serial split-K (still one launch),
    cuBLAS, parallel split-K (two launches), Stream-K (memset + kernel); then
    the smaller tile, fewer stages, smaller split.  The order the tuner has
    always used (``_pick_winner``), so re-tunes agree with the history.
    """
    choice = m.payload
    if choice == "torch":
        return (2, 0, 0, 0, 0)
    if isinstance(choice, str):
        return (0, 0, 0, 0, 0)
    if getattr(choice, "stream_k", False):
        rank = 4
    elif getattr(choice, "parallel_split_k", False):
        rank = 3
    elif getattr(choice, "split_k", 1) > 1:
        rank = 1
    else:
        rank = 0
    return (
        rank,
        getattr(choice, "block_m", getattr(choice, "tile_m", 1 << 30)),
        getattr(choice, "block_n", getattr(choice, "tile_n", 1 << 30)),
        getattr(choice, "kStages", 0),
        getattr(choice, "split_k", 1),
    )


# =============================================================================
# One case
# =============================================================================


def _alloc(case: GemmCase, copies: int):
    dt = case.torch_dtype
    dev = "cuda"
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    if case.op == "bmm":
        A = torch.randn(case.batch, case.M, case.K, device=dev, dtype=dt, generator=g)
        Bs = [
            torch.randn(case.batch, case.N, case.K, device=dev, dtype=dt, generator=g)
            for _ in range(copies)
        ]
        out = torch.empty(case.batch, case.M, case.N, device=dev, dtype=dt)
        return A, Bs, None, out
    A = torch.randn(case.M, case.K, device=dev, dtype=dt, generator=g)
    Bs = [
        torch.randn(case.N, case.K, device=dev, dtype=dt, generator=g) / math.sqrt(case.K)
        for _ in range(copies)
    ]
    C = torch.randn(case.N, device=dev, dtype=dt, generator=g)
    out = torch.empty(case.M, case.N, device=dev, dtype=dt)
    return A, Bs, C, out


def _reference(case: GemmCase, A, B, C):
    import torch.nn.functional as F

    if case.op == "bmm":
        return torch.matmul(A.float(), B.float().transpose(-1, -2))
    ref = torch.addmm(C.float(), A.float(), B.float().t())
    if case.op == "gemm_activation":
        return F.silu(ref)
    if case.op == "gemm_log_softmax":
        return F.log_softmax(ref, dim=-1)
    return ref


def _call_args(case: GemmCase, out, A, B, C) -> tuple:
    if case.op == "bmm":
        return (out, A, B)
    if case.op == "gemm_activation":
        return (out, A, B, C, _ACTIVATION_SWISH)
    return (out, A, B, C)


def benchmark_case(
    case: GemmCase,
    entries: Optional[Sequence] = None,
    *,
    policy: bench.BenchPolicy = bench.BenchPolicy(),
    working_set_bytes: Optional[int] = None,
    forced: Iterable[str] = ("default", "fused"),
    issue_cost: bool = True,
    only: Optional[Iterable[str]] = None,
) -> CaseResult:
    """Measure *case* over registry *entries* (default: every candidate).

    ``working_set_bytes`` is the served model's per-forward weight bytes; with
    it, weights are rotated exactly when they would be cold in production.
    ``only`` restricts to the named candidates (cross-evaluation re-measures).
    """
    if entries is None:
        entries = candidates(case.op)
    wanted = None if only is None else set(only)
    copies = bench.l2_copies(
        case.weight_bytes, working_set_bytes=working_set_bytes, max_copies=policy.max_rotation_calls
    )
    A, Bs, C, out = _alloc(case, copies)
    ref = _reference(case, A, Bs[0], C)

    named = []
    seen = set()
    for entry in entries:
        try:
            choice = choice_of(entry.tactic)
        except Exception:  # noqa: BLE001 -- an unknown tactic shape is not ours to rank
            continue
        name = choice_name(choice)
        if name in seen or (wanted is not None and name not in wanted):
            continue
        seen.add(name)
        named.append((name, choice, entry))

    errors: Dict[str, float] = {}
    rejected: Dict[str, str] = {}
    runners = {}
    for name, choice, entry in named:
        try:
            runner = entry.get_runner()
            runner(*_call_args(case, out, A, Bs[0], C))
            torch.cuda.synchronize()
            errors[name] = bench.numerics_error(out, ref)
            runners[name] = (runner, choice)
        except Exception as exc:  # noqa: BLE001
            bench._sync_or_sticky(f"first call of {name}")
            rejected[name] = f"{type(exc).__name__}: {exc}"[:300]
    reference_err = errors.get("torch")
    arms = []
    for name, (runner, choice) in runners.items():
        if not bench.numerics_ok(errors[name], reference_err, case.torch_dtype):
            rejected[name] = f"numerics: max|err| {errors[name]:.3g} vs torch {reference_err}"
            continue

        def make_call(i, r=runner):
            args = _call_args(case, out, A, Bs[i % copies], C)
            return lambda: r(*args)

        arms.append(bench.Arm(name, make_call, forced=name in set(forced), payload=choice))

    results, cond = bench.measure(arms, policy, copies=copies)
    if issue_cost:
        for name, m in results.items():
            if m.status == "ok" and m.eliminated_at is None:
                runner, _ = runners[name]
                args = _call_args(case, out, A, Bs[0], C)
                m.issue_us = bench.measure_issue_us(functools.partial(runner, *args))
    A = Bs = C = out = ref = None  # release this shape's operands before the next
    torch.cuda.empty_cache()
    return CaseResult(
        case, results, cond, errors, rejected, copies, {name: entry for name, _, entry in named}
    )
