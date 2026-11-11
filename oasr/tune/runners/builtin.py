# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The runners every tuning file already names, under the kinds it already stores.

Each family's own resolver dispatches its OASR runners through a specialised
fast path (a compiled variant looked up by name), so those runners describe
rather than dispatch.  cuBLAS is the exception: its ``"torch"`` kind is served
through :meth:`CublasGemm.bind`, the path every further runner takes.
"""

from __future__ import annotations

import functools
import importlib.metadata
from typing import Any, Callable, Dict, Optional

from oasr.tune.runners.base import Runner, RunnerTraits, register_runner

#: The GEMM family's operations (``oasr.tune.gemm_tune.GEMM_OPS``).
GEMM_OPS = ("gemm", "gemm_activation", "gemm_log_softmax", "bmm")


class OasrCutlassGemm(Runner):
    """The CUTLASS GEMM lanes: 2.x ``mma.sync`` and the native 3.x ones, plus the default.

    Not batch-invariant as a whole; its data-parallel 2.x class is, per config
    (:func:`oasr.jit.gemm.is_batch_invariant`).
    """

    name = "oasr_cutlass"
    family = "gemm"
    kinds = ("cutlass", "default")
    ops = GEMM_OPS
    backends = ("cutlass",)
    traits = RunnerTraits(capturable=True, compiled=True)
    issue_class = "oasr_launcher"


class OasrFusedGemm(Runner):
    """The single-call fused GEMM + log_softmax launcher of the CTC head."""

    name = "oasr_fused"
    family = "gemm"
    kinds = ("fused",)
    ops = ("gemm_log_softmax",)
    backends = ("cutlass_fused",)
    traits = RunnerTraits(capturable=True)
    issue_class = "oasr_launcher"


@functools.lru_cache(maxsize=1)
def cublas_version() -> Optional[str]:
    """The cuBLASLt this process loaded, as ``cublasLtGetVersion()`` reports it.

    Read from the library actually mapped, not from what a package claims: two
    wheels can put different builds on the path.  Falls back to the installed
    ``nvidia-cublas`` distribution, then to the CUDA toolkit torch was built for.
    """
    try:
        import ctypes

        with open("/proc/self/maps") as f:
            paths = sorted({ln.split()[-1] for ln in f if "libcublasLt.so" in ln})
        for path in paths:
            lib = ctypes.CDLL(path)
            lib.cublasLtGetVersion.restype = ctypes.c_size_t
            return str(int(lib.cublasLtGetVersion()))
    except (OSError, AttributeError, ValueError):
        pass
    for dist in ("nvidia-cublas", "nvidia-cublas-cu13", "nvidia-cublas-cu12"):
        try:
            return f"{dist}=={importlib.metadata.version(dist)}"
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        import torch

        return f"cuda=={torch.version.cuda}" if torch.version.cuda else None
    except ImportError:
        return None


class CublasGemm(Runner):
    """cuBLAS through torch -- the tuning file's ``"torch"`` kind.

    One tactic: cuBLAS's own heuristic pick for the shape, which is also why it
    is not batch-invariant (that pick, and its split, follow M).
    """

    name = "cublas"
    family = "gemm"
    kinds = ("torch",)
    ops = GEMM_OPS
    backends = ("torch",)
    traits = RunnerTraits(capturable=True, external=True)
    issue_class = "torch"

    def versions(self) -> Dict[str, Any]:
        import torch

        out: Dict[str, Any] = {"torch": torch.__version__}
        v = cublas_version()
        if v is not None:
            out["cublasLt"] = v
        return out

    def bind(self, op: str, choice: Any) -> Callable[..., None]:
        from oasr.functionals import gemm_torch

        fns: Dict[str, Callable[..., None]] = {
            "gemm": gemm_torch.torch_gemm,
            "gemm_activation": gemm_torch.torch_gemm_activation,
            "gemm_log_softmax": gemm_torch.torch_gemm_log_softmax,
            "bmm": gemm_torch.torch_bmm,
        }
        return fns[op]


class OasrCutlassConv1d(Runner):
    """The CUTLASS implicit-GEMM Conv1D variants and their default."""

    name = "oasr_cutlass"
    family = "conv1d"
    kinds = ("cutlass", "default")
    backends = ("cutlass",)
    traits = RunnerTraits(capturable=True, compiled=True)


class OasrScaledMm(Runner):
    """The 8-bit GEMM lanes (``mma.sync`` and native) and their shape heuristic.

    Every variant computes the same bits (an exact int32 accumulator, one fp8
    summation order), so the family's numerics gate is bitwise.
    """

    name = "oasr_cutlass"
    kinds = ("cutlass", "heuristic")
    backends = ("cutlass",)
    traits = RunnerTraits(capturable=True, numerics="bitwise", compiled=True)

    def __init__(self, family: str) -> None:
        self.family = family


register_runner(OasrCutlassGemm())
register_runner(OasrFusedGemm())
register_runner(CublasGemm())
register_runner(OasrCutlassConv1d())
register_runner(OasrScaledMm("scaled_mm"))
register_runner(OasrScaledMm("scaled_mm_fp8"))
