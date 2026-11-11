# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The runners every tuning file already names, under the kinds it already stores.

Each family's own resolver dispatches its OASR runners through a specialised
fast path (a compiled variant looked up by name), so those runners describe
rather than dispatch.  cuBLAS is the exception: its ``"torch"`` kind is served
through :meth:`CublasGemm.bind`, the path every further runner takes.

The attention family (``fmha``) dispatches all three of its runners -- the
CuTeDSL kernel, the C++ kernel, torch SDPA -- inside
:func:`oasr.functionals.attention.fmha`, which already owns every lane.
"""

from __future__ import annotations

import functools
import importlib.metadata
from typing import Any, Callable, Dict, Optional, Tuple

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


#: The attention family's one operation (``oasr.jit.attention_select.OP``).
FMHA_OPS = ("fmha",)


def _device_sm() -> Optional[int]:
    import torch

    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


class OasrCuteFmha(Runner):
    """The CuTeDSL attention kernel, one tactic per tile.

    Also owns ``"auto"``, the sentinel for "today's routing", which runs this
    kernel wherever it is built.
    """

    name = "oasr_cute"
    family = "fmha"
    kinds = ("cute", "auto")
    ops = FMHA_OPS
    traits = RunnerTraits(capturable=True, compiled=True)
    issue_class = "oasr_cute"

    def unavailable(self) -> Optional[str]:
        sm = _device_sm()
        if sm is None:
            return "no CUDA device"
        try:
            from oasr.jit.attention import _check_cutedsl_version
            from oasr.kernels.cute.attention.base import pick_arch_cls

            _check_cutedsl_version()
            pick_arch_cls(sm // 10, sm % 10)
        except Exception as exc:  # noqa: BLE001 -- any failure means it cannot run here
            return f"CuTeDSL attention is unavailable on sm{sm}: {exc}"
        return None

    def versions(self) -> Dict[str, Any]:
        import cutlass

        return {"cutedsl": str(getattr(cutlass, "__version__", "unknown"))}


class OasrCxxFmha(Runner):
    """The C++ CUTLASS/CuTe attention kernel, one tactic per split-KV setting."""

    name = "oasr_cxx"
    family = "fmha"
    kinds = ("cxx",)
    ops = FMHA_OPS
    traits = RunnerTraits(capturable=True, compiled=True)
    issue_class = "oasr_launcher"

    def unavailable(self) -> Optional[str]:
        from oasr.jit.fmha import SUPPORTED_SM

        sm = _device_sm()
        if sm is None:
            return "no CUDA device"
        from oasr.jit.core import sm_family

        if sm_family(sm // 10, sm % 10) not in SUPPORTED_SM:
            return f"the C++ attention lane is built for {SUPPORTED_SM}, not sm{sm}"
        return None


class SdpaFmha(Runner):
    """torch's ``scaled_dot_product_attention`` -- its own backend pick for the call.

    External: a region naming it is a policy hit at dispatch.
    """

    name = "sdpa"
    family = "fmha"
    kinds = ("sdpa",)
    ops = FMHA_OPS
    traits = RunnerTraits(capturable=True, external=True)
    issue_class = "torch"

    def versions(self) -> Dict[str, Any]:
        import torch

        out: Dict[str, Any] = {"torch": torch.__version__}
        cudnn = torch.backends.cudnn.version()
        if cudnn is not None:
            out["cudnn"] = int(cudnn)
        return out


# --------------------------------------------------------------------------
# The recurrent and gated-MLP paths (``oasr.jit.recurrent_select``,
# ``oasr.jit.mlp_select``): every path is OASR's own, dispatched by the
# family's functional.
# --------------------------------------------------------------------------


def _cutedsl_version() -> Dict[str, Any]:
    import cutlass

    return {"cutedsl": str(getattr(cutlass, "__version__", "unknown"))}


class _FamilyRunner(Runner):
    """A path of the recurrent or gated-MLP family, named per family."""

    def __init__(
        self,
        family: str,
        name: str,
        kinds: Tuple[str, ...],
        *,
        compiled: bool = True,
        issue_class: str = "oasr_launcher",
        cutedsl: bool = False,
    ) -> None:
        self.family = family
        self.name = name
        self.kinds = kinds
        self.traits = RunnerTraits(capturable=True, compiled=compiled)
        self.issue_class = issue_class
        self._cutedsl = cutedsl

    def unavailable(self) -> Optional[str]:
        if not self._cutedsl:
            return None
        sm = _device_sm()
        if sm is None:
            return "no CUDA device"
        try:
            import cutlass  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return f"CuTeDSL is unavailable: {exc}"
        return None

    def versions(self) -> Dict[str, Any]:
        return _cutedsl_version() if self._cutedsl else {}


register_runner(OasrCutlassGemm())
register_runner(OasrFusedGemm())
register_runner(CublasGemm())
register_runner(OasrCutlassConv1d())
register_runner(OasrScaledMm("scaled_mm"))
register_runner(OasrScaledMm("scaled_mm_fp8"))
register_runner(OasrCuteFmha())
register_runner(OasrCxxFmha())
register_runner(SdpaFmha())
for _runner in (
    # ``auto`` is today's hand-timed rules; owned by the path they default to.
    _FamilyRunner("recurrent", "oasr_scalar", ("scalar", "auto"), compiled=False),
    _FamilyRunner("recurrent", "oasr_gemm", ("gemm",)),
    _FamilyRunner("recurrent", "oasr_cxx_step", ("cxx",)),
    _FamilyRunner("recurrent", "oasr_cute_step", ("cute",), issue_class="oasr_cute", cutedsl=True),
    _FamilyRunner("gated_mlp", "oasr_cxx", ("cxx", "auto")),
    _FamilyRunner("gated_mlp", "oasr_cute", ("cute",), issue_class="oasr_cute", cutedsl=True),
    _FamilyRunner("gated_mlp", "oasr_gemm", ("gemm",)),
):
    register_runner(_runner)
