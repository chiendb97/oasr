# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Which kernel serves a quantized GEMM on this GPU.

A lane declares what it can run — :meth:`ScaledMMLane.is_supported` (the
device) and :meth:`ScaledMMLane.can_implement` (the scheme and shape) — the way
vLLM's ``ScaledMMLinearKernel`` s do, and :func:`choose_lane` picks one once per
layer at bind time.  A scheme no lane serves on this device is either a declared
kernel gap (``oasr.layers._backend.KERNEL_GAPS``, counted, then the torch
reference) or a refusal — never a silent reroute (AGENTS.md rule 3).

``OASR_SCALED_MM_BACKEND`` (``auto`` | ``cutlass`` | ``torch``), read once at
import, is the A/B switch: naming a lane requires it.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

# Called through the module, not a name bound at import, so a shape census
# (``oasr.tune.capture``) that patches ``oasr.functionals.scaled_mm`` sees the call.
import oasr.functionals.scaled_mm as _smm

if TYPE_CHECKING:  # pragma: no cover
    from ..spec import QuantScheme

_BACKEND = os.environ.get("OASR_SCALED_MM_BACKEND", "auto").strip().lower()
_BACKENDS = ("auto", "cutlass", "torch")
if _BACKEND not in _BACKENDS:
    raise ValueError(f"OASR_SCALED_MM_BACKEND={_BACKEND!r}; expected one of {_BACKENDS}")


def w8a8_format(scheme: "QuantScheme") -> Optional[str]:
    """``"int8"`` / ``"fp8"`` for a W8A8 scheme the scaled GEMMs serve, else ``None``.

    Per-channel symmetric 8-bit weights x 8-bit activations of the same type, with
    a per-token dynamic or one static per-tensor activation scale.
    """
    w, a = scheme.weights, scheme.input_activations
    if (
        a is None
        or w.num_bits != 8
        or a.num_bits != 8
        or w.type != a.type
        or w.granularity != "channel"
        or not (w.symmetric and a.symmetric)
    ):
        return None
    if not (
        (a.granularity == "token" and a.dynamic) or (a.granularity == "tensor" and not a.dynamic)
    ):
        return None
    return "int8" if w.type == "int" else "fp8"


def _is_int8_w8a8(scheme: "QuantScheme") -> bool:
    """int8 per-channel W x int8 A: per-token dynamic, or one static per-tensor scale."""
    return w8a8_format(scheme) == "int8"


def _is_fp8_w8a8(scheme: "QuantScheme") -> bool:
    """e4m3 per-channel W x e4m3 A: per-token dynamic, or one static per-tensor scale."""
    return w8a8_format(scheme) == "fp8"


class ScaledMMLane(ABC):
    """One way of running ``act(a_scale ⊗ b_scale · (a @ bᵀ) + bias)``."""

    name: ClassVar[str]
    backend: ClassVar[str]

    @classmethod
    @abstractmethod
    def is_supported(cls, sm: int) -> Tuple[bool, str]:
        """Whether this lane runs on an SM family, and why not."""

    @classmethod
    @abstractmethod
    def can_implement(
        cls, scheme: "QuantScheme", n: int, k: int, out_dtype: torch.dtype
    ) -> Tuple[bool, str]:
        """Whether this lane runs *scheme* at ``[*, k] @ [n, k]ᵀ``, and why not."""

    @classmethod
    def is_native(cls, sm: int) -> bool:
        """Whether this lane is the SM's native tensor-core path (for the report)."""
        return True

    @abstractmethod
    def run(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: Optional[str],
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """The GEMM: 8-bit ``a`` ``[..., K]``, 8-bit ``b`` ``[N, K]``; returns ``[..., N]``.

        ``a_scale`` holds one scale per row of *a*, or one per-tensor scale.
        """

    def run_quant(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: Optional[str],
        out_mul: torch.Tensor,
        round_dtype: torch.dtype,
    ) -> torch.Tensor:
        """The GEMM, its output quantized for a static-scale consumer (``[..., N]``).

        In the operands' format.  ``quantize_static(run(...), out_mul)`` by default
        -- bit-identical to a lane that fuses it.
        """
        from oasr.functionals.quant import quantize_static

        q: torch.Tensor = quantize_static(
            self.run(a, b, a_scale, b_scale, bias, activation, round_dtype),
            out_mul,
            dtype=a.dtype,
        )
        return q


class CutlassSm80Int8Lane(ScaledMMLane):
    """CUTLASS 2.x int8 ``mma.sync`` with the EVT dequant epilogue (``oasr.scaled_mm``)."""

    name = "cutlass_sm80_int8"
    backend = "cutlass"

    @classmethod
    def is_supported(cls, sm: int) -> Tuple[bool, str]:
        from oasr.jit.scaled_mm import lane_available

        if lane_available(sm):
            return True, ""
        return False, f"no int8 mma.sync lane is built for sm{sm}"

    @classmethod
    def is_native(cls, sm: int) -> bool:
        from oasr.jit.scaled_mm import is_native_lane

        return is_native_lane(sm)

    @classmethod
    def can_implement(
        cls, scheme: "QuantScheme", n: int, k: int, out_dtype: torch.dtype
    ) -> Tuple[bool, str]:
        from oasr.jit.scaled_mm import K_ALIGNMENT, N_ALIGNMENT

        if not _is_int8_w8a8(scheme):
            return False, "serves int8 per-channel W x int8 A (per-token dynamic or static) only"
        if out_dtype not in (torch.float16, torch.bfloat16):
            return False, f"output dtype {out_dtype} (fp16/bf16 only)"
        if k % K_ALIGNMENT or n % N_ALIGNMENT:
            return False, f"N={n} K={k}: needs K % {K_ALIGNMENT} == 0 and N % {N_ALIGNMENT} == 0"
        return True, ""

    def run(self, a, b, a_scale, b_scale, bias, activation, out_dtype):
        return _smm.scaled_mm(a, b, a_scale, b_scale, bias, activation, out_dtype=out_dtype)

    def run_quant(self, a, b, a_scale, b_scale, bias, activation, out_mul, round_dtype):
        # One launch: the epilogue quantizes for the consumer.
        return _smm.scaled_mm_quant(a, b, a_scale, b_scale, bias, activation, out_mul, round_dtype)


_TORCH_ACTIVATION = {
    None: lambda y: y,
    "relu": F.relu,
    "gelu_tanh": lambda y: F.gelu(y, approximate="tanh"),
    "swish": F.silu,
    "silu": F.silu,
    "gelu": F.gelu,
}


class TorchIntMMLane(ScaledMMLane):
    """``torch._int_mm`` (cuBLASLt int8) plus a torch dequant epilogue.

    A baseline and A/B arm, not the default: the epilogue is four extra
    elementwise launches and an int32 round trip through global memory.
    ``_int_mm`` refuses M <= 16, so small M is padded with zero rows.
    """

    name = "torch_int_mm"
    backend = "torch"

    @classmethod
    def is_supported(cls, sm: int) -> Tuple[bool, str]:
        return (sm >= 75), ("" if sm >= 75 else f"sm{sm} has no int8 tensor cores")

    @classmethod
    def can_implement(
        cls, scheme: "QuantScheme", n: int, k: int, out_dtype: torch.dtype
    ) -> Tuple[bool, str]:
        if not _is_int8_w8a8(scheme):
            return False, "serves int8 per-channel W x int8 A (per-token dynamic or static) only"
        if k % 8 or n % 8:
            return False, f"N={n} K={k}: torch._int_mm needs both 8-aligned"
        return True, ""

    def run(self, a, b, a_scale, b_scale, bias, activation, out_dtype):
        k = a.shape[-1]
        a2 = a.reshape(-1, k)
        m = a2.shape[0]
        if m <= 16:
            a2 = torch.cat([a2, a2.new_zeros(17 - m, k)])
        acc = torch._int_mm(a2, b.t())[:m]
        y = acc.float() * b_scale.reshape(1, -1) * a_scale.reshape(-1, 1)
        if bias is not None:
            y = y + bias.float()
        y = _TORCH_ACTIVATION[activation](y)
        return y.to(out_dtype).reshape(*a.shape[:-1], b.shape[0])


class CutlassSm89Fp8Lane(ScaledMMLane):
    """CUTLASS 2.x fp8 e4m3 ``mma.sync`` (Ada's, run by sm_120 too) with the same EVT epilogue.

    An fp32 accumulator with one summation order for every tile, so the lane is
    batch-invariant (``oasr/jit/scaled_mm.py``).  Below sm89 there are no fp8
    tensor cores: :meth:`is_supported` says so, and the method refuses the scheme.
    """

    name = "cutlass_sm89_fp8"
    backend = "cutlass"

    @classmethod
    def is_supported(cls, sm: int) -> Tuple[bool, str]:
        from oasr.jit.scaled_mm import lane_available

        if lane_available(sm, "fp8"):
            return True, ""
        return False, f"no fp8 mma.sync lane is built for sm{sm} (fp8 tensor cores need sm89+)"

    @classmethod
    def is_native(cls, sm: int) -> bool:
        from oasr.jit.scaled_mm import is_native_lane

        return is_native_lane(sm, "fp8")

    @classmethod
    def can_implement(
        cls, scheme: "QuantScheme", n: int, k: int, out_dtype: torch.dtype
    ) -> Tuple[bool, str]:
        from oasr.jit.scaled_mm import K_ALIGNMENT, N_ALIGNMENT

        if not _is_fp8_w8a8(scheme):
            return False, "serves e4m3 per-channel W x e4m3 A (per-token dynamic or static) only"
        if out_dtype not in (torch.float16, torch.bfloat16):
            return False, f"output dtype {out_dtype} (fp16/bf16 only)"
        if k % K_ALIGNMENT or n % N_ALIGNMENT:
            return False, f"N={n} K={k}: needs K % {K_ALIGNMENT} == 0 and N % {N_ALIGNMENT} == 0"
        return True, ""

    def run(self, a, b, a_scale, b_scale, bias, activation, out_dtype):
        return _smm.scaled_mm(a, b, a_scale, b_scale, bias, activation, out_dtype=out_dtype)

    def run_quant(self, a, b, a_scale, b_scale, bias, activation, out_mul, round_dtype):
        return _smm.scaled_mm_quant(a, b, a_scale, b_scale, bias, activation, out_mul, round_dtype)


class TorchScaledMMLane(ScaledMMLane):
    """``torch._scaled_mm`` (cuBLASLt fp8) plus a torch dequant epilogue.

    A baseline and A/B arm, not the default.  The GEMM runs with unit per-tensor
    scales and an fp32 output -- the plain fp32 sum of the e4m3 products -- and
    the per-token x per-channel dequant, bias and activation follow in torch.
    cuBLASLt's *rowwise* scaling is not used: on this torch build
    (2.12 / CUDA 13.2, RTX 5090) it accepts an fp16 output and returns values
    ~64 off on unit scales, silently, so a lane built on it would be wrong.
    Not batch-invariant: cuBLASLt picks its algorithm per M.
    """

    name = "torch_scaled_mm"
    backend = "torch"

    @classmethod
    def is_supported(cls, sm: int) -> Tuple[bool, str]:
        return (sm >= 89), ("" if sm >= 89 else f"sm{sm} has no fp8 tensor cores")

    @classmethod
    def can_implement(
        cls, scheme: "QuantScheme", n: int, k: int, out_dtype: torch.dtype
    ) -> Tuple[bool, str]:
        if not _is_fp8_w8a8(scheme):
            return False, "serves e4m3 per-channel W x e4m3 A (per-token dynamic or static) only"
        if k % 16 or n % 16:
            return False, f"N={n} K={k}: torch._scaled_mm needs both 16-aligned"
        return True, ""

    def run(self, a, b, a_scale, b_scale, bias, activation, out_dtype):
        k = a.shape[-1]
        a2 = a.reshape(-1, k)
        one = _unit_scale(a.device)
        acc = torch._scaled_mm(a2, b.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
        y = acc * b_scale.reshape(1, -1) * a_scale.reshape(-1, 1)
        if bias is not None:
            y = y + bias.float()
        y = _TORCH_ACTIVATION[activation](y)
        return y.to(out_dtype).reshape(*a.shape[:-1], b.shape[0])


_UNIT_SCALES: dict = {}


def _unit_scale(device: torch.device) -> torch.Tensor:
    """A cached fp32 scalar ``1.0`` on *device* (one allocation per device, capture-safe)."""
    one = _UNIT_SCALES.get(device)
    if one is None:
        one = _UNIT_SCALES[device] = torch.ones((), dtype=torch.float32, device=device)
    return one


#: Priority order for ``auto``.
_LANES: List[Type[ScaledMMLane]] = [
    CutlassSm80Int8Lane,
    CutlassSm89Fp8Lane,
    TorchIntMMLane,
    TorchScaledMMLane,
]


def register_lane(cls: Type[ScaledMMLane], *, first: bool = False) -> Type[ScaledMMLane]:
    """Add a lane (e.g. a native 3.x kernel) to the candidates."""
    if first:
        _LANES.insert(0, cls)
    else:
        _LANES.append(cls)
    return cls


def choose_lane(
    scheme: "QuantScheme", n: int, k: int, out_dtype: torch.dtype, sm: int
) -> Tuple[Optional[ScaledMMLane], str]:
    """``(lane, note)``: the first lane that serves this layer, else ``(None, why)``.

    With ``OASR_SCALED_MM_BACKEND`` naming a backend, only that backend's lanes
    are candidates, and the caller must refuse rather than fall back.
    """
    reasons = []
    for cls in _LANES:
        if _BACKEND != "auto" and cls.backend != _BACKEND:
            continue
        ok, why = cls.is_supported(sm)
        if ok:
            ok, why = cls.can_implement(scheme, n, k, out_dtype)
        if ok:
            note = "native" if cls.is_native(sm) else f"non-native lane on sm{sm}"
            return cls(), note
        reasons.append(f"{cls.name}: {why}")
    return None, "; ".join(reasons) or f"no lane for backend {_BACKEND!r}"


def forced_backend() -> Optional[str]:
    """The backend ``OASR_SCALED_MM_BACKEND`` names, or ``None`` for ``auto``."""
    return None if _BACKEND == "auto" else _BACKEND
