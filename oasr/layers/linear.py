# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Linear projections — the waist's GEMM entry point.

Every model's projections go through :class:`Linear` (or one of the
tensor-parallel-shaped aliases below) instead of ``nn.Linear``, so a GEMM
improvement, a fused epilogue or a future quantized kernel lands once and
reaches every architecture.  Parameter layout is ``nn.Linear``'s exactly
(``weight (out, in)``, ``bias (out,)``), so a checkpoint loads 1:1 and a
migrated module keeps its upstream state-dict keys.

Backend choice lives in :mod:`oasr.layers._backend`: ``oasr.gemm`` on CUDA
fp16/bf16 with both dimensions 8-aligned, ``F.linear`` otherwise.  A projection
pinned with ``batch_invariant = True`` takes ``oasr.gemm(...,
batch_invariant=True)`` on that path at every M, below the row floor included.

**Tensor-parallel naming.**  :class:`ColumnParallelLinear` and
:class:`RowParallelLinear` are today exact aliases of :class:`Linear` — they
carry no communication and no sharding.  What they carry is the *shard axis*
recorded at the definition site (``tp_dim``: 0 = split ``out_features`` across
ranks and concatenate, 1 = split ``in_features`` and all-reduce the partial
sums).  Writing the axis down while the code is single-GPU is what makes
tensor parallelism (M3) a fill-in rather than an audit of every model.  The
convention follows vLLM: QKV and gate/up projections are column-parallel,
output and down projections are row-parallel.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

import oasr

from ._backend import GEMM_ALIGNMENT, GEMM_DTYPES, layers_backend, use_gemm_kernel
from ._quantizable import QuantizableGemm

#: Torch equivalent of each fused CUDA epilogue. Exact-erf ``gelu`` and the
#: ``gelu_tanh`` approximation deliberately remain separate names so checkpoint
#: semantics cannot change when a projection becomes fused.
_TORCH_ACTIVATION = {
    "relu": F.relu,
    "swish": F.silu,
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
}

#: Names accepted by :class:`LinearActivation`, mapped to the kernel's id.
_FUSED_ACTIVATION_ID = {
    "relu": "relu",
    "swish": "swish",
    "silu": "silu",
    "gelu": "gelu_erf",
    "gelu_tanh": "gelu",
}


def _use_pinned_gemm(x: torch.Tensor, in_features: int, out_features: int) -> bool:
    """Can a ``batch_invariant`` projection take the pinned CUDA kernel path?

    The kernel path's own preconditions -- CUDA, fp16/bf16, both dimensions
    8-aligned, the ``oasr`` backend -- and deliberately *not* its row floor
    (``GEMM_MIN_ROWS``): below the floor the unpinned path is ``F.linear``,
    i.e. cuBLAS choosing its kernel from M, which is the very dependence the
    pin removes.  Anything else (CPU, fp32, ``OASR_LAYERS_BACKEND=torch``, an
    unaligned shape) keeps its existing route: the pin is a property of the
    CUDA kernel path, not a second backend.
    """
    return (
        x.is_cuda
        and x.dtype in GEMM_DTYPES
        and in_features % GEMM_ALIGNMENT == 0
        and out_features % GEMM_ALIGNMENT == 0
        and layers_backend() != "torch"
    )


class Linear(QuantizableGemm, nn.Module):
    """``y = x @ weightᵀ + bias``, on the OASR GEMM kernel where it can run.

    Parameters
    ----------
    in_features, out_features : int
        As ``nn.Linear``.  The kernel additionally wants both 8-aligned; a
        shape that is not (an unpadded vocabulary, say) transparently uses
        ``F.linear``.
    bias : bool
        Whether to allocate a bias parameter.

    Attributes
    ----------
    batch_invariant : bool
        ``False`` by default.  Set it on an instance (``layer.batch_invariant =
        True``) to *pin* the projection: on the CUDA fp16/bf16 kernel path every
        output row is then bitwise independent of how many rows share the call,
        at any M, including below ``GEMM_MIN_ROWS`` (``oasr.gemm(...,
        batch_invariant=True)``: CUTLASS data-parallel variants only, never
        cuBLAS, split-K or Stream-K).  For a full-precision layer whose ulps a
        later quantizer would amplify into a batch-dependent result.  It costs
        speed where the unpinned choice was a K decomposition or cuBLAS, and
        changes nothing off the kernel path (CPU, fp32,
        ``OASR_LAYERS_BACKEND=torch``) or for a quantized layer, whose
        ``quant_method`` runs first.  Not a parameter or buffer: it is not in
        the state dict.
    """

    #: Axis this projection would be sharded on under tensor parallelism.
    #: ``None`` = replicated (the shard axis is not implied by the module).
    tp_dim: Optional[int] = None

    #: Pin every output row to be independent of M (see the class docstring).
    batch_invariant: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (*, in_features) -> (*, out_features)``."""
        if self.quant_method is not None:
            return self.quant_method.apply(self, x)
        if self.batch_invariant and _use_pinned_gemm(x, self.in_features, self.out_features):
            pinned: torch.Tensor = oasr.gemm(x, self.weight, self.bias, batch_invariant=True)
            return pinned
        if use_gemm_kernel(x, self.in_features, self.out_features):
            return oasr.gemm(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        ) + (", batch_invariant=True" if self.batch_invariant else "")


class ColumnParallelLinear(Linear):
    """:class:`Linear` whose ``out_features`` is the tensor-parallel shard axis.

    Single-GPU today; the name records that under TP each rank would hold a
    slice of the rows of ``weight`` and produce a slice of the output (no
    collective needed until the consumer asks for the full tensor).  Use for
    Q/K/V and gate/up projections.
    """

    tp_dim = 0


class RowParallelLinear(Linear):
    """:class:`Linear` whose ``in_features`` is the tensor-parallel shard axis.

    Single-GPU today; under TP each rank holds a slice of the columns of
    ``weight``, consumes the matching slice of the input and the partial sums
    are all-reduced.  Use for output and down projections.
    """

    tp_dim = 1


class LinearActivation(QuantizableGemm, nn.Module):
    """``y = activation(x @ weightᵀ + bias)`` as one fused GEMM epilogue.

    ``activation`` is one of ``relu`` / ``swish`` (== ``silu``) / exact-erf
    ``gelu`` / ``gelu_tanh``. The two GELU spellings select distinct epilogues.

    ``batch_invariant`` pins the projection exactly as on :class:`Linear`
    (``oasr.gemm_activation(..., batch_invariant=True)``).
    """

    tp_dim: Optional[int] = None

    #: Pin every output row to be independent of M (see :class:`Linear`).
    batch_invariant: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_type: str = "swish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if activation_type not in _FUSED_ACTIVATION_ID:
            raise ValueError(
                f"activation_type={activation_type!r} is not fusable; "
                f"expected one of {sorted(_FUSED_ACTIVATION_ID)}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation_type
        self.fused_activation = activation_type
        self.activation_type = oasr.get_activation_type_id(_FUSED_ACTIVATION_ID[activation_type])
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (*, in_features) -> (*, out_features)``."""
        if self.quant_method is not None:
            return self.quant_method.apply(self, x)
        if self.batch_invariant and _use_pinned_gemm(x, self.in_features, self.out_features):
            pinned: torch.Tensor = oasr.gemm_activation(
                x, self.weight, self.bias, self.activation_type, batch_invariant=True
            )
            return pinned
        if use_gemm_kernel(x, self.in_features, self.out_features):
            return oasr.gemm_activation(x, self.weight, self.bias, self.activation_type)
        return _TORCH_ACTIVATION[self.activation_name](F.linear(x, self.weight, self.bias))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, activation={self.activation_name}"
        ) + (", batch_invariant=True" if self.batch_invariant else "")


__all__ = [
    "ColumnParallelLinear",
    "Linear",
    "LinearActivation",
    "RowParallelLinear",
]
