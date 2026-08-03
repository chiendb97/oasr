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
fp16/bf16 with both dimensions 8-aligned, ``F.linear`` otherwise.

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

from ._backend import use_gemm_kernel

#: Activation ids whose CUDA epilogue is bit-comparable to a torch function.
#: ``gelu`` is deliberately absent: the OASR epilogue is the **tanh
#: approximation** (``include/oasr/common/math.h``) while ``F.gelu`` defaults to
#: the exact erf form, and Whisper / Qwen2-Audio are trained with the latter.
#: Ask for ``"gelu_tanh"`` to opt into the fused kernel knowingly.
_TORCH_ACTIVATION = {
    "relu": F.relu,
    "swish": F.silu,
    "silu": F.silu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
}

#: Names accepted by :class:`LinearActivation`, mapped to the kernel's id.
_FUSED_ACTIVATION_ID = {
    "relu": "relu",
    "swish": "swish",
    "silu": "silu",
    "gelu_tanh": "gelu",
}


class Linear(nn.Module):
    """``y = x @ weightᵀ + bias``, on the OASR GEMM kernel where it can run.

    Parameters
    ----------
    in_features, out_features : int
        As ``nn.Linear``.  The kernel additionally wants both 8-aligned; a
        shape that is not (an unpadded vocabulary, say) transparently uses
        ``F.linear``.
    bias : bool
        Whether to allocate a bias parameter.
    """

    #: Axis this projection would be sharded on under tensor parallelism.
    #: ``None`` = replicated (the shard axis is not implied by the module).
    tp_dim: Optional[int] = None

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
        if use_gemm_kernel(x, self.in_features, self.out_features):
            return oasr.gemm(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


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


class LinearActivation(nn.Module):
    """``y = activation(x @ weightᵀ + bias)`` as one fused GEMM epilogue.

    ``activation`` is one of ``relu`` / ``swish`` (== ``silu``) /
    ``gelu_tanh``.  There is no plain ``"gelu"``: the CUDA epilogue implements
    the tanh approximation, so fusing it under that name would silently change
    the math of every erf-GELU model.  Spell the approximation out or use a
    :class:`Linear` plus ``F.gelu``.
    """

    tp_dim: Optional[int] = None

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
        if use_gemm_kernel(x, self.in_features, self.out_features):
            return oasr.gemm_activation(x, self.weight, self.bias, self.activation_type)
        return _TORCH_ACTIVATION[self.activation_name](F.linear(x, self.weight, self.bias))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, activation={self.activation_name}"
        )


__all__ = [
    "ColumnParallelLinear",
    "Linear",
    "LinearActivation",
    "RowParallelLinear",
]
