# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for the fused gated MLP (SwiGLU / GeGLU).

``gated_mlp`` is the first two thirds of a gated feed-forward block in one
launch::

    activation(x @ w_gateᵀ + b_gate) * (x @ w_upᵀ + b_up)
"""

from typing import Any, Optional, Tuple

import torch

from oasr.api_logging import oasr_api

# ``torch.cuda.current_stream()`` builds a Python ``Stream`` object to hand back
# one integer, which measured 4.1 us against a 6 us kernel.  Hoisted here, and
# cached against the raw pointer, for the same reason as in
# ``oasr/functionals/attention.py``; see :mod:`oasr.jit.cute_runtime`.
from oasr.jit.cute_runtime import current_stream as _current_stream
from oasr.jit.mlp import ALIGNMENT, routed_gated_mlp

#: Gate activations the fused kernel implements, spelled the way ``oasr.layers``
#: spells them.  ``gelu`` is the exact-erf form and ``gelu_tanh`` the tanh
#: approximation; they stay separate names because they are different epilogues.
GATED_MLP_ACTIVATIONS = frozenset({"silu", "swish", "relu", "gelu", "gelu_tanh", "identity"})

_DTYPES = (torch.float16, torch.bfloat16)

#: One dummy per (device, dtype), handed to the kernel in place of an absent
#: bias.  The kernel is compiled with ``has_bias=False`` there, so the tensor is
#: never dereferenced -- it exists only to give the compiled signature an operand
#: of the right rank and dtype.
_DUMMY: dict = {}


def _dummy_bias(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (dtype, device.type, device.index)
    buf = _DUMMY.get(key)
    if buf is None:
        buf = torch.zeros(ALIGNMENT, dtype=dtype, device=device)
        _DUMMY[key] = buf
    return buf


def _shape(x: torch.Tensor, w_gate: torch.Tensor) -> Tuple[int, int, int]:
    """``(rows, n, k)`` for an ``x`` of any leading rank."""
    return x.numel() // x.shape[-1], w_gate.shape[0], x.shape[-1]


def gated_mlp_available(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    *,
    activation: str = "silu",
    has_bias: bool = False,
) -> bool:
    """Would :func:`gated_mlp` run the fused kernel for these operands?

    ``False`` means "use the two-GEMM path", and covers both *cannot* (wrong
    dtype, device, alignment, arch, CuTeDSL absent) and *should not* (outside the
    measured band).  Compiles on the first ``True`` for a configuration and then
    memoises, so a per-layer call is a dict lookup.
    """
    return _route(x, w_gate, activation=activation, has_bias=has_bias)[0] is not None


def _route(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    *,
    activation: str,
    has_bias: bool,
) -> Tuple[Optional[Any], int, int, int]:
    """``(kernel_or_None, rows, n, k)``.

    Every precondition the kernel's own contract does not cover lives here, and
    each one is a *silent* failure if it is left out rather than an error:

    * ``x`` non-contiguous -- ``reshape`` would copy, and the 2-D kernel would
      write into the copy.
    * ``w_up`` non-contiguous, or a shape that disagrees with ``w_gate`` --
      ``w_up`` is the operand :func:`gated_mlp_available` cannot see, so the
      check has to be here rather than at the call site.
    * ``w_gate.shape[1] != x.shape[-1]`` -- ``K`` is taken from ``x``, so a
      mismatch reads the weight at the wrong stride instead of raising.
    """
    if activation not in GATED_MLP_ACTIVATIONS:
        return None, 0, 0, 0
    if x.dtype not in _DTYPES or w_gate.dtype is not x.dtype:
        return None, 0, 0, 0
    if not x.is_cuda or x.dim() < 2 or w_gate.dim() != 2:
        return None, 0, 0, 0
    if not x.is_contiguous() or not w_gate.is_contiguous():
        return None, 0, 0, 0
    if w_gate.shape[1] != x.shape[-1]:
        return None, 0, 0, 0
    rows, n, k = _shape(x, w_gate)
    dtype_str = "float16" if x.dtype is torch.float16 else "bfloat16"
    fn = routed_gated_mlp(
        dtype_str=dtype_str, activation=activation, has_bias=has_bias, rows=rows, n=n, k=k
    )
    return fn, rows, n, k


@oasr_api
def gated_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    bias_gate: Optional[torch.Tensor] = None,
    bias_up: Optional[torch.Tensor] = None,
    *,
    activation: str = "silu",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``activation(x @ w_gateᵀ + bias_gate) * (x @ w_upᵀ + bias_up)``.

    Parameters
    ----------
    x : torch.Tensor
        ``(*, K)``, fp16 or bf16 on CUDA, contiguous.
    w_gate, w_up : torch.Tensor
        ``(N, K)`` each -- ``nn.Linear``'s own ``(out, in)`` layout.
    bias_gate, bias_up : torch.Tensor, optional
        ``(N,)`` each.  Either both or neither.
    activation : str
        One of :data:`GATED_MLP_ACTIVATIONS`, applied to the **gate** half only.
    out : torch.Tensor, optional
        ``(*, N)`` destination.

    Returns
    -------
    torch.Tensor
        ``(*, N)``.

    Raises
    ------
    ValueError
        If the fused kernel cannot serve this call.  Ask
        :func:`gated_mlp_available` first -- this function does not fall back,
        because a silent reroute to torch is what makes a missing kernel
        invisible (``oasr/layers/_backend.py``).
    """
    if (bias_gate is None) != (bias_up is None):
        raise ValueError("gated_mlp needs both biases or neither")
    has_bias = bias_gate is not None
    fn, rows, n, k = _route(x, w_gate, activation=activation, has_bias=has_bias)
    if fn is None:
        raise ValueError(
            "oasr.gated_mlp cannot serve this call "
            f"(shape={tuple(x.shape)} w_gate={tuple(w_gate.shape)} dtype={x.dtype} "
            f"activation={activation!r}); ask gated_mlp_available() first"
        )
    if w_up.shape != w_gate.shape or not w_up.is_contiguous():
        raise ValueError(
            f"w_up must match w_gate {tuple(w_gate.shape)} and be contiguous; "
            f"got {tuple(w_up.shape)} contiguous={w_up.is_contiguous()}"
        )
    if bias_gate is not None and bias_up is not None:
        if not (bias_gate.is_contiguous() and bias_up.is_contiguous()):
            raise ValueError("gated_mlp needs contiguous biases (the kernel reads them 1-D)")
    if out is not None and not out.is_contiguous():
        # ``reshape`` on a non-contiguous ``out`` copies, and the kernel would
        # then write into the copy and the caller would observe nothing.
        raise ValueError("gated_mlp needs a contiguous out= (it is written through a view)")
    if out is None:
        out = torch.empty(x.shape[:-1] + (n,), device=x.device, dtype=x.dtype)
    dummy = None if has_bias else _dummy_bias(x.dtype, x.device)
    # The kernel is 2-D; a (B, T, K) caller reshapes for free because x is
    # contiguous, and ``out`` is written through the same view.
    fn(
        x.reshape(rows, k),
        w_gate,
        w_up,
        bias_gate if has_bias else dummy,
        bias_up if has_bias else dummy,
        out.reshape(rows, n),
        _current_stream(),
    )
    return out


__all__ = ["gated_mlp", "gated_mlp_available", "GATED_MLP_ACTIVATIONS"]
