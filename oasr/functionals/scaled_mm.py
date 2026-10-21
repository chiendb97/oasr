# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for the int8 scaled GEMM.

``out = act(a_scale[m] * b_scale[n] * (a @ bᵀ)[m, n] + bias[n])`` with int8
``a`` (quantized activations: one scale per row, or one static per-tensor
scale) and ``b`` (per-channel-quantized weights, the ``nn.Linear`` ``[N, K]``
layout), an exact int32 accumulator and an fp16/bf16 output.
:func:`scaled_mm_quant` is the same GEMM quantizing its output straight to int8
for a consumer with a static input scale.  :func:`reference_scaled_mm` and
:func:`reference_scaled_mm_quant` are the torch twins.
"""

import functools
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from oasr.api_logging import oasr_api

#: Layer-level activation names (as :class:`oasr.layers.LinearActivation`
#: spells them: ``gelu`` is exact erf, ``gelu_tanh`` the tanh approximation)
#: to the kernel's ``ActivationType`` ids.
_ACTIVATION_ID = {
    None: 3,  # IDENTITY
    "relu": 0,
    "gelu_tanh": 1,
    "swish": 2,
    "silu": 2,
    "gelu": 4,  # GELU_ERF
}

_TORCH_ACTIVATION: Dict[Optional[str], Callable[[torch.Tensor], torch.Tensor]] = {
    None: lambda y: y,
    "relu": F.relu,
    "gelu_tanh": lambda y: F.gelu(y, approximate="tanh"),
    "swish": F.silu,
    "silu": F.silu,
    "gelu": F.gelu,
}


def activation_id(name: Optional[str]) -> int:
    """The kernel id of a layer-level activation name (``None`` = identity)."""
    try:
        return _ACTIVATION_ID[name]
    except KeyError:
        raise ValueError(
            f"scaled_mm cannot fuse activation {name!r}; expected one of "
            f"{sorted(k for k in _ACTIVATION_ID if k)} or None"
        ) from None


@functools.cache
def _get_scaled_mm_module():
    from oasr.jit.scaled_mm import gen_scaled_mm_module

    return gen_scaled_mm_module().build_and_load()


@oasr_api
def scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    out_dtype: torch.dtype = torch.float16,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Int8 GEMM with per-token × per-channel dequantization fused in the epilogue.

    Args:
        a: int8 ``[..., K]`` CUDA tensor, contiguous (``K % 16 == 0``).
        b: int8 ``[N, K]`` weight, contiguous (``N % 8 == 0``).
        a_scale: fp32, one scale per row of *a* (``a.numel() // K`` entries), or a
            single per-tensor scale (one entry).
        b_scale: fp32 ``[N]``, one scale per output channel.
        bias: Optional ``[N]`` in the output dtype.
        activation: Layer-level activation name, or ``None``.
        out_dtype: fp16 or bf16, when *out* is not given.
        out: Optional pre-allocated ``[..., N]`` output.

    Returns:
        ``[..., N]`` tensor in the output dtype.
    """
    from oasr.jit.scaled_mm import scaled_mm_func_name, select_config

    k = a.shape[-1]
    m = a.numel() // k if k else 0
    n = b.shape[0]
    if out is None:
        out = torch.empty((*a.shape[:-1], n), dtype=out_dtype, device=a.device)
    cfg = select_config(m, n, k)
    fn = getattr(_get_scaled_mm_module(), scaled_mm_func_name(cfg))
    fn(out, a, b, a_scale, b_scale, bias, activation_id(activation))
    return out


@oasr_api
def scaled_mm_quant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    out_mul: torch.Tensor,
    round_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """:func:`scaled_mm`, quantized straight to int8 for the consuming GEMM.

    ``q = clamp(rne(f32(round_dtype(act(...))) * out_mul[n]), -127, 127)``: the
    value :func:`scaled_mm` would have stored in *round_dtype*, quantized with the
    consumer's static multiplier (:func:`oasr.functionals.quant.static_multiplier`)
    -- bit-identical to ``quantize_static(scaled_mm(...), out_mul)``, without the
    fp round trip through global memory or the second launch.

    Args:
        a, b, a_scale, b_scale, activation: As :func:`scaled_mm`.
        bias: Optional ``[N]`` in *round_dtype*.
        out_mul: fp32 ``[N]``, the consumer's per-channel quantize multiplier.
        round_dtype: fp16 or bf16 -- the layer's floating dtype.
        out: Optional pre-allocated int8 ``[..., N]``.

    Returns:
        int8 ``[..., N]``.
    """
    from oasr.jit.scaled_mm import scaled_mm_quant_func_name, select_quant_config

    k = a.shape[-1]
    m = a.numel() // k if k else 0
    n = b.shape[0]
    if out is None:
        out = torch.empty((*a.shape[:-1], n), dtype=torch.int8, device=a.device)
    code = _ROUND_DTYPE_CODE.get(round_dtype)
    if code is None:
        raise ValueError(f"scaled_mm_quant rounds to fp16 or bf16, not {round_dtype}")
    cfg = select_quant_config(m, n, k)
    fn = getattr(_get_scaled_mm_module(), scaled_mm_quant_func_name(cfg))
    fn(out, a, b, a_scale, b_scale, bias, out_mul, activation_id(activation), code)
    return out


#: ``round_dtype`` -> the launcher's code.
_ROUND_DTYPE_CODE = {torch.float16: 0, torch.bfloat16: 1}


def reference_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Torch twin of :func:`scaled_mm` on any device.

    The int32 accumulator is computed exactly (fp64 products and sums of int8
    values stay below 2**53 for any realistic K), converted to fp32 with one
    rounding as the kernel's epilogue does, then scaled, biased and activated in
    fp32 and rounded once to the output dtype.  The kernel's epilogue uses an FMA
    for ``a_scale * t + bias``, so the two agree to within one output ulp.
    """
    activation_id(activation)  # validate the name
    k = a.shape[-1]
    a2 = a.reshape(-1, k)
    acc = (a2.double() @ b.double().t()).to(torch.int64).to(torch.float32)
    t = acc * b_scale.reshape(1, -1).float()
    y = t * a_scale.reshape(-1, 1).float()
    if bias is not None:
        y = y + bias.float().reshape(1, -1)
    y = _TORCH_ACTIVATION[activation](y)
    return y.to(out_dtype).reshape(*a.shape[:-1], b.shape[0])


def reference_scaled_mm_quant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    out_mul: torch.Tensor,
    round_dtype: torch.dtype,
) -> torch.Tensor:
    """Torch twin of :func:`scaled_mm_quant`: the reference GEMM, then the static quantize.

    Within one int8 step of the kernel at a rounding tie (the GEMM half agrees
    with the kernel to one output ulp; see :func:`reference_scaled_mm`).
    """
    from oasr.functionals.quant import reference_quantize_static

    y = reference_scaled_mm(a, b, a_scale, b_scale, bias, activation, round_dtype)
    return reference_quantize_static(y, out_mul)
