# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for activation quantization.

Two 8-bit formats, chosen by the output dtype: ``torch.int8`` (``Q = 127``,
round half to even, clamp) and ``torch.float8_e4m3fn`` (``Q = 448``, round to
nearest even in the fp8 grid, saturating).  The numerics are pinned
(``include/oasr/quant/quantize.cuh``) and mirrored bit for bit by
:func:`reference_quantize_per_token` / :func:`reference_quantize_static`, which
are both the parity oracles and the torch path of a quantized layer on CPU / fp32.
"""

import functools
from typing import Optional, Tuple

import torch

from oasr.api_logging import oasr_api

#: Floor on a row's amax.  An all-zero row (padding) quantizes to zeros with a
#: finite scale instead of dividing by zero.
QUANT_EPS = 1e-8

#: Symmetric int8 range: -128 is never produced, so negation is exact.
INT8_QMAX = 127

#: The largest finite ``float8_e4m3fn`` (its all-ones pattern is NaN; no infinity).
FP8_E4M3_QMAX = 448.0

#: The 8-bit formats the quantize kernels write, and each one's symmetric range.
QUANT_QMAX = {torch.int8: float(INT8_QMAX), torch.float8_e4m3fn: FP8_E4M3_QMAX}


def quant_qmax(dtype: torch.dtype) -> float:
    """The symmetric range of a quantized dtype (127 for int8, 448 for e4m3)."""
    try:
        return QUANT_QMAX[dtype]
    except KeyError:
        raise ValueError(
            f"no 8-bit quantization format {dtype}; expected one of {sorted(map(str, QUANT_QMAX))}"
        ) from None


def round_to_quant(y: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """fp32 *y* (already scaled into range) rounded to the quantized *dtype*.

    int8: round half to even, then clamp to ±127.  e4m3: clamp to ±448, then
    torch's round-to-nearest-even conversion -- the kernel's satfinite ``cvt``.
    """
    if dtype == torch.int8:
        return torch.round(y).clamp_(-INT8_QMAX, INT8_QMAX).to(torch.int8)
    qmax = quant_qmax(dtype)
    return y.clamp(-qmax, qmax).to(dtype)


@functools.cache
def _get_quant_module():
    from oasr.jit.quant import gen_quant_module

    return gen_quant_module().build_and_load()


@oasr_api
def quantize_per_token(
    input: torch.Tensor,
    smooth_inv: Optional[torch.Tensor] = None,
    eps: float = QUANT_EPS,
    out: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token 8-bit quantization of the last axis.

    Args:
        input: fp16/bf16 CUDA tensor ``[..., K]``, contiguous.
        smooth_inv: Optional fp32 ``[K]`` multiplied into each row before the
            amax (SmoothQuant's runtime smoother).
        eps: Floor on each row's amax.
        out: Optional pre-allocated output shaped like *input*; its dtype wins
            over *dtype*.
        scale: Optional pre-allocated fp32 tensor with one entry per row.
        dtype: ``torch.int8`` or ``torch.float8_e4m3fn``, when *out* is not given.

    Returns:
        ``(q, scale)``: 8-bit values shaped like *input* and the fp32 per-row
        scale (``input ≈ q * scale[:, None]``), shaped ``input.shape[:-1]``.
    """
    if out is None:
        quant_qmax(dtype)
        out = torch.empty(input.shape, dtype=dtype, device=input.device)
    if scale is None:
        scale = torch.empty(input.shape[:-1], dtype=torch.float32, device=input.device)
    _get_quant_module().quantize_per_token(out, scale, input, smooth_inv, eps)
    return out, scale


def reference_quantize_per_token(
    input: torch.Tensor,
    smooth_inv: Optional[torch.Tensor] = None,
    eps: float = QUANT_EPS,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch twin of :func:`quantize_per_token`, bit-exact on any device.

    Every step is an IEEE-rounded fp32 operation that the kernel performs in the
    same order: one multiply by the smoother, the amax, one division for the
    reciprocal, one multiply per element, then the format's rounding
    (:func:`round_to_quant`).
    """
    x = input.float()
    if smooth_inv is not None:
        x = x * smooth_inv.float()
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    # Tensor / tensor on purpose: torch lowers ``tensor / python_scalar`` to a
    # multiply by the scalar's reciprocal, one ulp away from the kernel's
    # ``__fdiv_rn``.
    qmax = torch.full_like(amax, quant_qmax(dtype))
    inv = qmax / amax
    q = round_to_quant(x * inv, dtype)
    scale = (amax / qmax).squeeze(-1)
    return q, scale


def static_multiplier(
    input_scale: torch.Tensor, smooth_inv: Optional[torch.Tensor], in_features: int
) -> torch.Tensor:
    """The per-channel multiplier a static-scale layer quantizes its input with.

    ``mul[k] = smooth_inv[k] * (1 / input_scale)`` (``1 / input_scale`` alone without
    a runtime smoother), fp32 ``[in_features]`` on ``input_scale``'s device.
    Pinned: one IEEE division, then one IEEE multiply per channel -- computed
    once when a layer is bound, and identically by the torch reference, so the
    kernel and the reference quantize with the same bits.
    """
    scale = input_scale.float().reshape(1)
    inv = torch.ones_like(scale) / scale  # tensor / tensor: an IEEE division
    if smooth_inv is None:
        return inv.expand(in_features).contiguous()
    return (smooth_inv.float().to(scale.device) * inv).contiguous()


@oasr_api
def quantize_static(
    input: torch.Tensor,
    mul: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    """Static 8-bit quantization: ``q = round_to_quant(f32(input) * mul)``.

    Args:
        input: fp16/bf16 CUDA tensor ``[..., K]``, contiguous.
        mul: fp32 ``[K]`` per-channel multiplier (:func:`static_multiplier`).
        out: Optional pre-allocated output shaped like *input*; its dtype wins
            over *dtype*.
        dtype: ``torch.int8`` or ``torch.float8_e4m3fn``, when *out* is not given.

    Returns:
        8-bit values shaped like *input*; the matching scale is the layer's
        calibrated per-tensor ``input_scale``.
    """
    if out is None:
        quant_qmax(dtype)
        out = torch.empty(input.shape, dtype=dtype, device=input.device)
    _get_quant_module().quantize_static(out, input, mul)
    return out


def reference_quantize_static(
    input: torch.Tensor, mul: torch.Tensor, dtype: torch.dtype = torch.int8
) -> torch.Tensor:
    """Torch twin of :func:`quantize_static`, bit-exact on any device."""
    return round_to_quant(input.float() * mul.float(), dtype)
