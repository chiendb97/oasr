# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for activation quantization.

The numerics are pinned (``include/oasr/quant/quantize.cuh``) and mirrored bit
for bit by :func:`reference_quantize_per_token`, which is both the parity oracle
and the torch path of a quantized layer on CPU / fp32.
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token int8 quantization of the last axis.

    Args:
        input: fp16/bf16 CUDA tensor ``[..., K]``, contiguous.
        smooth_inv: Optional fp32 ``[K]`` multiplied into each row before the
            amax (SmoothQuant's runtime smoother).
        eps: Floor on each row's amax.
        out: Optional pre-allocated int8 tensor shaped like *input*.
        scale: Optional pre-allocated fp32 tensor with one entry per row.

    Returns:
        ``(q, scale)``: int8 values shaped like *input* and the fp32 per-row
        scale (``input ≈ q * scale[:, None]``), shaped ``input.shape[:-1]``.
    """
    if out is None:
        out = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    if scale is None:
        scale = torch.empty(input.shape[:-1], dtype=torch.float32, device=input.device)
    _get_quant_module().quantize_per_token(out, scale, input, smooth_inv, eps)
    return out, scale


def reference_quantize_per_token(
    input: torch.Tensor,
    smooth_inv: Optional[torch.Tensor] = None,
    eps: float = QUANT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch twin of :func:`quantize_per_token`, bit-exact on any device.

    Every step is an IEEE-rounded fp32 operation that the kernel performs in the
    same order: one multiply by the smoother, the amax, one division for the
    reciprocal, one multiply per element, round-half-to-even, clamp.
    """
    x = input.float()
    if smooth_inv is not None:
        x = x * smooth_inv.float()
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    # Tensor / tensor on purpose: torch lowers ``tensor / python_scalar`` to a
    # multiply by the scalar's reciprocal, one ulp away from the kernel's
    # ``__fdiv_rn``.
    qmax = torch.full_like(amax, float(INT8_QMAX))
    inv = qmax / amax
    q = torch.round(x * inv).clamp_(-INT8_QMAX, INT8_QMAX).to(torch.int8)
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
    input: torch.Tensor, mul: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Static int8 quantization: ``q = clamp(rne(f32(input) * mul), -127, 127)``.

    Args:
        input: fp16/bf16 CUDA tensor ``[..., K]``, contiguous.
        mul: fp32 ``[K]`` per-channel multiplier (:func:`static_multiplier`).
        out: Optional pre-allocated int8 tensor shaped like *input*.

    Returns:
        int8 values shaped like *input*; the matching scale is the layer's
        calibrated per-tensor ``input_scale``.
    """
    if out is None:
        out = torch.empty(input.shape, dtype=torch.int8, device=input.device)
    _get_quant_module().quantize_static(out, input, mul)
    return out


def reference_quantize_static(input: torch.Tensor, mul: torch.Tensor) -> torch.Tensor:
    """Torch twin of :func:`quantize_static`, bit-exact on any device."""
    return torch.round(input.float() * mul.float()).clamp_(-INT8_QMAX, INT8_QMAX).to(torch.int8)
