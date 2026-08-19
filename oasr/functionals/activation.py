# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for activation operations."""

import functools
from typing import Optional

import torch

from oasr.api_logging import oasr_api

# Activation type integer constants matching ActivationType enum in include/oasr/common/types.h
ACTIVATION_RELU = 0
ACTIVATION_GELU = 1
ACTIVATION_SWISH = 2
ACTIVATION_GELU_ERF = 4

_ACTIVATION_NAME_TO_ID = {
    "relu": ACTIVATION_RELU,
    "gelu": ACTIVATION_GELU,
    "gelu_tanh": ACTIVATION_GELU,
    "gelu_erf": ACTIVATION_GELU_ERF,
    "swish": ACTIVATION_SWISH,
    "silu": ACTIVATION_SWISH,
}


def get_activation_type_id(name: str) -> int:
    """Map an activation name to its integer ID for TVM-FFI kernels."""
    return _ACTIVATION_NAME_TO_ID[name.lower()]


@functools.cache
def _get_activation_module():
    from oasr.jit.activation import gen_activation_module

    return gen_activation_module().build_and_load()


@oasr_api
def glu(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Gated Linear Unit activation.

    Computes: output = input[..., :channels] * sigmoid(input[..., channels:])

    Args:
        input: Input tensor [..., 2 * channels].
        out: Optional pre-allocated output tensor [..., channels].

    Returns:
        Output tensor [..., channels].
    """
    if out is None:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    _get_activation_module().glu(out, input)
    return out


@oasr_api
def swish(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Swish (SiLU) activation: x * sigmoid(x).

    Args:
        input: Input tensor.
        out: Optional pre-allocated output tensor.

    Returns:
        Output tensor with same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_activation_module().swish(out, input)
    return out


@oasr_api
def gelu(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Exact-erf GELU: ``0.5 * x * (1 + erf(x / sqrt(2)))``.

    Args:
        input: Input tensor. Non-contiguous inputs are materialized once.
        out: Optional pre-allocated contiguous output tensor.

    Returns:
        Output tensor with the same shape as input.
    """
    input = input.contiguous()
    if out is None:
        out = torch.empty_like(input)
    _get_activation_module().gelu_erf(out, input)
    return out


def _prepare_elementwise(
    input: torch.Tensor, out: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Channel slices can have contiguous rows separated by a regular padded
    # stride. The CUDA kernel consumes that
    # layout directly; only arbitrary transposes/expands need materialization.
    regular_rows = input.dim() == 0 or (
        input.stride(-1) == 1
        and all(
            input.stride(i) == input.shape[i + 1] * input.stride(i + 1)
            for i in range(input.dim() - 2)
        )
    )
    if not regular_rows:
        input = input.contiguous()
    if out is None:
        out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    return input, out


@oasr_api
def sigmoid(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Elementwise logistic sigmoid.

    Contiguous rows with a regular padded stride are consumed directly; other
    non-contiguous inputs are materialized once. ``out``, when supplied, must be
    contiguous and have the same shape, dtype, and device as ``input``.
    """
    input, out = _prepare_elementwise(input, out)
    _get_activation_module().sigmoid(out, input)
    return out


@oasr_api
def tanh(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Elementwise hyperbolic tangent."""
    input, out = _prepare_elementwise(input, out)
    _get_activation_module().tanh(out, input)
    return out


@oasr_api
def relu(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Elementwise rectified linear unit: ``max(input, 0)``."""
    input, out = _prepare_elementwise(input, out)
    _get_activation_module().relu(out, input)
    return out


@oasr_api
def swoosh_l(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Zipformer Swoosh-L activation: ``log(1 + exp(x - 4)) - 0.08 x - 0.035``.

    Elementwise over a flat contiguous buffer (shape-agnostic). A non-contiguous
    input is made contiguous first.

    Args:
        input: Input tensor.
        out: Optional pre-allocated output tensor.

    Returns:
        Output tensor with same shape as input.
    """
    input = input.contiguous()
    if out is None:
        out = torch.empty_like(input)
    _get_activation_module().swoosh_l(out, input)
    return out


@oasr_api
def swoosh_r(input: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Zipformer Swoosh-R activation: ``log(1 + exp(x - 1)) - 0.08 x - 0.313261687``.

    Elementwise over a flat contiguous buffer (shape-agnostic). A non-contiguous
    input is made contiguous first.

    Args:
        input: Input tensor.
        out: Optional pre-allocated output tensor.

    Returns:
        Output tensor with same shape as input.
    """
    input = input.contiguous()
    if out is None:
        out = torch.empty_like(input)
    _get_activation_module().swoosh_r(out, input)
    return out
