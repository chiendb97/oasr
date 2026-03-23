# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for activation operations."""

import functools
from typing import Optional

import torch

# Activation type integer constants matching ActivationType enum in include/oasr/common/types.h
ACTIVATION_RELU = 0
ACTIVATION_GELU = 1
ACTIVATION_SWISH = 2

_ACTIVATION_NAME_TO_ID = {
    "relu": ACTIVATION_RELU,
    "gelu": ACTIVATION_GELU,
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
