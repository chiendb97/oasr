# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Mappings between Python/PyTorch types and OASR C extension enums/classes.

Every function lazily imports ``oasr._C`` (or ``oasr.layers``) so that this
module can be loaded during ``oasr`` package initialization without triggering
circular imports.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def get_dtype(dtype: torch.dtype):
    """Map a ``torch.dtype`` to the corresponding ``oasr.DataType`` enum."""
    from oasr import _C  # type: ignore[attr-defined]

    return {
        torch.float32: _C.DataType.FP32,
        torch.float16: _C.DataType.FP16,
        torch.bfloat16: _C.DataType.BF16,
    }.get(dtype, _C.DataType.FP16)


def get_activation_type(activation_type: str):
    """Map an activation name to the corresponding ``oasr.ActivationType`` enum."""
    from oasr import _C  # type: ignore[attr-defined]

    return {
        "swish": _C.ActivationType.SWISH,
        "silu": _C.ActivationType.SWISH,
        "relu": _C.ActivationType.RELU,
        "gelu": _C.ActivationType.GELU,
    }.get(activation_type.lower(), _C.ActivationType.SWISH)


def get_activation(activation_type: str) -> nn.Module:
    """Map an activation name to a fresh ``torch.nn`` activation module."""
    cls = {
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
    }.get(activation_type.lower(), nn.SiLU)
    return cls()


def get_norm_type(norm_type: str):
    """Map a norm name to the corresponding ``oasr.NormType`` enum."""
    from oasr import _C  # type: ignore[attr-defined]

    return {
        "layer_norm": _C.NormType.LAYER_NORM,
        "rms_norm": _C.NormType.RMS_NORM,
        "batch_norm": _C.NormType.BATCH_NORM,
        "group_norm": _C.NormType.GROUP_NORM,
    }.get(norm_type.lower(), _C.NormType.BATCH_NORM)


def get_norm(norm_type: str):
    """Map a norm name to the corresponding OASR norm layer *class*."""
    from oasr.layers import norm as _norm

    return {
        "layer_norm": _norm.LayerNorm,
        "rms_norm": _norm.RMSNorm,
        "batch_norm": _norm.BatchNorm1d,
        "group_norm": _norm.GroupNorm,
    }.get(norm_type.lower(), _norm.BatchNorm1d)
