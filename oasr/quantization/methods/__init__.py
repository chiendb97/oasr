# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Quantization methods: how a quantized GEMM layer stores and runs its weights."""

from .base import QuantMethod, method_for, register_quant_method
from .w8a8 import W8A8Method, quantize_weight_per_channel

__all__ = [
    "QuantMethod",
    "W8A8Method",
    "method_for",
    "quantize_weight_per_channel",
    "register_quant_method",
]
