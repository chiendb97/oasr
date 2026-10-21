# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Quantization algorithms: how the numbers are chosen (offline)."""

from .base import (
    ForwardLoop,
    QuantAlgorithm,
    build_algorithm,
    list_algorithms,
    register_quant_algorithm,
)
from .rtn import RoundToNearest
from .smoothquant import SmoothQuant, SmoothQuantOptions
from .static_inputs import StaticInputs, StaticInputsOptions

__all__ = [
    "ForwardLoop",
    "QuantAlgorithm",
    "RoundToNearest",
    "SmoothQuant",
    "SmoothQuantOptions",
    "StaticInputs",
    "StaticInputsOptions",
    "build_algorithm",
    "list_algorithms",
    "register_quant_algorithm",
]
