# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Quantization: the ninth registry axis.

Three planes joined by one artifact and one interface:

* **offline** (:mod:`.calibration`, :mod:`.algorithms`, :mod:`.export`,
  ``oasr quantize``): calibrate on audio, smooth, quantize, write a native
  checkpoint whose ``oasr_config.json`` carries a :class:`QuantizationSpec`;
* **the artifact**: that checkpoint (int8 weights + fp32 scales);
* **runtime** (:mod:`.methods`, :mod:`.kernels`, :mod:`.apply`): every
  :class:`~oasr.layers.QuantizableGemm` the spec selects hands its forward to a
  :class:`~oasr.quantization.methods.QuantMethod`, which runs a kernel lane chosen
  once per layer.

Adding a format is a preset + a method + its lanes; a backend is a lane; an
algorithm is a registration; an architecture is its ``quantization_layout()``.
See ``.artifacts/quantization_design.md`` and ``docs/quantization.md``.
"""

from .apply import (
    apply_quantization,
    bind_quantization,
    check_quantized_state_dict,
    describe_quantization,
    finalize_quantization,
    format_quant_report,
    quantizable_layers,
    quantized_layers,
    validate_runtime,
)
from .layout import FOLD_KINDS, GemmChain, QuantLayout, SmoothGroup
from .methods import QuantMethod, method_for, register_quant_method
from .spec import (
    FP8_W8A8,
    FP8_W8A8_STATIC,
    INT8_W8A8,
    INT8_W8A8_STATIC,
    PRESETS,
    QuantArgs,
    QuantGroup,
    QuantizationSpec,
    QuantScheme,
    get_scheme,
)

__all__ = [
    "FOLD_KINDS",
    "FP8_W8A8",
    "FP8_W8A8_STATIC",
    "GemmChain",
    "INT8_W8A8",
    "INT8_W8A8_STATIC",
    "PRESETS",
    "QuantArgs",
    "QuantGroup",
    "QuantLayout",
    "QuantMethod",
    "QuantScheme",
    "QuantizationSpec",
    "SmoothGroup",
    "apply_quantization",
    "bind_quantization",
    "check_quantized_state_dict",
    "describe_quantization",
    "finalize_quantization",
    "format_quant_report",
    "get_scheme",
    "method_for",
    "quantizable_layers",
    "quantized_layers",
    "register_quant_method",
    "validate_runtime",
]
