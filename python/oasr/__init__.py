# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
OASR - Open Automatic Speech Recognition
High-performance ASR inference with CUDA kernels (conv, gemm, norm, attention).
"""

__version__ = "0.1.0"

# C extension: kernels and enums
try:
    from oasr import _C  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional extension
    _C = None

if _C is not None:
    kernels = _C.kernels
    DataType = _C.DataType
    ConvType = _C.ConvType
    ActivationType = _C.ActivationType
    NormType = getattr(_C, "NormType", None)
    synchronize = _C.synchronize
else:
    kernels = None
    DataType = None
    ConvType = None
    ActivationType = None
    NormType = None
    synchronize = None

# Layers package (primary home of Python kernel wrappers)
from . import layers
from .layers import (
    Conv1d,
    DepthwiseConv1d,
    PointwiseConv1d,
    Linear,
    LayerNorm,
    RMSNorm,
    GroupNorm,
    BatchNorm1d,
    AddLayerNorm,
)

__all__ = [
    "__version__",
    "kernels",
    "DataType",
    "ConvType",
    "ActivationType",
    "NormType",
    "synchronize",
    "layers",
    # Conv
    "Conv1d",
    "DepthwiseConv1d",
    "PointwiseConv1d",
    # Linear
    "Linear",
    # Norm
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BatchNorm1d",
    "AddLayerNorm",
]
