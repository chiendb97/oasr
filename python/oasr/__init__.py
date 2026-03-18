# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
OASR - Open Automatic Speech Recognition
High-performance ASR inference with CUDA kernels (conv, gemm, norm, attention).
"""

__version__ = "0.1.0"

import importlib as _importlib
import sys as _sys
import types as _types

# Layers package (primary home of Python kernel wrappers)
from . import layers
from .layers import (
    DepthwiseConv1d,
    PointwiseConv1d,
    Conv2d,
    Conv2dActivation,
    Linear,
    LayerNorm,
    RMSNorm,
    GroupNorm,
    BatchNorm1d,
    AddLayerNorm,
)


def _register_c_extension():
    """Load the C extension and register its submodules (e.g. ``decoder``)
    in ``sys.modules`` so that ``from oasr.decoder import ...`` works."""
    try:
        _C = _importlib.import_module("oasr._C")
    except ImportError:
        return
    globals()["_C"] = _C
    for _attr_name in dir(_C):
        _attr = getattr(_C, _attr_name)
        if isinstance(_attr, _types.ModuleType):
            _sys.modules[f"{__name__}.{_attr_name}"] = _attr
            globals()[_attr_name] = _attr


_register_c_extension()


def __getattr__(name: str):
    """Lazily expose C extension symbols (kernels, enums, synchronize, …).

    On first access the compiled ``oasr._C`` module is imported and the
    requested attribute is cached in the package globals so that subsequent
    look-ups are instant and skip this function entirely.
    """
    _C = globals().get("_C")
    if _C is None:
        _C = _importlib.import_module("oasr._C")
        globals()["_C"] = _C
    if name == "_C":
        return _C
    try:
        attr = getattr(_C, name)
    except AttributeError:
        raise AttributeError(
            f"module 'oasr' has no attribute {name!r}"
        ) from None
    globals()[name] = attr
    return attr


__all__ = [
    "__version__",
    # C extension (loaded lazily via __getattr__)
    "kernels",
    "DataType",
    "ConvType",
    "ActivationType",
    "NormType",
    "layers",
    # Conv
    "DepthwiseConv1d",
    "PointwiseConv1d",
    "Conv2d",
    "Conv2dActivation",
    # Linear
    "Linear",
    # Norm
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BatchNorm1d",
    "AddLayerNorm",
]
