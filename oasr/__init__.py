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

# =============================================================================
# Functional API (FlashInfer style)
# =============================================================================
from .activation import (
    glu, swish,
    ACTIVATION_RELU, ACTIVATION_GELU, ACTIVATION_SWISH,
    get_activation_type_id,
)
from .norm import (
    layer_norm, rms_norm, batch_norm_1d, group_norm, add_layer_norm,
    layer_norm_activation, rms_norm_activation, batch_norm_activation, batch_norm_swish,
    cmvn,
)
from .conv import (
    depthwise_conv1d, conv2d,
    depthwise_conv1d_silu, causal_conv1d, conv2d_activation,
)
from .gemm import gemm, bmm, group_gemm, gemm_activation
from .ctc_decode import (
    ctc_beam_search_decode,
    GpuStreamingDecoder,
    GpuDecoderConfig,
    GpuDecoderResult,
)

# =============================================================================
# Autotuning
# =============================================================================
from . import tune
from .tune import autotune, enable_autotune, disable_autotune

# =============================================================================
# nn.Module wrappers
# =============================================================================
from . import layers
from .layers import (
    DepthwiseConv1d,
    PointwiseConv1d,
    Conv2d as Conv2dModule,
    Conv2dActivation,
    Linear,
    LayerNorm,
    RMSNorm,
    GroupNorm,
    BatchNorm1d,
    AddLayerNorm,
    GlobalCMVN,
)


# =============================================================================
# Legacy C extension support (backward compatibility)
# =============================================================================

def _register_c_extension():
    """Load the C extension and register its submodules in ``sys.modules``.

    The ``decoder`` submodule is intentionally skipped here because
    ``oasr/decoder/`` is a real Python package that handles its own imports
    from the C extension.
    """
    try:
        _C = _importlib.import_module("oasr._C")
    except ImportError:
        return
    globals()["_C"] = _C
    for _attr_name in dir(_C):
        if _attr_name == "decoder":
            # oasr.decoder is a Python package; do not overwrite it.
            continue
        _attr = getattr(_C, _attr_name)
        if isinstance(_attr, _types.ModuleType):
            _sys.modules[f"{__name__}.{_attr_name}"] = _attr
            globals()[_attr_name] = _attr


_register_c_extension()

# =============================================================================
# Streaming cache manager
# =============================================================================
from . import cache
from .cache import (
    CacheConfig,
    BlockPool,
    AttentionCacheManager,
    CnnCacheManager,
    CtcStateCacheManager,
    StreamContext,
)

# =============================================================================
# High-level decoder API
# =============================================================================
from .decode import Decoder, DecoderConfig, DecoderResult


def __getattr__(name: str):
    """Lazily expose C extension symbols (kernels, enums, synchronize, ...)."""
    _C = globals().get("_C")
    if _C is None:
        try:
            _C = _importlib.import_module("oasr._C")
            globals()["_C"] = _C
        except ImportError:
            raise AttributeError(
                f"module 'oasr' has no attribute {name!r}"
            ) from None
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
    # Activation constants
    "ACTIVATION_RELU",
    "ACTIVATION_GELU",
    "ACTIVATION_SWISH",
    "get_activation_type_id",
    # Functional API
    "glu",
    "swish",
    "layer_norm",
    "rms_norm",
    "batch_norm_1d",
    "group_norm",
    "add_layer_norm",
    "layer_norm_activation",
    "rms_norm_activation",
    "batch_norm_activation",
    "batch_norm_swish",
    "cmvn",
    "depthwise_conv1d",
    "conv2d",
    "depthwise_conv1d_silu",
    "causal_conv1d",
    "conv2d_activation",
    "gemm",
    "bmm",
    "group_gemm",
    "gemm_activation",
    # nn.Module wrappers
    "layers",
    "DepthwiseConv1d",
    "PointwiseConv1d",
    "Conv2dModule",
    "Conv2dActivation",
    "Linear",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BatchNorm1d",
    "AddLayerNorm",
    "GlobalCMVN",
    # Autotuning
    "tune",
    "autotune",
    "enable_autotune",
    "disable_autotune",
    # High-level decoder API
    "Decoder",
    "DecoderConfig",
    "DecoderResult",
    # GPU CTC decoder
    "ctc_beam_search_decode",
    "GpuStreamingDecoder",
    "GpuDecoderConfig",
    "GpuDecoderResult",
    # Streaming cache manager
    "cache",
    "CacheConfig",
    "BlockPool",
    "AttentionCacheManager",
    "CnnCacheManager",
    "CtcStateCacheManager",
    "StreamContext",
    # Legacy C extension (loaded lazily via __getattr__)
    "DataType",
    "ConvType",
    "ActivationType",
    "NormType",
]
