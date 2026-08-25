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
# Autotuning
# =============================================================================
# =============================================================================
# nn.Module wrappers
# =============================================================================
from . import functionals, layers, tune
from .features import (
    BatchedStreamingFeatureExtractor,
    FeatureConfig,
    extract_features_batch,
    fbank_batch,
    mfcc_batch,
)

# =============================================================================
# Functional API (FlashInfer style)
# =============================================================================
from .functionals.activation import (
    ACTIVATION_GELU,
    ACTIVATION_GELU_ERF,
    ACTIVATION_RELU,
    ACTIVATION_SWISH,
    gelu,
    get_activation_type_id,
    glu,
    relu,
    sigmoid,
    swish,
    swoosh_l,
    swoosh_r,
    tanh,
)
from .functionals.attention import fmha
from .functionals.conv import (
    causal_conv1d,
    conv1d,
    conv1d_activation,
    conv2d,
    conv2d_activation,
    depthwise_conv1d,
    depthwise_conv1d_silu,
)
from .functionals.ctc_decode import (
    GpuDecoderConfig,
    GpuDecoderResult,
    GpuStreamingDecoder,
    StreamHandle,
    StreamState,
    ctc_beam_search_decode,
)
from .functionals.feature import (
    dct_lifter,
    fbank_preprocess,
    lfr_gather,
    mel_log,
    stft_frame,
    whisper_logmel,
)
from .functionals.fft import rfft, rfft_power
from .functionals.gemm import bmm, gemm, gemm_activation, gemm_log_softmax, group_gemm
from .functionals.norm import (
    add_layer_norm,
    add_layer_norm_residual,
    add_rms_norm,
    add_rms_norm_residual,
    batch_norm_1d,
    batch_norm_activation,
    batch_norm_swish,
    bias_norm,
    cmvn,
    group_norm,
    layer_norm,
    layer_norm_activation,
    rms_norm,
    rms_norm_activation,
)
from .functionals.pooling import avg_pool1d
from .functionals.recurrent import (
    lstm_gemm_layer,
    lstm_layer,
    lstm_slot_step,
    rnn_gemm_layer,
    rnn_layer,
    rnn_slot_step,
)
from .functionals.softmax import log_softmax, masked_softmax, softmax
from .functionals.topk import topk
from .layers import (
    LSTM,
    RNN,
    AddLayerNorm,
    AddRMSNorm,
    AvgPool1d,
    BatchNorm1d,
    BiasNorm,
    Conv1d,
    Conv1dActivation,
    Conv2d as Conv2dModule,
    Conv2dActivation,
    DepthwiseConv1d,
    Fbank,
    Gelu as GeluModule,
    GlobalCMVN,
    Glu as GluModule,
    GroupNorm,
    LayerNorm,
    Linear,
    MaskedSoftmax as MaskedSoftmaxModule,
    Mfcc,
    PointwiseConv1d,
    Relu as ReluModule,
    RMSNorm,
    Sigmoid as SigmoidModule,
    Softmax as SoftmaxModule,
    Tanh as TanhModule,
    TopK as TopKModule,
)
from .tune import autotune, disable_autotune, enable_autotune

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
    AttentionCacheManager,
    BlockPool,
    CacheConfig,
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
    # Model loader — lazy so ``import oasr`` doesn't eagerly pull in every
    # architecture package (they self-register on first use).
    if name == "from_pretrained":
        from oasr.models import from_pretrained as _fp

        globals()["from_pretrained"] = _fp
        return _fp

    _C = globals().get("_C")
    if _C is None:
        try:
            _C = _importlib.import_module("oasr._C")
            globals()["_C"] = _C
        except ImportError:
            raise AttributeError(f"module 'oasr' has no attribute {name!r}") from None
    if name == "_C":
        return _C
    try:
        attr = getattr(_C, name)
    except AttributeError:
        raise AttributeError(f"module 'oasr' has no attribute {name!r}") from None
    globals()[name] = attr
    return attr


__all__ = [
    "__version__",
    # Model loading
    "from_pretrained",
    # Activation constants
    "ACTIVATION_RELU",
    "ACTIVATION_GELU",
    "ACTIVATION_GELU_ERF",
    "ACTIVATION_SWISH",
    "get_activation_type_id",
    # Functional API
    "functionals",
    "gelu",
    "glu",
    "relu",
    "sigmoid",
    "swish",
    "swoosh_l",
    "swoosh_r",
    "tanh",
    "layer_norm",
    "rms_norm",
    "bias_norm",
    "batch_norm_1d",
    "group_norm",
    "add_layer_norm",
    "add_layer_norm_residual",
    "add_rms_norm",
    "add_rms_norm_residual",
    "layer_norm_activation",
    "rms_norm_activation",
    "batch_norm_activation",
    "batch_norm_swish",
    "cmvn",
    "avg_pool1d",
    "lstm_layer",
    "lstm_slot_step",
    "rnn_layer",
    "rnn_slot_step",
    "lstm_gemm_layer",
    "rnn_gemm_layer",
    "depthwise_conv1d",
    "conv1d",
    "conv1d_activation",
    "conv2d",
    "depthwise_conv1d_silu",
    "causal_conv1d",
    "conv2d_activation",
    "gemm",
    "bmm",
    "group_gemm",
    "gemm_activation",
    "gemm_log_softmax",
    "log_softmax",
    "masked_softmax",
    "softmax",
    "topk",
    "rfft",
    "rfft_power",
    # Recurrent modules
    "LSTM",
    "RNN",
    # Feature extraction (batched)
    "fbank_batch",
    "mfcc_batch",
    "extract_features_batch",
    "FeatureConfig",
    "BatchedStreamingFeatureExtractor",
    # Low-level CUDA feature ops
    "stft_frame",
    "fbank_preprocess",
    "mel_log",
    "dct_lifter",
    "whisper_logmel",
    "lfr_gather",
    # nn.Module wrappers
    "layers",
    "DepthwiseConv1d",
    "Conv1d",
    "Conv1dActivation",
    "PointwiseConv1d",
    "Conv2dModule",
    "Conv2dActivation",
    "GluModule",
    "GeluModule",
    "ReluModule",
    "SigmoidModule",
    "TanhModule",
    "Fbank",
    "Mfcc",
    "Linear",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BiasNorm",
    "BatchNorm1d",
    "AddLayerNorm",
    "AddRMSNorm",
    "AvgPool1d",
    "GlobalCMVN",
    "SoftmaxModule",
    "MaskedSoftmaxModule",
    "TopKModule",
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
    "StreamState",
    "StreamHandle",
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

# Opt-in GEMM-shape capture: if OASR_CAPTURE_GEMM=<path> is set, wrap the
# functional GEMM entries to record real (op, M, N, K) shapes and dump on exit.
# Cheap no-op otherwise.
from .tune.capture import maybe_autostart as _maybe_autostart_capture

_maybe_autostart_capture()
