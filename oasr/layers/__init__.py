"""The narrow waist every model implementation goes through.

A model built from these modules picks up the OASR CUDA kernels, CUDA-graph
capture and (later) quantization automatically, and still loads its upstream
checkpoint 1:1 — the layers are ``nn.Module``-compatible in parameter layout
and fall back to torch wherever a kernel cannot run.  See
``docs/architecture.md`` and :mod:`oasr.layers._backend`;
``tests/test_layer_waist.py`` is what keeps architectures inside it.
"""

from ._backend import (
    KERNEL_GAPS,
    format_gap_report,
    gap_hits,
    layers_backend,
    layers_backend_override,
    policy_hits,
    reset_backend_stats,
    set_layers_backend,
)
from .activation import Gelu, Relu, Sigmoid, Tanh
from .attention import Attention, RelPositionMultiHeadedAttention
from .conv import (
    Conv1d,
    Conv1dActivation,
    Conv2d,
    Conv2dActivation,
    DepthwiseConv1d,
    Glu,
    PointwiseConv1d,
)
from .ctc import CtcProjection
from .embedding import Embedding, VocabParallelEmbedding
from .feature import Fbank, Mfcc
from .linear import ColumnParallelLinear, Linear, LinearActivation, RowParallelLinear
from .mlp import FeedForward, GatedMLP
from .norm import (
    ESPNET_EPS,
    QWEN2_RMS_EPS,
    TORCH_EPS,
    AddLayerNorm,
    BatchNorm1d,
    BiasNorm,
    GlobalCMVN,
    GroupNorm,
    LayerNorm,
    RMSNorm,
)
from .pooling import AvgPool1d
from .rotary_embedding import NeoxRotaryEmbedding, RotaryEmbedding, apply_rotary_pos_emb
from .softmax import Softmax
from .topk import TopK

__all__ = [
    # Backend selection + kernel-coverage reporting
    "layers_backend",
    "layers_backend_override",
    "set_layers_backend",
    "KERNEL_GAPS",
    "format_gap_report",
    "gap_hits",
    "policy_hits",
    "reset_backend_stats",
    # Attention
    "Attention",
    "RelPositionMultiHeadedAttention",
    # Activation
    "Gelu",
    "Relu",
    "Sigmoid",
    "Tanh",
    # Convolution
    "Conv1d",
    "Conv1dActivation",
    "DepthwiseConv1d",
    "PointwiseConv1d",
    "Conv2d",
    "Conv2dActivation",
    "Glu",
    # CTC projection
    "CtcProjection",
    # Embedding
    "Embedding",
    "VocabParallelEmbedding",
    # Feature extraction
    "Fbank",
    "Mfcc",
    # Linear
    "Linear",
    "LinearActivation",
    "ColumnParallelLinear",
    "RowParallelLinear",
    # Feed-forward
    "FeedForward",
    "GatedMLP",
    # Normalization
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BiasNorm",
    "BatchNorm1d",
    "AddLayerNorm",
    "GlobalCMVN",
    "TORCH_EPS",
    "ESPNET_EPS",
    "QWEN2_RMS_EPS",
    # Pooling
    "AvgPool1d",
    # Rotary embedding
    "RotaryEmbedding",
    "NeoxRotaryEmbedding",
    "apply_rotary_pos_emb",
    # Softmax
    "Softmax",
    # TopK
    "TopK",
]
