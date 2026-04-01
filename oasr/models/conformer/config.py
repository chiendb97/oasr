# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Conformer model configuration.

Follows the config pattern used in vLLM model_executor (e.g. dedicated
config dataclass) while matching Conformer hyperparameters from WeNet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConformerEncoderConfig:
    """Configuration for the Conformer encoder (WeNet-style defaults)."""

    # Input / output
    input_size: int = 80
    output_size: int = 256
    num_blocks: int = 12

    # Attention
    attention_heads: int = 4

    # Feed-forward
    linear_units: int = 2048

    # Conformer-specific
    positionwise_conv_kernel_size: int = 1
    macaron_style: bool = True
    use_cnn_module: bool = True
    cnn_module_kernel: int = 15
    causal: bool = False
    cnn_module_norm: str = "batch_norm"
    conv_bias: bool = True
    conv_norm_eps: float = 1e-5
    conv_inner_factor: int = 2

    # Normalization
    normalize_before: bool = True
    final_norm: bool = True
    layer_norm_type: str = "layer_norm"
    norm_eps: float = 1e-5

    # Activation (Conformer typically uses swish)
    activation_type: str = "swish"

    # Input layer: "conv2d", "linear", "conv2d6", "conv2d8"
    input_layer: str = "conv2d"
    # Whether to apply LayerNorm after the embed linear projection
    embed_layer_norm: bool = True
    # Positional encoding: "rel_pos" for Conformer
    pos_enc_layer_type: str = "rel_pos"

    # Optional (MQA/GQA; usually not used in Conformer)
    n_kv_head: Optional[int] = None
    head_dim: Optional[int] = None


@dataclass
class ConformerModelConfig:
    """Top-level Conformer model config (encoder-only or encoder + head)."""

    encoder: ConformerEncoderConfig = field(
        default_factory=ConformerEncoderConfig)
    # For ASR: vocab_size if adding a CTC/decoder head later
    vocab_size: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> ConformerModelConfig:
        """Build from a dict (e.g. HuggingFace config)."""
        encoder_dict = d.get("encoder", d)
        encoder = ConformerEncoderConfig(
            **{k: v for k, v in encoder_dict.items() if hasattr(ConformerEncoderConfig, k)})
        return cls(encoder=encoder, vocab_size=d.get("vocab_size"))
