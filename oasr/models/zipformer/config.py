# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Zipformer model configuration.

Defaults match the icefall LibriSpeech non-streaming "M" recipe
(``egs/librispeech/ASR/zipformer``).  icefall checkpoints do not store the
architecture config (it comes from CLI args), so these defaults are the source
of truth and are overridable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from ..base import BaseModelConfig, CacheSpec
from .encoder import _to_tuple


@dataclass
class ZipformerEncoderConfig:
    """Configuration for the Zipformer2 encoder (icefall LibriSpeech "M" defaults).

    Tuple-valued fields are per-stack; a length-1 tuple broadcasts to all stacks
    (matching icefall's CLI behavior).  ``causal=True`` enables the chunk-wise
    streaming path.
    """

    feature_dim: int = 80
    output_downsampling_factor: int = 2
    downsampling_factor: Tuple[int, ...] = (1, 2, 4, 8, 4, 2)
    encoder_dim: Tuple[int, ...] = (192, 256, 384, 512, 384, 256)
    num_encoder_layers: Tuple[int, ...] = (2, 2, 3, 4, 3, 2)
    query_head_dim: Tuple[int, ...] = (32,)
    pos_head_dim: Tuple[int, ...] = (4,)
    value_head_dim: Tuple[int, ...] = (12,)
    num_heads: Tuple[int, ...] = (4, 4, 4, 8, 4, 4)
    feedforward_dim: Tuple[int, ...] = (512, 768, 1024, 1536, 1024, 768)
    cnn_module_kernel: Tuple[int, ...] = (31, 31, 15, 15, 15, 31)
    pos_dim: int = 48
    causal: bool = False
    chunk_size: Tuple[int, ...] = (-1,)
    left_context_frames: Tuple[int, ...] = (-1,)

    @property
    def num_stacks(self) -> int:
        return len(self.downsampling_factor)


@dataclass
class ZipformerModelConfig(BaseModelConfig):
    """Top-level Zipformer model config (encoder + CTC head)."""

    model_type: str = "zipformer"
    vocab_size: Optional[int] = 500  # icefall LibriSpeech BPE
    encoder: ZipformerEncoderConfig = field(default_factory=ZipformerEncoderConfig)

    @property
    def cache_spec(self) -> CacheSpec:
        """Engine cache descriptor.

        Zipformer streams with its own per-layer cache (not the engine's paged-KV
        / slot-CNN model), so this only sizes the paged cache the engine
        allocates at init; it is unused for Zipformer.  ``conv_kernel_size == 1``
        means no CNN cache.
        """
        enc = self.encoder
        n = enc.num_stacks
        num_heads = _to_tuple(enc.num_heads, n)
        value_head_dim = _to_tuple(enc.value_head_dim, n)
        query_head_dim = _to_tuple(enc.query_head_dim, n)
        num_layers = sum(_to_tuple(enc.num_encoder_layers, n))
        return CacheSpec(
            num_layers=num_layers,
            n_kv_head=max(num_heads),
            head_dim=max(max(value_head_dim), max(query_head_dim)),
            hidden_dim=max(enc.encoder_dim),
            conv_kernel_size=1,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ZipformerModelConfig":
        """Build from a dict (e.g. icefall args). Unknown keys are ignored."""
        encoder_dict = d.get("encoder", d)
        fields = {f for f in ZipformerEncoderConfig.__dataclass_fields__}
        encoder = ZipformerEncoderConfig(
            **{k: tuple(v) if isinstance(v, list) else v
               for k, v in encoder_dict.items() if k in fields}
        )
        return cls(encoder=encoder, vocab_size=d.get("vocab_size", 500))
