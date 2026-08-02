# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Attention layers.

:class:`Attention` is the generic compute core every architecture shares;
:class:`RelPositionMultiHeadedAttention` is the Conformer-specific
Transformer-XL rel-pos variant, which owns its own fused QKV projection
because the rel-pos bias is entangled with it.
"""

from .attention import RelPositionMultiHeadedAttention
from .core import Attention, kv_length_mask, merge_heads, split_heads

__all__ = [
    "Attention",
    "RelPositionMultiHeadedAttention",
    "kv_length_mask",
    "merge_heads",
    "split_heads",
]
