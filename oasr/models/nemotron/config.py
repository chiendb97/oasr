# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Nemotron ASR (FastConformer + RNN-T) configs.

Field names mirror ``transformers``'s ``Nemotron3_5AsrConfig`` /
``NemotronAsrStreamingEncoderConfig`` so a converter is a rename-free copy of
``config.json``, and so the two can be diffed when upstream moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

from ..base import BaseModelConfig


@dataclass
class NemotronEncoderConfig:
    """FastConformer encoder with causal depthwise-separable 8x subsampling.

    The three fields that are *not* ordinary Conformer hyperparameters, and what
    each one costs to get wrong:

    ``sliding_window``
        Attention **left** context is ``sliding_window - 1`` frames.  Not a
        cache size — it is part of the trained mask, so it applies offline too.
    ``default_num_lookahead_tokens``
        Attention **right** context, in subsampled frames.  Together with the
        left context it defines the ``chunked_limited`` mask: the chunk is
        ``right + 1`` frames wide and a query may see its own chunk plus
        ``(sliding_window - 1) // (right + 1)`` earlier ones.  Larger values are
        more accurate and less streamable; the model was trained on the set in
        :attr:`supported_num_lookahead_tokens` and only those are meaningful.
    ``scale_input``
        Whether the subsampling output is multiplied by ``sqrt(hidden_size)``.
        ``False`` on the released 0.6B checkpoint — and it is a factor of 32, so
        a wrong default is not a subtle accuracy drift.
    """

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    intermediate_size: int = 4096
    hidden_act: str = "silu"
    #: Bias on the attention and feed-forward projections (``False`` upstream).
    attention_bias: bool = False
    #: Bias on the convolution module's three convolutions (``False`` upstream).
    convolution_bias: bool = False
    conv_kernel_size: int = 9
    num_mel_bins: int = 128
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    max_position_embeddings: int = 5000
    scale_input: bool = False
    sliding_window: int = 57
    default_num_lookahead_tokens: int = 3
    supported_num_lookahead_tokens: Tuple[int, ...] = (3, 0, 6, 13)

    @property
    def num_subsampling_layers(self) -> int:
        """Number of stride-2 conv stages (``log2(subsampling_factor)``)."""
        return int(math.log2(self.subsampling_factor))

    @property
    def subsampling_out_hidden_size(self) -> int:
        """Flattened width out of the subsampling stack: ``channels * freq_bins``.

        The frequency axis is padded ``(kernel - 1, stride - 1)`` and convolved
        with ``padding=0``, so 128 mels become 65 → 33 → 17 bins and the
        released checkpoint's projection input is ``256 * 17 = 4352``.
        """
        total_pad = (self.subsampling_conv_kernel_size - 1) + (self.subsampling_conv_stride - 1)
        bins = self.num_mel_bins
        for _ in range(self.num_subsampling_layers):
            bins = (
                bins + total_pad - self.subsampling_conv_kernel_size
            ) // self.subsampling_conv_stride + 1
        return self.subsampling_conv_channels * bins

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def output_size(self) -> int:
        return self.hidden_size


@dataclass
class NemotronModelConfig(BaseModelConfig):
    """Nemotron ASR RNN-T: encoder + language-prompt projector + LSTM predictor.

    The joint network is *additive with no per-branch projections of its own*:
    ``head(relu(encoder_projected + decoder_projected))``.  The two projections
    live upstream as top-level ``encoder_projector`` and
    ``decoder.decoder_projector``; OASR places them on the joiner as
    ``encoder_proj`` / ``decoder_proj`` (the icefall layout its decode strategy
    already drives) and remaps the two keys on load.  Same arithmetic, one
    strategy.
    """

    model_type: str = "nemotron"
    vocab_size: int = 13088
    #: Blank id — the *last* vocabulary entry, not 0 (that is ``<unk>``).
    blank_token_id: int = 13087
    decoder_hidden_size: int = 640
    num_decoder_layers: int = 2
    #: Joint-network activation (``relu`` upstream), applied to the sum.
    hidden_act: str = "relu"
    max_symbols_per_step: int = 10
    #: Width of the one-hot language prompt spliced onto the encoder output.
    num_prompts: int = 128
    prompt_intermediate_size: int = 2048
    #: Prompt slot used when the caller names no language; 101 == ``"auto"`` on
    #: the released checkpoint (the model does its own language ID).
    default_prompt_id: int = 101
    encoder: NemotronEncoderConfig = field(default_factory=NemotronEncoderConfig)

    @property
    def blank_id(self) -> int:
        """Alias matching the other transducer configs."""
        return self.blank_token_id
