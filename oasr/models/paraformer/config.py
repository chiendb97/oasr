# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Paraformer model configuration (FunASR SANM + CIF + NAR decoder)."""

from __future__ import annotations

from dataclasses import dataclass

from oasr.models.base import BaseModelConfig


@dataclass
class ParaformerModelConfig(BaseModelConfig):
    """Configuration for the non-autoregressive Paraformer model.

    Field values default to the ``paraformer-zh`` (paraformer-large) recipe.
    ``input_size`` is the **post-LFR** feature dimension (80 mels × lfr_m 7 =
    560); the LFR stacking itself is a feature-extraction concern carried by
    the converter-emitted :class:`~oasr.features.FeatureSpec`.
    """

    model_type: str = "paraformer"
    vocab_size: int = 8404
    input_size: int = 560

    # SANM encoder
    encoder_output_size: int = 512
    encoder_attention_heads: int = 4
    encoder_linear_units: int = 2048
    encoder_num_blocks: int = 50
    encoder_kernel_size: int = 11
    encoder_sanm_shift: int = 0

    # SANM NAR decoder
    decoder_attention_heads: int = 4
    decoder_linear_units: int = 2048
    decoder_num_blocks: int = 16
    decoder_att_layer_num: int = 16
    decoder_kernel_size: int = 11
    decoder_sanm_shift: int = 0

    # CIF predictor (CifPredictorV2)
    predictor_idim: int = 512
    predictor_threshold: float = 1.0
    predictor_l_order: int = 1
    predictor_r_order: int = 1
    predictor_tail_threshold: float = 0.45

    # Special token ids (FunASR convention: <blank>=0, <s>=1, </s>=2)
    blank_id: int = 0
    sos_id: int = 1
    eos_id: int = 2
