# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive decoder contracts for non-CTC ASR families.

CTC models project encoder hidden states with a :class:`~oasr.models.base.BaseHead`
and decode non-autoregressively.  Transducer (RNNT), AED (attention
encoder-decoder), and LLM-based families instead run an **autoregressive**
decoder token-by-token; this package defines the contract the engine's
``DecodeStrategy`` drives for those families.

Besides the interfaces, this package ships the WeNet/ESPnet-compatible
:class:`TransformerDecoder` / :class:`BiTransformerDecoder` (U2++ attention
rescoring + AED generation).  See ``oasr/engine/decode/`` for the matching
decode strategies.
"""

from .base import BaseDecoder, DecoderState, Joiner, PredictionNetwork, TransducerPredictor
from .transformer_decoder import (
    BiTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderConfig,
    add_sos_eos,
    reverse_pad_list,
)

__all__ = [
    "BaseDecoder",
    "BiTransformerDecoder",
    "DecoderState",
    "Joiner",
    "PredictionNetwork",
    "TransducerPredictor",
    "TransformerDecoder",
    "TransformerDecoderConfig",
    "add_sos_eos",
    "reverse_pad_list",
]
