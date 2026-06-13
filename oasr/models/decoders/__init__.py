# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive decoder contracts for non-CTC ASR families.

CTC models project encoder hidden states with a :class:`~oasr.models.base.BaseHead`
and decode non-autoregressively.  Transducer (RNNT), AED (attention
encoder-decoder), and LLM-based families instead run an **autoregressive**
decoder token-by-token; this package defines the contract the engine's
``DecodeStrategy`` drives for those families.

Only the interfaces live here today — concrete decoders ship with their model
families.  See ``oasr/engine/decode/`` for the matching decode strategies.
"""

from .base import BaseDecoder, DecoderState, Joiner, PredictionNetwork

__all__ = ["BaseDecoder", "DecoderState", "Joiner", "PredictionNetwork"]
