# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Whisper model configuration (mirrors the HF ``WhisperConfig`` fields OASR uses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from ..base import BaseModelConfig


@dataclass
class WhisperModelConfig(BaseModelConfig):
    """Encoder-decoder Whisper hyperparameters + generation control ids.

    The generation fields (``decoder_start_token_id`` / ``forced_decoder_ids``
    / suppress lists) travel here because they are checkpoint properties (from
    ``config.json`` / ``generation_config.json``), not engine choices — the
    ``aed`` decode strategy reads them to build the SOT prompt and the logit
    suppression masks.
    """

    model_type: str = "whisper"
    # vocab_size inherited from BaseModelConfig (51865 for multilingual tiny).
    d_model: int = 384
    encoder_layers: int = 4
    decoder_layers: int = 4
    encoder_attention_heads: int = 6
    decoder_attention_heads: int = 6
    encoder_ffn_dim: int = 1536
    decoder_ffn_dim: int = 1536
    num_mel_bins: int = 80
    max_source_positions: int = 1500
    max_target_positions: int = 448
    activation_function: str = "gelu"

    # -- generation control (checkpoint-derived) ---------------------------
    decoder_start_token_id: int = 50258  # <|startoftranscript|>
    eos_token_id: int = 50257  # <|endoftext|>
    # [(position, token_id)] forced after SOT — language / task / notimestamps.
    forced_decoder_ids: List[Tuple[int, int]] = field(default_factory=list)
    # Token ids whose logits are set to -inf at every generation step.
    suppress_tokens: List[int] = field(default_factory=list)
    # Token ids suppressed only at the first *generated* step.
    begin_suppress_tokens: List[int] = field(default_factory=list)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.encoder_attention_heads

    def sot_sequence(self) -> List[int]:
        """The decoder prompt: ``<|startoftranscript|>`` + forced ids in
        position order (language, task, ``<|notimestamps|>``)."""
        prompt = [self.decoder_start_token_id]
        for _pos, tok in sorted(self.forced_decoder_ids, key=lambda pt: pt[0]):
            prompt.append(int(tok))
        return prompt
