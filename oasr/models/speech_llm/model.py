# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Speech-LLM ASR model (Qwen2-Audio-style): audio tower + projector + LLM.

Module names mirror the HF ``Qwen2AudioForConditionalGeneration`` layout
(``audio_tower`` / ``multi_modal_projector`` / ``language_model``) so
``load_weights`` is a near-1:1 copy and the native format round-trips on our
own canonical keys.  Offline-only (``streaming_kind == "none"``): the engine
drives it through the ``llm`` decode strategy (``consumes="hidden"``,
incremental) — :meth:`encode_offline` runs tower + projector into per-audio
LLM-space embeddings, the strategy splices them into the ChatML prompt and
generates via the LM's ``prefill`` / ``step`` / ``select`` surface under the
engine's per-tick :class:`~oasr.engine.generation.StepBudget`.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Tuple

import torch
from torch import nn

from oasr.layers import Linear

from ..base import BaseAsrModel, LoadReport
from .audio_tower import Qwen2AudioTower
from .config import SpeechLlmModelConfig
from .llm import Qwen2Lm

logger = logging.getLogger(__name__)


class _Projector(nn.Module):
    """Audio → LLM space projection (HF nests the Linear as ``.linear``)."""

    def __init__(self, config: SpeechLlmModelConfig) -> None:
        super().__init__()
        self.linear = Linear(config.audio_d_model, config.text_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SpeechLlmModel(BaseAsrModel):
    """Qwen2-Audio-style speech-LLM for OASR (offline LLM decoding)."""

    def __init__(self, config: SpeechLlmModelConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_tower = Qwen2AudioTower(config)
        self.multi_modal_projector = _Projector(config)
        self.language_model = Qwen2Lm(config)

    @classmethod
    def from_config(cls, config: SpeechLlmModelConfig, **aux: Any) -> "SpeechLlmModel":
        del aux
        return cls(config)

    # -- engine-facing metadata ---------------------------------------------
    @property
    def encoder(self) -> Qwen2AudioTower:
        return self.audio_tower

    @property
    def decoder(self) -> Qwen2Lm:
        return self.language_model

    @property
    def default_decode_type(self) -> str:
        return "llm"

    @property
    def capabilities(self) -> frozenset:
        return frozenset({"llm"})

    # -- weights --------------------------------------------------------------
    #: Accepted checkpoint layouts, normalized to our canonical keys:
    #: * HF 4.x publishes: ``audio_tower.*`` / ``multi_modal_projector.*`` /
    #:   ``language_model.model.*`` / ``language_model.lm_head.weight``
    #: * HF 5.x resaves: same, except the LM trunk gains an extra nesting
    #:   level (``language_model.model.model.*``) — the leading
    #:   ``language_model.model…`` chain collapses to ``language_model.``
    #: * ``model.``-prefixed variants (5.x internal layout) and a top-level
    #:   ``lm_head.weight`` are accepted too
    #: * native format: our canonical keys verbatim.
    @staticmethod
    def _canonical_key(key: str) -> str:
        if key.startswith("model."):
            key = key[len("model.") :]
        if key == "lm_head.weight":
            return "language_model.lm_head.weight"
        while key.startswith("language_model.model."):
            key = "language_model." + key[len("language_model.model.") :]
        return key

    @property
    def decoder_cache_spec(self):
        """Per-layer KV geometry of the language model, for admission budgeting.

        Uses ``text_num_key_value_heads`` rather than the attention-head count:
        with GQA the KV cache is sized by the *key/value* heads, which is the
        whole point of GQA.  On the shipped Qwen2-Audio-7B the two are equal.
        """
        from oasr.models.base import CacheSpec

        cfg = self.config
        return CacheSpec(
            num_layers=int(cfg.text_num_hidden_layers),
            n_kv_head=int(cfg.text_num_key_value_heads),
            head_dim=int(cfg.text_head_dim),
            hidden_dim=int(cfg.text_hidden_size),
        )

    def load_weights(
        self, state_dict: Mapping[str, torch.Tensor], *, strict: bool = False
    ) -> LoadReport:
        sd = {}
        dropped = []
        for k, v in state_dict.items():
            key = self._canonical_key(k)
            if key.startswith(("audio_tower.", "multi_modal_projector.", "language_model.")):
                sd[key] = v
            else:
                dropped.append(k)
        if (
            self.config.text_tie_word_embeddings
            and "language_model.lm_head.weight" not in sd
            and "language_model.embed_tokens.weight" in sd
        ):
            sd["language_model.lm_head.weight"] = sd["language_model.embed_tokens.weight"]
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        if unexpected:
            logger.warning("Unexpected keys in speech-LLM checkpoint: %s", unexpected[:8])
        if missing:
            logger.warning("Speech-LLM model keys not filled: %s", missing[:8])
        return LoadReport.build(sd, missing, unexpected, dropped)

    # -- engine-facing compute ------------------------------------------------
    def encode_offline(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Log-mel ``(B, 3000, n_mels)`` + real frame counts → LLM-space audio
        embeddings ``(B, 750, hidden_size)`` + valid embedding counts."""
        hidden, masks = self.audio_tower(features, lengths)
        audio_embeds = self.multi_modal_projector(hidden)
        return audio_embeds, self._lengths_from_mask(masks)
