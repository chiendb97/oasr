# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Speech-LLM model configuration (Qwen2-Audio-style audio tower + Qwen2 LM).

Field names mirror the HF ``Qwen2AudioConfig`` split: ``audio_*`` fields come
from ``config.json: audio_config`` (a Whisper-large-v3-geometry encoder) and
``text_*`` fields from ``config.json: text_config`` (a ``Qwen2Config``, whose
**defaults fill any omitted key** — the published Qwen2-Audio-7B checkpoint
relies on them for ``hidden_size`` / ``num_hidden_layers`` / head counts).

The prompt-template fields travel here because they are checkpoint properties
(the processor's ChatML template with the ``Audio 1: <|audio_bos|>…`` audio
slot), not engine choices; ``EngineConfig.llm_prompt`` overrides only the user
text inside the template.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Mapping

from ..base import _UNSET, BaseModelConfig

logger = logging.getLogger(__name__)

# The processor-side ChatML template around the audio slot (verbatim from
# Qwen2AudioProcessor.default_chat_template with one user turn holding one
# audio + one text content and add_generation_prompt=True).  ``{prompt}`` in
# the suffix is replaced with the user prompt text.
QWEN2_AUDIO_PROMPT_PREFIX = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nAudio 1: <|audio_bos|>"
)
QWEN2_AUDIO_PROMPT_SUFFIX = "<|audio_eos|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

#: The user turn the template above carries by default: an instruction to transcribe.
#:
#: The template is the *Instruct* checkpoint's ChatML, and the Instruct model treats
#: the user turn as a question to answer.  The previous default, the base model's ASR
#: eval prompt (:data:`QWEN2_AUDIO_BASE_ASR_PROMPT`), got answers such as "The speech
#: is in English, and it is a male singing in a neutral mood": 94.9 % WER on
#: LibriSpeech test-clean (bf16, VAD on), against 3.19 % with this prompt.
QWEN2_AUDIO_DEFAULT_USER_PROMPT = (
    "Transcribe the audio into text. Output only the transcription, with no extra words."
)

#: Qwen2-Audio's own ASR prompt, from its speech-recognition evals.  It was written
#: for the *base* model and its plain-text template, not for a chat turn.  It is the
#: default every native bundle exported before 2026-11-09 stored, which
#: :meth:`SpeechLlmModelConfig.from_dict` reads as the stale default it is.
QWEN2_AUDIO_BASE_ASR_PROMPT = "Detect the language and recognize the speech."


def _user_prompt_from_dict(d: Mapping[str, Any]) -> Any:
    """``default_user_prompt`` of a native bundle, with the stale default migrated.

    The field is written by ``asdict`` like every other, so a bundle exported before
    the default changed stores the base model's prompt.  That is a copy of a code
    default, not a property of the weights, and it is the defect itself: served as a
    ChatML user turn, it gets a description instead of a transcript.  Only that
    exact string under the ChatML suffix is replaced.  Any other stored prompt is
    kept, and a deployment that wants the old one sets ``llm_prompt``.
    """
    prompt = d.get("default_user_prompt", _UNSET)
    suffix = d.get("prompt_suffix", QWEN2_AUDIO_PROMPT_SUFFIX)
    if prompt == QWEN2_AUDIO_BASE_ASR_PROMPT and suffix == QWEN2_AUDIO_PROMPT_SUFFIX:
        logger.warning(
            "this speech-LLM bundle stores the base model's ASR prompt (%r) as its "
            "default; the Instruct template answers it with a description, so the "
            "transcription prompt is used instead. Re-export the bundle to store it, "
            "or set llm_prompt to keep the old one.",
            prompt,
        )
        return QWEN2_AUDIO_DEFAULT_USER_PROMPT
    return prompt


@dataclass
class SpeechLlmModelConfig(BaseModelConfig):
    """Qwen2-Audio-style speech-LLM hyperparameters + generation control ids."""

    model_type: str = "speech_llm"
    # vocab_size inherited from BaseModelConfig (156032 for Qwen2-Audio).

    # -- audio tower (HF ``audio_config``; Whisper-encoder geometry) --------
    audio_num_mel_bins: int = 128
    audio_d_model: int = 1280
    audio_encoder_layers: int = 32
    audio_encoder_attention_heads: int = 20
    audio_encoder_ffn_dim: int = 5120
    audio_max_source_positions: int = 1500

    # -- text LM (HF ``text_config``; Qwen2 architecture) -------------------
    text_hidden_size: int = 4096
    text_num_hidden_layers: int = 32
    text_num_attention_heads: int = 32
    text_num_key_value_heads: int = 32
    text_intermediate_size: int = 22016
    text_rope_theta: float = 10000.0
    text_rms_norm_eps: float = 1e-6
    text_max_position_embeddings: int = 32768
    text_tie_word_embeddings: bool = False

    # -- generation control (checkpoint-derived) ----------------------------
    #: Placeholder id the processor expands per audio frame (<|AUDIO|>).
    audio_token_id: int = 151646
    #: Stop set — generation_config.json may pin several (<|endoftext|>, <|im_end|>).
    eos_token_ids: List[int] = field(default_factory=lambda: [151643, 151645])
    pad_token_id: int = 151643
    #: ChatML template around the audio embeddings; encoded with the
    #: checkpoint's tokenizer at strategy build time.
    prompt_prefix: str = QWEN2_AUDIO_PROMPT_PREFIX
    prompt_suffix: str = QWEN2_AUDIO_PROMPT_SUFFIX
    default_user_prompt: str = QWEN2_AUDIO_DEFAULT_USER_PROMPT

    _from_dict_overrides: ClassVar[Mapping[str, Any]] = {
        "default_user_prompt": _user_prompt_from_dict,
    }

    @property
    def audio_head_dim(self) -> int:
        return self.audio_d_model // self.audio_encoder_attention_heads

    @property
    def text_head_dim(self) -> int:
        return self.text_hidden_size // self.text_num_attention_heads
