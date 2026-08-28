# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace Whisper checkpoint converter.

Reads an HF Whisper snapshot (``config.json`` + ``model.safetensors`` /
``pytorch_model.bin`` + ``tokenizer.json`` + optional
``generation_config.json``) and emits the complete
:class:`~oasr.checkpoints.ConvertedCheckpoint` bundle: model config, weights,
a ``whisper`` TokenizerSpec, a ``whisper_logmel`` FeatureSpec
(``audio_scale = 1.0`` — Whisper consumes [-1, 1] floats, unlike the Kaldi
int16-scale frontends), and AED decoding defaults.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Mapping, Optional, Tuple

import torch

from ..converter import BaseCheckpointConverter
from ..registry import DETECT_KEYED_VALUE
from .config import WhisperModelConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


#: A Whisper language tag is ``<|xx|>`` / ``<|xxx|>`` — two or three ASCII
#: letters.  Every control token that is *not* a language (``<|translate|>``,
#: ``<|notimestamps|>``, ``<|startofprev|>``, …) is longer, and the timestamp
#: tokens (``<|0.00|>``) are not letters, so the shape alone separates them.
_LANGUAGE_TOKEN = re.compile(r"^<\|([a-z]{2,3})\|>$")

#: The two task tokens, named rather than numbered: their ids move between
#: Whisper releases (large-v3 added a language and shifted everything after it).
_TASK_TOKENS = {"<|transcribe|>": "transcribe", "<|translate|>": "translate"}


#: Spellings of the no-speech control token across Whisper vocabularies.  The
#: original release used ``<|nocaptions|>``; everything since uses
#: ``<|nospeech|>``, and both appear in snapshots still in circulation.
_NO_SPEECH_TOKENS: Tuple[str, ...] = ("<|nospeech|>", "<|nocaptions|>")


def _no_speech_token_id(tok_json: Path) -> Optional[int]:
    """The ``<|nospeech|>`` id from an HF ``tokenizer.json``, or ``None``.

    Read by name rather than derived as ``no_timestamps_token_id - 1``.  The
    arithmetic happens to hold for the multilingual vocabularies and is exactly
    the kind of assumption that produces a confident wrong answer on the one
    snapshot where it does not — here, a probability read off an unrelated
    token's logit, which would look like a plausible number forever.
    """
    try:
        added = _read_json(tok_json).get("added_tokens") or []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("whisper: could not read %s for the no-speech token: %s", tok_json, exc)
        return None
    for entry in added:
        content, tid = entry.get("content"), entry.get("id")
        if isinstance(content, str) and isinstance(tid, int) and content in _NO_SPEECH_TOKENS:
            return int(tid)
    return None


def _control_token_tables(tok_json: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """``(task_token_ids, language_token_ids)`` from an HF ``tokenizer.json``.

    Read here, at conversion time, so the engine never needs the tokenizer to
    build a decoder prompt — and so the tables round-trip through the native
    format like every other checkpoint-derived fact.  A snapshot without them
    (an English-only Whisper has no language tokens) yields empty tables, and
    the per-request options then fail with a message saying so.
    """
    tasks: Dict[str, int] = {}
    languages: Dict[str, int] = {}
    try:
        added = _read_json(tok_json).get("added_tokens") or []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("whisper: could not read %s for control tokens: %s", tok_json, exc)
        return tasks, languages
    for entry in added:
        content, tid = entry.get("content"), entry.get("id")
        if not isinstance(content, str) or not isinstance(tid, int):
            continue
        if (task := _TASK_TOKENS.get(content)) is not None:
            tasks[task] = tid
        elif (m := _LANGUAGE_TOKEN.match(content)) is not None:
            languages[m.group(1)] = tid
    return tasks, languages


class HFWhisperConverter(BaseCheckpointConverter):
    """Checkpoint converter for HF ``model_type: "whisper"`` snapshots."""

    #: ``proj_out.weight`` is tied to ``decoder.embed_tokens.weight`` — a
    #: silent, expected drop when the snapshot materializes it.
    expected_unused_prefixes: Tuple[str, ...] = ("proj_out.",)
    capability_drop_hints: Dict[str, str] = {}

    architecture: ClassVar[str] = "whisper"
    source_format: ClassVar[str] = "huggingface"
    default_checkpoint_name: ClassVar[str] = "model.safetensors"
    default_decode_type: ClassVar[str] = "aed"
    #: the architecture is named in ``config.json`` (``model_type == "whisper"``), so this claim outranks a weaker one
    #: (see :func:`oasr.models.registry.resolve_architecture`).
    detect_specificity: ClassVar[int] = DETECT_KEYED_VALUE

    def detect(self, ckpt_dir: Path) -> bool:
        cfg_path = Path(ckpt_dir) / "config.json"
        if not cfg_path.exists():
            return False
        try:
            return _read_json(cfg_path).get("model_type") == "whisper"
        except (json.JSONDecodeError, OSError):
            return False

    # -- 4-method converter protocol ----------------------------------------

    def build_config(self, ckpt_dir: Path) -> WhisperModelConfig:
        raw = _read_json(Path(ckpt_dir) / "config.json")
        # generation_config.json (when shipped) is the authoritative source
        # for the generation control ids; config.json carries older copies.
        gen_path = Path(ckpt_dir) / "generation_config.json"
        gen = _read_json(gen_path) if gen_path.exists() else {}

        def pick(key: str, default):
            v = gen.get(key)
            if v is None:
                v = raw.get(key)
            return default if v is None else v

        # generation_config marks auto-detected slots with null (e.g.
        # ``[[1, null]]`` for language detection); fall back to config.json's
        # concrete id per position (the recipe default, <|en|> on multilingual
        # snapshots) and drop any slot neither file pins — greedy decoding
        # here does not run language detection.
        gen_forced = {int(p): t for p, t in (gen.get("forced_decoder_ids") or [])}
        raw_forced = {int(p): t for p, t in (raw.get("forced_decoder_ids") or [])}
        positions = sorted(set(gen_forced) | set(raw_forced))
        forced = []
        for p in positions:
            tok = gen_forced.get(p)
            if tok is None:
                tok = raw_forced.get(p)
            if tok is not None:
                forced.append((p, int(tok)))
        tok_json = Path(ckpt_dir) / "tokenizer.json"
        tasks, languages = _control_token_tables(tok_json)
        no_speech = _no_speech_token_id(tok_json)
        # Published alignment heads ride in generation_config.json only; a
        # snapshot without them still converts, and word timestamps fall back
        # to the upper decoder layers (noisier, and much more transient memory).
        heads = [
            (int(layer), int(head))
            for layer, head in (gen.get("alignment_heads") or raw.get("alignment_heads") or [])
        ]
        return WhisperModelConfig(
            vocab_size=int(raw["vocab_size"]),
            d_model=int(raw["d_model"]),
            encoder_layers=int(raw["encoder_layers"]),
            decoder_layers=int(raw["decoder_layers"]),
            encoder_attention_heads=int(raw["encoder_attention_heads"]),
            decoder_attention_heads=int(raw["decoder_attention_heads"]),
            encoder_ffn_dim=int(raw["encoder_ffn_dim"]),
            decoder_ffn_dim=int(raw["decoder_ffn_dim"]),
            num_mel_bins=int(raw.get("num_mel_bins", 80)),
            max_source_positions=int(raw.get("max_source_positions", 1500)),
            max_target_positions=int(raw.get("max_target_positions", 448)),
            decoder_start_token_id=int(pick("decoder_start_token_id", 50258)),
            eos_token_id=int(pick("eos_token_id", 50257)),
            forced_decoder_ids=[(int(p), int(t)) for p, t in forced],
            suppress_tokens=[int(t) for t in (pick("suppress_tokens", []) or [])],
            begin_suppress_tokens=[int(t) for t in (pick("begin_suppress_tokens", []) or [])],
            task_token_ids=tasks,
            language_token_ids=languages,
            alignment_heads=heads,
            no_speech_token_id=no_speech,
        )

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]:
        return self.load_hf_state_dict(ckpt_dir, map_location)

    # -- complete-bundle conversion ------------------------------------------

    def build_tokenizer_spec(self, ckpt_dir: Path):
        from oasr.tokenizers import TokenizerSpec

        tok_path = Path(ckpt_dir) / "tokenizer.json"
        if not tok_path.exists():
            return None
        raw = _read_json(Path(ckpt_dir) / "config.json")
        return TokenizerSpec(
            kind="whisper",
            files={"tokenizer": str(tok_path)},
            options={"eot_id": int(raw.get("eos_token_id", 50257))},
        )

    def build_feature_spec(self, ckpt_dir: Path):
        raw = _read_json(Path(ckpt_dir) / "config.json")
        return self.whisper_logmel_spec(raw.get("num_mel_bins", 80))

    def build_decoding_defaults(self, config, ckpt_dir: Path):
        from oasr.checkpoints import DecodingDefaults

        return DecodingDefaults(
            default_decode_type=self.default_decode_type,
            blank_id=0,
            sos_id=config.decoder_start_token_id,
            eos_id=config.eos_token_id,
        )
