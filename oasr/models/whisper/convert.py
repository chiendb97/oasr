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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

import torch

from .config import WhisperModelConfig

if TYPE_CHECKING:
    from oasr.checkpoints import ConvertedCheckpoint

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class HFWhisperConverter:
    """Checkpoint converter for HF ``model_type: "whisper"`` snapshots."""

    #: ``proj_out.weight`` is tied to ``decoder.embed_tokens.weight`` — a
    #: silent, expected drop when the snapshot materializes it.
    expected_unused_prefixes: Tuple[str, ...] = ("proj_out.",)
    capability_drop_hints: Dict[str, str] = {}

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
        )

    def build_aux(self, ckpt_dir: Path) -> Dict[str, Any]:
        return {}

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]:
        ckpt_dir = Path(ckpt_dir)
        st_path = ckpt_dir / "model.safetensors"
        if st_path.exists():
            from safetensors.torch import load_file

            return load_file(str(st_path), device=str(map_location))
        bin_path = ckpt_dir / "pytorch_model.bin"
        if bin_path.exists():
            return torch.load(str(bin_path), map_location=map_location)
        raise FileNotFoundError(f"no model.safetensors or pytorch_model.bin under {ckpt_dir}")

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
        from oasr.features import FeatureSpec

        raw = _read_json(Path(ckpt_dir) / "config.json")
        return FeatureSpec(
            kind="whisper_logmel",
            sample_rate=16000,
            feature_dim=int(raw.get("num_mel_bins", 80)),
            frame_length_ms=25.0,  # n_fft 400 @ 16 kHz
            frame_shift_ms=10.0,  # hop 160
            dither=0.0,
            audio_scale=1.0,
        )

    def convert(
        self, ckpt_dir: Path, checkpoint_name: str = "model.safetensors", map_location: Any = "cpu"
    ) -> "ConvertedCheckpoint":
        from oasr.checkpoints import ConvertedCheckpoint, DecodingDefaults

        ckpt_dir = Path(ckpt_dir)
        config = self.build_config(ckpt_dir)
        return ConvertedCheckpoint(
            architecture="whisper",
            model_config=config,
            aux={},
            state_dict=self.load_state_dict(ckpt_dir, checkpoint_name, map_location),
            tokenizer=self.build_tokenizer_spec(ckpt_dir),
            features=self.build_feature_spec(ckpt_dir),
            decoding=DecodingDefaults(
                default_decode_type="aed",
                blank_id=0,
                sos_id=config.decoder_start_token_id,
                eos_id=config.eos_token_id,
            ),
            source_format="huggingface",
        )
