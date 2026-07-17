# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace Qwen2-Audio checkpoint converter.

Reads an HF Qwen2-Audio snapshot (``config.json`` with
``model_type: "qwen2_audio"``, sharded or single ``model.safetensors``,
``tokenizer.json``, ``generation_config.json``) and emits the complete
:class:`~oasr.checkpoints.ConvertedCheckpoint` bundle: model config, weights,
a ``huggingface`` TokenizerSpec, a 128-mel ``whisper_logmel`` FeatureSpec
(``audio_scale = 1.0``), and LLM decoding defaults.

The published checkpoint's ``text_config`` stores only the fields that differ
from ``Qwen2Config`` defaults — :data:`_QWEN2_TEXT_DEFAULTS` fills the rest
(hidden size, layer/head counts), matching what ``transformers`` does.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Tuple

import torch

from .config import SpeechLlmModelConfig

if TYPE_CHECKING:
    from oasr.checkpoints import ConvertedCheckpoint

logger = logging.getLogger(__name__)

#: ``Qwen2Config`` defaults for the fields OASR reads (transformers ≥4.37).
_QWEN2_TEXT_DEFAULTS: Dict[str, Any] = {
    "vocab_size": 151936,
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "intermediate_size": 22016,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 32768,
    "tie_word_embeddings": False,
}


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class HFQwen2AudioConverter:
    """Checkpoint converter for HF ``model_type: "qwen2_audio"`` snapshots."""

    #: Rotary tables are computed buffers when a snapshot materializes them.
    expected_unused_prefixes: Tuple[str, ...] = ()
    capability_drop_hints: Dict[str, str] = {}

    def detect(self, ckpt_dir: Path) -> bool:
        cfg_path = Path(ckpt_dir) / "config.json"
        if not cfg_path.exists():
            return False
        try:
            return _read_json(cfg_path).get("model_type") == "qwen2_audio"
        except (json.JSONDecodeError, OSError):
            return False

    # -- 4-method converter protocol ----------------------------------------

    def build_config(self, ckpt_dir: Path) -> SpeechLlmModelConfig:
        raw = _read_json(Path(ckpt_dir) / "config.json")
        audio = raw.get("audio_config", {})
        text = dict(_QWEN2_TEXT_DEFAULTS)
        text.update(raw.get("text_config", {}))

        gen_path = Path(ckpt_dir) / "generation_config.json"
        gen = _read_json(gen_path) if gen_path.exists() else {}
        eos = gen.get("eos_token_id", text.get("eos_token_id", 151645))
        eos_ids: List[int] = [int(t) for t in (eos if isinstance(eos, list) else [eos])]
        pad = gen.get("pad_token_id", eos_ids[0])

        vocab_size = int(text.get("vocab_size") or raw.get("vocab_size") or 156032)
        return SpeechLlmModelConfig(
            vocab_size=vocab_size,
            audio_num_mel_bins=int(audio.get("num_mel_bins", 128)),
            audio_d_model=int(audio.get("d_model", 1280)),
            audio_encoder_layers=int(audio.get("encoder_layers", 32)),
            audio_encoder_attention_heads=int(audio.get("encoder_attention_heads", 20)),
            audio_encoder_ffn_dim=int(audio.get("encoder_ffn_dim", 5120)),
            audio_max_source_positions=int(audio.get("max_source_positions", 1500)),
            text_hidden_size=int(text["hidden_size"]),
            text_num_hidden_layers=int(text["num_hidden_layers"]),
            text_num_attention_heads=int(text["num_attention_heads"]),
            text_num_key_value_heads=int(text["num_key_value_heads"]),
            text_intermediate_size=int(text["intermediate_size"]),
            text_rope_theta=float(text["rope_theta"]),
            text_rms_norm_eps=float(text["rms_norm_eps"]),
            text_max_position_embeddings=int(text["max_position_embeddings"]),
            text_tie_word_embeddings=bool(text["tie_word_embeddings"]),
            audio_token_id=int(raw.get("audio_token_index", 151646)),
            eos_token_ids=eos_ids,
            pad_token_id=int(pad),
        )

    def build_aux(self, ckpt_dir: Path) -> Dict[str, Any]:
        return {}

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]:
        ckpt_dir = Path(ckpt_dir)
        index_path = ckpt_dir / "model.safetensors.index.json"
        if index_path.exists():
            from safetensors.torch import load_file

            shards = sorted(set(_read_json(index_path)["weight_map"].values()))
            sd: Dict[str, torch.Tensor] = {}
            for shard in shards:
                sd.update(load_file(str(ckpt_dir / shard), device=str(map_location)))
            return sd
        st_path = ckpt_dir / "model.safetensors"
        if st_path.exists():
            from safetensors.torch import load_file

            return load_file(str(st_path), device=str(map_location))
        bin_path = ckpt_dir / "pytorch_model.bin"
        if bin_path.exists():
            return torch.load(str(bin_path), map_location=map_location)
        raise FileNotFoundError(
            f"no model.safetensors[.index.json] or pytorch_model.bin under {ckpt_dir}"
        )

    # -- complete-bundle conversion ------------------------------------------

    def build_tokenizer_spec(self, ckpt_dir: Path):
        from oasr.tokenizers import TokenizerSpec

        tok_path = Path(ckpt_dir) / "tokenizer.json"
        if not tok_path.exists():
            return None
        files = {"tokenizer": str(tok_path)}
        # Qwen2-Audio declares its audio / timestamp special tokens only in
        # tokenizer_config.json (added_tokens_decoder) — required for prompt
        # encoding (<|audio_bos|> etc. must not BPE-split).
        cfg_path = Path(ckpt_dir) / "tokenizer_config.json"
        if cfg_path.exists():
            files["tokenizer_config"] = str(cfg_path)
        return TokenizerSpec(kind="huggingface", files=files)

    def build_feature_spec(self, ckpt_dir: Path):
        from oasr.features import FeatureSpec

        raw = _read_json(Path(ckpt_dir) / "config.json")
        n_mels = int(raw.get("audio_config", {}).get("num_mel_bins", 128))
        return FeatureSpec(
            kind="whisper_logmel",
            sample_rate=16000,
            feature_dim=n_mels,
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
            architecture="speech_llm",
            model_config=config,
            aux={},
            state_dict=self.load_state_dict(ckpt_dir, checkpoint_name, map_location),
            tokenizer=self.build_tokenizer_spec(ckpt_dir),
            features=self.build_feature_spec(ckpt_dir),
            decoding=DecodingDefaults(
                default_decode_type="llm",
                blank_id=0,
                sos_id=config.pad_token_id,
                eos_id=config.eos_token_ids[-1],
            ),
            source_format="huggingface",
        )
