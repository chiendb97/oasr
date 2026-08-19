# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Convert supported Nemotron ASR snapshots to the native checkpoint contract.

Frontend geometry and waveform scale come from processor metadata. The blank ID
is a special token beyond the tokenizer vocabulary. Detection claims only the
validated ``nemotron3_5_asr`` model type.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Tuple

import torch

from ..converter import BaseCheckpointConverter
from ..registry import DETECT_KEYED_VALUE
from .config import NemotronEncoderConfig, NemotronModelConfig

logger = logging.getLogger(__name__)

#: ``config.json: model_type`` values this converter claims.
_MODEL_TYPES = ("nemotron3_5_asr",)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded: Dict[str, Any] = json.load(f)
    return loaded


class HFNemotronConverter(BaseCheckpointConverter):
    """Converter for HF ``model_type: "nemotron3_5_asr"`` snapshots."""

    architecture: ClassVar[str] = "nemotron"
    source_format: ClassVar[str] = "huggingface"
    default_checkpoint_name: ClassVar[str] = "model.safetensors"
    default_decode_type: ClassVar[str] = "transducer"
    #: The architecture is named in ``config.json``, so this claim outranks any
    #: filename-based one (see :func:`oasr.models.registry.resolve_architecture`).
    detect_specificity: ClassVar[int] = DETECT_KEYED_VALUE
    #: Nothing is expected to be dropped: the released state dict maps 1:1 (the
    #: two joint projections are *renamed*, not discarded), so a non-empty
    #: ``LoadReport.dropped`` here means the format moved and should be loud.
    expected_unused_prefixes: ClassVar[Tuple[str, ...]] = ()

    def detect(self, ckpt_dir: Path) -> bool:
        cfg_path = Path(ckpt_dir) / "config.json"
        if not cfg_path.exists():
            return False
        try:
            return _read_json(cfg_path).get("model_type") in _MODEL_TYPES
        except (json.JSONDecodeError, OSError):
            return False

    # -- 4-method converter protocol -----------------------------------------

    def build_config(self, ckpt_dir: Path) -> NemotronModelConfig:
        raw = _read_json(Path(ckpt_dir) / "config.json")
        enc = dict(raw.get("encoder_config") or {})
        defaults = NemotronEncoderConfig()
        supported = enc.get("supported_num_lookahead_tokens") or list(
            defaults.supported_num_lookahead_tokens
        )
        encoder = NemotronEncoderConfig(
            hidden_size=int(enc.get("hidden_size", defaults.hidden_size)),
            num_hidden_layers=int(enc.get("num_hidden_layers", defaults.num_hidden_layers)),
            num_attention_heads=int(enc.get("num_attention_heads", defaults.num_attention_heads)),
            num_key_value_heads=int(
                enc.get("num_key_value_heads")
                or enc.get("num_attention_heads")
                or defaults.num_key_value_heads
            ),
            intermediate_size=int(enc.get("intermediate_size", defaults.intermediate_size)),
            hidden_act=str(enc.get("hidden_act", defaults.hidden_act)),
            attention_bias=bool(enc.get("attention_bias", defaults.attention_bias)),
            convolution_bias=bool(enc.get("convolution_bias", defaults.convolution_bias)),
            conv_kernel_size=int(enc.get("conv_kernel_size", defaults.conv_kernel_size)),
            num_mel_bins=int(enc.get("num_mel_bins", defaults.num_mel_bins)),
            subsampling_factor=int(enc.get("subsampling_factor", defaults.subsampling_factor)),
            subsampling_conv_channels=int(
                enc.get("subsampling_conv_channels", defaults.subsampling_conv_channels)
            ),
            subsampling_conv_kernel_size=int(
                enc.get("subsampling_conv_kernel_size", defaults.subsampling_conv_kernel_size)
            ),
            subsampling_conv_stride=int(
                enc.get("subsampling_conv_stride", defaults.subsampling_conv_stride)
            ),
            max_position_embeddings=int(
                enc.get("max_position_embeddings", defaults.max_position_embeddings)
            ),
            scale_input=bool(enc.get("scale_input", defaults.scale_input)),
            sliding_window=int(enc.get("sliding_window", defaults.sliding_window)),
            default_num_lookahead_tokens=int(
                enc.get("default_num_lookahead_tokens", defaults.default_num_lookahead_tokens)
            ),
            supported_num_lookahead_tokens=tuple(int(v) for v in supported),
        )
        model_defaults = NemotronModelConfig()
        return NemotronModelConfig(
            vocab_size=int(raw["vocab_size"]),
            blank_token_id=int(raw.get("blank_token_id", int(raw["vocab_size"]) - 1)),
            decoder_hidden_size=int(
                raw.get("decoder_hidden_size", model_defaults.decoder_hidden_size)
            ),
            num_decoder_layers=int(
                raw.get("num_decoder_layers", model_defaults.num_decoder_layers)
            ),
            hidden_act=str(raw.get("hidden_act", model_defaults.hidden_act)),
            max_symbols_per_step=int(
                raw.get("max_symbols_per_step", model_defaults.max_symbols_per_step)
            ),
            num_prompts=int(raw.get("num_prompts", 0)),
            prompt_intermediate_size=int(
                raw.get("prompt_intermediate_size", model_defaults.prompt_intermediate_size)
            ),
            default_prompt_id=int(raw.get("default_prompt_id", model_defaults.default_prompt_id)),
            encoder=encoder,
        )

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]:
        return self.load_hf_state_dict(ckpt_dir, map_location)

    # -- complete-bundle conversion -------------------------------------------

    def build_tokenizer_spec(self, ckpt_dir: Path):
        from oasr.tokenizers import TokenizerSpec

        ckpt_dir = Path(ckpt_dir)
        tok_path = ckpt_dir / "tokenizer.json"
        if not tok_path.exists():
            return None
        raw = _read_json(ckpt_dir / "config.json")
        blank = int(raw.get("blank_token_id", int(raw["vocab_size"]) - 1))
        files = {"tokenizer": str(tok_path)}
        cfg_path = ckpt_dir / "tokenizer_config.json"
        if cfg_path.exists():
            files["tokenizer_config"] = str(cfg_path)
        # The blank id sits *past* the tokenizer's vocabulary (13087 with 13087
        # pieces), so it has to be filtered before the backend sees it — the
        # language tags are already flagged ``special`` inside tokenizer.json and
        # ``skip_special_tokens`` drops them.
        return TokenizerSpec(
            kind="huggingface",
            files=files,
            options={"special_ids": [blank]},
        )

    def build_feature_spec(self, ckpt_dir: Path):
        from oasr.features import FeatureSpec

        ckpt_dir = Path(ckpt_dir)
        raw = _read_json(ckpt_dir / "config.json")
        mel_bins = int((raw.get("encoder_config") or {}).get("num_mel_bins", 128))
        fe: Dict[str, Any] = {}
        proc_path = ckpt_dir / "processor_config.json"
        if proc_path.exists():
            fe = dict(_read_json(proc_path).get("feature_extractor") or {})
        sample_rate = int(fe.get("sampling_rate", 16000))
        hop = int(fe.get("hop_length", 160))
        win = int(fe.get("win_length", 400))
        return FeatureSpec(
            kind="nemotron_logmel",
            sample_rate=sample_rate,
            feature_dim=int(fe.get("feature_size", mel_bins)),
            frame_length_ms=win * 1000.0 / sample_rate,
            frame_shift_ms=hop * 1000.0 / sample_rate,
            dither=0.0,
            preemphasis=float(fe.get("preemphasis", 0.97)),
            # NeMo consumes the [-1, 1] waveform, like icefall/lhotse and unlike
            # WeNet's 1 << 15.  See docs/checkpoints.md.
            audio_scale=1.0,
        )

    def build_decoding_defaults(self, config, ckpt_dir: Path):
        from oasr.checkpoints import DecodingDefaults

        return DecodingDefaults(
            default_decode_type=self.default_decode_type,
            blank_id=int(config.blank_token_id),
        )


__all__ = ["HFNemotronConverter"]
