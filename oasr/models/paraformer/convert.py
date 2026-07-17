# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""FunASR Paraformer checkpoint converter.

Reads a FunASR model dir (``config.yaml`` + ``model.pt`` + ``am.mvn`` +
``tokens.json`` [+ ``seg_dict``]) and emits the complete
:class:`~oasr.checkpoints.ConvertedCheckpoint` bundle: model config, weights
(with the ``am.mvn`` CMVN injected as ``encoder.cmvn_shift`` /
``encoder.cmvn_scale`` buffers so it rides the ordinary weight path and
round-trips through the native format), a ``funasr_char`` TokenizerSpec, and a
``kaldi_fbank`` FeatureSpec carrying the hamming window + LFR 7/6 stacking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

import torch

from .config import ParaformerModelConfig

if TYPE_CHECKING:
    from oasr.checkpoints import ConvertedCheckpoint

logger = logging.getLogger(__name__)

#: Module classes this converter knows how to map.  Other FunASR variants
#: (BiCif / SeACo / contextual models) need their own converters.
_SUPPORTED = {
    "model": ("Paraformer",),
    "encoder": ("SANMEncoder",),
    "decoder": ("ParaformerSANMDecoder",),
    "predictor": ("CifPredictorV2",),
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_am_mvn(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse a Kaldi-nnet ``am.mvn`` file → ``(add_shift, rescale)`` fp32 tensors.

    FunASR applies it as ``(x + add_shift) * rescale`` on the post-LFR features.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    shift = None
    scale = None
    for i, line in enumerate(lines):
        item = line.split()
        if not item:
            continue
        if item[0] in ("<AddShift>", "<Rescale>"):
            values = lines[i + 1].split()
            if values and values[0] == "<LearnRateCoef>":
                vec = torch.tensor([float(v) for v in values[3 : len(values) - 1]])
                if item[0] == "<AddShift>":
                    shift = vec
                else:
                    scale = vec
    if shift is None or scale is None:
        raise ValueError(f"{path}: could not parse <AddShift>/<Rescale> vectors")
    return shift, scale


class FunASRParaformerConverter:
    """Checkpoint converter for FunASR ``model: Paraformer`` dirs."""

    expected_unused_prefixes: Tuple[str, ...] = ()
    capability_drop_hints: Dict[str, str] = {
        "ctc.": "a CTC branch (CTC decoding of Paraformer checkpoints is not wired)",
        "bias_encoder": "a contextual-biasing branch (hotword models are unsupported)",
    }

    def detect(self, ckpt_dir: Path) -> bool:
        cfg_path = Path(ckpt_dir) / "config.yaml"
        if not cfg_path.exists():
            return False
        try:
            raw = _read_yaml(cfg_path)
        except Exception:  # noqa: BLE001 — malformed yaml just means "not ours"
            return False
        return isinstance(raw, dict) and raw.get("model") in _SUPPORTED["model"]

    # -- 4-method converter protocol ----------------------------------------

    def build_config(self, ckpt_dir: Path) -> ParaformerModelConfig:
        raw = _read_yaml(Path(ckpt_dir) / "config.yaml")
        for slot in ("model", "encoder", "decoder", "predictor"):
            value = raw.get(slot)
            if value not in _SUPPORTED[slot]:
                raise ValueError(
                    f"Unsupported FunASR {slot} {value!r}; this converter handles "
                    f"{slot} in {_SUPPORTED[slot]}"
                )
        enc = raw.get("encoder_conf", {})
        dec = raw.get("decoder_conf", {})
        pred = raw.get("predictor_conf", {})
        tokens_path = Path(ckpt_dir) / "tokens.json"
        with open(tokens_path, "r", encoding="utf-8") as f:
            vocab_size = len(json.load(f))
        num_blocks = int(dec.get("num_blocks", 16))
        return ParaformerModelConfig(
            vocab_size=vocab_size,
            input_size=int(raw.get("input_size", 560)),
            encoder_output_size=int(enc.get("output_size", 512)),
            encoder_attention_heads=int(enc.get("attention_heads", 4)),
            encoder_linear_units=int(enc.get("linear_units", 2048)),
            encoder_num_blocks=int(enc.get("num_blocks", 50)),
            encoder_kernel_size=int(enc.get("kernel_size", 11)),
            encoder_sanm_shift=int(enc.get("sanm_shfit", 0)),
            decoder_attention_heads=int(dec.get("attention_heads", 4)),
            decoder_linear_units=int(dec.get("linear_units", 2048)),
            decoder_num_blocks=num_blocks,
            decoder_att_layer_num=int(dec.get("att_layer_num", num_blocks)),
            decoder_kernel_size=int(dec.get("kernel_size", 11)),
            decoder_sanm_shift=int(dec.get("sanm_shfit", 0)),
            predictor_idim=int(pred.get("idim", 512)),
            predictor_threshold=float(pred.get("threshold", 1.0)),
            predictor_l_order=int(pred.get("l_order", 1)),
            predictor_r_order=int(pred.get("r_order", 1)),
            predictor_tail_threshold=float(pred.get("tail_threshold", 0.45)),
        )

    def build_aux(self, ckpt_dir: Path) -> Dict[str, Any]:
        return {}

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]:
        ckpt_dir = Path(ckpt_dir)
        name = checkpoint_name if checkpoint_name and checkpoint_name != "final.pt" else "model.pt"
        ckpt = torch.load(str(ckpt_dir / name), map_location=map_location, weights_only=True)
        sd = dict(ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)
        mvn_path = ckpt_dir / "am.mvn"
        if mvn_path.exists():
            shift, scale = load_am_mvn(mvn_path)
            sd["encoder.cmvn_shift"] = shift
            sd["encoder.cmvn_scale"] = scale
        else:
            logger.warning("%s: no am.mvn found — CMVN will be identity", ckpt_dir)
        return sd

    # -- complete-bundle conversion ------------------------------------------

    def build_tokenizer_spec(self, ckpt_dir: Path):
        from oasr.tokenizers import TokenizerSpec

        tokens_path = Path(ckpt_dir) / "tokens.json"
        if not tokens_path.exists():
            return None
        files = {"tokens": str(tokens_path)}
        seg_dict = Path(ckpt_dir) / "seg_dict"
        if seg_dict.exists():
            files["seg_dict"] = str(seg_dict)
        return TokenizerSpec(
            kind="funasr_char",
            files=files,
            options={"special_ids": [0, 1, 2]},
        )

    def build_feature_spec(self, ckpt_dir: Path):
        from oasr.features import FeatureSpec

        raw = _read_yaml(Path(ckpt_dir) / "config.yaml")
        fe = raw.get("frontend_conf", {})
        return FeatureSpec(
            kind="kaldi_fbank",
            sample_rate=int(fe.get("fs", 16000)),
            feature_dim=int(fe.get("n_mels", 80)),
            frame_length_ms=float(fe.get("frame_length", 25)),
            frame_shift_ms=float(fe.get("frame_shift", 10)),
            dither=0.0,
            lfr_m=int(fe.get("lfr_m", 1)),
            lfr_n=int(fe.get("lfr_n", 1)),
            window_type=str(fe.get("window", "hamming")),
            normalize=None,  # CMVN is model-side (encoder.cmvn_* buffers)
        )

    def convert(
        self, ckpt_dir: Path, checkpoint_name: str = "model.pt", map_location: Any = "cpu"
    ) -> "ConvertedCheckpoint":
        from oasr.checkpoints import ConvertedCheckpoint, DecodingDefaults

        ckpt_dir = Path(ckpt_dir)
        config = self.build_config(ckpt_dir)
        return ConvertedCheckpoint(
            architecture="paraformer",
            model_config=config,
            aux={},
            state_dict=self.load_state_dict(ckpt_dir, checkpoint_name, map_location),
            tokenizer=self.build_tokenizer_spec(ckpt_dir),
            features=self.build_feature_spec(ckpt_dir),
            decoding=DecodingDefaults(
                default_decode_type="paraformer",
                blank_id=config.blank_id,
                sos_id=config.sos_id,
                eos_id=config.eos_id,
            ),
            source_format="funasr",
        )
