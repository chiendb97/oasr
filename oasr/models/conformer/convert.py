# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Convert and load pretrained WeNet Conformer checkpoints into OASR models.

Usage (as module)::

    from oasr.models.conformer.convert import load_wenet_checkpoint

    model = load_wenet_checkpoint(
        "/path/to/20210610_u2pp_conformer_exp_librispeech",
    )

Usage (CLI)::

    python -m oasr.models.conformer.convert \
        /path/to/20210610_u2pp_conformer_exp_librispeech

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Mapping, Optional, Tuple

import torch
import yaml

from oasr.layers.norm import GlobalCMVN

from ..converter import BaseCheckpointConverter
from ..decoders.transformer_decoder import TransformerDecoderConfig
from ..registry import DETECT_NAMED_CONFIG
from .config import ConformerEncoderConfig, ConformerModelConfig
from .model import ConformerModel

if TYPE_CHECKING:
    from oasr.checkpoints import ConvertedCheckpoint

logger = logging.getLogger(__name__)


def parse_wenet_yaml(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def build_config_from_wenet(raw: Dict[str, Any]) -> ConformerModelConfig:
    """Translate a WeNet ``train.yaml`` into a :class:`ConformerModelConfig`."""
    enc = raw.get("encoder_conf", {})
    # ``concat_after`` replaces each sub-layer's residual add with
    # ``concat_linear(cat([x, sublayer_out]))``.  OASR's encoder implements only
    # the residual form, so honouring a checkpoint that asks for the other one is
    # impossible — and *ignoring* it would load a plausible-looking model that
    # silently computes something else.  Fail where the information is, rather
    # than shipping wrong transcripts.  (Checkpoints trained with the default
    # still *carry* the unused ``concat_linear`` parameters, which is why the
    # converter declares them expected-unused.)
    for section in ("encoder_conf", "decoder_conf"):
        if raw.get(section, {}).get("concat_after"):
            raise ValueError(
                f"train.yaml sets {section}.concat_after=True, which this "
                "Conformer implementation does not support (it uses the residual "
                "form). Loading it would silently compute a different model."
            )
    encoder_cfg = ConformerEncoderConfig(
        input_size=raw.get("input_dim", 80),
        output_size=enc.get("output_size", 256),
        num_blocks=enc.get("num_blocks", 12),
        attention_heads=enc.get("attention_heads", 4),
        linear_units=enc.get("linear_units", 2048),
        cnn_module_kernel=enc.get("cnn_module_kernel", 15),
        use_cnn_module=enc.get("use_cnn_module", True),
        activation_type=enc.get("activation_type", "swish"),
        normalize_before=enc.get("normalize_before", True),
        macaron_style=enc.get("macaron_style", True),
        causal=enc.get("causal", False),
        cnn_module_norm=enc.get("cnn_module_norm", "batch_norm"),
        input_layer=enc.get("input_layer", "conv2d"),
        embed_layer_norm=False,
    )

    raw_vocab = raw.get("output_dim")
    if raw_vocab is None:
        raise ValueError(
            "train.yaml declares no `output_dim`, so the CTC vocabulary size is "
            "unknown. Add it to train.yaml, or pass an explicit model config."
        )
    raw_vocab = int(raw_vocab)
    vocab_size = raw_vocab
    if vocab_size % 8 != 0:
        vocab_size = (vocab_size // 8 + 1) * 8

    # U2/U2++ AED decoder branch: keep the (bi)transformer decoder for the
    # ``ctc_aed_rescoring`` capability.  The decoder's vocab is the *raw*
    # (unpadded) output_dim; WeNet's <sos/eos> is the last raw unit.
    decoder_cfg = None
    dec_name = raw.get("decoder")
    dec = raw.get("decoder_conf") or {}
    if dec_name in ("transformer", "bitransformer") and raw_vocab:
        model_conf = raw.get("model_conf") or {}
        decoder_cfg = TransformerDecoderConfig(
            vocab_size=int(raw_vocab),
            encoder_output_size=encoder_cfg.output_size,
            attention_heads=dec.get("attention_heads", 4),
            linear_units=dec.get("linear_units", 2048),
            num_blocks=dec.get("num_blocks", 6),
            r_num_blocks=dec.get("r_num_blocks", 0) if dec_name == "bitransformer" else 0,
            sos_id=int(raw_vocab) - 1,
            eos_id=int(raw_vocab) - 1,
            reverse_weight=float(model_conf.get("reverse_weight", 0.0)),
        )
    elif dec_name:
        logger.warning(
            "train.yaml declares decoder=%r, which OASR does not model; the "
            "checkpoint's decoder.* weights will be dropped and attention "
            "rescoring will be unavailable",
            dec_name,
        )

    return ConformerModelConfig(
        encoder=encoder_cfg,
        vocab_size=vocab_size,
        decoder=decoder_cfg,
    )


def load_global_cmvn(cmvn_path: str) -> GlobalCMVN:
    """Load a WeNet JSON-format ``global_cmvn`` file into a :class:`GlobalCMVN` module."""
    with open(cmvn_path, "r") as f:
        raw = json.load(f)
    mean_stat = torch.tensor(raw["mean_stat"], dtype=torch.float32)
    var_stat = torch.tensor(raw["var_stat"], dtype=torch.float32)
    frame_num = raw["frame_num"]

    mean = mean_stat / frame_num
    variance = var_stat / frame_num - mean**2
    istd = 1.0 / torch.sqrt(torch.clamp(variance, min=1e-20))
    return GlobalCMVN(mean, istd)


class WenetConverter(BaseCheckpointConverter):
    """Checkpoint converter for WeNet Conformer experiment directories.

    Implements the :class:`~oasr.models.registry.CheckpointConverter` protocol:
    it owns the *format*-specific concerns (parse ``train.yaml`` → config, load
    ``global_cmvn``, read ``final.pt`` → raw state-dict).  The architecture's
    name-mapping / vocab-padding lives in
    :meth:`ConformerModel.load_weights`.
    """

    #: Weight-drop accounting (read by the registry after ``load_weights``):
    #: nothing in a WeNet dir is *expected* to be dropped silently.  The U2++
    #: (bi)transformer decoder loads as the ``ctc_aed_rescoring`` capability;
    #: this hint only fires for decoder types OASR does not model (the
    #: ``decoder.*`` weights are then dropped by ``load_weights``).
    #: WeNet's ``ConformerEncoderLayer`` / ``DecoderLayer`` build
    #: ``concat_linear`` unconditionally and only *use* it when
    #: ``concat_after=True``, so a checkpoint trained with the default carries
    #: the parameters unused.  Declaring them keeps a normal checkpoint quiet
    #: (they were reported as two dozen "unrecognized tensors", and the
    #: ``decoder.*`` ones fell through to the capability hint below, which then
    #: wrongly announced that attention rescoring was unavailable).
    #:
    #: OASR does not implement ``concat_after``; ``build_config_from_wenet``
    #: rejects a checkpoint that asks for it rather than silently ignoring it.
    expected_unused_prefixes: Tuple[str, ...] = ("concat_linear",)
    capability_drop_hints: Dict[str, str] = {
        "decoder.": (
            "the attention-decoder (CTC+AED rescoring) branch was not loaded "
            "(unsupported decoder type in train.yaml); attention rescoring is "
            "unavailable for this checkpoint"
        ),
    }

    architecture: ClassVar[str] = "conformer"
    source_format: ClassVar[str] = "wenet"
    default_checkpoint_name: ClassVar[str] = "final.pt"
    default_decode_type: ClassVar[str] = "ctc"
    #: ``train.yaml`` identifies WeNet, not a specific architecture, so this claim outranks a weaker one
    #: (see :func:`oasr.models.registry.resolve_architecture`).
    detect_specificity: ClassVar[int] = DETECT_NAMED_CONFIG

    def detect(self, ckpt_dir: Path) -> bool:
        """A WeNet experiment dir is identified by its ``train.yaml``."""
        return (Path(ckpt_dir) / "train.yaml").exists()

    def build_config(self, ckpt_dir: Path) -> ConformerModelConfig:
        ckpt_dir = Path(ckpt_dir)
        yaml_path = ckpt_dir / "train.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Required file not found: {yaml_path}")
        config = build_config_from_wenet(parse_wenet_yaml(str(yaml_path)))
        logger.info("Encoder config: %s", config.encoder)
        return config

    def build_aux(self, ckpt_dir: Path) -> Dict[str, Any]:
        """Build the ``global_cmvn`` buffer (optional) passed to ``from_config``."""
        cmvn_path = Path(ckpt_dir) / "global_cmvn"
        global_cmvn: Optional[GlobalCMVN] = None
        if cmvn_path.exists():
            global_cmvn = load_global_cmvn(str(cmvn_path))
            logger.info("Loaded global CMVN from %s", cmvn_path)
        return {"global_cmvn": global_cmvn}

    def load_state_dict(
        self, ckpt_dir: Path, checkpoint_name: str, map_location: Any
    ) -> Mapping[str, torch.Tensor]:
        ckpt_path = Path(ckpt_dir) / checkpoint_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Required file not found: {ckpt_path}")
        return torch.load(str(ckpt_path), map_location=map_location, weights_only=True)

    # -- complete-bundle conversion (tokenizer / feature / decoding specs) ----

    def build_tokenizer_spec(self, ckpt_dir: Path):
        """WeNet decodes with ``units.txt`` unit ids → a ``symbol_table`` spec.

        The SentencePiece ``.model`` (when shipped) is *not* the decode
        tokenizer — its piece ids differ from the CTC unit ids.
        """
        from oasr.tokenizers import TokenizerSpec

        for fname in ("units.txt", "words.txt"):
            table = Path(ckpt_dir) / fname
            if table.exists():
                return TokenizerSpec(kind="symbol_table", files={"table": str(table)})
        return None

    def build_feature_spec(self, ckpt_dir: Path, raw: Optional[Dict[str, Any]] = None):
        """FBANK geometry from ``train.yaml``'s ``dataset_conf`` (engine defaults
        stop being blind guesses).  Dither is forced to 0.0 — a training-time
        augmentation, always off at inference."""
        from oasr.features import FeatureSpec

        if raw is None:
            raw = parse_wenet_yaml(str(Path(ckpt_dir) / "train.yaml"))
        dataset_conf = raw.get("dataset_conf") or {}
        fbank = dataset_conf.get("fbank_conf") or {}
        resample = dataset_conf.get("resample_conf") or {}
        return FeatureSpec(
            kind="kaldi_fbank",
            sample_rate=int(resample.get("resample_rate", 16000)),
            feature_dim=int(fbank.get("num_mel_bins", raw.get("input_dim", 80))),
            frame_length_ms=float(fbank.get("frame_length", 25)),
            frame_shift_ms=float(fbank.get("frame_shift", 10)),
            dither=0.0,
            normalize="global_cmvn" if (Path(ckpt_dir) / "global_cmvn").exists() else None,
            audio_scale=32768.0,
        )

    def build_decoding_defaults(self, config, ckpt_dir: Path):
        from oasr.checkpoints import DecodingDefaults

        # WeNet convention: <blank>=0, <unk>=1, <sos/eos> = last unit of the
        # *unpadded* vocab (output_dim - 1), which only train.yaml knows —
        # config.vocab_size is 8-padded for the GEMM kernels.
        raw_vocab = parse_wenet_yaml(str(Path(ckpt_dir) / "train.yaml")).get("output_dim")
        sos_eos = int(raw_vocab) - 1 if raw_vocab else None
        return DecodingDefaults(
            default_decode_type=self.default_decode_type,
            blank_id=0,
            unk_id=1,
            sos_id=sos_eos,
            eos_id=sos_eos,
        )


def load_wenet_checkpoint(
    ckpt_dir: str,
    checkpoint_name: str = "final.pt",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[ConformerModel, ConformerModelConfig]:
    """Load a WeNet pretrained Conformer checkpoint directory.

    Thin back-compat wrapper around
    :func:`oasr.models.registry.build_model_from_checkpoint`.

    Args:
        ckpt_dir: Path to the WeNet experiment directory (must contain
            ``train.yaml``, ``global_cmvn``, and the checkpoint file).
        checkpoint_name: Filename of the ``.pt`` checkpoint inside
            *ckpt_dir*.  Defaults to ``"final.pt"``.
        device: Device to map tensors onto.
        dtype: Optional dtype to cast model parameters into after loading.

    Returns:
        A tuple of ``(model, config)`` where *model* has weights loaded
        and is set to eval mode.
    """
    from ..registry import build_model_from_checkpoint

    return build_model_from_checkpoint(
        ckpt_dir,
        checkpoint_name=checkpoint_name,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Load a WeNet Conformer checkpoint into OASR.")
    parser.add_argument(
        "ckpt_dir",
        help="Path to the WeNet experiment directory.",
    )
    parser.add_argument(
        "--checkpoint",
        default="final.pt",
        help="Checkpoint filename (default: final.pt).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load onto (default: cpu).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="If set, save the converted OASR state dict to this path.",
    )
    args = parser.parse_args()

    model, config = load_wenet_checkpoint(
        args.ckpt_dir,
        checkpoint_name=args.checkpoint,
        device=args.device,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")
    print(f"Config: {config}")

    if args.save:
        save_path = Path(args.save)
        torch.save(model.state_dict(), str(save_path))
        print(f"Saved converted state dict to {save_path}")
