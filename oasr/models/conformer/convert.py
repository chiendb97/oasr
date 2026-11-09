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
    pass

logger = logging.getLogger(__name__)


def parse_wenet_yaml(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def wenet_sos_eos(raw: Dict[str, Any], raw_vocab: int) -> Tuple[int, int]:
    """The decoder's ``(<sos>, <eos>)`` ids, by WeNet's own rule.

    ``tokenizer_conf.special_tokens`` names them when present -- WeNet's
    ``ASRModel`` reads ``special_tokens.get("<sos>", vocab_size - 1)`` -- and
    only a config without that table falls back to the legacy convention, the
    last unit of the unpadded vocab.  The re-released LibriSpeech U2++ puts
    ``<sos/eos>`` at id 2; taking ``output_dim - 1`` there primes and closes
    every rescoring hypothesis with the BPE piece ``ZZ``, which rewards a
    hypothesis cut mid-word: attention rescoring scored 6.09 % WER on
    LibriSpeech test-clean, worse than CTC alone (3.77 %), and 3.37 % fixed.
    """
    special = (raw.get("tokenizer_conf") or {}).get("special_tokens") or {}
    last = int(raw_vocab) - 1
    return int(special.get("<sos>", last)), int(special.get("<eos>", last))


#: Every ``encoder_conf`` key WeNet's ``ConformerEncoder`` has taken, v1.0 through
#: main, by what this converter does with it.  WeNet passes ``encoder_conf`` to
#: that constructor as keyword arguments, so a key outside these tables comes from
#: another encoder class or a newer WeNet, and may change what the encoder computes.
#:
#: Read by :func:`build_config_from_wenet`.
_ENCODER_READ = frozenset(
    {
        "output_size",
        "attention_heads",
        "linear_units",
        "num_blocks",
        "input_layer",
        "normalize_before",
        "macaron_style",
        "use_cnn_module",
        "cnn_module_kernel",
        "causal",
        "conv_bias",
        "conv_norm_eps",
        "conv_inner_factor",
        "norm_eps",
        "final_norm",
        "static_chunk_size",
        "use_dynamic_chunk",
    }
)
#: No effect on what the encoder computes at inference.
_ENCODER_INERT = frozenset(
    {
        "dropout_rate",
        "positional_dropout_rate",
        "attention_dropout_rate",
        "use_dynamic_left_chunk",
        "gradient_checkpointing",
        "use_sdpa",
        # Accepted and never read, in every WeNet version.
        "positionwise_conv_kernel_size",
        # Read only by ``mlp_type="moe"``, which is refused.
        "n_expert",
        "n_expert_activated",
    }
)
_RMS_NORM = "WeNet's RMSNorm scales by 1 + weight, and OASR's by weight"
_BIAS = "every attention and feed-forward projection here carries a bias"
#: Keys this Conformer implements at some values only: the values, and why.
_ENCODER_VALUES: Dict[str, Tuple[Tuple[Any, ...], str]] = {
    # WeNet picks the attention class by ``pos_enc_layer_type`` before v3 and by
    # ``selfattention_layer_type`` from v3, so both must name the relative form.
    "pos_enc_layer_type": (("rel_pos",), "only WeNet's RelPositionalEncoding is implemented"),
    "selfattention_layer_type": (
        ("rel_selfattn",),
        "only WeNet's RelPositionMultiHeadedAttention is implemented",
    ),
    # WeNet's ``gelu`` is exact erf; the convolution module's fused one is the tanh
    # approximation.
    "activation_type": (
        ("swish", "relu"),
        "the convolution module's gelu is the tanh approximation, and hardtanh, tanh "
        "and selu are not implemented",
    ),
    "layer_norm_type": (("layer_norm",), _RMS_NORM),
    "cnn_module_norm": (("batch_norm", "layer_norm"), _RMS_NORM),
    # ``concat_after`` replaces each sub-layer's residual add with
    # ``concat_linear(cat([x, out]))``.  Checkpoints trained without it still
    # carry ``concat_linear``, hence ``expected_unused_prefixes``.
    "concat_after": ((False,), "it uses the residual form"),
    "mlp_type": (
        ("position_wise_feed_forward",),
        "only WeNet's PositionwiseFeedForward is implemented",
    ),
    "query_bias": ((True,), _BIAS),
    "key_bias": ((True,), _BIAS),
    "value_bias": ((True,), _BIAS),
    "mlp_bias": ((True,), _BIAS),
    "n_kv_head": ((None,), "grouped-query attention is not checked against WeNet's"),
    "head_dim": ((None,), "grouped-query attention is not checked against WeNet's"),
}

#: The same for ``decoder_conf`` and WeNet's ``(Bi)TransformerDecoder``.  A
#: decoder this converter cannot build costs attention rescoring only, so it is
#: dropped rather than refused.
_DECODER_READ = frozenset({"attention_heads", "linear_units", "num_blocks", "r_num_blocks"})
_DECODER_INERT = frozenset(
    {
        "dropout_rate",
        "positional_dropout_rate",
        "self_attention_dropout_rate",
        "src_attention_dropout_rate",
        "gradient_checkpointing",
        "use_sdpa",
        "n_expert",
        "n_expert_activated",
        # Tied, both tensors are saved and equal; a missing one fails the load.
        "tie_word_embedding",
    }
)
_DECODER_VALUES: Dict[str, Tuple[Tuple[Any, ...], str]] = {
    key: (values, "the decoder implements WeNet's default only")
    for key, values in {
        "input_layer": ("embed",),
        "use_output_layer": (True,),
        "normalize_before": (True,),
        "src_attention": (True,),
        "activation_type": ("relu",),
        "layer_norm_type": ("layer_norm",),
        "norm_eps": (1e-5,),
        "mlp_type": ("position_wise_feed_forward",),
        "mlp_bias": (True,),
        "query_bias": (True,),
        "key_bias": (True,),
        "value_bias": (True,),
        "src_query_bias": (True,),
        "src_key_bias": (True,),
        "src_value_bias": (True,),
        "n_kv_head": (None,),
        "head_dim": (None,),
        "concat_after": (False,),
    }.items()
}

#: WeNet model classes OASR serves: both train the CTC head OASR decodes, and
#: WeNet decodes ``K2Model`` (an ``ASRModel`` trained with LF-MMI) through it too.
_CTC_MODELS = ("asr_model", "k2_model")


def _unsupported(
    section: str,
    conf: Mapping[str, Any],
    known: frozenset,
    values: Mapping[str, Tuple[Tuple[Any, ...], str]],
) -> Optional[str]:
    """The first thing in a ``train.yaml`` section OASR cannot build, or ``None``."""
    for key, value in conf.items():
        if key in values:
            supported, why = values[key]
            if isinstance(value, str) and any(isinstance(s, float) for s in supported):
                # YAML 1.1 reads ``1e-5`` (no dot) as a string.
                try:
                    value = float(value)
                except ValueError:
                    pass
            if value not in supported:
                return f"sets {section}.{key}={value}, which is not supported ({why})"
        elif key not in known:
            return (
                f"sets {section}.{key}, which no WeNet version this converter knows takes, "
                "and which may change what the model computes"
            )
    return None


def build_config_from_wenet(raw: Dict[str, Any]) -> ConformerModelConfig:
    """Translate a WeNet ``train.yaml`` into a :class:`ConformerModelConfig`.

    Refuses a ``train.yaml`` whose encoder or model class this Conformer does not
    implement: its weights could load by name and compute something else.
    """
    # WeNet builds the encoder class ``encoder`` names, ``conformer`` by default.
    encoder = raw.get("encoder", "conformer")
    if encoder != "conformer":
        raise ValueError(
            f"train.yaml declares encoder={encoder!r}, and the WeNet converter builds "
            "only WeNet's ConformerEncoder ('conformer'). The checkpoint is refused "
            "rather than built as a Conformer."
        )
    # Before v3, WeNet picked the model class by these keys instead of ``model``.
    model = raw.get("model") or (
        "transducer" if "predictor" in raw else "paraformer" if "paraformer" in raw else "asr_model"
    )
    if model not in _CTC_MODELS:
        raise ValueError(
            f"train.yaml declares a WeNet {model!r} model. OASR decodes a WeNet "
            f"checkpoint through the CTC head that {' and '.join(_CTC_MODELS)} train, "
            "and that head need not be trained in this one."
        )
    if float((raw.get("model_conf") or {}).get("ctc_weight", 0.5)) == 0.0:
        raise ValueError(
            "train.yaml sets model_conf.ctc_weight=0, so the checkpoint's CTC head was "
            "never trained, and OASR decodes a WeNet checkpoint through it."
        )
    enc = raw.get("encoder_conf", {})
    problem = _unsupported("encoder_conf", enc, _ENCODER_READ | _ENCODER_INERT, _ENCODER_VALUES)
    if problem is not None:
        raise ValueError(
            f"train.yaml {problem}. The checkpoint is refused rather than loaded as a "
            "Conformer that computes something else."
        )
    # A static chunk without a dynamic one masks attention to that chunk whatever
    # the decoding chunk (WeNet's ``add_optional_chunk_mask``), offline included,
    # and OASR's offline forward attends to the whole utterance.
    static_chunk = int(enc.get("static_chunk_size", 0))
    if static_chunk > 0 and not enc.get("use_dynamic_chunk", False):
        raise ValueError(
            f"train.yaml sets encoder_conf.static_chunk_size={static_chunk} without "
            "use_dynamic_chunk, so WeNet decodes it with that chunk's attention mask "
            "even offline, and OASR's offline forward attends to the whole utterance."
        )
    encoder_cfg = ConformerEncoderConfig(
        input_size=raw.get("input_dim", 80),
        output_size=enc.get("output_size", 256),
        num_blocks=enc.get("num_blocks", 6),
        attention_heads=enc.get("attention_heads", 4),
        linear_units=enc.get("linear_units", 2048),
        cnn_module_kernel=enc.get("cnn_module_kernel", 15),
        use_cnn_module=enc.get("use_cnn_module", True),
        activation_type=enc.get("activation_type", "swish"),
        normalize_before=enc.get("normalize_before", True),
        macaron_style=enc.get("macaron_style", True),
        causal=enc.get("causal", False),
        cnn_module_norm=enc.get("cnn_module_norm", "batch_norm"),
        # Structural, with WeNet's defaults: dropping one builds a model the
        # weights still load into, and it computes something else.
        layer_norm_type=enc.get("layer_norm_type", "layer_norm"),
        norm_eps=float(enc.get("norm_eps", 1e-5)),
        conv_bias=enc.get("conv_bias", True),
        conv_norm_eps=float(enc.get("conv_norm_eps", 1e-5)),
        conv_inner_factor=enc.get("conv_inner_factor", 2),
        final_norm=enc.get("final_norm", True),
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
    # (unpadded) output_dim; <sos>/<eos> come from ``wenet_sos_eos``.
    decoder_cfg = None
    dec_name = raw.get("decoder")
    dec = raw.get("decoder_conf") or {}
    problem = _unsupported("decoder_conf", dec, _DECODER_READ | _DECODER_INERT, _DECODER_VALUES)
    if dec_name in ("transformer", "bitransformer") and problem is not None:
        logger.warning(
            "train.yaml %s; the checkpoint's decoder.* weights will be dropped and "
            "attention rescoring will be unavailable",
            problem,
        )
    elif dec_name in ("transformer", "bitransformer") and raw_vocab:
        model_conf = raw.get("model_conf") or {}
        sos_id, eos_id = wenet_sos_eos(raw, raw_vocab)
        decoder_cfg = TransformerDecoderConfig(
            vocab_size=int(raw_vocab),
            encoder_output_size=encoder_cfg.output_size,
            attention_heads=dec.get("attention_heads", 4),
            linear_units=dec.get("linear_units", 2048),
            num_blocks=dec.get("num_blocks", 6),
            r_num_blocks=dec.get("r_num_blocks", 0) if dec_name == "bitransformer" else 0,
            sos_id=sos_id,
            eos_id=eos_id,
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

    #: Upstream checkpoints create ``concat_linear`` even when unused. The
    #: converter rejects configurations that enable unsupported ``concat_after``.
    expected_unused_prefixes: Tuple[str, ...] = ("concat_linear",)
    capability_drop_hints: Dict[str, str] = {
        "decoder.": (
            "the attention-decoder (CTC+AED rescoring) branch was not loaded "
            "(train.yaml declares a decoder type or structure OASR does not model); "
            "attention rescoring is unavailable for this checkpoint"
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

        # WeNet convention: <blank>=0, <unk>=1; <sos>/<eos> from train.yaml's
        # special_tokens, else the last unit of the *unpadded* vocab
        # (output_dim - 1), which only train.yaml knows — config.vocab_size is
        # 8-padded for the GEMM kernels.
        raw = parse_wenet_yaml(str(Path(ckpt_dir) / "train.yaml"))
        raw_vocab = raw.get("output_dim")
        sos_id, eos_id = wenet_sos_eos(raw, int(raw_vocab)) if raw_vocab else (None, None)
        return DecodingDefaults(
            default_decode_type=self.default_decode_type,
            blank_id=0,
            unk_id=1,
            sos_id=sos_id,
            eos_id=eos_id,
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
