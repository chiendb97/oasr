# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""WeNet ``train.yaml`` → Conformer: the structure the converter builds.

``test_conformer.py`` checks numbers against upstream WeNet, but only where the
``wenet`` package is installed, and CI does not install it.  These run anywhere:
each one pins a field the converter used to drop or misroute, which loaded a
checkpoint whose weights fit by name and then computed something else.
"""

from __future__ import annotations

import logging

import pytest
import torch

from oasr.layers.norm import BatchNormActivation, LayerNorm, LayerNormActivation
from oasr.models.conformer import ConformerEncoder
from oasr.models.conformer.config import ConformerEncoderConfig
from oasr.models.conformer.convert import build_config_from_wenet

#: The fields every case shares; WeNet's own defaults fill the rest.
_BASE = {"output_size": 64, "attention_heads": 2, "linear_units": 128, "num_blocks": 2}


def _encoder(**encoder_conf) -> ConformerEncoder:
    raw = {"input_dim": 80, "output_dim": 32, "encoder_conf": {**_BASE, **encoder_conf}}
    return ConformerEncoder(build_config_from_wenet(raw).encoder)


class TestNormConv:
    """The conv block's pre-norm follows ``layer_norm_type``, as in WeNet.

    It used to follow ``cnn_module_norm``, the norm *inside* the conv module.
    WeNet's default for that is ``batch_norm``, so any yaml that omits it built
    ``norm_conv`` as a ``BatchNorm1d``: the checkpoint's LayerNorm weights loaded
    into it and its running statistics stayed at their initial values.
    """

    @pytest.mark.parametrize("cnn_module_norm", [None, "batch_norm", "layer_norm"])
    def test_is_a_layer_norm_whatever_the_conv_module_uses(self, cnn_module_norm):
        conf = {} if cnn_module_norm is None else {"cnn_module_norm": cnn_module_norm}
        layer = _encoder(**conf).encoders[0]
        assert isinstance(layer.norm_conv, LayerNorm)
        inner = LayerNormActivation if cnn_module_norm == "layer_norm" else BatchNormActivation
        assert isinstance(layer.conv_module.norm_activation, inner)

    def test_holds_exactly_the_tensors_a_wenet_checkpoint_has(self):
        """WeNet's ``norm_conv`` is a LayerNorm: a weight and a bias, no statistics."""
        state = _encoder().state_dict()
        for i in range(_BASE["num_blocks"]):
            prefix = f"encoders.{i}.norm_conv."
            assert sorted(k[len(prefix) :] for k in state if k.startswith(prefix)) == [
                "bias",
                "weight",
            ]


class TestStructuralFields:
    """Fields the converter used to drop, falling back to OASR's defaults."""

    def test_norm_eps_reaches_every_block_norm(self):
        enc = _encoder(norm_eps=1e-3)
        layer = enc.encoders[0]
        for norm in (layer.norm_ff, layer.norm_mha, layer.norm_conv, layer.norm_final):
            assert norm.eps == pytest.approx(1e-3)
        assert enc.after_norm.eps == pytest.approx(1e-3)

    def test_conv_norm_eps_reaches_the_conv_module(self):
        layer = _encoder(conv_norm_eps=1e-3).encoders[0]
        assert layer.conv_module.norm_activation.eps == pytest.approx(1e-3)

    def test_final_norm(self):
        assert _encoder(final_norm=False).final_norm is False

    def test_conv_bias_off_removes_every_conv_bias(self):
        """Including ``pointwise_conv1``'s, which ignored ``bias`` altogether."""
        module = _encoder(conv_bias=False).encoders[0].conv_module
        names = {k for k, _ in module.named_parameters()}
        assert not {k for k in names if k.endswith("bias") and "norm" not in k}

    def test_conv_inner_factor_widens_the_depthwise_conv(self):
        """WeNet's depthwise conv runs at ``conv_inner_factor * channels // 2``;
        OASR's ran at ``channels``, which only agrees at the default factor 2."""
        module = _encoder(conv_inner_factor=4).encoders[0].conv_module
        assert module.pointwise_conv1.out_channels == 4 * 64
        assert module.depthwise_conv.channels == 2 * 64
        assert module.pointwise_conv2.in_channels == 2 * 64

    @pytest.mark.parametrize("key", ["layer_norm_type", "cnn_module_norm"])
    def test_rms_norm_is_refused(self, key):
        """WeNet's RMSNorm scales by ``1 + weight``; OASR's by ``weight``."""
        with pytest.raises(ValueError, match=f"{key}=rms_norm"):
            _encoder(**{key: "rms_norm"})


#: Every ``encoder_conf`` key at WeNet's default: main's ``ConformerEncoder``
#: signature, plus v1's ``concat_after``.  ``input_size`` and ``global_cmvn`` are
#: passed by WeNet itself, so a yaml cannot hold them.
_WENET_ENCODER_DEFAULTS = {
    "output_size": 256,
    "attention_heads": 4,
    "linear_units": 2048,
    "num_blocks": 6,
    "dropout_rate": 0.1,
    "positional_dropout_rate": 0.1,
    "attention_dropout_rate": 0.0,
    "input_layer": "conv2d",
    "pos_enc_layer_type": "rel_pos",
    "normalize_before": True,
    "concat_after": False,
    "static_chunk_size": 0,
    "use_dynamic_chunk": False,
    "use_dynamic_left_chunk": False,
    "positionwise_conv_kernel_size": 1,
    "macaron_style": True,
    "selfattention_layer_type": "rel_selfattn",
    "activation_type": "swish",
    "use_cnn_module": True,
    "cnn_module_kernel": 15,
    "causal": False,
    "cnn_module_norm": "batch_norm",
    "query_bias": True,
    "key_bias": True,
    "value_bias": True,
    "conv_bias": True,
    "gradient_checkpointing": False,
    "use_sdpa": False,
    "layer_norm_type": "layer_norm",
    "norm_eps": 1e-5,
    "n_kv_head": None,
    "head_dim": None,
    "mlp_type": "position_wise_feed_forward",
    "mlp_bias": True,
    "n_expert": 8,
    "n_expert_activated": 2,
    "conv_norm_eps": 1e-5,
    "conv_inner_factor": 2,
    "final_norm": True,
}

#: The same for ``decoder_conf``: main's ``BiTransformerDecoder`` signature, the
#: ``src_*_bias`` keys its ``TransformerDecoder`` adds, and v1's ``concat_after``.
_WENET_DECODER_DEFAULTS = {
    "attention_heads": 4,
    "linear_units": 2048,
    "num_blocks": 6,
    "r_num_blocks": 0,
    "dropout_rate": 0.1,
    "positional_dropout_rate": 0.1,
    "self_attention_dropout_rate": 0.0,
    "src_attention_dropout_rate": 0.0,
    "input_layer": "embed",
    "use_output_layer": True,
    "normalize_before": True,
    "src_attention": True,
    "query_bias": True,
    "key_bias": True,
    "value_bias": True,
    "src_query_bias": True,
    "src_key_bias": True,
    "src_value_bias": True,
    "activation_type": "relu",
    "gradient_checkpointing": False,
    "tie_word_embedding": False,
    "use_sdpa": False,
    "layer_norm_type": "layer_norm",
    "norm_eps": 1e-5,
    "n_kv_head": None,
    "head_dim": None,
    "mlp_type": "position_wise_feed_forward",
    "mlp_bias": True,
    "n_expert": 8,
    "n_expert_activated": 2,
    "concat_after": False,
}


def _config(encoder_conf=None, **raw):
    return build_config_from_wenet(
        {"input_dim": 80, "output_dim": 32, "encoder_conf": encoder_conf or {}, **raw}
    )


class TestWhatTheConverterAccepts:
    """Each key WeNet's encoder takes is read, inert, or checked against its value."""

    def test_every_wenet_default_spelled_out_builds_what_omitting_it_does(self):
        """So every key is known, and each omitted one defaults as in WeNet."""
        assert _config(_WENET_ENCODER_DEFAULTS).encoder == _config().encoder

    def test_the_decoder_too(self):
        bi = {"decoder": "bitransformer"}
        spelled = _config(decoder_conf=_WENET_DECODER_DEFAULTS, **bi).decoder
        assert spelled is not None
        assert spelled == _config(decoder_conf={}, **bi).decoder

    def test_num_blocks_defaults_to_wenets_six(self):
        """It defaulted to 12, which a checkpoint of WeNet's default cannot fill."""
        assert _config().encoder.num_blocks == 6

    def test_an_unknown_key_is_refused(self):
        """WeNet passes ``encoder_conf`` as keyword arguments, so an unknown key
        is from another encoder class or a newer WeNet."""
        with pytest.raises(ValueError, match="encoder_conf.stride_layer_idx"):
            _config({"stride_layer_idx": [3, 7]})

    def test_the_real_u2pp_yaml_is_accepted(self):
        """``20210610_u2pp_conformer_exp_librispeech``'s ``encoder_conf``."""
        cfg = _config(
            {
                "output_size": 256,
                "attention_heads": 4,
                "linear_units": 2048,
                "num_blocks": 12,
                "dropout_rate": 0.1,
                "positional_dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "input_layer": "conv2d",
                "normalize_before": True,
                "cnn_module_kernel": 15,
                "use_cnn_module": True,
                "activation_type": "swish",
                "pos_enc_layer_type": "rel_pos",
                "selfattention_layer_type": "rel_selfattn",
                "causal": True,
                "use_dynamic_chunk": True,
                "cnn_module_norm": "layer_norm",
                "use_dynamic_left_chunk": False,
            }
        )
        assert cfg.encoder.causal and cfg.encoder.num_blocks == 12


class TestTheEncoderClass:
    """Only WeNet's ``ConformerEncoder`` is built; ``encoder`` names the class."""

    @pytest.mark.parametrize(
        "encoder",
        ["transformer", "squeezeformer", "efficientConformer", "branchformer", "dual_conformer"],
    )
    def test_another_encoder_is_refused(self, encoder):
        with pytest.raises(ValueError, match=f"encoder='{encoder}'"):
            _config(encoder=encoder)

    @pytest.mark.parametrize("raw", [{}, {"encoder": "conformer"}], ids=["default", "named"])
    def test_conformer_is_built(self, raw):
        assert _config(**raw).encoder.output_size == 256


class TestPositionalEncoding:
    """Only WeNet's relative form is implemented.  WeNet picks the attention class
    by ``pos_enc_layer_type`` before v3 and by ``selfattention_layer_type`` from
    v3, so the weights of another form can load by name."""

    @pytest.mark.parametrize("value", ["abs_pos", "no_pos", "rope_pos", "rel_pos_firered"])
    def test_another_pos_enc_layer_type_is_refused(self, value):
        with pytest.raises(ValueError, match=f"pos_enc_layer_type={value}"):
            _config({"pos_enc_layer_type": value})

    @pytest.mark.parametrize("value", ["selfattn", "shaw_rel_selfattn", "grouped_rel_selfattn"])
    def test_another_selfattention_layer_type_is_refused(self, value):
        with pytest.raises(ValueError, match=f"selfattention_layer_type={value}"):
            _config({"selfattention_layer_type": value})

    def test_a_native_config_naming_another_is_refused_by_the_model(self):
        """The config field was never read, so the embed built ``rel_pos`` anyway."""
        with pytest.raises(NotImplementedError, match="pos_enc_layer_type=abs_pos"):
            ConformerEncoder(ConformerEncoderConfig(**{**_BASE, "pos_enc_layer_type": "abs_pos"}))


class TestValuesOnlySomeOfWhichAreImplemented:
    @pytest.mark.parametrize("activation", ["swish", "relu"])
    def test_an_activation_the_kernels_compute_exactly_is_accepted(self, activation):
        layer = _encoder(activation_type=activation).encoders[0]
        assert layer.conv_module.norm_activation.activation_name == activation

    @pytest.mark.parametrize("activation", ["gelu", "hardtanh", "tanh", "selu"])
    def test_another_is_refused(self, activation):
        with pytest.raises(ValueError, match=f"activation_type={activation}"):
            _config({"activation_type": activation})

    def test_the_conv_modules_gelu_is_not_wenets(self):
        """Why ``gelu`` is refused: WeNet's is ``nn.GELU()``, exact erf, and the
        convolution module's fused activation spells ``gelu`` as the tanh
        approximation, so it would have loaded and been ~1e-3 off."""
        from oasr.layers.norm import kernel_activation

        x = torch.linspace(-4, 4, 1001)
        diff = (kernel_activation(x, "gelu") - torch.nn.GELU()(x)).abs().max()
        assert diff > 1e-4

    @pytest.mark.parametrize(
        "key,value",
        [
            ("mlp_type", "gated"),
            ("mlp_type", "moe"),
            ("query_bias", False),
            ("key_bias", False),
            ("value_bias", False),
            ("mlp_bias", False),
            ("n_kv_head", 1),
            ("head_dim", 32),
        ],
    )
    def test_a_structure_this_conformer_does_not_build_is_refused(self, key, value):
        with pytest.raises(ValueError, match=f"encoder_conf.{key}={value}"):
            _config({key: value})


class TestAStaticChunk:
    """With a static chunk and no dynamic one, WeNet masks attention to that
    chunk even offline, and OASR's offline forward attends to everything."""

    def test_is_refused(self):
        with pytest.raises(ValueError, match="static_chunk_size=16"):
            _config({"static_chunk_size": 16})

    def test_beside_a_dynamic_chunk_is_accepted(self):
        """A dynamic chunk takes precedence, and decodes full-context offline."""
        assert _config({"static_chunk_size": 16, "use_dynamic_chunk": True}) is not None


class TestTheModelClass:
    """OASR decodes a WeNet checkpoint through its CTC head."""

    @pytest.mark.parametrize("model", ["transducer", "paraformer", "ctl_model", "whisper"])
    def test_another_model_class_is_refused(self, model):
        with pytest.raises(ValueError, match=f"'{model}' model"):
            _config(model=model)

    @pytest.mark.parametrize(
        "raw,model",
        [({"predictor": "rnn"}, "transducer"), ({"paraformer": True}, "paraformer")],
        ids=["transducer", "paraformer"],
    )
    def test_before_v3_the_class_is_named_by_its_keys(self, raw, model):
        with pytest.raises(ValueError, match=f"'{model}' model"):
            _config(**raw)

    @pytest.mark.parametrize("model", ["asr_model", "k2_model"])
    def test_a_ctc_model_is_accepted(self, model):
        assert _config(model=model) is not None

    def test_an_untrained_ctc_head_is_refused(self):
        with pytest.raises(ValueError, match="ctc_weight=0"):
            _config(model_conf={"ctc_weight": 0.0})


class TestAnUnsupportedDecoderIsDropped:
    """It costs attention rescoring only, so the checkpoint still loads for CTC."""

    @pytest.mark.parametrize(
        "decoder_conf",
        [
            {"activation_type": "swish"},
            {"normalize_before": False},
            {"norm_eps": "1e-6"},
            {"concat_after": True},
            {"mystery": 1},
        ],
        ids=["activation", "post_norm", "norm_eps", "concat_after", "unknown"],
    )
    def test_with_a_warning(self, decoder_conf, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = _config(decoder="bitransformer", decoder_conf=decoder_conf)
        assert cfg.decoder is None
        assert any("rescoring will be unavailable" in r.getMessage() for r in caplog.records)

    def test_a_yaml_float_without_a_dot_is_read_as_one(self):
        """YAML 1.1 reads ``1e-5`` as a string."""
        cfg = _config(decoder="bitransformer", decoder_conf={"norm_eps": "1e-5"})
        assert cfg.decoder is not None
