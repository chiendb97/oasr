# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""WeNet ``train.yaml`` → Conformer: the structure the converter builds.

``test_conformer.py`` checks numbers against upstream WeNet, but only where the
``wenet`` package is installed, and CI does not install it.  These run anywhere:
each one pins a field the converter used to drop or misroute, which loaded a
checkpoint whose weights fit by name and then computed something else.
"""

from __future__ import annotations

import pytest

from oasr.layers.norm import BatchNormActivation, LayerNorm, LayerNormActivation
from oasr.models.conformer import ConformerEncoder
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
