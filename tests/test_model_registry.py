# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the model base abstractions + registry (no checkpoint / GPU needed)."""

import pytest

from oasr.models import (
    BaseAsrModel,
    BaseEncoder,
    BaseHead,
    CacheSpec,
    ConformerModel,
    ConformerModelConfig,
    CTCHead,
    get_model_entry,
    list_models,
    register_model,
    resolve_architecture,
)
from oasr.models.conformer import CTC, WenetConverter, load_wenet_checkpoint
from oasr.models.conformer.config import ConformerEncoderConfig


def _tiny_config() -> ConformerModelConfig:
    enc = ConformerEncoderConfig(
        output_size=64, num_blocks=2, attention_heads=4, linear_units=128, cnn_module_kernel=15
    )
    return ConformerModelConfig(encoder=enc, vocab_size=32)


class TestRegistry:
    def test_conformer_registered(self):
        assert "conformer" in list_models()
        entry = get_model_entry("conformer")
        assert entry.model_cls is ConformerModel
        assert entry.config_cls is ConformerModelConfig
        assert isinstance(entry.converter, WenetConverter)

    def test_unknown_arch_raises(self):
        with pytest.raises(KeyError):
            get_model_entry("does-not-exist")

    def test_resolve_architecture(self, tmp_path):
        # WeNet dirs are identified by train.yaml; default fallback is conformer.
        (tmp_path / "train.yaml").write_text("encoder: conformer\n")
        assert resolve_architecture(tmp_path) == "conformer"
        with pytest.warns(DeprecationWarning, match="falling back"):
            assert resolve_architecture(tmp_path / "missing") == "conformer"  # fallback

    def test_resolve_architecture_explicit_override(self, tmp_path):
        # The override wins with no sniffing, and is validated eagerly.
        assert resolve_architecture(tmp_path, architecture="zipformer") == "zipformer"
        with pytest.raises(KeyError):
            resolve_architecture(tmp_path, architecture="does-not-exist")

    def test_resolve_architecture_ambiguity_raises(self, tmp_path):
        (tmp_path / "train.yaml").write_text("encoder: conformer\n")

        class AlwaysDetects:
            def detect(self, ckpt_dir):
                return True

            def build_config(self, ckpt_dir):
                raise NotImplementedError

            def build_aux(self, ckpt_dir):
                return {}

            def load_state_dict(self, ckpt_dir, checkpoint_name, map_location):
                return {}

        from oasr.models import registry as R

        register_model(
            "greedy-test-arch",
            model_cls=ConformerModel,
            config_cls=ConformerModelConfig,
            converter=AlwaysDetects(),
        )
        try:
            with pytest.raises(ValueError, match="Ambiguous checkpoint format"):
                resolve_architecture(tmp_path)
        finally:
            del R._REGISTRY["greedy-test-arch"]

    def test_icefall_detect_tightened(self, tmp_path):
        import torch

        # A bare arbitrarily-named .pt no longer detects as icefall.
        loose = tmp_path / "loose"
        loose.mkdir()
        torch.save({}, loose / "whatever.pt")
        with pytest.warns(DeprecationWarning):
            resolve_architecture(loose)

        # Conventional icefall layouts still detect.
        named = tmp_path / "named"
        named.mkdir()
        torch.save({}, named / "pretrained.pt")
        assert resolve_architecture(named) == "zipformer"

        exp = tmp_path / "exp_layout"
        (exp / "exp").mkdir(parents=True)
        torch.save({}, exp / "exp" / "epoch-30.pt")
        assert resolve_architecture(exp) == "zipformer"

        tokens = tmp_path / "tokens_layout"
        tokens.mkdir()
        (tokens / "tokens.txt").write_text("<blk> 0\n")
        assert resolve_architecture(tokens) == "zipformer"

    def test_register_is_idempotent(self):
        before = get_model_entry("conformer")
        register_model(
            "conformer",
            model_cls=ConformerModel,
            config_cls=ConformerModelConfig,
            converter=WenetConverter(),
        )
        after = get_model_entry("conformer")
        assert after.model_cls is before.model_cls

    def test_back_compat_aliases(self):
        assert CTC is CTCHead
        assert callable(load_wenet_checkpoint)


class TestBaseContract:
    def test_abcs_not_instantiable(self):
        for cls in (BaseEncoder, BaseHead):
            with pytest.raises(TypeError):
                cls()

    def test_base_model_abstract_methods(self):
        assert BaseAsrModel.__abstractmethods__ == frozenset({"from_config", "load_weights"})

    def test_decode_type_and_cache_spec(self):
        cfg = _tiny_config()
        model = ConformerModel.from_config(cfg)
        assert model.decode_type == "ctc"
        assert isinstance(model.cache_spec, CacheSpec)
        # Live-model cache spec must match the config-derived one (the engine
        # builds caches from the live model; tests build from the config).
        assert model.cache_spec == cfg.cache_spec
        assert model.cache_spec == CacheSpec(
            num_layers=2, n_kv_head=4, head_dim=16, hidden_dim=64, conv_kernel_size=15
        )

    def test_head_aliases_ctc(self):
        model = ConformerModel.from_config(_tiny_config())
        assert model.head is model.ctc
        assert isinstance(model.head, CTCHead)

    def test_conv_kernel_size_zeroed_without_cnn(self):
        enc = ConformerEncoderConfig(output_size=64, num_blocks=1, use_cnn_module=False)
        cfg = ConformerModelConfig(encoder=enc, vocab_size=16)
        assert cfg.cache_spec.conv_kernel_size == 1
        assert ConformerModel.from_config(cfg).cache_spec.conv_kernel_size == 1


class TestLoadWeights:
    def test_load_weights_pads_ctc_vocab(self):
        """load_weights zero-pads an unpadded-vocab CTC head up to the model's vocab."""
        model = ConformerModel.from_config(_tiny_config())  # vocab=32 (8-aligned)
        sd = dict(model.state_dict())
        # Simulate a checkpoint whose CTC vocab (30) is smaller than the model's.
        sd["ctc.ctc_lo.weight"] = sd["ctc.ctc_lo.weight"][:30].clone()
        sd["ctc.ctc_lo.bias"] = sd["ctc.ctc_lo.bias"][:30].clone()

        model.load_weights(sd)  # must not raise

        assert model.ctc.ctc_lo.weight.shape[0] == 32
        assert model.ctc.ctc_lo.bias.shape[0] == 32
        # The padded rows are zero-filled.
        assert model.ctc.ctc_lo.weight[30:].abs().sum().item() == 0.0

    def test_load_weights_returns_report(self):
        """Every checkpoint key is accounted for: mapped or dropped, never silent."""
        import torch

        from oasr.models import LoadReport

        model = ConformerModel.from_config(_tiny_config())
        sd = dict(model.state_dict())
        sd["decoder.some.branch.weight"] = torch.zeros(1)

        report = model.load_weights(sd)
        assert isinstance(report, LoadReport)
        assert report.dropped == ["decoder.some.branch.weight"]
        assert not report.missing
        assert set(report.mapped) == {k for k in sd if k != "decoder.some.branch.weight"}
        assert "dropped" in report.summary()
