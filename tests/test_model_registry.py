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
        # An unrecognized dir is now refused instead of being guessed as conformer:
        # the guess used to surface as a shape error deep inside weight loading.
        with pytest.raises(ValueError, match="No registered converter recognized"):
            resolve_architecture(tmp_path / "missing")

    def test_resolve_architecture_explicit_override(self, tmp_path):
        # The override wins with no sniffing, and is validated eagerly.
        assert resolve_architecture(tmp_path, architecture="zipformer") == "zipformer"
        with pytest.raises(KeyError):
            resolve_architecture(tmp_path, architecture="does-not-exist")

    @staticmethod
    def _always_detects(specificity=None):
        """A converter that claims every directory, at a chosen specificity."""

        class AlwaysDetects:
            def detect(self, ckpt_dir):
                return True

            def build_config(self, ckpt_dir):
                raise NotImplementedError

            def build_aux(self, ckpt_dir):
                return {}

            def load_state_dict(self, ckpt_dir, checkpoint_name, map_location):
                return {}

        if specificity is not None:
            AlwaysDetects.detect_specificity = specificity
        return AlwaysDetects()

    def _with_greedy(self, converter):
        """Register ``converter`` as a throwaway architecture, then clean up."""
        from contextlib import contextmanager

        from oasr.models import registry as R

        @contextmanager
        def _ctx():
            register_model(
                "greedy-test-arch",
                model_cls=ConformerModel,
                config_cls=ConformerModelConfig,
                converter=converter,
            )
            try:
                yield
            finally:
                del R._REGISTRY["greedy-test-arch"]

        return _ctx()

    def test_resolve_architecture_ambiguity_raises(self, tmp_path):
        """A **tie** at the top specificity is still an error.

        Note what changed: two matches at *different* specificities are no longer
        ambiguous (see the next test), because that is the normal case — a FunASR
        dir also satisfies icefall's filename rule.  Only an unresolvable tie is.
        """
        from oasr.models.registry import DETECT_NAMED_CONFIG

        (tmp_path / "train.yaml").write_text("encoder: conformer\n")  # conformer matches
        with self._with_greedy(self._always_detects(DETECT_NAMED_CONFIG)):
            with pytest.raises(ValueError, match="Ambiguous checkpoint format"):
                resolve_architecture(tmp_path)

    def test_more_specific_detect_wins_over_a_weaker_one(self, tmp_path):
        """Ranking replaced the negative guards that used to live inside
        ``IcefallConverter.detect`` (``return False`` if ``train.yaml`` exists)."""
        from oasr.models.registry import DETECT_ASSET_LAYOUT

        (tmp_path / "train.yaml").write_text("encoder: conformer\n")
        with self._with_greedy(self._always_detects(DETECT_ASSET_LAYOUT)):
            # conformer declares DETECT_NAMED_CONFIG (20) > 10 — no ambiguity.
            assert resolve_architecture(tmp_path) == "conformer"

    def test_a_converter_declaring_nothing_gets_the_weakest_level(self, tmp_path):
        (tmp_path / "train.yaml").write_text("encoder: conformer\n")
        with self._with_greedy(self._always_detects()):  # no detect_specificity
            assert resolve_architecture(tmp_path) == "conformer"

    def test_specificity_levels_are_ordered(self):
        from oasr.models.registry import (
            DETECT_ASSET_LAYOUT,
            DETECT_KEYED_VALUE,
            DETECT_NAMED_CONFIG,
        )

        assert DETECT_ASSET_LAYOUT < DETECT_NAMED_CONFIG < DETECT_KEYED_VALUE

    def test_every_builtin_converter_declares_its_specificity(self):
        """A converter that forgets to declare falls back to the weakest level,
        which would silently lose to anything — make the omission visible."""
        from oasr.models.registry import get_model_entry, list_models

        for arch in list_models():
            converter = get_model_entry(arch).converter
            assert hasattr(converter, "detect_specificity"), (
                f"{arch}'s converter does not declare detect_specificity; it would "
                "default to the weakest level and lose every contested directory"
            )

    def test_icefall_detect_declares_no_negative_guards(self, tmp_path):
        """The regression this replaces: FunASR / WeNet markers hardcoded inside
        *icefall's* detector, so adding a format meant editing an unrelated file.

        Both dirs still resolve correctly — now because the other converter's claim
        is more specific, not because ``IcefallConverter`` knows about it."""
        import torch

        from oasr.models.registry import get_model_entry

        icefall = get_model_entry("zipformer").converter

        funasr = tmp_path / "funasr"
        funasr.mkdir()
        (funasr / "config.yaml").write_text("model: Paraformer\n")
        torch.save({}, funasr / "model.pt")  # also an icefall-conventional name
        assert icefall.detect(funasr) is True, "icefall's own rule still matches"
        assert resolve_architecture(funasr) == "paraformer", "but the specific one wins"

        wenet = tmp_path / "wenet"
        wenet.mkdir()
        (wenet / "train.yaml").write_text("encoder: conformer\n")
        (wenet / "tokens.txt").write_text("<blank> 0\n")  # an icefall asset marker
        assert icefall.detect(wenet) is True
        assert resolve_architecture(wenet) == "conformer"

    def test_icefall_detect_tightened(self, tmp_path):
        import torch

        # A bare arbitrarily-named .pt no longer detects as icefall.
        loose = tmp_path / "loose"
        loose.mkdir()
        torch.save({}, loose / "whatever.pt")
        with pytest.raises(ValueError, match="No registered converter recognized"):
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


class TestModelDiscovery:
    """Adding an architecture must not mean editing the registry (N8).

    The registry used to carry a hand-written six-branch if-chain, and
    ``oasr/models/__init__`` a separate hand-written import list — which had
    already drifted: ``__init__`` exported conformer / transducer / zipformer
    only, while the registry knew all six.  One list plus entry-point discovery
    removes both the drift and the edit.
    """

    def test_builtin_list_matches_the_registry(self):
        from oasr.models.registry import _BUILTIN_PACKAGES, list_models

        assert set(_BUILTIN_PACKAGES) == set(list_models())

    def test_every_builtin_package_is_exported(self):
        """``oasr.models`` must re-export each package, not just the old three."""
        import oasr.models as m
        from oasr.models.registry import _BUILTIN_PACKAGES

        for pkg in _BUILTIN_PACKAGES:
            mod = __import__(f"oasr.models.{pkg}", fromlist=["*"])
            exported = [n for n in getattr(mod, "__all__", []) if n.endswith("Model")]
            assert exported, f"{pkg} exports no *Model name"
            for name in exported:
                assert name in m.__all__, f"{pkg}: {name} missing from oasr.models.__all__"

    def test_entry_point_plugins_are_loaded(self, monkeypatch):
        """A third-party architecture registers via an entry point, no edit here."""
        from oasr.models import registry

        loaded = []

        class _EP:
            name = "plugin_arch"

            def load(self):
                loaded.append(self.name)

        monkeypatch.setattr(registry, "_builtins_loaded", False)
        monkeypatch.setattr(registry, "_load_entry_point_models", lambda: _EP().load())
        registry.list_models()
        assert loaded == ["plugin_arch"]

    def test_a_broken_plugin_does_not_break_the_builtins(self, monkeypatch, caplog):
        """An incompatible plugin must degrade to a warning, not an import error.

        Otherwise one bad third-party package makes the whole framework
        unusable, including the architectures that ship in-tree.
        """
        import logging

        from oasr.models import registry

        class _BadEP:
            name = "broken"

            def load(self):
                raise ImportError("boom")

        # Drive the loader directly with a failing entry point.
        import importlib.metadata as md

        monkeypatch.setattr(md, "entry_points", lambda **kw: [_BadEP()])
        with caplog.at_level(logging.WARNING):
            registry._load_entry_point_models()
        assert any("broken" in r.getMessage() for r in caplog.records)
        # The built-ins are still reachable.
        assert "conformer" in registry.list_models()
