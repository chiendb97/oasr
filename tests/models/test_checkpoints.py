# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the model base abstractions + registry (no checkpoint / GPU needed)."""

import json
import logging
import sys
import types

import pytest
import torch
import yaml

from oasr.checkpoints import (
    ConvertedCheckpoint,
    DecodingDefaults,
    convert_checkpoint,
    is_native_checkpoint,
    read_native_config,
)
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
from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle


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


class TestExpectedUnusedMatching:
    """A normal WeNet checkpoint must not report "unrecognized tensors".

    WeNet builds ``concat_linear`` in every encoder/decoder layer unconditionally
    and only *uses* it when ``concat_after=True``, so a checkpoint trained with
    the default carries ~24 unused parameters.  ``expected_unused_prefixes``
    matched by key **prefix** only, which cannot express
    ``encoder.encoders.N.concat_linear.*`` — so they were reported as
    unrecognized, and the ``decoder.*`` ones fell through to the capability hint,
    which then announced that attention rescoring was unavailable **on a
    checkpoint whose decoder had loaded fine**.  A false capability warning is
    worse than a noisy one.
    """

    def _report(self, dropped):
        from oasr.models.base import LoadReport

        return LoadReport(mapped=["encoder.x"], dropped=list(dropped), missing=[])

    def _warn(self, caplog, dropped, expected=(), hints=None):
        import logging
        from types import SimpleNamespace

        from oasr.models import registry

        conv = SimpleNamespace(
            expected_unused_prefixes=tuple(expected),
            capability_drop_hints=dict(hints or {}),
        )
        with caplog.at_level(logging.WARNING):
            registry._log_load_report(self._report(dropped), conv, "conformer")  # noqa: SLF001
        return [r.getMessage() for r in caplog.records]

    def test_a_dotted_component_is_matched_not_just_a_prefix(self, caplog):
        dropped = [
            "encoder.encoders.0.concat_linear.weight",
            "decoder.left_decoder.decoders.0.concat_linear1.bias",
        ]
        msgs = self._warn(caplog, dropped, expected=("concat_linear",))
        assert not msgs, f"expected silence, got {msgs}"

    def test_a_genuinely_unknown_tensor_still_warns(self, caplog):
        msgs = self._warn(caplog, ["encoder.mystery.weight"], expected=("concat_linear",))
        assert any("unrecognized" in m for m in msgs)

    def test_a_prefix_declaration_still_works(self, caplog):
        """icefall's ``simple_am_proj`` is a real prefix — do not regress it."""
        msgs = self._warn(caplog, ["simple_am_proj.weight"], expected=("simple_am_proj",))
        assert not msgs, msgs

    def test_undeclared_concat_linear_would_trip_the_capability_hint(self, caplog):
        """The bug being fixed, pinned: without the declaration the hint lies."""
        msgs = self._warn(
            caplog,
            ["decoder.left_decoder.decoders.0.concat_linear1.bias"],
            expected=(),
            hints={"decoder.": "attention rescoring is unavailable"},
        )
        assert any("attention rescoring is unavailable" in m for m in msgs)

    def test_the_wenet_converter_declares_it(self):
        from oasr.models.conformer.convert import WenetConverter

        assert "concat_linear" in WenetConverter.expected_unused_prefixes


class TestConcatAfterIsRejected:
    """OASR's Conformer implements only the residual form.

    ``concat_after=True`` replaces each sub-layer's residual add with
    ``concat_linear(cat([x, out]))``.  Silently ignoring it loads a
    plausible-looking model that computes something else — the same
    raise-instead-of-guess rule the icefall shape-inference fallback follows.
    """

    def _yaml(self, **enc):
        return {"input_dim": 80, "output_dim": 100, "encoder_conf": dict(enc)}

    def test_encoder_concat_after_raises(self):
        from oasr.models.conformer.convert import build_config_from_wenet

        with pytest.raises(ValueError, match="concat_after"):
            build_config_from_wenet(self._yaml(concat_after=True))

    def test_decoder_concat_after_raises(self):
        from oasr.models.conformer.convert import build_config_from_wenet

        raw = self._yaml()
        raw["decoder_conf"] = {"concat_after": True}
        with pytest.raises(ValueError, match="concat_after"):
            build_config_from_wenet(raw)

    def test_the_default_is_accepted(self):
        from oasr.models.conformer.convert import build_config_from_wenet

        cfg = build_config_from_wenet(self._yaml(output_size=256))
        assert cfg.encoder.output_size == 256

    def test_explicit_false_is_accepted(self):
        from oasr.models.conformer.convert import build_config_from_wenet

        assert build_config_from_wenet(self._yaml(concat_after=False)) is not None


# ---------------------------------------------------------------------------
# The native format and the converter bundle
#
# Built from a synthetic WeNet directory, so this half needs no real
# checkpoint -- which is the model the whole group should follow.
# ---------------------------------------------------------------------------


UNITS = "<blank> 0\n<unk> 1\n▁he 2\n▁hel 3\nlo 4\n▁wor 5\nld 6\n<sos/eos> 29\n"


def _tiny_model_config() -> ConformerModelConfig:
    enc = ConformerEncoderConfig(
        input_size=80,
        output_size=64,
        num_blocks=2,
        attention_heads=4,
        linear_units=128,
        cnn_module_kernel=15,
        embed_layer_norm=False,
    )
    return ConformerModelConfig(encoder=enc, vocab_size=32)


@pytest.fixture(scope="module")
def wenet_dir(tmp_path_factory):
    """A synthetic-but-complete WeNet experiment dir (tiny random Conformer)."""
    d = tmp_path_factory.mktemp("wenet_ckpt")
    (d / "train.yaml").write_text(
        yaml.safe_dump(
            {
                "input_dim": 80,
                "output_dim": 30,
                "encoder_conf": {
                    "output_size": 64,
                    "num_blocks": 2,
                    "attention_heads": 4,
                    "linear_units": 128,
                    "cnn_module_kernel": 15,
                },
                "dataset_conf": {
                    "fbank_conf": {
                        "num_mel_bins": 80,
                        "frame_shift": 10,
                        "frame_length": 25,
                        "dither": 0.1,
                    }
                },
            }
        )
    )
    (d / "units.txt").write_text(UNITS, encoding="utf-8")
    (d / "global_cmvn").write_text(
        json.dumps({"mean_stat": [1.0] * 80, "var_stat": [2.0] * 80, "frame_num": 2})
    )

    from oasr.layers.norm import GlobalCMVN

    torch.manual_seed(0)
    m = ConformerModel.from_config(
        _tiny_model_config(), global_cmvn=GlobalCMVN(torch.zeros(80), torch.ones(80))
    )
    sd = {k: v for k, v in m.state_dict().items() if not k.endswith("pos_enc.pe")}
    # Checkpoint vocab (30) below the model's 8-aligned 32, like real WeNet dirs.
    sd["ctc.ctc_lo.weight"] = sd["ctc.ctc_lo.weight"][:30].clone()
    sd["ctc.ctc_lo.bias"] = sd["ctc.ctc_lo.bias"][:30].clone()
    # A U2++ attention-decoder branch the CTC model must report as dropped.
    sd["decoder.left_decoder.embed.weight"] = torch.randn(30, 64)
    torch.save(sd, d / "final.pt")
    return d


@pytest.fixture(scope="module")
def native_dir(wenet_dir, tmp_path_factory):
    pytest.importorskip("safetensors")
    from oasr.checkpoints.convert import convert_to_native

    out = tmp_path_factory.mktemp("native_ckpt") / "bundle"
    convert_to_native(str(wenet_dir), str(out))
    return out


class TestBundleEmission:
    def test_wenet_bundle_specs(self, wenet_dir):
        arch, bundle = load_checkpoint_bundle(wenet_dir)
        assert arch == "conformer"
        assert bundle.source_format == "wenet"
        assert bundle.tokenizer is not None
        assert bundle.tokenizer.kind == "symbol_table"
        assert bundle.tokenizer.files["table"].endswith("units.txt")
        f = bundle.features
        assert f is not None
        assert (f.kind, f.feature_dim, f.sample_rate) == ("kaldi_fbank", 80, 16000)
        assert f.dither == 0.0  # forced off at inference regardless of train.yaml
        assert f.normalize == "global_cmvn"
        d = bundle.decoding
        assert (d.default_decode_type, d.blank_id, d.unk_id) == ("ctc", 0, 1)
        assert d.sos_id == d.eos_id == 29  # output_dim - 1

    def test_feature_spec_to_config_matches_engine_default(self, wenet_dir):
        """The standard WeNet spec materializes exactly the old engine default."""
        from oasr.features import FeatureConfig

        _, bundle = load_checkpoint_bundle(wenet_dir)
        assert bundle.features.to_feature_config() == FeatureConfig(dither=0.0)
        assert bundle.features.mismatches(FeatureConfig(dither=0.0)) == []
        diffs = bundle.features.mismatches(FeatureConfig(num_mel_bins=40, dither=0.0))
        assert len(diffs) == 1 and diffs[0].startswith("feature_dim")

    def test_legacy_converter_adapter(self, wenet_dir):
        """A 4-method converter (no convert()) goes through the sniffing adapter."""

        class Legacy4Method:
            def detect(self, ckpt_dir):
                return True

            def build_config(self, ckpt_dir):
                return _tiny_model_config()

            def build_aux(self, ckpt_dir):
                return {}

            def load_state_dict(self, ckpt_dir, checkpoint_name, map_location):
                return {"w": torch.zeros(1)}

        bundle = convert_checkpoint("conformer", Legacy4Method(), wenet_dir)
        assert bundle.source_format == "legacy"
        assert bundle.tokenizer is not None  # sniffed units.txt
        assert bundle.tokenizer.kind == "symbol_table"
        assert bundle.features is None  # legacy: engine-side default
        assert isinstance(bundle.decoding, DecodingDefaults)


class TestLoadReport:
    def test_dropped_decoder_branch_reported_and_warned(self, wenet_dir, caplog):
        arch, bundle = load_checkpoint_bundle(wenet_dir)
        with caplog.at_level(logging.WARNING, logger="oasr.models.registry"):
            model, _, report = instantiate_from_bundle(arch, bundle)
        assert report is not None
        assert any(k.startswith("decoder.") for k in report.dropped)
        assert not report.missing
        assert len(report.mapped) > 50
        joined = " ".join(r.message for r in caplog.records)
        assert "decoder.*" in joined and "rescoring" in joined

    def test_expected_prefixes_stay_silent(self, wenet_dir, caplog):
        """Keys under expected_unused_prefixes drop without a warning."""
        arch, bundle = load_checkpoint_bundle(wenet_dir)
        entry_converter = type(
            "C",
            (),
            {"expected_unused_prefixes": ("decoder.",), "capability_drop_hints": {}},
        )()
        from oasr.models.registry import _log_load_report

        model, _, report = instantiate_from_bundle(arch, bundle)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="oasr.models.registry"):
            _log_load_report(report, entry_converter, arch)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestNativeFormat:
    def test_layout_and_metadata(self, native_dir):
        assert is_native_checkpoint(native_dir)
        cfg = read_native_config(native_dir)
        assert cfg["architecture"] == "conformer"
        assert cfg["tokenizer"]["files"]["table"] == "tokenizer/units.txt"
        assert cfg["aux"]["global_cmvn"]["buffers"]["mean"] == [80]
        assert (native_dir / "model.safetensors").exists()
        assert (native_dir / "tokenizer" / "units.txt").exists()

    def test_round_trip_state_dict_identical(self, wenet_dir, native_dir):
        arch1, b1 = load_checkpoint_bundle(wenet_dir)
        m1, _, _ = instantiate_from_bundle(arch1, b1)
        arch2, b2 = load_checkpoint_bundle(native_dir)
        assert (arch2, b2.source_format) == ("conformer", "native")
        m2, _, _ = instantiate_from_bundle(arch2, b2)
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        assert set(sd1) == set(sd2)
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k]), k

    def test_round_trip_specs_and_tokenizer(self, wenet_dir, native_dir):
        from oasr.tokenizers import build_tokenizer

        _, b1 = load_checkpoint_bundle(wenet_dir)
        _, b2 = load_checkpoint_bundle(native_dir)
        assert b2.features == b1.features
        assert b2.decoding == b1.decoding
        t1, t2 = build_tokenizer(b1.tokenizer), build_tokenizer(b2.tokenizer)
        ids = [0, 2, 4, 1, 5, 6]
        assert t1.decode(ids) == t2.decode(ids) == "lo world"

    def test_unsupported_format_version(self, native_dir, tmp_path):
        import shutil

        bad = tmp_path / "bad"
        shutil.copytree(native_dir, bad)
        cfg = json.loads((bad / "oasr_config.json").read_text())
        cfg["format_version"] = 999
        (bad / "oasr_config.json").write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="format_version"):
            load_checkpoint_bundle(bad)

    def test_native_mismatched_weights_raise(self, native_dir):
        from oasr.checkpoints import load_native, load_native_weights

        bundle = load_native(native_dir)
        model = ConformerModel.from_config(_tiny_model_config())  # no CMVN module
        with pytest.raises(RuntimeError, match="does not match"):
            load_native_weights(model, dict(bundle.state_dict))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_round_trip_forward_identical_gpu(self, wenet_dir, native_dir):
        arch1, b1 = load_checkpoint_bundle(wenet_dir)
        m1, _, _ = instantiate_from_bundle(arch1, b1)
        arch2, b2 = load_checkpoint_bundle(native_dir)
        m2, _, _ = instantiate_from_bundle(arch2, b2)
        m1 = m1.cuda().to(torch.bfloat16)
        m2 = m2.cuda().to(torch.bfloat16)
        x = torch.randn(2, 67, 80, device="cuda", dtype=torch.bfloat16)
        lens = torch.tensor([67, 50], dtype=torch.int32, device="cuda")
        with torch.no_grad():
            o1, l1 = m1.forward_offline(x, lens)
            o2, l2 = m2.forward_offline(x, lens)
        assert torch.equal(o1, o2) and torch.equal(l1, l2)


@pytest.fixture(scope="module")
def wenet_hybrid_dir(tmp_path_factory):
    """A synthetic U2++-style dir: CTC + bitransformer decoder branch."""
    from oasr.models.decoders import TransformerDecoderConfig

    d = tmp_path_factory.mktemp("wenet_hybrid_ckpt")
    (d / "train.yaml").write_text(
        yaml.safe_dump(
            {
                "input_dim": 80,
                "output_dim": 30,
                "encoder_conf": {
                    "output_size": 64,
                    "num_blocks": 2,
                    "attention_heads": 4,
                    "linear_units": 128,
                    "cnn_module_kernel": 15,
                },
                "decoder": "bitransformer",
                "decoder_conf": {
                    "attention_heads": 2,
                    "linear_units": 64,
                    "num_blocks": 2,
                    "r_num_blocks": 1,
                },
                "model_conf": {"ctc_weight": 0.3, "reverse_weight": 0.3},
            }
        )
    )
    (d / "units.txt").write_text(UNITS, encoding="utf-8")

    cfg = _tiny_model_config()
    cfg.decoder = TransformerDecoderConfig(
        vocab_size=30,
        encoder_output_size=64,
        attention_heads=2,
        linear_units=64,
        num_blocks=2,
        r_num_blocks=1,
        sos_id=29,
        eos_id=29,
        reverse_weight=0.3,
    )
    torch.manual_seed(1)
    m = ConformerModel.from_config(cfg)
    sd = {k: v for k, v in m.state_dict().items() if not k.endswith("pos_enc.pe")}
    sd["ctc.ctc_lo.weight"] = sd["ctc.ctc_lo.weight"][:30].clone()
    sd["ctc.ctc_lo.bias"] = sd["ctc.ctc_lo.bias"][:30].clone()
    torch.save(sd, d / "final.pt")
    return d


class TestHybridDecoderBranch:
    def test_converter_builds_decoder_config(self, wenet_hybrid_dir):
        arch, bundle = load_checkpoint_bundle(wenet_hybrid_dir)
        dec = bundle.model_config.decoder
        assert dec is not None
        assert (dec.num_blocks, dec.r_num_blocks) == (2, 1)
        assert (dec.vocab_size, dec.sos_id, dec.eos_id) == (30, 29, 29)
        assert dec.reverse_weight == pytest.approx(0.3)

    def test_decoder_branch_loads_without_drops(self, wenet_hybrid_dir, caplog):
        arch, bundle = load_checkpoint_bundle(wenet_hybrid_dir)
        with caplog.at_level(logging.WARNING, logger="oasr.models.registry"):
            model, _, report = instantiate_from_bundle(arch, bundle)
        assert not [k for k in report.dropped if k.startswith("decoder.")]
        assert not report.missing
        assert sorted(model.capabilities) == ["ctc", "ctc_aed_rescoring"]
        assert model.default_decode_type == "ctc"
        assert "rescoring" not in " ".join(r.message for r in caplog.records)

    def test_native_round_trip_preserves_decoder(self, wenet_hybrid_dir, tmp_path_factory):
        pytest.importorskip("safetensors")
        from oasr.checkpoints.convert import convert_to_native

        out = tmp_path_factory.mktemp("native_hybrid") / "bundle"
        convert_to_native(str(wenet_hybrid_dir), str(out))

        arch1, b1 = load_checkpoint_bundle(wenet_hybrid_dir)
        m1, _, _ = instantiate_from_bundle(arch1, b1)
        arch2, b2 = load_checkpoint_bundle(out)
        assert b2.source_format == "native"
        m2, cfg2, _ = instantiate_from_bundle(arch2, b2)
        assert cfg2.decoder is not None and cfg2.decoder.r_num_blocks == 1
        assert sorted(m2.capabilities) == ["ctc", "ctc_aed_rescoring"]
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        assert set(sd1) == set(sd2)
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k]), k


class TestConvertedCheckpointDataclass:
    def test_defaults(self):
        b = ConvertedCheckpoint(architecture="conformer", model_config=_tiny_model_config())
        assert b.source_format == "legacy"
        assert b.tokenizer is None and b.features is None
        assert b.decoding == DecodingDefaults()

    def test_decoding_defaults_round_trip(self):
        d = DecodingDefaults(default_decode_type="ctc", blank_id=0, sos_id=29, eos_id=29)
        assert DecodingDefaults.from_dict(d.to_dict()) == d


# ---------------------------------------------------------------------------
# ``from_pretrained`` -- local dir vs Hub snapshot
# ---------------------------------------------------------------------------


def test_resolve_local_dir(tmp_path):
    from oasr.models.loaders import _resolve_to_local_dir

    assert _resolve_to_local_dir(tmp_path) == str(tmp_path)


def test_from_pretrained_local_passthrough(tmp_path, monkeypatch):
    import oasr.models.loaders as L

    seen = {}

    def fake_build(local_dir, checkpoint_name, device=None, dtype=None, architecture=None):
        seen["args"] = (str(local_dir), checkpoint_name, device, dtype)
        return ("MODEL", "CONFIG")

    monkeypatch.setattr(L, "build_model_from_checkpoint", fake_build)
    out = L.from_pretrained(tmp_path, checkpoint_name="final.pt", device="cpu")
    assert out == ("MODEL", "CONFIG")
    assert seen["args"][0] == str(tmp_path)
    assert seen["args"][1] == "final.pt"


def test_from_pretrained_hf_download(tmp_path, monkeypatch):
    """A non-local id is resolved via huggingface_hub.snapshot_download."""
    captured = {}

    hub = types.ModuleType("huggingface_hub")

    def fake_snapshot(repo_id, revision=None, cache_dir=None, allow_patterns=None):
        captured.update(repo_id=repo_id, revision=revision)
        return str(tmp_path)

    hub.snapshot_download = fake_snapshot
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    import oasr.models.loaders as L

    monkeypatch.setattr(
        L,
        "build_model_from_checkpoint",
        lambda d, n, device=None, dtype=None, architecture=None: ("M", "C"),
    )
    out = L.from_pretrained("some-org/some-asr-model", revision="v1")
    assert out == ("M", "C")
    assert captured == {"repo_id": "some-org/some-asr-model", "revision": "v1"}


def test_top_level_and_classmethod_exports():
    import oasr
    from oasr.models import from_pretrained as module_fp
    from oasr.models.base import BaseAsrModel

    assert oasr.from_pretrained is module_fp
    assert callable(module_fp)
    # Classmethod exists and is bound to the class (auto-detect loader).
    assert hasattr(BaseAsrModel, "from_pretrained")
