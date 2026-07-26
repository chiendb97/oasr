# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ConvertedCheckpoint bundle + native format (CPU; one GPU parity test).

Covers the Phase-0 seams of the multi-paradigm plan
(``.artifacts/multi_paradigm.md``): converter-emitted tokenizer / feature /
decoding specs, the legacy 4-method converter adapter, LoadReport accounting
(no silent weight drops), and the ``convert → native → load`` round trip being
identical to the direct load.
"""

from __future__ import annotations

import json
import logging

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
from oasr.models import ConformerModel
from oasr.models.conformer.config import ConformerEncoderConfig, ConformerModelConfig
from oasr.models.registry import instantiate_from_bundle, load_checkpoint_bundle

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
