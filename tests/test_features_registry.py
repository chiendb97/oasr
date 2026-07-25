# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the feature-extractor axis (F1).

Features were the one extension axis without a registry: dispatch was
``if feature_type == ...`` in the shared ``InputProcessor``, in the CUDA-graph
cache, and in ``FeatureConfig`` itself — so a new frontend meant editing the
engine.  These tests pin the three properties that make the axis an actual seam:

1. every registered kind resolves, and an unregistered one fails with a message
   naming the registered ones;
2. the declared properties (``supports_streaming``, ``window_seconds_attr``) agree
   with what the rest of the engine does with them — the streaming path refuses a
   non-streamable frontend, and the batching cost model sees a constant per-row
   cost for a fixed-window one;
3. registering a frontend from **outside** the package needs no edit inside it.

(3) is the whole point, so it is a real test rather than a comment: it registers a
throwaway extractor and drives a ``FeatureConfig`` + the offline batch path with it.
"""

from __future__ import annotations

import pytest
import torch

from oasr.features import (
    ExtractorSpec,
    FeatureConfig,
    build_extractor,
    list_extractors,
    register_extractor,
)
from oasr.features import registry as registry_mod


@pytest.fixture
def isolated_registry(monkeypatch):
    """A copy of the registry, so a test's registration cannot leak."""
    registry_mod._ensure_builtins()
    monkeypatch.setattr(registry_mod, "_REGISTRY", dict(registry_mod._REGISTRY))
    return registry_mod._REGISTRY


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class TestResolution:
    def test_builtin_kinds_registered(self):
        assert set(list_extractors()) >= {"fbank", "mfcc", "whisper_logmel"}

    @pytest.mark.parametrize("kind", ["fbank", "mfcc", "whisper_logmel"])
    def test_every_kind_resolves(self, kind):
        cfg = FeatureConfig(feature_type=kind, num_mel_bins=128 if "whisper" in kind else 80)
        spec = build_extractor(cfg)
        assert spec.kind == kind
        assert callable(spec)

    def test_unregistered_kind_names_the_alternatives(self, isolated_registry):
        # Bypass FeatureConfig's own validation (which now also consults the
        # registry) to test build_extractor's message directly.
        cfg = FeatureConfig(num_mel_bins=80)
        object.__setattr__(cfg, "feature_type", "wav2vec_raw")
        with pytest.raises(NotImplementedError, match=r"wav2vec_raw.*Registered:.*fbank"):
            build_extractor(cfg)

    def test_config_validates_against_the_registry(self):
        """A ``feature_type`` is legal exactly when an extractor is registered."""
        with pytest.raises(ValueError, match=r"feature_type must be one of.*raw"):
            FeatureConfig(feature_type="raw")

    def test_registration_makes_a_new_kind_legal(self, isolated_registry):
        register_extractor(ExtractorSpec(kind="raw", fn=lambda w, ln, c: (w, ln)))
        cfg = FeatureConfig(feature_type="raw")  # would have raised a moment ago
        assert build_extractor(cfg).kind == "raw"


# ---------------------------------------------------------------------------
# Declared properties, and the code that reads them
# ---------------------------------------------------------------------------


class TestDeclaredProperties:
    def test_kaldi_is_streamable_whisper_is_not(self):
        assert build_extractor(FeatureConfig(num_mel_bins=80)).supports_streaming is True
        whisper = FeatureConfig(feature_type="whisper_logmel", num_mel_bins=128)
        assert build_extractor(whisper).supports_streaming is False

    def test_fixed_window_is_declared_by_the_extractor_not_a_name_check(self):
        """``FeatureConfig.fixed_window_seconds`` reads the registration."""
        whisper = FeatureConfig(feature_type="whisper_logmel", num_mel_bins=128)
        assert build_extractor(whisper).window_seconds_attr == "whisper_chunk_seconds"
        assert whisper.fixed_window_seconds == 30.0
        assert whisper.fixed_window_frames == 3000

        kaldi = FeatureConfig(num_mel_bins=80)
        assert build_extractor(kaldi).window_seconds_attr is None
        assert kaldi.fixed_window_seconds is None
        assert kaldi.fixed_window_frames is None

    def test_window_width_stays_a_config_knob(self):
        """The registry declares *whether*; the config declares *how wide*."""
        cfg = FeatureConfig(
            feature_type="whisper_logmel", num_mel_bins=128, whisper_chunk_seconds=10.0
        )
        assert cfg.fixed_window_seconds == 10.0
        assert cfg.fixed_window_frames == 1000

    def test_a_registered_fixed_window_frontend_needs_no_config_edit(self, isolated_registry):
        """The extension case: a new fixed-window kind reusing the same knob."""
        register_extractor(
            ExtractorSpec(
                kind="raw",
                fn=lambda w, ln, c: (w, ln),
                supports_streaming=False,
                window_seconds_attr="whisper_chunk_seconds",
            )
        )
        cfg = FeatureConfig(feature_type="raw", whisper_chunk_seconds=4.0)
        assert cfg.fixed_window_seconds == 4.0
        assert cfg.fixed_window_frames == 400

    def test_unregistered_kind_makes_no_window_claim(self, isolated_registry):
        cfg = FeatureConfig(num_mel_bins=80)
        object.__setattr__(cfg, "feature_type", "wav2vec_raw")
        assert cfg.fixed_window_seconds is None


class TestPropertiesReachTheirConsumers:
    """The declarations are only worth having if the engine honours them."""

    def test_batching_cost_is_constant_under_a_fixed_window(self, isolated_registry):
        from oasr.engine.batching.base import request_cost_frames

        register_extractor(
            ExtractorSpec(
                kind="raw",
                fn=lambda w, ln, c: (w, ln),
                window_seconds_attr="whisper_chunk_seconds",
            )
        )

        class _Req:
            def __init__(self, n):
                self.num_frames = n

        class _Cfg:
            feature_config = FeatureConfig(feature_type="raw", whisper_chunk_seconds=1.0)

        short, long = _Req(7), _Req(93)
        cost_short = request_cost_frames(short, _Cfg())
        assert cost_short == request_cost_frames(long, _Cfg()) == 100

    def test_streaming_rejects_a_non_streamable_frontend(self):
        """``prepare_streaming`` gates on the declaration, not on a kind name."""
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor
        from oasr.engine.request import Request

        cfg = EngineConfig(
            ckpt_dir="/nonexistent",
            feature_config=FeatureConfig(feature_type="whisper_logmel", num_mel_bins=128),
        )
        proc = InputProcessor(cfg, torch.device("cpu"))
        with pytest.raises(NotImplementedError, match=r"whisper_logmel.*offline-only"):
            proc.prepare_streaming(Request(request_id="r0", audio=None, streaming=True))


# ---------------------------------------------------------------------------
# The offline batch path goes through the resolved extractor
# ---------------------------------------------------------------------------


class TestOfflinePathUsesTheRegistry:
    def test_input_processor_resolves_once_at_construction(self):
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        cfg = EngineConfig(ckpt_dir="/nonexistent", feature_config=FeatureConfig(num_mel_bins=80))
        proc = InputProcessor(cfg, torch.device("cpu"))
        assert proc._extractor.kind == "fbank"

    def test_unregistered_kind_fails_at_engine_construction(self, isolated_registry):
        """Not on the first request — the whole point of resolving eagerly."""
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        cfg = EngineConfig(ckpt_dir="/nonexistent", feature_config=FeatureConfig(num_mel_bins=80))
        object.__setattr__(cfg.feature_config, "feature_type", "wav2vec_raw")
        with pytest.raises(NotImplementedError, match="wav2vec_raw"):
            InputProcessor(cfg, torch.device("cpu"))

    def test_custom_extractor_drives_the_offline_batch(self, isolated_registry):
        """End to end: register, then let ``_fbank_batch`` call it."""
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        calls = []

        def _fake(wav, lengths, config):
            calls.append((tuple(wav.shape), lengths.tolist()))
            B = wav.size(0)
            return torch.zeros(B, 5, config.output_dim), torch.full((B,), 5, dtype=torch.int32)

        register_extractor(ExtractorSpec(kind="raw", fn=_fake))
        cfg = EngineConfig(
            ckpt_dir="/nonexistent",
            feature_config=FeatureConfig(feature_type="raw", num_mel_bins=80),
        )
        proc = InputProcessor(cfg, torch.device("cpu"))

        feats, lens = proc._fbank_batch(
            torch.zeros(2, 1600), torch.tensor([1600, 800], dtype=torch.int32)
        )
        assert calls == [((2, 1600), [1600, 800])]
        assert feats.shape == (2, 5, 80)
        assert lens.tolist() == [5, 5]

    def test_lfr_still_applies_over_any_extractor(self, isolated_registry):
        """LFR is a post-transform on the caller's side, not per-extractor."""
        from oasr.engine.config import EngineConfig
        from oasr.engine.input_processor import InputProcessor

        def _fake(wav, lengths, config):
            B = wav.size(0)
            return torch.zeros(B, 42, 80), torch.full((B,), 42, dtype=torch.int32)

        register_extractor(ExtractorSpec(kind="raw", fn=_fake))
        cfg = EngineConfig(
            ckpt_dir="/nonexistent",
            feature_config=FeatureConfig(feature_type="raw", num_mel_bins=80, lfr_m=7, lfr_n=6),
        )
        proc = InputProcessor(cfg, torch.device("cpu"))
        feats, lens = proc._fbank_batch(
            torch.zeros(1, 1600), torch.tensor([1600], dtype=torch.int32)
        )
        # 42 frames stacked 7-wide with stride 6 → ceil(42/6) = 7 output frames × 560
        assert feats.shape == (1, 7, 560)
        assert lens.tolist() == [7]
