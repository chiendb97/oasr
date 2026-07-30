# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The engine accepts waveforms at exactly one sample rate — the model's.

Nothing below PyO3 resamples: :class:`~oasr.engine.input_processor.InputProcessor`
derives every frame count from ``FeatureConfig.sample_rate`` and the mel
filterbank is built for it, while ``Request.sample_rate`` feeds nothing but
long-form window arithmetic.  So audio at another rate is interpreted as if it
were at the model's — 8 kHz telephony plays back to the frontend at double
speed — and the client gets a confident, wrong transcript.  These tests pin the
rejection.

Conversion happens one layer up, in the Rust front-end (``oasr-asr``'s
``resample`` module, tested by ``cargo test -p oasr-asr``), so nothing reaching
these checks in production should ever mismatch.  They exist for the direct
Python callers — benchmarks, notebooks, harnesses — and as the backstop that
makes a front-end regression loud.

CPU-only: no checkpoint, no CUDA.
"""

from __future__ import annotations

import pytest
import torch

from oasr.engine.config import EngineConfig
from oasr.engine.input_processor import InputProcessor
from oasr.engine.request import Request
from oasr.features import FeatureConfig


def _processor(sample_rate: int = 16000, **fcfg) -> InputProcessor:
    cfg = EngineConfig(
        ckpt_dir="/nonexistent",
        feature_config=FeatureConfig(sample_rate=sample_rate, num_mel_bins=80, **fcfg),
    )
    return InputProcessor(cfg, torch.device("cpu"))


class TestCheckSampleRate:
    def test_matching_rate_is_accepted(self):
        _processor().check_sample_rate(16000)

    def test_unspecified_rate_is_accepted(self):
        """``None`` means "the model's rate" — the engine resolves it before
        building the request, so the check must not treat it as a mismatch."""
        _processor().check_sample_rate(None)

    def test_mismatch_names_both_rates(self):
        with pytest.raises(ValueError, match=r"8000 Hz.*requires 16000 Hz"):
            _processor().check_sample_rate(8000)

    def test_mismatch_message_says_the_engine_does_not_resample(self):
        """The error has to tell the caller what to do; "invalid sample rate"
        alone sends them looking for a config knob that does not exist."""
        with pytest.raises(ValueError, match="does not resample"):
            _processor().check_sample_rate(44100)

    def test_the_accepted_rate_comes_from_the_feature_config(self):
        """Not a hardcoded 16000.  Every checkpoint in tree happens to run at
        16 kHz, which is exactly what would let a hardcoded constant survive."""
        proc = _processor(sample_rate=8000)
        proc.check_sample_rate(8000)
        with pytest.raises(ValueError, match=r"16000 Hz.*requires 8000 Hz"):
            proc.check_sample_rate(16000)


class TestPreparePaths:
    def test_prepare_offline_accepts_the_model_rate(self):
        proc = _processor()
        req = Request(audio=torch.zeros(16000), streaming=False, sample_rate=16000)
        proc.prepare_offline(req)
        assert req.num_frames > 0

    def test_prepare_offline_rejects_a_mismatch(self):
        proc = _processor()
        req = Request(audio=torch.zeros(8000), streaming=False, sample_rate=8000)
        with pytest.raises(ValueError, match="requires 16000 Hz"):
            proc.prepare_offline(req)

    def test_prepare_streaming_rejects_at_open(self):
        """Not on the first chunk: by then the client has been told the stream
        is live, and a mid-stream error is far harder to attribute."""
        proc = _processor()
        req = Request(audio=None, streaming=True, sample_rate=44100)
        with pytest.raises(ValueError, match="requires 16000 Hz"):
            proc.prepare_streaming(req)
        # And nothing was set up for it.
        assert req.audio_chunks is None

    def test_prepare_streaming_accepts_the_model_rate(self):
        proc = _processor()
        req = Request(audio=None, streaming=True, sample_rate=16000)
        proc.prepare_streaming(req)
        assert req.audio_chunks is not None
