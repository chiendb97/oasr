# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ASR inference engine (oasr/engine/)."""

from __future__ import annotations

import glob
import os
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers (paths from conftest: --ckpt-dir / CKPT_DIR, --wav-dir / WAV_DIR)
# ---------------------------------------------------------------------------


def _require_ckpt(ckpt_dir: str) -> None:
    if not ckpt_dir or not Path(ckpt_dir).exists():
        pytest.skip(
            "WeNet checkpoint dir not set or not found; set CKPT_DIR env var or --ckpt-dir"
        )


def _require_wav_dir(wav_dir: str) -> None:
    if not wav_dir or not Path(wav_dir).is_dir():
        pytest.skip(
            "WAV directory not set or not found; use --wav-dir or WAV_DIR"
        )
    if not glob.glob(os.path.join(wav_dir, "*.wav")):
        pytest.skip("No .wav files found in WAV directory")


def _wav_path(wav_dir: str, n: int = 0) -> str:
    wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    return wavs[n]


# ---------------------------------------------------------------------------
# Unit tests — EngineConfig
# ---------------------------------------------------------------------------


class TestEngineConfig:
    def test_default_feature_config(self):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir="/tmp/fake")
        assert cfg.feature_config is not None
        assert cfg.feature_config.dither == 0.0

    def test_computed_properties(self):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir="/tmp/fake", chunk_size=16)
        assert cfg.subsampling_rate == 4
        assert cfg.right_context == 6
        # stride = 4 * 16 = 64
        assert cfg.stride == 64
        # decoding_window = (16 - 1) * 4 + 6 + 1 = 67
        assert cfg.decoding_window == 67

    def test_autodetect_sentencepiece(self, ckpt_dir: str):
        from oasr.engine.config import EngineConfig

        _require_ckpt(ckpt_dir)
        cfg = EngineConfig(ckpt_dir=ckpt_dir)
        assert cfg.sentencepiece_model is not None
        assert cfg.sentencepiece_model.endswith(".model")
        assert os.path.exists(cfg.sentencepiece_model)

    def test_autodetect_unit_table(self, ckpt_dir: str):
        from oasr.engine.config import EngineConfig

        _require_ckpt(ckpt_dir)
        cfg = EngineConfig(ckpt_dir=ckpt_dir)
        assert cfg.unit_table is not None
        assert os.path.exists(cfg.unit_table)

    def test_build_cache_config(self):
        from oasr.engine.config import EngineConfig
        from oasr.models.conformer.config import ConformerEncoderConfig, ConformerModelConfig

        cfg = EngineConfig(ckpt_dir="/tmp/fake", chunk_size=16, max_num_blocks=512)
        enc_cfg = ConformerEncoderConfig(
            output_size=256, num_blocks=12, attention_heads=4, cnn_module_kernel=15
        )
        model_cfg = ConformerModelConfig(encoder=enc_cfg, vocab_size=5002)
        cc = cfg.build_cache_config(model_cfg)
        assert cc.num_layers == 12
        assert cc.hidden_dim == 256
        assert cc.kernel_size == 15
        assert cc.n_kv_head == 4
        assert cc.chunk_size == 16
        assert cc.max_num_blocks == 512


# ---------------------------------------------------------------------------
# Unit tests — Request
# ---------------------------------------------------------------------------


class TestRequest:
    def test_defaults(self):
        from oasr.engine.request import Request, RequestState

        req = Request("audio.wav")
        assert req.state == RequestState.WAITING
        assert req.streaming is False
        assert req.request_id  # non-empty

    def test_has_chunks_false_initially(self):
        from oasr.engine.request import Request

        req = Request("audio.wav")
        assert not req.has_chunks

    def test_has_chunks_true_after_assignment(self):
        from oasr.engine.request import Request

        req = Request("audio.wav")
        req.chunks_remaining = [torch.zeros(1, 67, 80)]
        assert req.has_chunks

    def test_custom_request_id(self):
        from oasr.engine.request import Request

        req = Request("audio.wav", request_id="my-id")
        assert req.request_id == "my-id"


# ---------------------------------------------------------------------------
# Unit tests — Scheduler
# ---------------------------------------------------------------------------


class TestScheduler:
    def _make_config(self, max_batch_size=4):
        from oasr.engine.config import EngineConfig

        cfg = EngineConfig(ckpt_dir="/tmp/fake", max_batch_size=max_batch_size)
        return cfg

    def _make_request(self, n_chunks=3):
        from oasr.engine.request import Request

        req = Request("audio.wav", streaming=True)
        req.chunks_remaining = [torch.zeros(1, 67, 80) for _ in range(n_chunks)]
        return req

    def test_add_and_schedule(self):
        from oasr.engine.scheduler import Scheduler

        sched = Scheduler(self._make_config())
        req = self._make_request()
        sched.add_request(req)

        output = sched.schedule()
        assert len(output.newly_admitted) == 1
        assert req in output.scheduled_requests
        assert sched.num_running == 1
        assert sched.num_waiting == 0

    def test_max_batch_size_respected(self):
        from oasr.engine.scheduler import Scheduler

        sched = Scheduler(self._make_config(max_batch_size=2))
        reqs = [self._make_request() for _ in range(5)]
        for r in reqs:
            sched.add_request(r)

        output = sched.schedule()
        assert len(output.newly_admitted) == 2
        assert sched.num_running == 2
        assert sched.num_waiting == 3

    def test_finish_request(self):
        from oasr.engine.scheduler import Scheduler

        sched = Scheduler(self._make_config())
        req = self._make_request()
        sched.add_request(req)
        sched.schedule()
        finished = sched.finish_request(req.request_id)
        assert finished is req
        assert sched.num_running == 0

    def test_to_finalize_when_no_chunks(self):
        from oasr.engine.scheduler import Scheduler

        sched = Scheduler(self._make_config())
        req = self._make_request(n_chunks=0)
        sched.add_request(req)
        output = sched.schedule()
        assert req in output.to_finalize
        assert req not in output.scheduled_requests

    def test_has_pending(self):
        from oasr.engine.scheduler import Scheduler

        sched = Scheduler(self._make_config())
        assert not sched.has_pending()
        req = self._make_request()
        sched.add_request(req)
        assert sched.has_pending()
        sched.schedule()
        sched.finish_request(req.request_id)
        assert not sched.has_pending()

    def test_abort_waiting(self):
        from oasr.engine.scheduler import Scheduler

        sched = Scheduler(self._make_config())
        req = self._make_request()
        sched.add_request(req)
        aborted = sched.abort_request(req.request_id)
        assert aborted is req
        assert not sched.has_pending()

    def test_fcfs_ordering(self):
        from oasr.engine.scheduler import Scheduler

        sched = Scheduler(self._make_config(max_batch_size=2))
        r1 = self._make_request()
        r2 = self._make_request()
        r3 = self._make_request()
        sched.add_request(r1)
        sched.add_request(r2)
        sched.add_request(r3)

        output = sched.schedule()
        admitted_ids = [r.request_id for r in output.newly_admitted]
        assert admitted_ids == [r1.request_id, r2.request_id]


# ---------------------------------------------------------------------------
# Unit tests — InputProcessor
# ---------------------------------------------------------------------------


class TestInputProcessor:
    def _make_config(self):
        from oasr.engine.config import EngineConfig

        return EngineConfig(ckpt_dir="/tmp/fake", chunk_size=16)

    def test_load_audio_from_tensor(self):
        from oasr.engine.input_processor import InputProcessor

        cfg = self._make_config()
        proc = InputProcessor(cfg, torch.device("cpu"))
        wav = torch.randn(16000)
        result = proc.load_audio(wav)
        assert result.shape == (16000,)
        assert result.dtype == torch.float32

    def test_load_audio_from_numpy(self):
        import numpy as np

        from oasr.engine.input_processor import InputProcessor

        cfg = self._make_config()
        proc = InputProcessor(cfg, torch.device("cpu"))
        wav_np = torch.randn(16000).numpy()
        result = proc.load_audio(wav_np)
        assert result.shape == (16000,)

    def test_load_audio_from_file(self, wav_dir: str):
        from oasr.engine.input_processor import InputProcessor

        _require_wav_dir(wav_dir)
        cfg = self._make_config()
        proc = InputProcessor(cfg, torch.device("cpu"))
        result = proc.load_audio(_wav_path(wav_dir, 0))
        assert result.dim() == 1
        assert result.shape[0] > 0

    def test_chunk_features_count(self):
        from oasr.engine.input_processor import InputProcessor

        cfg = self._make_config()
        proc = InputProcessor(cfg, torch.device("cpu"))
        # Build a synthetic feature tensor: (1, 400, 80)
        features = torch.randn(1, 400, 80)
        chunks = proc.chunk_features(features)
        assert len(chunks) > 0
        # Each chunk should have decoding_window frames (or less for the last)
        assert chunks[0].size(1) <= cfg.decoding_window
        assert chunks[0].size(2) == 80

    def test_chunk_features_stride(self):
        from oasr.engine.input_processor import InputProcessor

        cfg = self._make_config()
        proc = InputProcessor(cfg, torch.device("cpu"))
        # stride + window frames produces exactly 2 chunks:
        # range(0, n_frames - context + 1, stride) = [0, stride]
        n_frames = cfg.stride + cfg.decoding_window
        features = torch.randn(1, n_frames, 80)
        chunks = proc.chunk_features(features)
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# Unit tests — OutputProcessor (detokenization)
# ---------------------------------------------------------------------------


class TestOutputProcessorDetokenize:
    def test_detokenize_sentencepiece(self, ckpt_dir: str):
        from oasr.engine.config import EngineConfig
        from oasr.engine.output_processor import OutputProcessor

        _require_ckpt(ckpt_dir)
        cfg = EngineConfig(ckpt_dir=ckpt_dir)
        proc = OutputProcessor(cfg)
        # Blank and sos/eos tokens should be stripped
        text = proc.detokenize([0, 2])
        assert text == ""

    def test_detokenize_nonempty(self, ckpt_dir: str):
        from oasr.engine.config import EngineConfig
        from oasr.engine.output_processor import OutputProcessor

        _require_ckpt(ckpt_dir)
        cfg = EngineConfig(ckpt_dir=ckpt_dir)
        proc = OutputProcessor(cfg)
        # Try a known token id > 2 (should produce something)
        text = proc.detokenize([16])  # token 16 = '▁ABOUT'
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Integration tests — OfflineEngine
# ---------------------------------------------------------------------------


class TestOfflineEngine:
    def _make_engine(self, ckpt_dir: str, device: torch.device):
        from oasr.engine import EngineConfig, OfflineEngine

        cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            decoder_type="ctc_prefix_beam",
        )
        return OfflineEngine(cfg)

    def test_transcribe_single(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        engine = self._make_engine(ckpt_dir, device)
        text = engine.transcribe(_wav_path(wav_dir, 0))
        assert isinstance(text, str)
        assert len(text) > 0

    def test_transcribe_batch(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 4:
            pytest.skip(f"Need at least 4 .wav files in WAV directory, found {len(wavs)}")
        engine = self._make_engine(ckpt_dir, device)
        paths = [_wav_path(wav_dir, i) for i in range(4)]
        texts = engine.transcribe(paths)
        assert isinstance(texts, list)
        assert len(texts) == 4
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)

    def test_transcribe_returns_english(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        engine = self._make_engine(ckpt_dir, device)
        text = engine.transcribe(_wav_path(wav_dir, 0))
        # LJSpeech is English — result should contain only ASCII
        assert text.isascii(), f"Expected ASCII text, got: {text!r}"

    @pytest.mark.slow
    def test_transcribe_streaming_simulation(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        engine = self._make_engine(ckpt_dir, device)
        text = engine.transcribe_streaming(_wav_path(wav_dir, 0))
        assert isinstance(text, str)
        assert len(text) > 0


# ---------------------------------------------------------------------------
# Integration tests — ASREngine (streaming)
# ---------------------------------------------------------------------------


class TestASREngine:
    def _make_engine(self, ckpt_dir: str, device: torch.device):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            decoder_type="ctc_prefix_beam",
            chunk_size=16,
            num_left_chunks=-1,
            use_paged_cache=True,
        )
        return ASREngine(cfg)

    def test_transcribe_single(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        engine = self._make_engine(ckpt_dir, device)
        text = engine.transcribe(_wav_path(wav_dir, 0))
        assert isinstance(text, str)
        assert len(text) > 0

    def test_transcribe_batch(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 3:
            pytest.skip(f"Need at least 3 .wav files in WAV directory, found {len(wavs)}")
        engine = self._make_engine(ckpt_dir, device)
        paths = [_wav_path(wav_dir, i) for i in range(3)]
        texts = engine.transcribe(paths)
        assert isinstance(texts, list)
        assert len(texts) == 3
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)

    def test_run_returns_only_finished(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        engine = self._make_engine(ckpt_dir, device)
        rid = engine.add_request(_wav_path(wav_dir, 0))
        results = engine.run()
        assert all(r.finished for r in results)
        assert any(r.request_id == rid for r in results)

    def test_engine_idle_after_run(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        engine = self._make_engine(ckpt_dir, device)
        engine.add_request(_wav_path(wav_dir, 0))
        engine.run()
        assert engine.num_running == 0
        assert engine.num_waiting == 0

    @pytest.mark.slow
    def test_memory_cleanup_after_streaming(self, device, ckpt_dir: str, wav_dir: str):
        from oasr.engine import ASREngine, EngineConfig

        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 2:
            pytest.skip(f"Need at least 2 .wav files in WAV directory, found {len(wavs)}")

        cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            decoder_type="ctc_prefix_beam",
            chunk_size=16,
            use_paged_cache=True,
            max_num_blocks=512,
        )
        engine = ASREngine(cfg)
        # Record initial free block count
        initial_free = engine._model_runner._block_pool.num_free_blocks

        engine.add_request(_wav_path(wav_dir, 0))
        engine.add_request(_wav_path(wav_dir, 1))
        engine.run()

        # All blocks should be returned to the pool
        final_free = engine._model_runner._block_pool.num_free_blocks
        assert final_free == initial_free
