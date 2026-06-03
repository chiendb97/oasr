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


def _wav_waveform(wav_dir: str, n: int = 0):
    """Load one wav into a 1-D float32 CPU waveform tensor.

    The engine is waveform-only, so tests decode files here — exactly as the
    serving entry point (``oasr-asr``) and the bench harness do — and never
    hand a file path to the engine.
    """
    import torchaudio

    wav, _sr = torchaudio.load(_wav_path(wav_dir, n))  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).float()


def _wav_waveforms(wav_dir: str, count: int):
    """List of ``count`` waveform tensors (see :func:`_wav_waveform`)."""
    return [_wav_waveform(wav_dir, i) for i in range(count)]


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
        cc = cfg.build_cache_config(model_cfg.cache_spec)
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

        req = Request(torch.zeros(16000))
        assert req.state == RequestState.WAITING
        assert req.streaming is False
        assert req.request_id  # non-empty

    def test_has_pending_audio_false_initially(self):
        from oasr.engine.request import Request

        req = Request(torch.zeros(16000))
        assert not req.has_pending_audio

    def test_has_pending_audio_true_after_enqueue(self):
        from collections import deque

        from oasr.engine.request import Request

        req = Request(torch.zeros(16000), streaming=True)
        req.audio_chunks = deque([torch.zeros(16000)])
        req.audio_tail = torch.zeros(0)
        req.audio_final = True
        assert req.has_pending_audio

    def test_custom_request_id(self):
        from oasr.engine.request import Request

        req = Request(torch.zeros(16000), request_id="my-id")
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
        from collections import deque

        from oasr.engine.request import Request

        req = Request(torch.zeros(16000), streaming=True)
        # Enqueue ``n_chunks`` fake audio-sample tensors so the request is
        # "still streaming" under the new audio-chunk admission model.
        req.audio_chunks = deque([torch.zeros(16000) for _ in range(n_chunks)])
        req.audio_tail = torch.zeros(0)
        req.audio_final = True
        return req

    def test_add_and_schedule(self):
        from oasr.engine.scheduler import Scheduler

        sched = Scheduler(self._make_config())
        req = self._make_request()
        sched.add_request(req)

        output = sched.schedule()
        assert len(output.newly_admitted) == 1
        assert req in output.running_streams
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

    def test_running_streams_surfaces_all_admitted(self):
        from oasr.engine.scheduler import Scheduler

        sched = Scheduler(self._make_config())
        r1 = self._make_request()
        r2 = self._make_request(n_chunks=0)  # no audio — still admitted
        sched.add_request(r1)
        sched.add_request(r2)
        output = sched.schedule()
        assert r1 in output.running_streams
        assert r2 in output.running_streams

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
# Integration tests — ASREngine offline path (former OfflineEngine surface)
# ---------------------------------------------------------------------------


class TestOfflineTranscribe:
    """Cover the offline batched path via ``ASREngine.transcribe_offline`` /
    ``transcribe(..., streaming=False)``."""

    def _make_engine(self, ckpt_dir: str, device: torch.device):
        from oasr.engine import ASREngine, EngineConfig

        cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            service_mode="offline",
            decoder_type="ctc_gpu",
        )
        return ASREngine(cfg)

    def test_transcribe_single(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        engine = self._make_engine(ckpt_dir, device)
        text = engine.transcribe_offline(_wav_waveform(wav_dir, 0))
        assert isinstance(text, str)
        assert len(text) > 0

    def test_transcribe_batch(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 4:
            pytest.skip(f"Need at least 4 .wav files in WAV directory, found {len(wavs)}")
        engine = self._make_engine(ckpt_dir, device)
        waves = _wav_waveforms(wav_dir, 4)
        texts = engine.transcribe_offline(waves)
        assert isinstance(texts, list)
        assert len(texts) == 4
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)

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
            decoder_type="ctc_gpu",
            chunk_size=16,
            num_left_chunks=-1,
        )
        return ASREngine(cfg)

    def test_transcribe_single(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        engine = self._make_engine(ckpt_dir, device)
        text = engine.transcribe(_wav_waveform(wav_dir, 0))
        assert isinstance(text, str)
        assert len(text) > 0

    def test_transcribe_batch(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 3:
            pytest.skip(f"Need at least 3 .wav files in WAV directory, found {len(wavs)}")
        engine = self._make_engine(ckpt_dir, device)
        waves = _wav_waveforms(wav_dir, 3)
        texts = engine.transcribe(waves)
        assert isinstance(texts, list)
        assert len(texts) == 3
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)

    def test_run_returns_only_finished(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        engine = self._make_engine(ckpt_dir, device)
        rid = engine.add_request(_wav_waveform(wav_dir, 0))
        results = engine.run()
        assert all(r.finished for r in results)
        assert any(r.request_id == rid for r in results)

    def test_streaming_matches_offline_single_stream(
        self, device, ckpt_dir: str, wav_dir: str,
    ):
        """With ``max_batch_size=1`` streaming must reproduce offline exactly.

        Running streams one at a time through the paged forward bypasses the
        batched path, so we get a strict bitwise check on the core audio-chunk
        refactor: per-step fbank + forward_chunk_paged at B=1 has to agree
        frame-for-frame with the offline batched forward.

        This was briefly ``xfail`` while GPU-DEC-1 (``docs/known_issues.md``)
        was open: the ``ctc_gpu`` decoder's blank-frame-skip mislabelled a
        freshly emitted non-blank token as "ends in blank", so the next
        identical frame extended (CTC repeat) instead of collapsing and
        duplicated the token (e.g. ``EXHIBITION`` → ``EXHIBIT EXHIBITIONION``).
        With that kernel fix the skip path is bit-exact with the no-skip /
        CPU ``prefix_beam`` decode again, so streaming(B=1) == offline holds.
        """
        from oasr.engine import ASREngine, EngineConfig

        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 3:
            pytest.skip("Need at least 3 .wav files in WAV directory")

        waves = _wav_waveforms(wav_dir, 3)

        off_cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            service_mode="offline",
            decoder_type="ctc_gpu",
        )
        off = ASREngine(off_cfg)
        off_texts = off.transcribe_offline(waves)

        cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            decoder_type="ctc_gpu",
            chunk_size=16,
            num_left_chunks=-1,
            max_batch_size=1,
        )
        on = ASREngine(cfg)
        on_texts = on.transcribe(waves)
        for off_t, on_t in zip(off_texts, on_texts):
            assert on_t == off_t, \
                f"streaming(B=1) != offline\n  offline: {off_t!r}\n  stream : {on_t!r}"

    def test_streaming_batched_matches_offline_wer(
        self, device, ckpt_dir: str, wav_dir: str,
    ):
        """Batched streaming is numerically close to offline (fp16 ULP-level).

        Batched paged forward reorders fp16 reductions across B streams, so
        one-char differences at CTC decision boundaries are expected.  We
        check WER stays below a loose threshold rather than demanding a
        bit-exact match.
        """
        from oasr.engine import ASREngine, EngineConfig

        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 4:
            pytest.skip("Need at least 4 .wav files in WAV directory")

        waves = _wav_waveforms(wav_dir, 4)

        off_cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            service_mode="offline",
            decoder_type="ctc_gpu",
        )
        off = ASREngine(off_cfg)
        off_texts = off.transcribe_offline(waves)

        on = self._make_engine(ckpt_dir, device)  # max_batch_size=32 by default
        on_texts = on.transcribe(waves)

        def _wer(ref: str, hyp: str) -> float:
            r, h = ref.split(), hyp.split()
            # Levenshtein at word level
            dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
            for i in range(len(r) + 1):
                dp[i][0] = i
            for j in range(len(h) + 1):
                dp[0][j] = j
            for i in range(1, len(r) + 1):
                for j in range(1, len(h) + 1):
                    if r[i - 1] == h[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
            return dp[len(r)][len(h)] / max(1, len(r))

        total = sum(_wer(ref, hyp) for ref, hyp in zip(off_texts, on_texts))
        avg_wer = total / len(wavs)
        # Loose threshold: batched-fp16 vs offline-batched-fp16 typically
        # diverge by <5% WER on a handful of utterances; the drift comes
        # from reordered fp16 reductions in the per-layer matmuls and
        # paged attention, *not* from wrong streaming logic.
        assert avg_wer < 0.05, \
            f"Batched streaming diverged too far from offline: WER={avg_wer:.3f}"

    def test_engine_idle_after_run(self, device, ckpt_dir: str, wav_dir: str):
        _require_ckpt(ckpt_dir)
        _require_wav_dir(wav_dir)
        engine = self._make_engine(ckpt_dir, device)
        engine.add_request(_wav_waveform(wav_dir, 0))
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
            decoder_type="ctc_gpu",
            chunk_size=16,
            max_num_blocks=512,
        )
        engine = ASREngine(cfg)
        # Record initial free block count
        initial_free = engine._model_runner._block_pool.num_free_blocks

        engine.add_request(_wav_waveform(wav_dir, 0))
        engine.add_request(_wav_waveform(wav_dir, 1))
        engine.run()

        # All blocks should be returned to the pool
        final_free = engine._model_runner._block_pool.num_free_blocks
        assert final_free == initial_free
