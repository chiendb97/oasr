# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``ASREngine``: construction, the seams it dispatches through, and offline decode.

``test_engine_seams.py`` used to be separate, but the seams *are* the engine:
which decode strategy and batching policy a config resolves to, and how bulk
admission behaves, are the first things ``ASREngine.__init__`` does. Two files
also meant two ``OutputProcessor`` detokenize tests over the same fixture
data.

Batching and partition moved out to ``test_scheduler.py``, which is the module
that owns them; ``EngineConfig`` field validation moved to ``test_config.py``,
where five files used to assert it.
"""

from __future__ import annotations

import glob
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from helpers.audio import waveform, waveforms

from oasr.engine.decode import Detokenizer, build_decode_strategy
from oasr.engine.decode.base import _REGISTRY as DECODE_REGISTRY

# ---------------------------------------------------------------------------
# Helpers (paths from conftest: --ckpt-dir / CKPT_DIR, --wav-dir / WAV_DIR)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unit tests — EngineConfig
# ---------------------------------------------------------------------------


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


class TestOutputProcessorDetokenize:
    """Detokenization off a real checkpoint's ``units.txt``.

    ``OutputProcessor`` builds a decode strategy, and since the capability table
    landed (H2) a strategy with **no model** is rejected — "no surface means
    rejected", deliberately, because the old duck-typed checks disagreed with
    each other about ``model=None``.  These tests only exercise the detokenizer,
    so they supply the minimal CTC surface rather than asking the facade to
    accept a model-less construction.  (They had been failing since H2 and went
    unnoticed because every verification run had ``CKPT_DIR`` unset, which
    skipped them.)
    """

    @staticmethod
    def _ctc_model():
        """Smallest object satisfying ``CAPABILITIES["ctc"]``.

        The bodies raise: these tests must never reach a forward, and a stub that
        silently returned something would hide it if they did.
        """

        def _unreachable(*_a, **_k):
            raise AssertionError("detokenization must not run a forward")

        return SimpleNamespace(head=_unreachable, forward_offline=_unreachable)

    def _proc(self, ckpt_dir: str):
        from oasr.engine.config import EngineConfig
        from oasr.engine.output_processor import OutputProcessor

        cfg = EngineConfig(ckpt_dir=ckpt_dir)
        # The engine stamps ``_model_config`` after loading the checkpoint; the
        # CTC strategy needs its ``vocab_size`` to size the beam state and now
        # *raises* rather than falling back to a magic token ID.  Read the real
        # value off the checkpoint so the stub is not a fiction.
        with open(Path(ckpt_dir) / "units.txt", encoding="utf-8") as f:
            vocab = sum(1 for line in f if line.strip())
        cfg._model_config = SimpleNamespace(vocab_size=vocab)
        return OutputProcessor(cfg, model=self._ctc_model())

    def test_detokenize_sentencepiece(self, ckpt_dir: str):
        # Blank and sos/eos tokens should be stripped
        assert self._proc(ckpt_dir).detokenize([0, 2]) == ""

    def test_detokenize_nonempty(self, ckpt_dir: str):
        # Try a known token id > 2 (should produce something)
        text = self._proc(ckpt_dir).detokenize([16])  # token 16 = '▁ABOUT'
        assert isinstance(text, str) and text


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
            decoder_type="ctc_cuda",
        )
        return ASREngine(cfg)

    def test_transcribe_single(self, device, ckpt_dir: str, wav_dir: str):
        engine = self._make_engine(ckpt_dir, device)
        text = engine.transcribe_offline(waveform(wav_dir, 0))
        assert isinstance(text, str)
        assert len(text) > 0

    def test_transcribe_batch(self, device, ckpt_dir: str, wav_dir: str):
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 4:
            pytest.skip(f"Need at least 4 .wav files in WAV directory, found {len(wavs)}")
        engine = self._make_engine(ckpt_dir, device)
        waves = waveforms(wav_dir, 4)
        texts = engine.transcribe_offline(waves)
        assert isinstance(texts, list)
        assert len(texts) == 4
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)

    def test_audio_in_a_pinned_buffer_transcribes_identically(
        self, device, ckpt_dir: str, wav_dir: str
    ):
        """The contract the Rust front-end relies on: fill
        ``engine.new_audio_buffer(n)``, submit the tensor, get the same
        transcript as the heap path — which is what lets ``collate`` skip the
        pack and DMA each row straight into the padded batch."""
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 3:
            pytest.skip("Need at least 3 .wav files in WAV directory")
        waves = waveforms(wav_dir, 3)

        engine = self._make_engine(ckpt_dir, device)
        heap_texts = engine.transcribe_offline(waves)

        staged = []
        for wav in waves:
            buf = engine.new_audio_buffer(int(wav.numel()))
            assert buf is not None, "a CUDA engine should offer pinned audio buffers"
            assert buf.is_pinned()
            # Written through a numpy view, exactly as the front-end does; the
            # *tensor* is what gets submitted, so the async H2D can still be
            # event-tracked against the caching host allocator's block.
            buf.numpy()[:] = wav.numpy()
            staged.append(buf)
        pinned_texts = engine.transcribe_offline(staged)

        assert pinned_texts == heap_texts


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
            decoder_type="ctc_cuda",
            chunk_size=16,
            num_left_chunks=-1,
        )
        return ASREngine(cfg)

    def test_transcribe_single(self, device, ckpt_dir: str, wav_dir: str):
        engine = self._make_engine(ckpt_dir, device)
        text = engine.transcribe(waveform(wav_dir, 0))
        assert isinstance(text, str)
        assert len(text) > 0

    def test_transcribe_batch(self, device, ckpt_dir: str, wav_dir: str):
        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 3:
            pytest.skip(f"Need at least 3 .wav files in WAV directory, found {len(wavs)}")
        engine = self._make_engine(ckpt_dir, device)
        waves = waveforms(wav_dir, 3)
        texts = engine.transcribe(waves)
        assert isinstance(texts, list)
        assert len(texts) == 3
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)

    def test_run_returns_only_finished(self, device, ckpt_dir: str, wav_dir: str):
        engine = self._make_engine(ckpt_dir, device)
        rid = engine.add_request(waveform(wav_dir, 0))
        results = engine.run()
        assert all(r.finished for r in results)
        assert any(r.request_id == rid for r in results)

    def test_streaming_matches_offline_single_stream(
        self,
        device,
        ckpt_dir: str,
        wav_dir: str,
    ):
        """With ``max_batch_size=1`` streaming must reproduce offline exactly.

        Running streams one at a time through the paged forward bypasses the
        batched path, so we get a strict bitwise check on the core audio-chunk
        refactor: per-step fbank + forward_chunk_paged at B=1 has to agree
        frame-for-frame with the offline batched forward.

        This was briefly ``xfail`` while GPU-DEC-1 (``.artifacts/known_issues.md``)
        was open: the ``ctc_cuda`` decoder's blank-frame-skip mislabelled a
        freshly emitted non-blank token as "ends in blank", so the next
        identical frame extended (CTC repeat) instead of collapsing and
        duplicated the token (e.g. ``EXHIBITION`` → ``EXHIBIT EXHIBITIONION``).
        With that kernel fix the skip path is bit-exact with the no-skip /
        CPU ``prefix_beam`` decode again, so streaming(B=1) == offline holds.
        """
        from oasr.engine import ASREngine, EngineConfig

        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 3:
            pytest.skip("Need at least 3 .wav files in WAV directory")

        waves = waveforms(wav_dir, 3)

        off_cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            service_mode="offline",
            decoder_type="ctc_cuda",
        )
        off = ASREngine(off_cfg)
        off_texts = off.transcribe_offline(waves)

        cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            decoder_type="ctc_cuda",
            chunk_size=16,
            num_left_chunks=-1,
            max_batch_size=1,
        )
        on = ASREngine(cfg)
        on_texts = on.transcribe(waves)
        for off_t, on_t in zip(off_texts, on_texts):
            assert (
                on_t == off_t
            ), f"streaming(B=1) != offline\n  offline: {off_t!r}\n  stream : {on_t!r}"

    def test_streaming_batched_matches_offline_wer(
        self,
        device,
        ckpt_dir: str,
        wav_dir: str,
    ):
        """Batched streaming is numerically close to offline (fp16 ULP-level).

        Batched paged forward reorders fp16 reductions across B streams, so
        one-char differences at CTC decision boundaries are expected.  We
        check WER stays below a loose threshold rather than demanding a
        bit-exact match.
        """
        from oasr.engine import ASREngine, EngineConfig

        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 4:
            pytest.skip("Need at least 4 .wav files in WAV directory")

        waves = waveforms(wav_dir, 4)

        off_cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            service_mode="offline",
            decoder_type="ctc_cuda",
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
        assert avg_wer < 0.05, f"Batched streaming diverged too far from offline: WER={avg_wer:.3f}"

    def test_engine_idle_after_run(self, device, ckpt_dir: str, wav_dir: str):
        engine = self._make_engine(ckpt_dir, device)
        engine.add_request(waveform(wav_dir, 0))
        engine.run()
        assert engine.num_running == 0
        assert engine.num_waiting == 0

    @pytest.mark.parametrize("overlap", [False, True])
    def test_partial_readback_modes_agree(self, device, ckpt_dir: str, wav_dir: str, overlap: bool):
        """``overlap_partial_readback`` changes *when* a partial is emitted, not what.

        The overlapped read-back is off by default and was therefore never
        exercised: it had kept an 11-argument call into a launcher that grew a
        twelfth (the word-timing buffer), so every stream failed its forward and
        finalised with ``finish_reason="error"`` and an empty transcript.
        Nothing noticed, because the only configuration that selects the path is
        one no test set.  Both modes are checked here, together, so the pair
        cannot drift again.
        """
        from oasr.engine import ASREngine, EngineConfig

        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 2:
            pytest.skip("Need at least 2 .wav files in WAV directory")
        waves = waveforms(wav_dir, 2)

        cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            decoder_type="ctc_cuda",
            chunk_size=16,
            num_left_chunks=-1,
            max_batch_size=2,
            overlap_partial_readback=overlap,
        )
        engine = ASREngine(cfg)
        chunk_samples = engine._input_processor.streaming_audio_chunk_samples

        rids = []
        for wav in waves:
            rid = engine.add_streaming_request(sample_rate=16000)
            rids.append(rid)
            starts = list(range(0, int(wav.numel()), chunk_samples))
            for s in starts:
                engine.feed_chunk(rid, wav[s : s + chunk_samples], is_last=(s == starts[-1]))

        partials: list[str] = []
        finals: dict[str, str] = {}
        while engine.num_running or engine.num_waiting:
            for out in engine.step():
                if out.finished:
                    assert out.finish_reason != "error", f"stream failed: {out}"
                    finals[out.request_id] = out.text
                elif out.text:
                    partials.append(out.text)

        assert set(finals) == set(rids)
        assert all(t for t in finals.values()), f"empty final transcript: {finals}"
        # The whole point of the interim path: a partial has to actually arrive.
        assert partials, "no interim partial was emitted"

    @pytest.mark.slow
    def test_memory_cleanup_after_streaming(self, device, ckpt_dir: str, wav_dir: str):
        from oasr.engine import ASREngine, EngineConfig

        wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        if len(wavs) < 2:
            pytest.skip(f"Need at least 2 .wav files in WAV directory, found {len(wavs)}")

        cfg = EngineConfig(
            ckpt_dir=ckpt_dir,
            device=str(device),
            dtype=torch.float16,
            decoder_type="ctc_cuda",
            chunk_size=16,
            max_num_blocks=512,
        )
        engine = ASREngine(cfg)
        # Record initial free block count
        initial_free = engine._model_runner._block_pool.num_free_blocks

        engine.add_request(waveform(wav_dir, 0))
        engine.add_request(waveform(wav_dir, 1))
        engine.run()

        # All blocks should be returned to the pool
        final_free = engine._model_runner._block_pool.num_free_blocks
        assert final_free == initial_free


# ---------------------------------------------------------------------------
# The pluggable seams: decode-strategy and batching registries, bulk admission
#
# Exercised without CUDA or a checkpoint -- the point is the dispatch and the
# autoregressive extension-point contract, not what the kernels compute.
# ---------------------------------------------------------------------------


def _stub_config(decoder_type="ctc_cuda"):
    return SimpleNamespace(
        decoder_type=decoder_type,
        device="cpu",
        dtype=None,
        use_cuda_graphs=False,
        use_ctc_cuda_graphs=False,
        _model_config=SimpleNamespace(vocab_size=32),
        ctc_decoder_config=None,
        wfst_decoder_config=None,
        fst_path=None,
    )


def _stub_model(capability):
    """Smallest object satisfying ``capability``'s declared surface.

    Built from :data:`oasr.models.interfaces.CAPABILITIES` so it cannot drift from
    the contract it is standing in for: add a required member to a spec and every
    stub grows it automatically.
    """
    from oasr.models.interfaces import CAPABILITIES

    root = SimpleNamespace()
    for path in CAPABILITIES[capability].requires:
        cur = root
        parts = path.split(".")
        for part in parts[:-1]:
            if not hasattr(cur, part):
                setattr(cur, part, SimpleNamespace())
            cur = getattr(cur, part)
        setattr(cur, parts[-1], lambda *a, **k: None)
    return root


# --------------------------------------------------------------------------- #
# Decode-strategy registry / dispatch
# --------------------------------------------------------------------------- #


def test_decode_registry_has_builtins():
    assert set(DECODE_REGISTRY) == {
        "ctc_cuda",
        "ctc_wfst",
        "ctc_aed_rescoring",
        "transducer",
        "aed",
        "llm",
        "paraformer",
    }


def test_build_ctc_strategies_by_decoder_type():
    detok = Detokenizer(None, None)
    model = _stub_model("ctc")
    gpu = build_decode_strategy("ctc", _stub_config("ctc_cuda"), detok, model)
    wfst = build_decode_strategy("ctc", _stub_config("ctc_wfst"), detok, model)
    assert type(gpu).__name__ == "CtcGpuDecodeStrategy"
    assert type(wfst).__name__ == "CtcWfstDecodeStrategy"
    assert gpu.consumes == "log_probs" and gpu.decode_type == "ctc"


@pytest.mark.parametrize("dt", ["transducer"])
def test_ar_strategies_resolve_and_consume_hidden(dt):
    s = build_decode_strategy(dt, _stub_config(), Detokenizer(None, None), _stub_model(dt))
    assert s.decode_type == dt
    assert s.consumes == "hidden"


def test_aed_is_incremental_and_needs_a_capable_model():
    """``aed`` is a real strategy now: it declares the incremental protocol
    and refuses models without the batched prefill/step decoder surface."""
    from oasr.engine.decode import get_decode_strategy_class

    cls = get_decode_strategy_class("aed", _stub_config())
    assert cls.__name__ == "AedDecodeStrategy"
    assert cls.consumes == "hidden" and cls.incremental is True
    with pytest.raises(ValueError, match="prefill"):
        build_decode_strategy("aed", _stub_config(), Detokenizer(None, None))


def test_llm_is_incremental_and_needs_a_capable_model():
    """``llm`` is a real strategy now: it declares the incremental protocol
    and refuses models without the speech-LLM prompt/decoder surface."""
    from oasr.engine.decode import get_decode_strategy_class

    cls = get_decode_strategy_class("llm", _stub_config())
    assert cls.__name__ == "LlmDecodeStrategy"
    assert cls.consumes == "hidden" and cls.incremental is True
    with pytest.raises(ValueError, match="prefill"):
        build_decode_strategy("llm", _stub_config(), Detokenizer(None, None))


def test_transducer_offline_and_streaming_implemented():
    # transducer is a full strategy: decode_offline + streaming sessions (both
    # tested in test_transducer.py).  finalize on a request with no session
    # yields an empty final transcript rather than raising.
    s = build_decode_strategy(
        "transducer", _stub_config(), Detokenizer(None, None), _stub_model("transducer")
    )
    assert type(s).__name__ == "TransducerDecodeStrategy"
    out = s.finalize(SimpleNamespace(request_id="never-decoded"))
    assert out.finished and out.tokens == [[]] and out.text == ""


def test_build_unknown_decode_type_raises():
    with pytest.raises(NotImplementedError, match="No decode strategy"):
        build_decode_strategy("paraformer-aed", _stub_config(), Detokenizer(None, None))


# --------------------------------------------------------------------------- #
# Detokenizer
# --------------------------------------------------------------------------- #


def test_detokenize_strips_specials_and_word_boundary(tmp_path):
    units = tmp_path / "units.txt"
    units.write_text("<blank> 0\n<unk> 1\n▁ABOUT 16\nS 17\n▁THE 18\n", encoding="utf-8")
    detok = Detokenizer(None, str(units))
    assert detok.detokenize([0, 1, 2]) == ""  # all special
    assert detok.detokenize([16, 17]) == "ABOUTS"  # 'S' is a sub-word continuation
    assert detok.detokenize([16, 18]) == "ABOUT THE"  # ▁ -> word boundary, leading strip


# --------------------------------------------------------------------------- #
# Batching + partition policy registries
# --------------------------------------------------------------------------- #


def test_batching_registries_and_builders():
    from oasr.engine.batching import build_batching_policy, build_partition_policy
    from oasr.engine.batching.base import _BATCHING, _PARTITION

    assert set(_BATCHING) == {"fcfs", "bucket", "sjf"}
    assert set(_PARTITION) == {"count", "frames", "packing"}

    cfg = SimpleNamespace(
        schedule_policy="bucket",
        enable_sequence_packing=False,
        max_batch_frames=None,
    )
    assert type(build_batching_policy(cfg)).__name__ == "BucketPolicy"
    assert type(build_partition_policy(cfg)).__name__ == "CountPartition"

    cfg.enable_sequence_packing = True
    assert type(build_partition_policy(cfg)).__name__ == "PackingPartition"
    cfg.enable_sequence_packing = False
    cfg.max_batch_frames = 8000
    assert type(build_partition_policy(cfg)).__name__ == "FramePartition"


def test_build_unknown_batching_policy_raises():
    from oasr.engine.batching import build_batching_policy

    cfg = SimpleNamespace(schedule_policy="round-robin")
    with pytest.raises(NotImplementedError, match="No batching policy"):
        build_batching_policy(cfg)


# --------------------------------------------------------------------------- #
# Bulk admission fault isolation
# --------------------------------------------------------------------------- #


class _FakeExecutor:
    """Minimal Executor surface for the admission path."""

    streaming = False

    def __init__(self):
        self.admitted = []

    def admit(self, request):
        self.admitted.append(request.request_id)


class _FakeStrategy:
    """A decode family with no task / language control.

    ``validate_options`` is the **real** base implementation rather than a stub:
    admission's job here is to route a family's rejection into that spec's
    result dict, and a stub that never rejects would test nothing.
    """

    from oasr.engine.decode.base import DecodeStrategy as _Base

    decode_type = "ctc"
    selective_options = ()
    #: No alignment either — so a spec asking for word timestamps is rejected by
    #: the same real code path, not by a stub.
    word_timing_modes = ()
    _clock = None
    _SELECTIVE_UNSET = _Base._SELECTIVE_UNSET
    validate_options = _Base.validate_options
    _require_word_timings = _Base._require_word_timings


class _FakeOutputProcessor:
    """Just the ``strategy`` property admission reads."""

    strategy = _FakeStrategy()


def _admission_engine(*, overlap=False):
    """An ``ASREngine`` with only the attributes the admission path touches.

    ``ASREngine.__init__`` loads a checkpoint, which these tests deliberately
    avoid: bulk admission is pure request-construction + validation.
    """
    import queue
    import threading

    from oasr.engine.engine import ASREngine
    from oasr.engine.input_processor import InputProcessor
    from oasr.features import FeatureConfig

    eng = ASREngine.__new__(ASREngine)
    eng._lock = threading.RLock()
    eng._config = SimpleNamespace(
        service_mode="offline",
        # Admission resolves/validates the request rate against this.
        feature_config=FeatureConfig(sample_rate=16000),
    )
    eng._executor = _FakeExecutor()
    eng._output_processor = _FakeOutputProcessor()
    eng._overlap_admit = overlap
    eng._longform = None
    eng._input_processor = InputProcessor.__new__(InputProcessor)
    eng._input_processor._feature_config = eng._config.feature_config
    eng._input_processor.check_audio_duration = lambda audio: None
    eng._prep_in = queue.Queue()
    eng._admit_inflight = 0
    eng._admit_inflight_lock = threading.Lock()
    return eng


def test_bulk_admission_isolates_a_bad_spec():
    """One malformed spec must not fail its batch-mates.

    Regression: the dispatcher coalesces up to ``admit_threshold`` envelopes into
    one ``add_requests_batch`` call, so a batch-wide raise turned one client's
    out-of-range ``top_p`` into an INTERNAL error for dozens of unrelated
    requests.
    """
    eng = _admission_engine()
    specs = [
        {"request_id": "good-1", "streaming": False},
        {"request_id": "bad", "streaming": False, "decoding": {"top_p": 1.5}},
        {"request_id": "good-2", "streaming": False},
    ]
    results = eng.add_requests_batch_checked(specs)

    assert [r["request_id"] for r in results] == ["good-1", "bad", "good-2"]
    assert "error" not in results[0] and "error" not in results[2]
    assert "top_p" in results[1]["error"]
    # The rejected spec never reached the executor; the others did.
    assert eng._executor.admitted == ["good-1", "good-2"]


def test_bulk_admission_reports_mode_mismatch_per_spec():
    eng = _admission_engine()
    results = eng.add_requests_batch_checked(
        [{"request_id": "a", "streaming": False}, {"request_id": "b", "streaming": True}]
    )
    assert "error" not in results[0]
    assert "service_mode" in results[1]["error"]
    assert eng._executor.admitted == ["a"]


def test_overlap_admission_isolates_a_bad_spec():
    eng = _admission_engine(overlap=True)
    results = eng.add_requests_batch_checked(
        [
            {"request_id": "a", "streaming": False},
            {"request_id": "bad", "streaming": False, "decoding": {"temperature": 1e-30}},
        ]
    )
    assert "error" not in results[0]
    assert "temperature" in results[1]["error"]
    # Only the valid request was queued for prep and counted as in-flight.
    assert eng._admit_inflight == 1
    assert eng._prep_in.qsize() == 1


def test_add_requests_batch_still_raises_for_python_callers():
    eng = _admission_engine()
    with pytest.raises(ValueError, match="top_p"):
        eng.add_requests_batch(
            [{"request_id": "bad", "streaming": False, "decoding": {"top_p": 2.0}}]
        )
    # And returns plain ids on the happy path.
    assert eng.add_requests_batch([{"request_id": "ok", "streaming": False}]) == ["ok"]


# --------------------------------------------------------------------------- #
# Sample-rate admission
# --------------------------------------------------------------------------- #


def test_admission_defaults_the_sample_rate_to_the_models():
    """An omitted rate means the model's, not a hardcoded 16 kHz.

    ``Request.sample_rate`` used to default to 16000 independently of the
    checkpoint, which is only harmless because every in-tree checkpoint happens
    to run at 16 kHz.
    """
    import torch

    eng = _admission_engine()
    object.__setattr__(eng._config.feature_config, "sample_rate", 8000)
    eng.add_request(torch.zeros(800), request_id="r", streaming=False)
    assert eng._executor.admitted == ["r"]
    assert eng._resolve_sample_rate(None) == 8000


def test_admission_rejects_a_mismatched_sample_rate():
    """The engine does not resample, so a mismatch must fail, not transcribe.

    This is the only silent wrong-answer path a default configuration had: the
    rate rode all the way to ``Request.sample_rate`` and was then used for
    nothing, while features came out of a filterbank built for another rate.
    """
    import torch

    eng = _admission_engine()
    with pytest.raises(ValueError, match="8000 Hz.*requires 16000 Hz"):
        eng.add_request(torch.zeros(8000), request_id="r", sample_rate=8000, streaming=False)
    assert eng._executor.admitted == []


def test_bulk_admission_isolates_a_mismatched_sample_rate():
    """Per-spec, like every other admission rejection — one client's 44.1 kHz
    upload must not fail the coalesced batch it landed in."""
    import torch

    eng = _admission_engine()
    results = eng.add_requests_batch_checked(
        [
            {"request_id": "ok", "audio": torch.zeros(160), "streaming": False},
            {
                "request_id": "bad",
                "audio": torch.zeros(441),
                "sample_rate": 44100,
                "streaming": False,
            },
        ]
    )
    assert "error" not in results[0]
    assert "44100 Hz" in results[1]["error"]
    assert eng._executor.admitted == ["ok"]


def test_overlap_admission_rejects_on_the_callers_thread():
    """Under ``overlap_admit`` the same check inside ``prepare_offline`` runs on
    the prep thread, where a raise is only logged and the client waits forever
    for an output that never comes."""
    import torch

    eng = _admission_engine(overlap=True)
    with pytest.raises(ValueError, match="requires 16000 Hz"):
        eng.add_request(torch.zeros(8000), request_id="r", sample_rate=8000, streaming=False)
    assert eng._admit_inflight == 0
    assert eng._prep_in.qsize() == 0
