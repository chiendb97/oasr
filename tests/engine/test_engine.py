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

    def test_finished_requests_leave_the_scheduler(self, device, ckpt_dir: str, wav_dir: str):
        """An offline server's request index must not grow with its traffic."""
        engine = self._make_engine(ckpt_dir, device)
        for _ in range(2):
            engine.transcribe_offline(waveforms(wav_dir, 3))
        assert engine._scheduler._index == {}

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


def test_construction_refuses_a_device_of_another_architecture_before_loading(monkeypatch):
    """A process compiles its kernels for one architecture, the current device's.
    An engine on a device of another family would load its weights and then fail
    at the first kernel, so construction refuses it before the load."""
    import oasr.engine.engine as engine_mod
    import oasr.jit.core as jit_core
    from oasr.engine.config import EngineConfig

    def _no_load(*args, **kwargs):
        raise AssertionError("the checkpoint was loaded before the device was checked")

    caps = {0: (12, 0), 1: (9, 0)}
    monkeypatch.setattr(engine_mod, "load_pretrained", _no_load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: caps[index])
    monkeypatch.setattr(jit_core, "_detect_cuda_arch", lambda: caps[0])
    monkeypatch.setattr(jit_core, "_arch_from_env", lambda: None)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        engine_mod.ASREngine(EngineConfig(ckpt_dir="/nonexistent", device="cuda:1"))


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

    assert set(_BATCHING) == {"fcfs", "bucket", "sjf", "window"}
    assert set(_PARTITION) == {"count", "frames", "packing"}

    cfg = SimpleNamespace(
        schedule_policy="bucket",
        enable_sequence_packing=False,
        max_batch_frames=None,
    )
    assert type(build_batching_policy(cfg)).__name__ == "BucketPolicy"
    cfg.schedule_policy = "window"
    assert type(build_batching_policy(cfg)).__name__ == "WindowPolicy"
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
        self.requests = []

    def admit(self, request):
        self.admitted.append(request.request_id)
        self.requests.append(request)


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
    import threading

    from oasr.engine.admission_prep import AdmissionPrep
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
    # Not started: these tests read what was queued for the prep thread.
    eng._prep = AdmissionPrep(eng._input_processor.prepare_offline) if overlap else None
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
    assert eng._prep.in_flight == 1
    assert eng._prep._todo.qsize() == 1


@pytest.mark.parametrize("overlap", [False, True], ids=["direct", "overlap"])
def test_bulk_admission_clamps_a_negative_priority(overlap):
    """The dispatcher hands the wire's ``priority`` (an unchecked ``i32``) to this
    entry point.  Lower is sooner, so a negative value put that client's requests
    ahead of every other client's; it now lands in the default class."""
    eng = _admission_engine(overlap=overlap)
    specs = [
        {"request_id": "a", "streaming": False, "priority": -(2**31)},
        {"request_id": "b", "streaming": False, "priority": 4},
    ]
    results = eng.add_requests_batch_checked(specs)
    assert all("error" not in r for r in results)
    if overlap:
        admitted = [eng._prep._todo.get_nowait() for _ in specs]
    else:
        admitted = eng._executor.requests
    assert [r.priority for r in admitted] == [0, 4]


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
    assert eng._prep.in_flight == 0
    assert eng._prep._todo.qsize() == 0


# --------------------------------------------------------------------------- #
# Long-form and VAD fan-out on every admission path
# --------------------------------------------------------------------------- #


class _AbortingExecutor(_FakeExecutor):
    """Admits ``fail_at`` windows, then refuses; records what was aborted."""

    def __init__(self, fail_at=None):
        super().__init__()
        self.fail_at = fail_at
        self.aborted = []

    def admit(self, request):
        if self.fail_at is not None and len(self.admitted) == self.fail_at:
            raise RuntimeError("executor refused")
        super().admit(request)

    def abort(self, request_id):
        self.aborted.append(request_id)


def _fan_out_engine(*, overlap=False, window_s=1.0, spans=None):
    """:func:`_admission_engine` with a long-form tracker, and optionally a VAD
    splitter that returns ``spans``."""
    from oasr.engine.longform import LongFormTracker

    eng = _admission_engine(overlap=overlap)
    eng._executor = _AbortingExecutor()
    eng._longform = LongFormTracker()
    eng._longform_window_samples = int(16000 * window_s)
    eng._longform_overlap_samples = 0
    eng._vad_splitter = None if spans is None else SimpleNamespace(spans=lambda wave: spans)
    eng._vad_resplit_samples = 0
    eng._metrics = SimpleNamespace(incr=lambda *a: None)
    return eng


def _queued(eng):
    """The ids queued for the prep thread, in order."""
    ids = []
    while not eng._prep._todo.empty():
        ids.append(eng._prep._todo.get_nowait().request_id)
    return ids


@pytest.mark.parametrize("overlap", [False, True], ids=["admit", "overlap_admit"])
def test_bulk_admission_fans_out_long_audio(overlap):
    """The serving dispatcher's entry point splits over-window audio too.

    Regression: only ``add_request`` reached the fan-out, and the dispatcher
    admits through ``add_requests_batch_checked``, so ``oasr-server --long-form``
    decoded nothing past one window: Whisper refused every request over 30 s.
    """
    eng = _fan_out_engine(overlap=overlap)
    results = eng.add_requests_batch_checked(
        [
            {"request_id": "long", "audio": torch.zeros(40000), "streaming": False},
            {"request_id": "short", "audio": torch.zeros(8000), "streaming": False},
        ]
    )
    assert results == [{"request_id": "long"}, {"request_id": "short"}]
    children = [eng._longform.child_id("long", i) for i in range(3)]
    admitted = _queued(eng) if overlap else eng._executor.admitted
    assert admitted == children + ["short"]
    assert all(eng._longform.owns(c) for c in children)
    if overlap:
        assert eng._prep.in_flight == 4


@pytest.mark.parametrize("overlap", [False, True], ids=["admit", "overlap_admit"])
def test_bulk_admission_segments_at_the_vad_spans(overlap):
    """Same for ``--vad-mode segment``, which reuses the fan-out."""
    eng = _fan_out_engine(overlap=overlap, spans=[(0, 8000), (24000, 40000)])
    results = eng.add_requests_batch_checked(
        [{"request_id": "file", "audio": torch.zeros(48000), "streaming": False}]
    )
    assert results == [{"request_id": "file"}]
    admitted = _queued(eng) if overlap else eng._executor.admitted
    assert admitted == [eng._longform.child_id("file", i) for i in range(2)]


@pytest.mark.parametrize(
    "overlap_samples,spans,expected",
    [
        # Fixed windows of 16000 with no overlap: each ends where the next starts.
        (0, None, [(0, 16000), (16000, 32000), (32000, 40000)]),
        # 4000 samples of overlap: stride 12000.
        (4000, None, [(0, 16000), (12000, 28000), (24000, 40000)]),
        # VAD spans, cut in silence: a gap between them.
        (0, [(0, 8000), (24000, 40000)], [(0, 8000), (24000, 40000)]),
    ],
    ids=["touching", "overlapping", "vad_gap"],
)
def test_the_fan_out_registers_where_each_window_ends(overlap_samples, spans, expected):
    """The merge drops a repeat only where windows share audio, so it is told
    where each one ends -- from sample counts, so windows that only touch
    compare equal rather than overlapping by a rounding error."""
    eng = _fan_out_engine(spans=spans)
    eng._longform_overlap_samples = overlap_samples
    eng.add_requests_batch_checked(
        [{"request_id": "long", "audio": torch.zeros(40000), "streaming": False}]
    )
    entry = eng._longform._pending["long"]
    assert entry.window_starts_s == [s / 16000 for s, _ in expected]
    assert entry.window_ends_s == [e / 16000 for _, e in expected]


def test_a_rejected_long_request_registers_no_windows():
    """Options are validated before the fan-out, not by each window.

    Regression: the windows were registered first and then each rejected, so the
    caller got its error and the tracker kept a parent that could never finish.
    """
    eng = _fan_out_engine()
    with pytest.raises(ValueError, match="top_p"):
        eng.add_request(torch.zeros(40000), streaming=False, decoding={"top_p": 1.5})
    results = eng.add_requests_batch_checked(
        [
            {
                "request_id": "bad",
                "audio": torch.zeros(40000),
                "streaming": False,
                "decoding": {"top_p": 1.5},
            }
        ]
    )
    assert "top_p" in results[0]["error"]
    assert not eng._longform
    assert eng._executor.admitted == []


def test_a_window_the_executor_refuses_aborts_its_siblings():
    eng = _fan_out_engine()
    eng._executor.fail_at = 1
    results = eng.add_requests_batch_checked(
        [{"request_id": "long", "audio": torch.zeros(40000), "streaming": False}]
    )
    assert "executor refused" in results[0]["error"]
    assert not eng._longform
    assert eng._executor.aborted == [eng._longform.child_id("long", i) for i in range(3)]


# --------------------------------------------------------------------------- #
# Overlapped admission: failures, aborts and the audio count (audit L8)
# --------------------------------------------------------------------------- #


class _PreppedExecutor(_AbortingExecutor):
    """Finishes every request admitted to it on the next step.

    ``reject_prep`` is the real executor's, so the error output under test is the
    one a served engine returns.
    """

    def __init__(self):
        super().__init__()
        self.prepared = []
        self._ready = []

    def admit(self, request):  # pragma: no cover - overlap admits through admit_prepared
        raise AssertionError("an overlapped admission must not prepare a request twice")

    def admit_prepared(self, request):
        self.prepared.append(request.request_id)
        self._ready.append(request)

    def reject_prep(self, request, exc):
        from oasr.engine.executor.offline import OfflineExecutor

        return OfflineExecutor.reject_prep(self, request, exc)

    def _reject(self, request, reason, stage="unknown"):
        from oasr.engine.executor.offline import OfflineExecutor

        return OfflineExecutor._reject(self, request, reason, stage)

    def abort(self, request_id):
        super().abort(request_id)
        self._ready = [r for r in self._ready if r.request_id != request_id]

    def step(self):
        from oasr.engine.request import RequestOutput

        done, self._ready = self._ready, []
        return [
            RequestOutput(request_id=r.request_id, text="ok", tokens=[[1]], finished=True)
            for r in done
        ]

    def num_waiting(self):
        return 0

    def has_pending(self):
        return bool(self._ready)


def _overlap_engine(prepare, *, window_s=None):
    """An overlapped-admission engine whose prep thread runs ``prepare``.

    ``window_s`` adds a long-form splitter, so a request fans out into windows.
    """
    from oasr.engine.admission_prep import AdmissionPrep
    from oasr.engine.longform import LongFormTracker
    from oasr.engine.metrics import EngineMetrics

    eng = _admission_engine(overlap=True)
    eng._executor = _PreppedExecutor()
    eng._metrics = EngineMetrics()
    eng._input_processor = SimpleNamespace(
        check_audio_duration=lambda audio: None, wants_pinned_trim=False
    )
    eng._prep = AdmissionPrep(prepare)
    if window_s is not None:
        eng._longform = LongFormTracker()
        eng._longform_window_samples = int(16000 * window_s)
        eng._longform_overlap_samples = 0
        eng._vad_splitter = None
        eng._vad_resplit_samples = 0
    eng._prep.start()
    return eng


def _wait_prepared(eng, n, timeout=5.0):
    """Block until the prep thread has finished with ``n`` requests."""
    import time

    deadline = time.monotonic() + timeout
    while eng._prep._done.qsize() < n:
        assert time.monotonic() < deadline, "the prep thread did not finish"
        time.sleep(0.001)


def _drain(eng, max_steps=50):
    """``eng.run()``, bounded: a request the engine never lets go fails the test.

    ``run()`` loops until nothing is waiting, so the regressions these tests pin
    (a count that is never released) would hang it rather than fail.
    """
    import time

    outs = []
    for _ in range(max_steps):
        outs += [o for o in eng.step() if o.finished]
        if not eng._executor.has_pending() and eng._prep.in_flight == 0:
            return outs
        time.sleep(0.01)  # the prep thread may still be preparing
    raise AssertionError(f"still waiting after {max_steps} steps: {eng._prep.in_flight} in prep")


def _refuse(names):
    """A ``prepare`` that raises for the request ids in ``names``."""

    def prepare(req):
        if req.request_id in names:
            raise TypeError(f"cannot read the audio of {req.request_id}")

    return prepare


def test_overlapped_admission_answers_a_failed_prep_with_an_error():
    """The caller already holds the id, so the failure has to come back as an output.

    Regression: the prep thread logged the exception and dropped the request,
    so the client waited for its timeout. The request now counts as waiting
    until a step has delivered its error, or the dispatcher could stop stepping
    an engine with an answer still to give.
    """
    eng = _overlap_engine(_refuse({"bad"}))
    try:
        eng.add_request(torch.zeros(160), request_id="bad", streaming=False)
        eng.add_request(torch.zeros(160), request_id="good", streaming=False)
        _wait_prepared(eng, 2)
        assert eng.num_waiting == 2
        outs = {o.request_id: o for o in eng.step()}
    finally:
        eng._prep.stop()
    assert outs["bad"].finished and outs["bad"].finish_reason == "error"
    assert outs["bad"].error_stage == "offline_prep"
    assert outs["good"].finish_reason is None and outs["good"].text == "ok"
    assert eng._executor.prepared == ["good"]
    assert eng._prep.in_flight == 0


def test_a_window_whose_prep_fails_fails_its_file():
    """A window's error output is stitched into its parent, like a forward failure."""
    eng = _overlap_engine(lambda req: None, window_s=1.0)
    child = eng._longform.child_id("long", 1)
    eng._prep._prepare = _refuse({child})
    try:
        eng.add_request(torch.zeros(40000), request_id="long", streaming=False)
        _wait_prepared(eng, 3)
        outs = eng.step()
    finally:
        eng._prep.stop()
    assert [(o.request_id, o.finish_reason, o.error_stage) for o in outs] == [
        ("long", "error", "offline_prep")
    ]


@pytest.mark.parametrize("stage", ["queued", "prepared"])
def test_an_abort_during_overlapped_admission_is_not_decoded(stage):
    """An abort reaches a request the scheduler has not seen yet.

    Regression: only the scheduler was searched, so a request still queued for
    prep, or prepared and waiting for the next step, was admitted afterwards
    and decoded for a client that had gone.
    """
    eng = _overlap_engine(lambda req: None)
    if stage == "queued":
        eng._prep.stop()  # nothing prepares it, so it stays queued
    try:
        eng.add_request(torch.zeros(160), request_id="gone", streaming=False)
        eng.add_request(torch.zeros(160), request_id="kept", streaming=False)
        if stage == "prepared":
            _wait_prepared(eng, 2)
        eng.abort_request("gone")
        if stage == "queued":
            eng._prep.start()
            _wait_prepared(eng, 1)
        outs = _drain(eng)
    finally:
        eng._prep.stop()
    assert [o.request_id for o in outs] == ["kept"]
    assert eng._executor.prepared == ["kept"]
    assert eng._prep.in_flight == 0


def test_aborting_a_long_file_withdraws_its_windows_from_prep():
    """The windows were queued for prep, not admitted; the abort has to find them there.

    On the old path each window was decoded anyway and, its parent gone, came
    back under a child id the client never saw.
    """
    eng = _overlap_engine(lambda req: None, window_s=1.0)
    try:
        eng.add_request(torch.zeros(40000), request_id="long", streaming=False)
        _wait_prepared(eng, 3)
        eng.abort_request("long")
        outs = _drain(eng)
    finally:
        eng._prep.stop()
    assert outs == []
    assert eng._executor.prepared == []
    assert eng._prep.in_flight == 0


def test_an_abort_spares_a_later_request_reusing_the_id():
    """Withdrawal is by request, not by id: nothing makes ids unique."""
    import threading

    entered, gate = threading.Event(), threading.Event()

    def prepare(req):
        entered.set()
        gate.wait(5.0)

    eng = _overlap_engine(prepare)
    try:
        eng.add_request(torch.zeros(160), request_id="r", streaming=False)
        assert entered.wait(5.0)  # the first one is being prepared
        eng.abort_request("r")
        eng.add_request(torch.zeros(160), request_id="r", streaming=False)
        gate.set()
        _wait_prepared(eng, 2)
        outs = _drain(eng)
    finally:
        eng._prep.stop()
    assert [o.request_id for o in outs] == ["r"]
    assert eng._executor.prepared == ["r"]


def test_an_overlapped_admission_counts_its_audio():
    """The drain admits through the executor, which is where the audio is counted.

    Regression: it went straight into the scheduler, so
    ``oasr_engine_audio_seconds_total`` stayed at zero under ``--overlap-admit``.
    """
    from oasr.engine import metrics as m
    from oasr.engine.executor.offline import OfflineExecutor

    added = []
    ex = OfflineExecutor.__new__(OfflineExecutor)
    ex._metrics = m.EngineMetrics()
    ex._scheduler = SimpleNamespace(add_request=added.append)
    req = SimpleNamespace(audio=torch.zeros(24000), sample_rate=16000)
    ex.admit_prepared(req)
    assert added == [req]
    assert ex._metrics.snapshot()["counters"][m.AUDIO_SECONDS] == 1.5


@pytest.fixture(scope="class", params=[False, True], ids=["direct", "overlap_admit"])
def admission_engine(request, device, ckpt_dir):
    """A real offline engine, with and without ``overlap_admit``, one at a time."""
    from helpers import make_engine

    eng = make_engine(ckpt_dir, device=str(device), overlap_admit=request.param)
    yield eng, request.param
    eng.shutdown()


class TestOverlappedAdmissionOnARealEngine:
    """What a caller sees must not depend on ``overlap_admit`` (audit L8).

    Public API only, so the same tests run unchanged against the old code.
    """

    def test_audio_that_cannot_be_read_is_reported(self, admission_engine):
        """Direct admission raises to the caller. Overlapped admission has already
        returned the id, so it answers with an error output."""
        import numpy as np

        eng, overlap = admission_engine
        try:
            eng.add_request(np.array(["not", "audio"]), request_id="bad", streaming=False)
        except TypeError:
            assert not overlap, "overlapped admission defers the read to its prep thread"
            return
        outs = eng.run()
        assert [(o.request_id, o.finish_reason, o.error_stage) for o in outs] == [
            ("bad", "error", "offline_prep")
        ]

    def test_an_aborted_request_is_not_decoded(self, admission_engine, wav_dir):
        eng, _ = admission_engine
        first, second = waveforms(wav_dir, 2)
        eng.add_request(first, request_id="gone", streaming=False)
        eng.abort_request("gone")
        eng.add_request(second, request_id="kept", streaming=False)
        outs = eng.run()
        assert [o.request_id for o in outs] == ["kept"]
        assert outs[0].text

    def test_the_audio_is_counted(self, admission_engine, wav_dir):
        from oasr.engine import metrics as m

        eng, _ = admission_engine
        waves = waveforms(wav_dir, 4)

        def counted():
            return eng.metrics_snapshot()["counters"].get(m.AUDIO_SECONDS, 0.0)

        before = counted()
        for i, w in enumerate(waves):
            eng.add_request(w, request_id=f"count-{i}", streaming=False)
        assert len(eng.run()) == len(waves)
        expected = sum(int(w.numel()) for w in waves) / 16000.0
        assert counted() - before == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Results the search could not complete (audit #13)
# ---------------------------------------------------------------------------


def test_step_counts_and_logs_results_the_search_could_not_complete(caplog):
    """A WFST lane that hit a capacity limit was returned as if it were complete.

    It is still returned — it is the best the decoder found — but the engine now
    counts every *finished* one and logs it.  Partials are not counted: a stream
    reports its overflow on every partial after it, so counting them would count
    one result once per tick.
    """
    import logging
    import threading

    from oasr.engine import metrics as m
    from oasr.engine.engine import ASREngine
    from oasr.engine.request import RequestOutput

    outs = [
        RequestOutput(request_id="clean", text="a", tokens=[[1]], finished=True),
        RequestOutput(
            request_id="short", text="a", tokens=[[1]], finished=True, search_overflow=16
        ),
        RequestOutput(
            request_id="partial", text="a", tokens=[[1]], finished=False, search_overflow=4
        ),
    ]
    eng = ASREngine.__new__(ASREngine)
    eng._lock = threading.RLock()
    eng._overlap_admit = False
    eng._longform = None
    eng._executor = SimpleNamespace(step=lambda: list(outs))
    eng._metrics = m.EngineMetrics()
    eng._input_processor = SimpleNamespace(wants_pinned_trim=False)  # no pinned audio

    with caplog.at_level(logging.WARNING, logger="oasr.engine.engine"):
        assert eng.step() == outs
    assert eng._metrics.snapshot()["counters"][m.SEARCH_OVERFLOW] == 1.0
    logged = [r.getMessage() for r in caplog.records if "overflowed" in r.getMessage()]
    assert len(logged) == 1 and "short" in logged[0]


def test_long_form_keeps_a_window_that_overflowed():
    """A stitched transcript is only as complete as its worst window."""
    from oasr.engine.longform import LongFormTracker
    from oasr.engine.request import RequestOutput

    tracker = LongFormTracker()
    tracker.register("file", ["file#0", "file#1"], [0.0, 30.0], [30.0, 60.0])
    parent = tracker.absorb(
        [
            RequestOutput(request_id="file#0", text="a b", tokens=[[1, 2]], finished=True),
            RequestOutput(
                request_id="file#1",
                text="c",
                tokens=[[3]],
                finished=True,
                search_overflow=16,
            ),
        ]
    )
    assert [p.request_id for p in parent] == ["file"]
    assert parent[0].search_overflow == 16

    tracker.register("clean", ["clean#0"], [0.0], [30.0])
    (out,) = tracker.absorb(
        [RequestOutput(request_id="clean#0", text="a", tokens=[[1]], finished=True)]
    )
    assert out.search_overflow is None
