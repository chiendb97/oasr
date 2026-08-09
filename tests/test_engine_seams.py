# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the engine's pluggable seams (decode strategies + batching policies).

These exercise the registries / dispatch + the autoregressive extension-point
contract without needing CUDA or a checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from oasr.engine.decode import Detokenizer, build_decode_strategy
from oasr.engine.decode.base import _REGISTRY as DECODE_REGISTRY


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
# Sample-rate admission (C2)
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
