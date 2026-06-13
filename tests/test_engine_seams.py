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


# --------------------------------------------------------------------------- #
# Decode-strategy registry / dispatch
# --------------------------------------------------------------------------- #


def test_decode_registry_has_builtins():
    assert set(DECODE_REGISTRY) == {"ctc_cuda", "ctc_wfst", "transducer", "aed", "llm"}


def test_build_ctc_strategies_by_decoder_type():
    detok = Detokenizer(None, None)
    gpu = build_decode_strategy("ctc", _stub_config("ctc_cuda"), detok)
    wfst = build_decode_strategy("ctc", _stub_config("ctc_wfst"), detok)
    assert type(gpu).__name__ == "CtcGpuDecodeStrategy"
    assert type(wfst).__name__ == "CtcWfstDecodeStrategy"
    assert gpu.consumes == "log_probs" and gpu.decode_type == "ctc"


@pytest.mark.parametrize("dt", ["transducer", "aed", "llm"])
def test_ar_strategies_resolve_and_consume_hidden(dt):
    s = build_decode_strategy(dt, _stub_config(), Detokenizer(None, None))
    assert s.decode_type == dt
    assert s.consumes == "hidden"


@pytest.mark.parametrize("dt", ["aed", "llm"])
def test_aed_llm_skeletons_raise_not_implemented(dt):
    s = build_decode_strategy(dt, _stub_config(), Detokenizer(None, None))
    with pytest.raises(NotImplementedError):
        s.decode_offline(None, None)
    with pytest.raises(NotImplementedError):
        s.finalize(None)


def test_transducer_offline_implemented_streaming_not():
    # transducer is a real strategy now: decode_offline works (tested in
    # test_transducer.py); only its streaming path is a follow-up.
    s = build_decode_strategy("transducer", _stub_config(), Detokenizer(None, None))
    assert type(s).__name__ == "TransducerDecodeStrategy"
    with pytest.raises(NotImplementedError):
        s.finalize(None)


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
