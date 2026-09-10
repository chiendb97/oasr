# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr._C.streaming.append_features`` against the Python loop it replaces.

The streaming feature-ring append exists twice: in C++ (used when ``_C`` is
built) and in Python (``InputProcessor._plan_append_features``, the reference,
and the fallback on a CPU-only install).  Two implementations of one rule is the
same hazard as the alignment pass — they can drift, and the symptom is a
transcript that is merely *slightly* wrong.  These tests are the oracle that
keeps them equal.

Set ``OASR_STREAMING_NATIVE=0`` to force the Python path at runtime.
"""

from __future__ import annotations

import random

import pytest
import torch

from oasr.engine.input_processor import (
    _FEATURE_HEADROOM_APPENDS,
    _FEATURE_HEADROOM_MAX,
    InputProcessor,
)

_C_streaming = None
try:
    from oasr import _C  # type: ignore[attr-defined]

    _C_streaming = _C.streaming
except (ImportError, AttributeError):  # pragma: no cover
    _C_streaming = None

needs_native = pytest.mark.skipif(_C_streaming is None, reason="oasr._C.streaming not built")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FEAT_DIM = 8


class _Req:
    """The four attributes the append reads and writes."""

    __slots__ = ("feature_buffer", "feature_frames", "feature_cursor", "feature_base")

    def __init__(self):
        self.feature_buffer = None
        self.feature_frames = 0
        self.feature_cursor = 0
        self.feature_base = 0

    def state(self):
        return (self.feature_frames, self.feature_cursor, self.feature_base)


def _python_step(reqs, feats, lens):
    dsts: list = []
    srcs: list = []
    for i, r in enumerate(reqs):
        if lens[i] > 0:
            InputProcessor._plan_append_features(
                None, r, feats[i, : lens[i], :], FEAT_DIM, dsts, srcs
            )
    if len(dsts) == 1:
        dsts[0].copy_(srcs[0])
    elif dsts:
        torch._foreach_copy_(dsts, srcs)


def _native_step(reqs, feats, lens):
    buffers, frames, cursors, base_delta, n_realloc = _C_streaming.append_features(
        [r.feature_buffer for r in reqs],
        [r.feature_frames for r in reqs],
        [r.feature_cursor for r in reqs],
        feats,
        list(lens),
        FEAT_DIM,
        _FEATURE_HEADROOM_APPENDS,
        _FEATURE_HEADROOM_MAX,
    )
    for r, b, f, c, d in zip(reqs, buffers, frames, cursors, base_delta):
        r.feature_buffer, r.feature_frames, r.feature_cursor = b, f, c
        r.feature_base += d
    return n_realloc


def _feats(B, T, start, device):
    return torch.arange(
        start, start + B * T * FEAT_DIM, dtype=torch.float32, device=device
    ).reshape(B, T, FEAT_DIM)


@needs_native
@needs_cuda
class TestMatchesThePythonReference:
    @pytest.mark.parametrize("seed", range(6))
    def test_random_streaming_history(self, seed):
        """Drive both over the same history and compare at every step."""
        dev = torch.device("cuda")
        rng = random.Random(seed)
        B = rng.randint(1, 6)
        py_reqs = [_Req() for _ in range(B)]
        cpp_reqs = [_Req() for _ in range(B)]
        counter = 0.0

        for step in range(30):
            # Zero-length appends are the common case for a stream that has no
            # full window this tick; they must pass through untouched.
            lens = [rng.choice([0, 1, 62, rng.randint(1, 70)]) for _ in range(B)]
            T = max(1, max(lens))
            feats = _feats(B, T, counter, dev)
            counter += B * T * FEAT_DIM

            _python_step(py_reqs, feats, lens)
            _native_step(cpp_reqs, feats, lens)

            for i, (a, b) in enumerate(zip(py_reqs, cpp_reqs)):
                where = f"seed {seed} step {step} stream {i}"
                assert a.state() == b.state(), f"{where}: bookkeeping diverged"
                assert (a.feature_buffer is None) == (b.feature_buffer is None), where
                if a.feature_buffer is not None:
                    assert a.feature_buffer.size(0) == b.feature_buffer.size(0), (
                        f"{where}: capacity {a.feature_buffer.size(0)} vs "
                        f"{b.feature_buffer.size(0)}"
                    )
                    n = a.feature_frames
                    assert torch.equal(
                        a.feature_buffer[:n], b.feature_buffer[:n]
                    ), f"{where}: live frames differ"

            # Consume like the engine does, keeping both sides in lockstep.
            for a, b in zip(py_reqs, cpp_reqs):
                a.feature_cursor = min(a.feature_cursor + rng.randint(0, 62), a.feature_frames)
                b.feature_cursor = a.feature_cursor


@needs_native
@needs_cuda
class TestContract:
    def test_a_stream_with_no_new_frames_is_untouched(self):
        dev = torch.device("cuda")
        reqs = [_Req(), _Req()]
        _native_step(reqs, _feats(2, 4, 0, dev), [4, 0])
        assert reqs[0].feature_frames == 4
        assert reqs[1].feature_frames == 0 and reqs[1].feature_buffer is None

    def test_compaction_preserves_the_absolute_frame_index(self):
        """``feature_base + feature_cursor`` is what the speech gate reads, so a
        rebase must move exactly what it removed."""
        dev = torch.device("cuda")
        r = _Req()
        for _ in range(200):
            before = r.feature_base + r.feature_cursor
            _native_step([r], _feats(1, 62, 0, dev), [62])
            assert r.feature_base + r.feature_cursor == before
            r.feature_cursor = min(r.feature_cursor + 62, r.feature_frames)

    def test_live_frames_survive_compaction_bit_exactly(self):
        dev = torch.device("cuda")
        r = _Req()
        expected: list = []
        counter = 0.0
        for _ in range(60):
            feats = _feats(1, 62, counter, dev)
            counter += 62 * FEAT_DIM
            _native_step([r], feats, [62])
            expected.extend(feats[0, :, 0].tolist())
            r.feature_cursor = max(0, r.feature_frames - 40)
            live = r.feature_buffer[r.feature_cursor : r.feature_frames, 0]
            assert live.tolist() == expected[-live.numel() :]

    def test_reallocation_is_rare_at_steady_state(self):
        """The same property ``test_feature_buffer_growth.py`` pins for Python."""
        dev = torch.device("cuda")
        r = _Req()
        reallocs = 0
        for _ in range(200):
            reallocs += _native_step([r], _feats(1, 62, 0, dev), [62])
            r.feature_cursor = min(r.feature_cursor + 62, r.feature_frames)
        assert reallocs / 200 < 0.15, f"{reallocs / 200:.1%} of appends reallocated"

    def test_mismatched_lengths_are_rejected(self):
        dev = torch.device("cuda")
        with pytest.raises((ValueError, RuntimeError)):
            _C_streaming.append_features(
                [None, None],
                [0],
                [0, 0],
                _feats(2, 4, 0, dev),
                [4, 4],
                FEAT_DIM,
                _FEATURE_HEADROOM_APPENDS,
                _FEATURE_HEADROOM_MAX,
            )
