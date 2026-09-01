# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-stream feature-buffer growth in ``InputProcessor._plan_append_features``.

At streaming steady state a stream consumes about as many feature frames per tick
as it gains, so ``feature_cursor`` tracks ``feature_frames``.  The original
compaction rule (``cursor >= have // 2``) was therefore true on almost every
append, and each compaction re-sized the buffer from the *live* window rather
than from the old capacity — to ``2 x live``, which the next append refills.
Measured on a real engine: **91.5 % of appends reallocated**, ~14 per tick at 16
streams, each an allocation plus a zero-fill launch.

These tests pin the loop closed.  They are pure host-side bookkeeping, so they
need neither CUDA nor a checkpoint.
"""

from __future__ import annotations

import torch

from oasr.engine.input_processor import InputProcessor

FEAT_DIM = 8
N_NEW = 62  # one encoder window's worth of new frames, as streaming produces


class _Req:
    """The attributes ``_plan_append_features`` reads and writes."""

    def __init__(self):
        self.feature_buffer = None
        self.feature_frames = 0
        self.feature_cursor = 0
        self.feature_base = 0


def _steady_state(ticks: int, n_new: int = N_NEW, consume: int | None = None):
    """Append ``n_new`` frames per tick and consume as many, as a live stream does.

    Returns ``(reallocations, appends, final_capacity)``.
    """
    consume = n_new if consume is None else consume
    req = _Req()
    reallocs = 0
    for _ in range(ticks):
        dsts: list = []
        srcs: list = []
        before = req.feature_buffer
        InputProcessor._plan_append_features(
            None, req, torch.zeros(n_new, FEAT_DIM), FEAT_DIM, dsts, srcs
        )
        if req.feature_buffer is not before:
            reallocs += 1
        for d, s in zip(dsts, srcs):
            d.copy_(s)
        req.feature_cursor = min(req.feature_cursor + consume, req.feature_frames)
    return reallocs, ticks, req.feature_buffer.size(0)


class TestSteadyStateDoesNotThrash:
    def test_reallocation_is_rare(self):
        reallocs, appends, _ = _steady_state(200)
        rate = reallocs / appends
        # The old rule reallocated on ~91 % of appends; anything near that is the
        # thrash returning.
        assert rate < 0.15, f"{rate:.1%} of appends reallocated (was ~91 % before)"

    def test_capacity_exceeds_the_live_window_by_real_headroom(self):
        """``2 x live`` is what made the loop self-perpetuating."""
        _, _, cap = _steady_state(200)
        assert cap >= 4 * N_NEW, f"capacity {cap} leaves no room to amortise"

    def test_buffer_does_not_grow_without_bound(self):
        """Compaction must still happen — ``have`` grows every tick forever."""
        _, _, cap = _steady_state(2000)
        assert cap < 40 * N_NEW, f"capacity {cap} suggests compaction stopped"


class TestFrameBookkeepingSurvivesCompaction:
    def test_absolute_frame_index_is_preserved(self):
        """``feature_base + feature_cursor`` is the stream's absolute input-frame
        index — the gate reads it to decide which seconds a window covers, so a
        compaction that rebases the cursor must move the same amount into the
        base."""
        req = _Req()
        for tick in range(120):
            dsts: list = []
            srcs: list = []
            before_abs = req.feature_base + req.feature_cursor
            InputProcessor._plan_append_features(
                None, req, torch.zeros(N_NEW, FEAT_DIM), FEAT_DIM, dsts, srcs
            )
            for d, s in zip(dsts, srcs):
                d.copy_(s)
            assert (
                req.feature_base + req.feature_cursor == before_abs
            ), f"tick {tick}: compaction moved the absolute frame index"
            req.feature_cursor = min(req.feature_cursor + N_NEW, req.feature_frames)

    def test_live_frames_survive_a_compaction_bit_exactly(self):
        """Whatever a compaction keeps must be the same frames, in order."""
        req = _Req()
        counter = 0.0
        expected: list[float] = []
        for _ in range(80):
            dsts: list = []
            srcs: list = []
            block = (
                torch.arange(counter, counter + N_NEW, dtype=torch.float32)
                .unsqueeze(1)
                .expand(N_NEW, FEAT_DIM)
                .contiguous()
            )
            counter += N_NEW
            InputProcessor._plan_append_features(None, req, block, FEAT_DIM, dsts, srcs)
            for d, s in zip(dsts, srcs):
                d.copy_(s)
            expected.extend(block[:, 0].tolist())
            # Consume all but a 40-frame tail, so something real is always live.
            keep = 40
            req.feature_cursor = max(0, req.feature_frames - keep)
            live = req.feature_buffer[req.feature_cursor : req.feature_frames, 0]
            assert live.tolist() == expected[-live.numel() :], "live frames corrupted"


class TestUnconsumedStreamStillGrows:
    def test_a_backlogged_stream_keeps_every_frame(self):
        """A stream nobody consumes must not lose frames to the new rule."""
        reallocs, _, cap = _steady_state(60, consume=0)
        assert cap >= 60 * N_NEW, "buffer failed to grow for an unconsumed stream"
        assert reallocs < 30, f"{reallocs} reallocations to grow 60 appends"


class TestFixedWindowFrontendHeadroomIsBounded:
    def test_a_3000_frame_window_does_not_reserve_tens_of_thousands(self):
        """``whisper_logmel`` hands over a whole 30 s window at once; headroom is
        capped in absolute frames so that does not multiply."""
        _, _, cap = _steady_state(6, n_new=3000)
        assert cap < 3000 * 4, f"capacity {cap} for a 3000-frame window"
