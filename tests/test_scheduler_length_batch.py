# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for length-aware (non-packing) offline batching.

Covers ``EngineConfig.max_batch_frames`` — a padded-frame budget that bounds
``max_len * batch_size`` per offline forward — across the two enforcement
points:

* ``Scheduler._build_offline_batch`` stops adding length-similar peers once
  the padded width would exceed the budget.
* ``Scheduler.split_offline_batch`` (frame path) re-splits a selected batch
  into micro-batches each under the budget, length-sorted to stay tight.

Both are pure-Python and need no GPU / model.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from oasr.engine.config import EngineConfig
from oasr.engine.request import Request
from oasr.engine.scheduler import Scheduler


def _make_config(
    *,
    max_batch_size: int = 16,
    schedule_policy: str = "bucket",
    max_batch_frames: Optional[int] = None,
    max_offline_pad_ratio: float = 0.0,
    length_bucket_ratio: float = 0.0,
) -> EngineConfig:
    return EngineConfig(
        ckpt_dir="/tmp/fake",
        max_batch_size=max_batch_size,
        schedule_policy=schedule_policy,
        max_batch_frames=max_batch_frames,
        # Disable the pad-ratio / bucket-ratio guards so the frame cap is the
        # only thing under test (otherwise they'd confound batch composition).
        max_offline_pad_ratio=max_offline_pad_ratio,
        length_bucket_ratio=length_bucket_ratio,
    )


def _make_offline(num_frames: int = 200) -> Request:
    req = Request("audio.wav", streaming=False)
    req.num_frames = num_frames
    return req


def _make_scheduler(*, max_batch_frames: Optional[int], mb: int = 64) -> Scheduler:
    """Build a scheduler whose ``split_offline_batch`` (frame path) is exercised."""
    return Scheduler(_make_config(max_batch_frames=max_batch_frames, max_batch_size=mb))


def _make_requests(num_frames_list: List[int]) -> List[Request]:
    out = []
    for n in num_frames_list:
        req = Request("audio.wav", streaming=False)
        req.num_frames = n
        out.append(req)
    return out


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_default_is_none(self):
        assert _make_config().max_batch_frames is None

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="max_batch_frames"):
            _make_config(max_batch_frames=0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="max_batch_frames"):
            _make_config(max_batch_frames=-5)


# ---------------------------------------------------------------------------
# Scheduler frame cap
# ---------------------------------------------------------------------------


class TestSchedulerFrameCap:
    def test_padded_width_respected(self):
        # L=200, budget=800 → at most 4 equal-length peers per batch.
        sched = Scheduler(_make_config(max_batch_frames=800))
        for _ in range(10):
            sched.add_request(_make_offline(num_frames=200))
        batch = sched.schedule_offline()
        assert len(batch) == 4
        max_len = max(r.num_frames for r in batch)
        assert max_len * len(batch) <= 800
        assert sched.num_waiting_offline == 6

    def test_lone_oversized_anchor_ships_alone(self):
        # An utterance larger than the whole budget still ships (can't split).
        sched = Scheduler(_make_config(max_batch_frames=800))
        sched.add_request(_make_offline(num_frames=1000))  # anchor, oversized
        sched.add_request(_make_offline(num_frames=100))
        sched.add_request(_make_offline(num_frames=100))
        batch = sched.schedule_offline()
        assert len(batch) == 1
        assert batch[0].num_frames == 1000
        # The two short ones remain for the next batch.
        assert sched.num_waiting_offline == 2

    def test_none_keeps_count_behaviour(self):
        # Without a frame cap, batch fills to the count cap (max_batch_size = 16).
        sched = Scheduler(_make_config(max_batch_frames=None))
        for _ in range(60):
            sched.add_request(_make_offline(num_frames=200))
        batch = sched.schedule_offline()
        assert len(batch) == 16

    def test_short_utts_pack_more_than_long(self):
        sched = Scheduler(_make_config(max_batch_frames=1000))
        for _ in range(20):
            sched.add_request(_make_offline(num_frames=100))  # 1000//100 = 10
        batch = sched.schedule_offline()
        assert len(batch) == 10


# ---------------------------------------------------------------------------
# Scheduler frame-budget split (split_offline_batch, frame path)
# ---------------------------------------------------------------------------


class TestSplitByFrames:
    def test_budget_respected_per_chunk(self):
        sched = _make_scheduler(max_batch_frames=800)
        reqs = _make_requests([200] * 10)
        chunks, _ = sched.split_offline_batch(reqs)
        # 200 * 4 = 800 fits; 200 * 5 = 1000 does not → chunks of 4.
        assert [len(c) for c in chunks] == [4, 4, 2]
        for c in chunks:
            assert max(r.num_frames for r in c) * len(c) <= 800

    def test_length_sorted_chunks(self):
        sched = _make_scheduler(max_batch_frames=600)
        reqs = _make_requests([100, 300, 50, 300, 100, 50])
        chunks, orig = sched.split_offline_batch(reqs)
        assert orig is not None
        flat = [r.num_frames for c in chunks for r in c]
        assert flat == sorted(flat)
        for c in chunks:
            assert max(r.num_frames for r in c) * len(c) <= 600

    def test_lone_oversized_ships_alone(self):
        sched = _make_scheduler(max_batch_frames=500)
        reqs = _make_requests([100, 100, 900, 100])
        chunks, _ = sched.split_offline_batch(reqs)
        # The 900-frame utt must be in a singleton chunk.
        singletons = [c for c in chunks if len(c) == 1]
        assert any(c[0].num_frames == 900 for c in singletons)

    def test_micro_batch_count_cap(self):
        # Budget is generous, but mb=3 caps each chunk at 3.
        sched = _make_scheduler(max_batch_frames=10_000, mb=3)
        reqs = _make_requests([10] * 9)
        chunks, _ = sched.split_offline_batch(reqs)
        assert [len(c) for c in chunks] == [3, 3, 3]


# ---------------------------------------------------------------------------
# Fixed-window frontends have constant cost per row
# ---------------------------------------------------------------------------


class TestFixedWindowCostModel:
    """A ``whisper_logmel`` frontend pads *and trims* every utterance to 30 s, so
    every row costs the same and the encoder throws the real lengths away.  The
    length-aware knobs must not split batches to avoid padding waste that does not
    exist, and ``max_batch_frames`` must count the real padded width.
    """

    @staticmethod
    def _cfg(**kw):
        from oasr.features import FeatureConfig

        return EngineConfig(
            device="cpu",
            service_mode="offline",
            feature_config=FeatureConfig(
                feature_type="whisper_logmel", num_mel_bins=128, dither=0.0
            ),
            **kw,
        )

    @staticmethod
    def _reqs(frames):
        out = []
        for i, n in enumerate(frames):
            r = Request(audio=None, request_id=f"r{i}", streaming=False)
            r.num_frames = n
            out.append(r)
        return out

    def test_cost_is_the_window_not_the_utterance(self):
        from oasr.engine.batching.base import request_cost_frames

        cfg = self._cfg()
        assert cfg.feature_config.fixed_window_frames == 3000
        short, long = self._reqs([98, 2900])
        assert request_cost_frames(short, cfg) == 3000
        assert request_cost_frames(long, cfg) == 3000

    def test_kaldi_frontend_still_costs_its_own_length(self):
        from oasr.engine.batching.base import request_cost_frames

        cfg = EngineConfig(device="cpu", service_mode="offline")
        assert cfg.feature_config.fixed_window_frames is None
        (r,) = self._reqs([137])
        assert request_cost_frames(r, cfg) == 137

    def test_mixed_lengths_batch_together(self):
        """The pad-ratio guard used to split a 1 s + 30 s pair for nothing."""
        cfg = self._cfg(max_batch_size=8, max_offline_pad_ratio=1.5, length_bucket_ratio=0.8)
        sched = Scheduler(cfg)
        for r in self._reqs([98, 2900, 300, 1500]):
            sched.add_request(r)
        batch = sched.schedule_offline()
        assert len(batch) == 4, "equal-cost rows must not be split by padding heuristics"

    def test_frame_budget_counts_the_real_window(self):
        """``max_batch_frames`` bounds padded frames; under a fixed window it must
        use 3000/row, not the ~98 a 1 s clip reports."""
        cfg = self._cfg(max_batch_size=8, max_batch_frames=6000)
        sched = Scheduler(cfg)
        for r in self._reqs([98] * 4):
            sched.add_request(r)
        batch = sched.schedule_offline()
        chunks, _ = sched.split_offline_batch(batch)
        assert all(len(c) <= 2 for c in chunks), (
            f"6000-frame budget at 3000/row allows 2 rows per micro-batch, got "
            f"{[len(c) for c in chunks]}"
        )
