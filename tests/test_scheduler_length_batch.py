# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for length-aware (non-packing) offline batching.

Covers ``EngineConfig.max_batch_frames`` — a padded-frame budget that bounds
``max_len * batch_size`` per offline forward — across the two enforcement
points:

* ``Scheduler._build_offline_batch`` stops adding length-similar peers once
  the padded width would exceed the budget.
* ``OfflinePipeline._split_by_frames`` re-splits an admitted pool into
  micro-batches each under the budget, length-sorted to stay tight.

Both are pure-Python and need no GPU / model.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

from oasr.engine.config import EngineConfig
from oasr.engine.pipeline import OfflinePipeline
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


def _make_pipeline(
    *, max_batch_frames: Optional[int], mb: int = 64
) -> OfflinePipeline:
    """Build a pipeline with stub IO — only ``_split_chunks`` is exercised."""
    inp = SimpleNamespace(_config=SimpleNamespace(dtype=torch.float32))
    return OfflinePipeline(
        scheduler=SimpleNamespace(),
        input_processor=inp,
        model_runner=SimpleNamespace(),
        output_processor=SimpleNamespace(),
        micro_batch_size=mb,
        device=torch.device("cpu"),
        max_batch_frames=max_batch_frames,
    )


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
# Pipeline frame-budget split
# ---------------------------------------------------------------------------


class TestSplitByFrames:
    def test_budget_respected_per_chunk(self):
        pipe = _make_pipeline(max_batch_frames=800)
        reqs = _make_requests([200] * 10)
        chunks, _ = pipe._split_chunks(reqs)
        # 200 * 4 = 800 fits; 200 * 5 = 1000 does not → chunks of 4.
        assert [len(c) for c in chunks] == [4, 4, 2]
        for c in chunks:
            assert max(r.num_frames for r in c) * len(c) <= 800

    def test_length_sorted_chunks(self):
        pipe = _make_pipeline(max_batch_frames=600)
        reqs = _make_requests([100, 300, 50, 300, 100, 50])
        chunks, orig = pipe._split_chunks(reqs)
        assert orig is not None
        flat = [r.num_frames for c in chunks for r in c]
        assert flat == sorted(flat)
        for c in chunks:
            assert max(r.num_frames for r in c) * len(c) <= 600

    def test_lone_oversized_ships_alone(self):
        pipe = _make_pipeline(max_batch_frames=500)
        reqs = _make_requests([100, 100, 900, 100])
        chunks, _ = pipe._split_chunks(reqs)
        # The 900-frame utt must be in a singleton chunk.
        singletons = [c for c in chunks if len(c) == 1]
        assert any(c[0].num_frames == 900 for c in singletons)

    def test_micro_batch_count_cap(self):
        # Budget is generous, but mb=3 caps each chunk at 3.
        pipe = _make_pipeline(max_batch_frames=10_000, mb=3)
        reqs = _make_requests([10] * 9)
        chunks, _ = pipe._split_chunks(reqs)
        assert [len(c) for c in chunks] == [3, 3, 3]
