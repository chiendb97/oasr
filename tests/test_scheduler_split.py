# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``Scheduler.split_offline_batch`` — offline micro-batch partition.

``schedule_offline`` selects *which* requests form a step's batch;
``split_offline_batch`` decides *how* they are grouped for the encoder forward:

* count / preferred-size partition (the default, no frame budget / packing),
* sequence packing (one gapless packed row per chunk, summed-token budget).

Pure-Python; no GPU / model required.  The frame-budget path is covered in
``test_scheduler_length_batch.py``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from oasr.engine.config import EngineConfig
from oasr.engine.request import Request
from oasr.engine.scheduler import Scheduler


def _make_scheduler(
    preferred_sizes: Optional[Sequence[int]],
    mb: int = 8,
    *,
    enable_packing: bool = False,
    max_packed_frames: int = 8192,
) -> Scheduler:
    """Build a scheduler whose ``split_offline_batch`` is exercised in isolation."""
    cfg = EngineConfig(
        ckpt_dir="/tmp/fake",
        service_mode="offline",
        max_batch_size=mb,
        preferred_batch_size=list(preferred_sizes) if preferred_sizes else None,
        enable_sequence_packing=enable_packing,
        max_packed_frames=max_packed_frames,
    )
    return Scheduler(cfg)


def _make_requests(num_frames_list: List[int]) -> List[Request]:
    out = []
    for n in num_frames_list:
        req = Request("audio.wav", streaming=False)
        req.num_frames = n
        out.append(req)
    return out


def _chunk_sizes(chunks):
    return [len(c) for c in chunks]


class TestSplitChunksPreferred:
    def test_greedy_peel_to_largest_preferred(self):
        # 11 requests, preferred [4, 8], mb=8 → [8, 3] (tail < min preferred)
        sched = _make_scheduler(preferred_sizes=[4, 8], mb=8)
        reqs = _make_requests([i * 10 for i in range(11)])
        chunks, _ = sched.split_offline_batch(reqs)
        assert _chunk_sizes(chunks) == [8, 3]

    def test_exact_multiple_no_tail(self):
        sched = _make_scheduler(preferred_sizes=[4, 8], mb=8)
        reqs = _make_requests([i * 10 for i in range(16)])
        chunks, _ = sched.split_offline_batch(reqs)
        assert _chunk_sizes(chunks) == [8, 8]

    def test_smaller_than_max_preferred_picks_lower(self):
        # 7 requests, preferred [4, 8] — first chunk snaps to 4, tail = 3
        sched = _make_scheduler(preferred_sizes=[4, 8], mb=8)
        reqs = _make_requests([i * 10 for i in range(7)])
        chunks, _ = sched.split_offline_batch(reqs)
        assert _chunk_sizes(chunks) == [4, 3]

    def test_micro_batch_caps_chunk_size(self):
        # preferred [4] with mb=4 caps each chunk at 4 (preferred <= mb always
        # holds — config rejects preferred values above max_batch_size).
        sched = _make_scheduler(preferred_sizes=[4], mb=4)
        reqs = _make_requests([i * 10 for i in range(12)])
        chunks, _ = sched.split_offline_batch(reqs)
        assert _chunk_sizes(chunks) == [4, 4, 4]


class TestSplitChunksLegacy:
    def test_balanced_split_when_pbs_none(self):
        # 11 requests, mb=8 — legacy balance picks 2 chunks ≈ [6, 5].
        sched = _make_scheduler(preferred_sizes=None, mb=8)
        reqs = _make_requests([i * 10 for i in range(11)])
        chunks, _ = sched.split_offline_batch(reqs)
        sizes = _chunk_sizes(chunks)
        assert sum(sizes) == 11
        # Balance keeps chunks within 1 of each other.
        assert max(sizes) - min(sizes) <= 1

    def test_single_chunk_when_n_le_mb(self):
        sched = _make_scheduler(preferred_sizes=None, mb=8)
        reqs = _make_requests([10] * 5)
        chunks, orig = sched.split_offline_batch(reqs)
        assert _chunk_sizes(chunks) == [5]
        assert orig is None


class TestSortByLength:
    def test_chunks_are_length_sorted(self):
        # Mixed lengths; chunks should land in ascending num_frames order.
        sched = _make_scheduler(preferred_sizes=[4], mb=8)
        reqs = _make_requests([100, 10, 50, 200, 80, 30, 60, 20])
        chunks, orig = sched.split_offline_batch(reqs)
        # 8 requests, preferred=[4] → [4, 4].
        assert _chunk_sizes(chunks) == [4, 4]
        # Within each chunk, num_frames is non-decreasing; across chunks
        # the first chunk holds the smallest 4.
        first_lens = sorted(r.num_frames for r in chunks[0])
        second_lens = sorted(r.num_frames for r in chunks[1])
        assert first_lens == [10, 20, 30, 50]
        assert second_lens == [60, 80, 100, 200]
        assert orig is not None


class TestSplitPacks:
    def test_packs_bounded_by_token_budget(self):
        # subsampling_rate=4 → each 200-frame utt is 50 post-subsampling tokens.
        # budget=120 → two utts (100) fit, a third (150) overflows → [2, 1].
        sched = _make_scheduler(preferred_sizes=None, enable_packing=True, max_packed_frames=120)
        reqs = _make_requests([200, 200, 200])
        chunks, orig = sched.split_offline_batch(reqs)
        assert _chunk_sizes(chunks) == [2, 1]
        assert orig is not None

    def test_packing_takes_precedence_over_frames(self):
        # enable_packing wins even when max_batch_frames is also set.
        cfg = EngineConfig(
            ckpt_dir="/tmp/fake",
            service_mode="offline",
            max_batch_size=64,
            enable_sequence_packing=True,
            max_packed_frames=120,
            max_batch_frames=800,
        )
        sched = Scheduler(cfg)
        chunks, _ = sched.split_offline_batch(_make_requests([200, 200, 200]))
        assert _chunk_sizes(chunks) == [2, 1]

    def test_oversized_utt_ships_as_own_pack(self):
        sched = _make_scheduler(preferred_sizes=None, enable_packing=True, max_packed_frames=60)
        # 200//4 = 50 ≤ 60 each, but two would be 100 > 60 → one per pack.
        chunks, _ = sched.split_offline_batch(_make_requests([200, 200]))
        assert _chunk_sizes(chunks) == [1, 1]


def test_empty_batch_returns_empty():
    sched = _make_scheduler(preferred_sizes=None)
    chunks, orig = sched.split_offline_batch([])
    assert chunks == []
    assert orig is None
