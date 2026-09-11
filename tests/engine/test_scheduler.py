# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr/engine/scheduler.py`` -- admission, batching and partition.

One module, one file. It was three (plus a fourth copy inside
``test_engine.py``), all importing exactly ``EngineConfig``, ``Request`` and
``Scheduler``, and two of them carried byte-identical ``_make_requests`` /
``_make_offline`` helpers. ``split_offline_batch`` was exercised from all
three, so a change to it broke tests in files that did not mention each other.

Three sections, matching the three decisions the scheduler makes:

* **the frame budget** -- ``EngineConfig.max_batch_frames`` bounds
  ``max_len * batch_size`` per offline forward, enforced both when building a
  batch and when re-splitting one;
* **the preferred-batch ladder** -- how long admission waits for a fuller
  batch before shipping what it has;
* **partition** -- splitting an admitted cohort into micro-batches.

All pure Python: no GPU, no model.
"""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Sequence

import torch

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


# ---------------------------------------------------------------------------
# The preferred-batch-size ladder
# ---------------------------------------------------------------------------


def _preferred_config(
    *,
    max_batch_size: int = 16,
    preferred_batch_size=None,
    max_wait_time: float = 0.2,
    schedule_policy: str = "bucket",
) -> EngineConfig:
    return EngineConfig(
        ckpt_dir="/tmp/fake",
        max_batch_size=max_batch_size,
        preferred_batch_size=preferred_batch_size,
        max_wait_time=max_wait_time,
        schedule_policy=schedule_policy,
    )


def _preferred_streaming(n_chunks: int = 3) -> Request:
    req = Request("audio.wav", streaming=True)
    req.audio_chunks = deque([torch.zeros(16000) for _ in range(n_chunks)])
    req.audio_tail = torch.zeros(0)
    req.audio_final = True
    return req


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Streaming admission
# ---------------------------------------------------------------------------


class TestStreamingAdmission:
    def test_snap_admits_largest_preferred(self):
        sched = Scheduler(_preferred_config(preferred_batch_size=[4, 8]))
        for _ in range(5):
            sched.add_request(_preferred_streaming())

        out = sched.schedule()
        assert len(out.newly_admitted) == 4
        assert sched.num_running == 4
        assert sched.num_waiting == 1

    def test_holds_when_below_min_preferred(self):
        # 3 waiting, smallest preferred is 4 — admit none, wait.
        sched = Scheduler(_preferred_config(preferred_batch_size=[4, 8]))
        for _ in range(3):
            sched.add_request(_preferred_streaming())

        out = sched.schedule()
        assert len(out.newly_admitted) == 0
        assert sched.num_waiting == 3

    def test_force_flush_on_wait_deadline(self):
        # Below min preferred, but oldest has waited past max_wait_time.
        sched = Scheduler(_preferred_config(preferred_batch_size=[4, 8], max_wait_time=0.05))
        for _ in range(3):
            req = _preferred_streaming()
            req.arrival_time = time.monotonic() - 1.0  # 1 s ago
            sched.add_request(req)

        out = sched.schedule()
        assert len(out.newly_admitted) == 3
        assert sched.num_running == 3

    def test_grows_existing_pool_to_next_preferred(self):
        # Admit 4, then more arrive — running grows to 8 (next preferred).
        sched = Scheduler(_preferred_config(preferred_batch_size=[4, 8]))
        for _ in range(4):
            sched.add_request(_preferred_streaming())
        sched.schedule()  # admits 4

        for _ in range(5):
            sched.add_request(_preferred_streaming())
        out = sched.schedule()
        assert len(out.newly_admitted) == 4  # 4 + 4 = 8
        assert sched.num_running == 8
        assert sched.num_waiting == 1

    def test_no_admission_when_running_already_preferred(self):
        # Running = 8 (preferred). Adding 3 more would land at 11 — not
        # preferred and below the next jump (16). Hold.
        sched = Scheduler(_preferred_config(preferred_batch_size=[4, 8, 16]))
        for _ in range(8):
            sched.add_request(_preferred_streaming())
        sched.schedule()  # admits 8
        assert sched.num_running == 8

        for _ in range(3):
            sched.add_request(_preferred_streaming())
        out = sched.schedule()
        assert len(out.newly_admitted) == 0
        assert sched.num_running == 8

    def test_pbs_none_keeps_greedy_admission(self):
        sched = Scheduler(_preferred_config(max_batch_size=4, preferred_batch_size=None))
        for _ in range(5):
            sched.add_request(_preferred_streaming())
        out = sched.schedule()
        assert len(out.newly_admitted) == 4
        assert sched.num_waiting == 1


# ---------------------------------------------------------------------------
# Offline batch construction
# ---------------------------------------------------------------------------


class TestOfflineBatch:
    def test_trims_built_batch_to_preferred(self):
        # max_batch_size=16 → cap=16. 10 requests, PBS=[4, 8] →
        # built batch sized 10, trimmed down to 8. Remaining 2 stay queued.
        sched = Scheduler(
            _preferred_config(
                max_batch_size=16,
                preferred_batch_size=[4, 8],
            )
        )
        for _ in range(10):
            sched.add_request(_make_offline(num_frames=200))

        out = sched.schedule()
        assert len(out.offline_batch) == 8
        assert sched.num_waiting_offline == 2

    def test_holds_when_below_min_preferred(self):
        sched = Scheduler(_preferred_config(preferred_batch_size=[4, 8]))
        for _ in range(3):
            sched.add_request(_make_offline())
        out = sched.schedule()
        assert out.offline_batch == []
        assert sched.num_waiting_offline == 3

    def test_force_flush_ships_sub_preferred(self):
        sched = Scheduler(_preferred_config(preferred_batch_size=[4, 8], max_wait_time=0.05))
        for _ in range(3):
            req = _make_offline()
            req.arrival_time = time.monotonic() - 1.0
            sched.add_request(req)

        out = sched.schedule()
        assert len(out.offline_batch) == 3

    def test_pbs_none_caps_at_max_batch_size(self):
        # With no preferred sizes the offline batch is capped at max_batch_size.
        # 10 requests, max_batch_size=4 → batch of 4.
        sched = Scheduler(
            _preferred_config(
                max_batch_size=4,
                preferred_batch_size=None,
            )
        )
        for _ in range(10):
            sched.add_request(_make_offline())
        out = sched.schedule()
        assert len(out.offline_batch) == 4

    def test_overflow_returned_to_head_in_order(self):
        # Tag requests with their arrival order; after trim, the overflow
        # should remain at the head of the queue in arrival order.
        sched = Scheduler(_preferred_config(preferred_batch_size=[4]))
        for i in range(6):
            req = _make_offline(num_frames=100 + i)
            req.request_id = f"req-{i:02d}"
            sched.add_request(req)

        out = sched.schedule()
        assert len(out.offline_batch) == 4
        remaining_ids = [r.request_id for r in sched._offline_waiting]
        # The two trailing requests should still be present; the exact pair
        # depends on bucket selection but their relative order must be FIFO.
        assert len(remaining_ids) == 2
        idx0 = int(remaining_ids[0].split("-")[1])
        idx1 = int(remaining_ids[1].split("-")[1])
        assert idx0 < idx1


# ---------------------------------------------------------------------------
# Partition: splitting an admitted cohort into micro-batches
# ---------------------------------------------------------------------------


def _split_scheduler(
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


def _chunk_sizes(chunks):
    return [len(c) for c in chunks]


class TestSplitChunksPreferred:
    def test_greedy_peel_to_largest_preferred(self):
        # 11 requests, preferred [4, 8], mb=8 → [8, 3] (tail < min preferred)
        sched = _split_scheduler(preferred_sizes=[4, 8], mb=8)
        reqs = _make_requests([i * 10 for i in range(11)])
        chunks, _ = sched.split_offline_batch(reqs)
        assert _chunk_sizes(chunks) == [8, 3]

    def test_exact_multiple_no_tail(self):
        sched = _split_scheduler(preferred_sizes=[4, 8], mb=8)
        reqs = _make_requests([i * 10 for i in range(16)])
        chunks, _ = sched.split_offline_batch(reqs)
        assert _chunk_sizes(chunks) == [8, 8]

    def test_smaller_than_max_preferred_picks_lower(self):
        # 7 requests, preferred [4, 8] — first chunk snaps to 4, tail = 3
        sched = _split_scheduler(preferred_sizes=[4, 8], mb=8)
        reqs = _make_requests([i * 10 for i in range(7)])
        chunks, _ = sched.split_offline_batch(reqs)
        assert _chunk_sizes(chunks) == [4, 3]

    def test_micro_batch_caps_chunk_size(self):
        # preferred [4] with mb=4 caps each chunk at 4 (preferred <= mb always
        # holds — config rejects preferred values above max_batch_size).
        sched = _split_scheduler(preferred_sizes=[4], mb=4)
        reqs = _make_requests([i * 10 for i in range(12)])
        chunks, _ = sched.split_offline_batch(reqs)
        assert _chunk_sizes(chunks) == [4, 4, 4]


class TestSplitChunksLegacy:
    def test_balanced_split_when_pbs_none(self):
        # 11 requests, mb=8 — legacy balance picks 2 chunks ≈ [6, 5].
        sched = _split_scheduler(preferred_sizes=None, mb=8)
        reqs = _make_requests([i * 10 for i in range(11)])
        chunks, _ = sched.split_offline_batch(reqs)
        sizes = _chunk_sizes(chunks)
        assert sum(sizes) == 11
        # Balance keeps chunks within 1 of each other.
        assert max(sizes) - min(sizes) <= 1

    def test_single_chunk_when_n_le_mb(self):
        sched = _split_scheduler(preferred_sizes=None, mb=8)
        reqs = _make_requests([10] * 5)
        chunks, orig = sched.split_offline_batch(reqs)
        assert _chunk_sizes(chunks) == [5]
        assert orig is None


class TestSortByLength:
    def test_chunks_are_length_sorted(self):
        # Mixed lengths; chunks should land in ascending num_frames order.
        sched = _split_scheduler(preferred_sizes=[4], mb=8)
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
        sched = _split_scheduler(preferred_sizes=None, enable_packing=True, max_packed_frames=120)
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
        sched = _split_scheduler(preferred_sizes=None, enable_packing=True, max_packed_frames=60)
        # 200//4 = 50 ≤ 60 each, but two would be 100 > 60 → one per pack.
        chunks, _ = sched.split_offline_batch(_make_requests([200, 200]))
        assert _chunk_sizes(chunks) == [1, 1]


def test_empty_batch_returns_empty():
    sched = _split_scheduler(preferred_sizes=None)
    chunks, orig = sched.split_offline_batch([])
    assert chunks == []
    assert orig is None


# ---------------------------------------------------------------------------
# Admission bookkeeping
#
# This class lived in ``test_engine.py``, which is how the scheduler ended up
# with a fourth request builder and a fourth config builder.
# ---------------------------------------------------------------------------


class TestScheduler:
    """Admission bookkeeping: what ``schedule()`` returns and what it counts.

    ``_make_config`` / ``_make_request`` are gone: this file's
    ``_config`` and ``_preferred_streaming`` build the same objects, and three
    near-identical request builders were the reason the scheduler's tests
    could disagree with each other about what a streaming request looks like.
    """

    def _make_config(self, max_batch_size=4):
        return _preferred_config(max_batch_size=max_batch_size)

    def _make_request(self, n_chunks=3):
        return _preferred_streaming(n_chunks=n_chunks)

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
