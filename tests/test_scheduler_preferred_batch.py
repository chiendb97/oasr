# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``EngineConfig.preferred_batch_size`` scheduling behaviour.

Covers both queues:

* Streaming admission snaps the running pool to a preferred B; below the
  smallest preferred value admission waits unless ``max_wait_time`` has
  expired (force-flush escape valve).
* Offline batch construction trims the built batch down to a preferred
  size, returning the overflow to the head of the queue so it ships on
  the next step with length-similar peers.
"""

from __future__ import annotations

import time
from collections import deque

import pytest
import torch

from oasr.engine.config import EngineConfig
from oasr.engine.request import Request
from oasr.engine.scheduler import Scheduler


def _make_config(
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


def _make_streaming(n_chunks: int = 3) -> Request:
    req = Request("audio.wav", streaming=True)
    req.audio_chunks = deque([torch.zeros(16000) for _ in range(n_chunks)])
    req.audio_tail = torch.zeros(0)
    req.audio_final = True
    return req


def _make_offline(num_frames: int = 200) -> Request:
    req = Request("audio.wav", streaming=False)
    req.num_frames = num_frames
    return req


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigNormalisation:
    def test_default_is_none(self):
        cfg = _make_config()
        assert cfg.preferred_batch_size is None

    def test_dedupe_and_sort(self):
        cfg = _make_config(preferred_batch_size=[8, 4, 8, 2])
        assert cfg.preferred_batch_size == [2, 4, 8]

    def test_rejects_value_above_cap(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            _make_config(max_batch_size=8, preferred_batch_size=[4, 16])

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match=">= 1"):
            _make_config(preferred_batch_size=[0, 4])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            _make_config(preferred_batch_size=[])

    def test_defaults_feature_buckets(self):
        cfg = _make_config(preferred_batch_size=[4, 8])
        assert cfg.feature_graph_batch_buckets == [4, 8]

    def test_explicit_feature_buckets_win(self):
        cfg = EngineConfig(
            ckpt_dir="/tmp/fake",
            max_batch_size=16,
            preferred_batch_size=[4, 8],
            feature_graph_batch_buckets=[16],
        )
        assert cfg.feature_graph_batch_buckets == [16]


# ---------------------------------------------------------------------------
# Streaming admission
# ---------------------------------------------------------------------------


class TestStreamingAdmission:
    def test_snap_admits_largest_preferred(self):
        sched = Scheduler(_make_config(preferred_batch_size=[4, 8]))
        for _ in range(5):
            sched.add_request(_make_streaming())

        out = sched.schedule()
        assert len(out.newly_admitted) == 4
        assert sched.num_running == 4
        assert sched.num_waiting == 1

    def test_holds_when_below_min_preferred(self):
        # 3 waiting, smallest preferred is 4 — admit none, wait.
        sched = Scheduler(_make_config(preferred_batch_size=[4, 8]))
        for _ in range(3):
            sched.add_request(_make_streaming())

        out = sched.schedule()
        assert len(out.newly_admitted) == 0
        assert sched.num_waiting == 3

    def test_force_flush_on_wait_deadline(self):
        # Below min preferred, but oldest has waited past max_wait_time.
        sched = Scheduler(
            _make_config(preferred_batch_size=[4, 8], max_wait_time=0.05)
        )
        for _ in range(3):
            req = _make_streaming()
            req.arrival_time = time.monotonic() - 1.0  # 1 s ago
            sched.add_request(req)

        out = sched.schedule()
        assert len(out.newly_admitted) == 3
        assert sched.num_running == 3

    def test_grows_existing_pool_to_next_preferred(self):
        # Admit 4, then more arrive — running grows to 8 (next preferred).
        sched = Scheduler(_make_config(preferred_batch_size=[4, 8]))
        for _ in range(4):
            sched.add_request(_make_streaming())
        sched.schedule()  # admits 4

        for _ in range(5):
            sched.add_request(_make_streaming())
        out = sched.schedule()
        assert len(out.newly_admitted) == 4  # 4 + 4 = 8
        assert sched.num_running == 8
        assert sched.num_waiting == 1

    def test_no_admission_when_running_already_preferred(self):
        # Running = 8 (preferred). Adding 3 more would land at 11 — not
        # preferred and below the next jump (16). Hold.
        sched = Scheduler(_make_config(preferred_batch_size=[4, 8, 16]))
        for _ in range(8):
            sched.add_request(_make_streaming())
        sched.schedule()  # admits 8
        assert sched.num_running == 8

        for _ in range(3):
            sched.add_request(_make_streaming())
        out = sched.schedule()
        assert len(out.newly_admitted) == 0
        assert sched.num_running == 8

    def test_pbs_none_keeps_greedy_admission(self):
        sched = Scheduler(_make_config(max_batch_size=4, preferred_batch_size=None))
        for _ in range(5):
            sched.add_request(_make_streaming())
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
            _make_config(
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
        sched = Scheduler(_make_config(preferred_batch_size=[4, 8]))
        for _ in range(3):
            sched.add_request(_make_offline())
        out = sched.schedule()
        assert out.offline_batch == []
        assert sched.num_waiting_offline == 3

    def test_force_flush_ships_sub_preferred(self):
        sched = Scheduler(
            _make_config(preferred_batch_size=[4, 8], max_wait_time=0.05)
        )
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
            _make_config(
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
        sched = Scheduler(_make_config(preferred_batch_size=[4]))
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
