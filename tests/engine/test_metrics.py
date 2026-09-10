# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Engine-side metric collection.

The exporter lives in Rust; what is checked here is the half Python owns — the
drain protocol's semantics, the bounded buffers, the stage vocabulary, and the
one property an observability change must never break: that it does not change
the thing it observes.
"""

from __future__ import annotations

import time

import pytest
import torch

from oasr.engine import metrics as m


@pytest.fixture
def collector():
    return m.EngineMetrics(device=None)


# ---------------------------------------------------------------------------
# Drain protocol
# ---------------------------------------------------------------------------


def test_counters_are_absolute_and_survive_a_drain(collector):
    """Counters are totals, not deltas.

    The front-end replays them with ``Counter::absolute``, so a drain that is
    missed must lose nothing and a drain replayed twice must double nothing.
    Clearing counters here would break both.
    """
    collector.incr(m.TOKENS_GENERATED, 5)
    assert collector.snapshot()["counters"][m.TOKENS_GENERATED] == 5
    collector.incr(m.TOKENS_GENERATED, 3)
    assert collector.snapshot()["counters"][m.TOKENS_GENERATED] == 8
    # A drain with nothing added still reports the same total.
    assert collector.snapshot()["counters"][m.TOKENS_GENERATED] == 8


def test_histogram_samples_are_handed_over_and_cleared(collector):
    """Samples, unlike counters, must not be replayed twice."""
    collector.observe(m.BATCH_PADDING_RATIO, 0.25)
    collector.observe_keyed(m.BATCH_SIZE, m.MODE_OFFLINE, 16.0)
    first = collector.snapshot()
    assert first["hist"][m.BATCH_PADDING_RATIO] == [0.25]
    assert first["keyed_hist"][m.BATCH_SIZE][m.MODE_OFFLINE] == [16.0]

    second = collector.snapshot()
    assert second["hist"] == {}
    assert second["keyed_hist"] == {}
    # The handed-over lists are the collector's former buffers; a second drain
    # must not be able to reach them.
    assert first["hist"][m.BATCH_PADDING_RATIO] == [0.25]


def test_snapshot_always_has_the_four_sections(collector):
    """The Rust side reads four fixed keys; a missing one would silently
    extract as empty rather than fail."""
    assert set(collector.snapshot()) == {"counters", "gauges", "hist", "keyed_hist"}


# ---------------------------------------------------------------------------
# Bounded buffers
# ---------------------------------------------------------------------------


def test_sample_buffers_are_capped_and_the_overflow_is_counted(collector):
    """A stalled drain must not grow without limit — and the truncation must
    be visible, since a capped histogram otherwise passes for a complete one."""
    over = m.MAX_SAMPLES_PER_SERIES + 100
    for _ in range(over):
        collector.observe(m.BATCH_PADDING_RATIO, 0.5)
    snap = collector.snapshot()
    assert len(snap["hist"][m.BATCH_PADDING_RATIO]) == m.MAX_SAMPLES_PER_SERIES
    assert snap["counters"][m.SAMPLES_DROPPED] == 100


def test_the_cap_is_per_series_not_global(collector):
    """One busy stage must not starve another out of its samples."""
    for _ in range(m.MAX_SAMPLES_PER_SERIES + 10):
        collector.observe_stage("offline.encode", 0.001)
    collector.observe_stage("offline.decode", 0.002)
    keyed = collector.snapshot()["keyed_hist"][m.STAGE_HOST_SECONDS]
    assert len(keyed["offline.encode"]) == m.MAX_SAMPLES_PER_SERIES
    assert keyed["offline.decode"] == [0.002]


# ---------------------------------------------------------------------------
# Stage vocabulary
# ---------------------------------------------------------------------------


def test_an_undeclared_stage_is_rejected(collector):
    """Stage names are Prometheus label values, so the set has to be closed.

    The NVTX ranges next door interpolate the batch size into the range name
    (``offline.micro_batch[B=16]``), which is right for a profiler timeline and
    would be a cardinality bomb here — one series per distinct batch size,
    forever.  Copying that pattern must fail on the first call.
    """
    with pytest.raises(ValueError, match="undeclared engine stage"):
        collector.observe_stage("offline.micro_batch[B=16]", 0.001)


@pytest.mark.parametrize("stage", sorted(m.STAGES))
def test_every_declared_stage_is_recordable(collector, stage):
    collector.observe_stage(stage, 0.5)
    assert collector.snapshot()["keyed_hist"][m.STAGE_HOST_SECONDS][stage] == [0.5]


def test_stage_names_are_prefixed_by_their_executor():
    """``encode`` happens in both executors and means different work in each;
    one ``stage`` label can only tell them apart if the name says which."""
    assert all(s.startswith(("offline.", "streaming.")) for s in m.STAGES)


# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------


def test_padding_ratio_is_the_fraction_of_the_padded_batch_that_is_padding(collector):
    # Rows of 1, 2 and 3 units pad to 3 each: 9 padded, 6 real, 1/3 wasted.
    collector.observe_padding([1, 2, 3])
    assert collector.snapshot()["hist"][m.BATCH_PADDING_RATIO] == pytest.approx([1 / 3])


def test_a_uniform_batch_has_no_padding(collector):
    collector.observe_padding([100, 100, 100])
    assert collector.snapshot()["hist"][m.BATCH_PADDING_RATIO] == pytest.approx([0.0])


@pytest.mark.parametrize("lengths", [[], [0, 0]])
def test_padding_ratio_skips_batches_it_cannot_divide(collector, lengths):
    """An empty or all-zero batch has no ratio; recording 0.0 would claim
    perfect packing for a batch that packed nothing."""
    collector.observe_padding(lengths)
    assert m.BATCH_PADDING_RATIO not in collector.snapshot()["hist"]


def test_queue_wait_is_measured_from_arrival(collector):
    class FakeRequest:
        def __init__(self, arrival_time):
            self.arrival_time = arrival_time

        arrival_time: float

    now = time.monotonic()
    collector.observe_queue_wait(m.MODE_OFFLINE, [FakeRequest(now - 0.05), FakeRequest(now)])
    waits = collector.snapshot()["keyed_hist"][m.QUEUE_WAIT_SECONDS][m.MODE_OFFLINE]
    assert len(waits) == 2
    assert waits[0] == pytest.approx(0.05, abs=0.05)
    assert waits[0] > waits[1]


# ---------------------------------------------------------------------------
# The disable switch
# ---------------------------------------------------------------------------


def test_null_metrics_records_nothing_but_accepts_everything():
    """``OASR_METRICS=0`` must not change any call site's shape."""
    null = m.NullMetrics()
    assert not null.enabled
    null.incr(m.TOKENS_GENERATED, 1)
    null.set_gauge(m.KV_BLOCKS_USED, 1.0)
    null.observe(m.BATCH_PADDING_RATIO, 0.5)
    null.observe_stage("offline.encode", 0.1)
    null.observe_batch(m.MODE_OFFLINE, 8)
    null.observe_padding([1, 2])
    null.observe_queue_wait(m.MODE_OFFLINE, [])
    null.observe_gpu()
    snap = null.snapshot()
    assert snap == {"counters": {}, "gauges": {}, "hist": {}, "keyed_hist": {}}


def test_null_metrics_does_not_reject_an_undeclared_stage():
    """Disabled means disabled: the validation is a recording-path check, and
    turning metrics off must not turn a new failure on."""
    m.NullMetrics().observe_stage("anything at all", 0.1)


@pytest.mark.parametrize(
    "env,expect_enabled",
    [("0", False), ("1", True), (None, True)],
)
def test_build_metrics_follows_the_environment(monkeypatch, env, expect_enabled):
    if env is None:
        monkeypatch.delenv("OASR_METRICS", raising=False)
    else:
        monkeypatch.setenv("OASR_METRICS", env)
    assert m.build_metrics(None).enabled is expect_enabled


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_recording_does_not_synchronise_the_device():
    """The one property that would make this change a regression.

    Padding is derived from host-side lengths precisely so the collate path
    gains no device-to-host read.  Deriving it from the device ``lengths``
    tensor instead would need a ``.sum().item()``, and a synchronisation added
    to the hot loop by a *metric* is the observation paying for itself in wall
    clock.  A CPU tensor's ``.item()`` does not sync, so this has to run on
    CUDA to mean anything.
    """
    if not torch.cuda.is_available():
        pytest.skip("a host-side value cannot be shown not to sync without a device")
    collector = m.EngineMetrics(device=torch.device("cuda"))
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        # Queue real work, then record.  If any recording path read a device
        # tensor, the event below would already be complete.
        big = torch.randn(4096, 4096, device="cuda")
        for _ in range(20):
            big = big @ big.T
        done = torch.cuda.Event()
        done.record()
        collector.observe_batch(m.MODE_OFFLINE, 32)
        collector.observe_padding([16000, 32000, 48000])
        collector.observe_stage("offline.collate", 0.001)
        collector.incr(m.TOKENS_GENERATED, 12)
        assert not done.query(), "a recording path synchronised on the device"
    torch.cuda.synchronize()


def test_recording_is_cheap_enough_for_the_step_loop(collector):
    """A guard, not a benchmark.

    The engine step loop is interpreter-bound at batch and holds the GIL for
    every request the engine finishes, so a per-batch helper that crept into
    tens of microseconds would show up as throughput.  The bound is loose
    enough to survive a slow CI box and tight enough to catch an accidental
    per-sample loop.
    """
    n = 20_000
    t0 = time.perf_counter()
    for _ in range(n):
        collector.observe_stage("offline.encode", 0.001)
    per_call_us = (time.perf_counter() - t0) / n * 1e6
    assert per_call_us < 10.0, f"observe_stage cost {per_call_us:.1f} us/call"
