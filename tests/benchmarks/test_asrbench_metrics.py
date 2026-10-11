# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Statistics and the metric vocabulary of the asrbench suite."""

from __future__ import annotations

import math

import pytest

from benchmarks.asrbench import metrics as m


@pytest.mark.parametrize("n", [1, 2, 3, 10, 19, 20, 99, 100, 101, 257])
def test_percentile_is_the_repository_rule(n):
    """Restated, not imported -- so pin it to `benchmarks.core.metrics.percentile`."""
    core = pytest.importorskip("benchmarks.core.metrics")  # imports torch
    values = [float((k * 37) % (n + 3)) for k in range(n)]
    for p in (0, 1, 50, 90, 95, 99, 100):
        assert m.percentile(values, p) == core.percentile(values, p)


def test_percentile_edges():
    assert math.isnan(m.percentile([], 50))
    xs = [5.0, 1.0, 3.0]
    assert m.percentile(xs, 0) == 1.0
    assert m.percentile(xs, 100) == 5.0
    assert m.percentile([7.0], 99) == 7.0
    assert m.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_summarize_skips_missing_and_counts():
    s = m.summarize([1.0, None, float("nan"), 3.0, 2.0], "lat")
    assert s["lat.n"] == 3
    assert s["lat.mean"] == pytest.approx(2.0)
    assert s["lat.min"] == 1.0 and s["lat.max"] == 3.0
    assert s["lat.p50"] == 2.0
    assert s["lat.std"] == pytest.approx(1.0)
    assert m.summarize([], "x") == {"x.n": 0.0}


def test_summarize_reps():
    s = m.summarize_reps([10.0, 12.0, 14.0], "rtfx")
    assert s["rtfx.rep_mean"] == pytest.approx(12.0)
    assert s["rtfx.rep_std"] == pytest.approx(2.0)
    assert s["rtfx.reps"] == 3
    assert m.summarize_reps([], "rtfx") == {}


def test_min_samples_for():
    assert m.min_samples_for(50) == 2
    assert m.min_samples_for(90) == 10
    assert m.min_samples_for(95) == 20
    assert m.min_samples_for(99) == 100
    assert m.min_samples_for(100) == 1


def test_low_sample_percentiles_flags_unresolvable_tails():
    s = m.summarize(range(50), "latency_ms")
    flagged = m.low_sample_percentiles(s)
    assert "latency_ms.p99" in flagged
    assert "latency_ms.p95" not in flagged  # 50 >= 20
    assert "latency_ms.p50" not in flagged


def test_rtf_and_rtfx_are_reciprocal_and_guarded():
    assert m.rtfx(10.0, 2.0) == 5.0
    assert m.rtf(2.0, 10.0) == 0.2
    assert m.rtfx(10.0, 0.0) is None
    assert m.rtf(1.0, 0.0) is None


@pytest.mark.parametrize(
    "key,expected",
    [
        ("latency_ms.p50", m.LOWER),
        ("bucket.typical.latency_ms.p95", m.LOWER),
        ("rtfx", m.HIGHER),
        ("rtfx.rep_mean", m.HIGHER),
        ("throughput.req_per_s", m.HIGHER),
        ("final_latency_ms.p95", m.LOWER),
        ("latency_ms.n", m.INFO),
        ("latency_ms.std", m.INFO),
        ("rtfx.rep_std", m.INFO),
        ("cpu_pct.mean", m.INFO),
        ("send_lag_ms.p99", m.INFO),
        ("error_rate", m.LOWER),
        ("achieved_ratio", m.HIGHER),
        ("no_such_metric", m.INFO),
    ],
)
def test_direction(key, expected):
    assert m.direction(key) == expected


def test_every_metric_has_unit_and_description():
    for d in m.METRICS:
        assert d.description and d.better in (m.LOWER, m.HIGHER, m.INFO), d.pattern
    assert m.unit("latency_ms.p50") == "ms"
    assert m.unit("bucket.short.latency_ms.p50") == "ms"
