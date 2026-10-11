# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Resource telemetry: window attribution, energy, and the out-of-process sampler."""

from __future__ import annotations

import os
import time

import pytest

from benchmarks.asrbench.monitor import (
    ResourceMonitor,
    read_samples,
    summarize_window,
    visible_gpu_indices,
)


def samples(n=11, dt=0.1, energy=True):
    out = []
    for k in range(n):
        s = {"t": 100.0 + k * dt, "cpu_pct": 50.0 + k, "rss_mb": 1000.0 + k, "gpu_power_w": 200.0}
        if energy:
            s["gpu_energy_mj"] = 1e6 + k * dt * 200.0 * 1000.0  # 200 W
        out.append(s)
    return out


def test_window_attribution_and_stats():
    out = summarize_window(samples(), 100.2, 100.5)
    assert out["cpu_pct.n"] == 4
    assert out["cpu_pct.mean"] == pytest.approx(53.5)
    assert out["rss_mb.max"] == pytest.approx(1005.0)
    assert "gpu_util_pct.mean" not in out  # absent probe = absent metric, never zero


def test_energy_prefers_the_driver_counter_and_needs_a_long_enough_window():
    out = summarize_window(samples(), 100.0, 101.0, audio_s=3600.0)
    assert out["energy_j"] == pytest.approx(200.0)  # 1 s at 200 W
    assert out["energy_j_per_audio_h"] == pytest.approx(200.0)
    integrated = summarize_window(samples(energy=False), 100.0, 101.0)
    assert integrated["energy_j"] == pytest.approx(200.0)
    short = summarize_window(samples(), 100.0, 100.3)
    assert "energy_j" not in short and short["gpu_power_w.mean"] == 200.0


def test_short_window_is_bracketed():
    out = summarize_window(samples(), 100.21, 100.24)  # between two samples
    assert out["rss_mb.n"] >= 2


def test_visible_gpu_indices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    assert visible_gpu_indices() == [2, 0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    assert visible_gpu_indices() is None
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert visible_gpu_indices() is None


def test_sampler_process_watches_a_pid(tmp_path):
    pytest.importorskip("psutil")
    mon = ResourceMonitor(tmp_path / "res.jsonl", interval_s=0.05, gpu=False)
    mon.start(os.getpid())
    deadline = time.monotonic() + 10
    while len(mon.samples()) < 4 and time.monotonic() < deadline:
        time.sleep(0.05)
    mon.stop()
    head = mon.header()
    assert head["psutil"] is True and head["nvml"] is False and head["interval_s"] == 0.05
    got = mon.samples()
    assert len(got) >= 4 and all("rss_mb" in s and "t" in s for s in got)
    assert [s["t"] for s in got] == sorted(s["t"] for s in got)
    # A second sampler never interleaves with the first file.
    mon.start(os.getpid())
    time.sleep(0.15)
    mon.stop()
    assert mon.out_path.name != "res.jsonl" and read_samples(tmp_path / "res.jsonl")
