# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Records -> summaries: every reported number, checked against hand arithmetic."""

from __future__ import annotations

import pytest

from benchmarks.asrbench.analysis import (
    emission_delays_ms,
    phase_wall,
    summarize_requests,
    summarize_sessions,
)
from benchmarks.asrbench.results import RequestRecord, SessionRecord


def req(
    seq,
    sched,
    submit,
    done,
    dur=2.0,
    status="ok",
    rep=0,
    phase="measure",
    batch=None,
    bucket="short",
    text="hi",
):  # noqa: E501
    return RequestRecord(
        result_id="r", rep=rep, phase=phase, seq=seq, item=f"i{seq}", duration_s=dur, bucket=bucket,
        batch=batch, t_sched=sched, t_submit=submit, t_done=done, status=status, text=text,
    )  # fmt: skip


def test_sequential_latency_rtf_and_rtfx():
    recs = [req(0, 0.0, 0.0, 0.1), req(1, 0.2, 0.2, 0.5, dur=3.0), req(2, 0.6, 0.6, 0.8)]
    s = summarize_requests(recs, wall_mode="sum_service")
    assert s["requests.total"] == 3 and s["requests.ok"] == 3 and s["error_rate"] == 0
    assert s["latency_ms.p50"] == pytest.approx(200.0)
    assert s["latency_ms.max"] == pytest.approx(300.0)
    # wall = sum of service times = 0.6 s; audio = 7 s
    assert s["wall_s"] == pytest.approx(0.6)
    assert s["rtfx"] == pytest.approx(7.0 / 0.6)
    assert s["throughput.req_per_s"] == pytest.approx(3 / 0.6)
    assert s["rtf.mean"] == pytest.approx((0.05 + 0.1 + 0.1) / 3)


def test_failures_are_counted_not_ranked():
    recs = [
        req(0, 0.0, 0.0, 0.1),
        req(1, 0.0, 0.0, None, status="timeout"),
        req(2, 0.0, 0.0, 0.05, status="failed"),
        req(3, 0.0, 0.0, 0.05, status="rejected"),
    ]
    s = summarize_requests(recs, wall_mode="span")
    assert s["requests.ok"] == 1 and s["requests.timeout"] == 1
    assert s["requests.failed"] == 1 and s["requests.rejected"] == 1
    assert s["error_rate"] == 0.75 and s["timeout_rate"] == 0.25
    assert s["latency_ms.n"] == 1 and s["latency_ms.max"] == pytest.approx(100.0)
    # Only successful audio counts toward throughput.
    assert s["audio_s"] == 2.0


def test_warmup_records_are_excluded():
    recs = [req(0, 0.0, 0.0, 5.0, phase="warmup"), req(1, 0.0, 0.0, 0.1)]
    s = summarize_requests(recs, wall_mode="sum_service")
    assert s["requests.total"] == 1 and s["latency_ms.max"] == pytest.approx(100.0)


def test_open_loop_latency_is_from_schedule_and_lag_is_reported():
    # Client fell 50 ms behind; the server took 20 ms.  Latency must include the lag.
    recs = [req(0, 1.00, 1.05, 1.07), req(1, 1.10, 1.15, 1.17)]
    s = summarize_requests(recs, wall_mode="span", send_lag=True)
    assert s["latency_ms.p50"] == pytest.approx(70.0)
    assert s["service_ms.p50"] == pytest.approx(20.0)
    assert s["send_lag_ms.max"] == pytest.approx(50.0)
    assert s["wall_s"] == pytest.approx(0.17)


def test_batch_wall_and_batch_latency():
    recs = [
        req(0, 0.0, 0.0, 0.3, batch=0), req(1, 0.0, 0.0, 0.4, batch=0),
        req(2, 1.0, 1.0, 1.2, batch=1), req(3, 1.0, 1.0, 1.2, batch=1),
    ]  # fmt: skip
    assert phase_wall(recs, "batches") == pytest.approx(0.6)
    s = summarize_requests(recs, wall_mode="batches")
    assert s["batch_latency_ms.max"] == pytest.approx(400.0)
    assert s["batch_latency_ms.min"] == pytest.approx(200.0)
    assert s["rtfx"] == pytest.approx(8.0 / 0.6)


def test_repetitions_give_dispersion_and_buckets_break_down():
    recs = [
        req(0, 0, 0, 0.1, rep=0, bucket="short"),
        req(1, 0, 0, 0.2, rep=1, bucket="typical", dur=6.0),
    ]
    s = summarize_requests(recs, wall_mode="sum_service")
    assert s["rtfx.reps"] == 2
    assert s["rtfx.rep_mean"] == pytest.approx((20.0 + 30.0) / 2)
    assert s["bucket.short.latency_ms.p50"] == pytest.approx(100.0)
    assert s["bucket.typical.rtf.mean"] == pytest.approx(0.2 / 6.0)


def test_empty_transcript_rate():
    recs = [req(0, 0, 0, 0.1, text=""), req(1, 0, 0, 0.1, text="ok"), req(2, 0, 0, 0.1, text=None)]
    assert summarize_requests(recs, wall_mode="span")["empty_rate"] == 0.5


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def session(paced=True, events=(), words=None, due=(), sent=(), status="ok", dur=2.0, audio0=10.0):
    return SessionRecord(
        result_id="s", rep=0, phase="measure", seq=0, item="x", duration_s=dur, chunk_ms=500,
        paced=paced, t_audio0=audio0, chunk_due=list(due), chunk_sent=list(sent),
        events=[{"kind": k, "t": t, "text": x} for k, t, x in events], status=status,
        text=events[-1][2] if events else None, words=words,
    )  # fmt: skip


def test_paced_session_latencies():
    s = session(
        events=[("partial", 10.7, ""), ("partial", 10.8, "hello"), ("partial", 11.3, "hello world"), ("final", 12.25, "hello world")],
        due=[10.5, 11.0, 11.5, 12.0],
        sent=[10.5, 11.0, 11.52, 12.2],
    )  # fmt: skip
    out = summarize_sessions([s], late_tolerance_ms=50.0)
    assert out["first_partial_ms.p50"] == pytest.approx(800.0)  # first *non-empty* text
    assert out["final_latency_ms.p50"] == pytest.approx(250.0)  # vs end of audio (12.0)
    assert out["final_after_last_chunk_ms.p50"] == pytest.approx(50.0)  # vs last send
    assert out["late_chunk_pct"] == pytest.approx(25.0)  # only the 200 ms one
    assert out["chunk_lag_ms.max"] == pytest.approx(200.0)
    assert out["partials_per_session.mean"] == 3
    assert "rtfx" not in out  # paced: bounded by real time by construction


def test_unpaced_session_measures_throughput_not_real_time():
    s = session(paced=False, events=[("final", 10.5, "a b")], sent=[10.0, 10.01, 10.02])
    out = summarize_sessions([s], late_tolerance_ms=10.0)
    assert out["final_latency_ms.p50"] == pytest.approx(480.0)  # from last chunk sent
    assert out["rtfx"] == pytest.approx(2.0 / 0.5)
    assert "late_chunk_pct" not in out


def test_emission_delay_counts_only_stable_prefixes():
    words = [
        {"word": "the", "start": 0.0, "end": 0.3},
        {"word": "cat", "start": 0.3, "end": 0.6},
        {"word": "sat", "start": 0.6, "end": 1.0},
    ]
    s = session(
        events=[
            ("partial", 10.5, "the"),
            ("partial", 10.9, "the hat"),  # wrong second word: not an emission of "cat"
            ("partial", 11.2, "the cat sat"),
            ("final", 11.4, "The cat sat."),
        ],
        words=words,
    )
    delays = emission_delays_ms(s)
    assert delays == pytest.approx([200.0, 600.0, 200.0])
    out = summarize_sessions([s], late_tolerance_ms=10.0)
    assert out["emission_delay_ms.n"] == 3
    unpaced = session(paced=False, events=[("final", 11.4, "the cat sat")], words=words)
    assert "emission_delay_ms.n" not in summarize_sessions([unpaced], late_tolerance_ms=10.0)


def test_failed_sessions_are_counted():
    ok = session(events=[("final", 12.1, "x")])
    bad = session(status="timeout")
    out = summarize_sessions([ok, bad], late_tolerance_ms=10.0)
    assert out["sessions.total"] == 2 and out["sessions.timeout"] == 1
    assert out["error_rate"] == 0.5 and out["final_latency_ms.n"] == 1
