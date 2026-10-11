# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Raw records -> summaries.  The only place a reported number is computed.

Runners call these at the end of each measurement, and ``recompute`` calls them
on the stored records, so the two cannot disagree about what a metric means.

Conventions (units are in the key names):

* Only ``phase == "measure"`` records count.  Warm-up records are stored, so
  they can be inspected, but never summarised.
* Latency percentiles are over **successful** requests; failures, timeouts and
  rejections are counted and rated instead.  A timed-out request has no
  latency to rank, and silently ranking its timeout value would let the timeout
  setting move the p99.
* ``rtfx`` is aggregate: total audio of successful requests over the wall time
  of the phase, where "wall" depends on the benchmark (see :func:`phase_wall`).
  The per-request ``rtf`` distribution is reported separately; the two are not
  reciprocals of each other unless every request has the same duration.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from benchmarks.asrbench.metrics import summarize, summarize_reps
from benchmarks.asrbench.results import FAILED, OK, REJECTED, TIMEOUT, RequestRecord, SessionRecord

#: How a repetition's wall time is measured.
#: ``sum_service``  sequential requests: the sum of their service times
#: ``batches``      batch calls: the sum of each batch's first-submit..last-done span
#: ``span``         concurrent or backlog: first scheduled start .. last completion
WALL_MODES = ("sum_service", "batches", "span")


def _measured(records: Iterable[Any]) -> List[Any]:
    return [r for r in records if r.phase == "measure"]


def _counts(records: Sequence[Any], unit: str) -> Dict[str, float]:
    total = len(records)
    by = defaultdict(int)
    for r in records:
        by[r.status] += 1
    out = {
        f"{unit}.total": float(total),
        f"{unit}.ok": float(by[OK]),
        f"{unit}.failed": float(by[FAILED]),
        f"{unit}.timeout": float(by[TIMEOUT]),
        f"{unit}.rejected": float(by[REJECTED]),
    }
    if total:
        out["error_rate"] = (total - by[OK]) / total
        out["timeout_rate"] = by[TIMEOUT] / total
    return out


def _empty_rate(texts: Sequence[Optional[str]]) -> Optional[float]:
    stored = [t for t in texts if t is not None]
    if not stored:
        return None
    return sum(1 for t in stored if not t.strip()) / len(stored)


def phase_wall(records: Sequence[RequestRecord], wall_mode: str) -> float:
    """Wall seconds of one repetition's measured requests (see :data:`WALL_MODES`)."""
    done = [r for r in records if r.t_done is not None]
    if not done:
        return 0.0
    if wall_mode == "sum_service":
        return sum(r.t_done - r.t_submit for r in done if r.status == OK)
    if wall_mode == "batches":
        spans: Dict[Any, List[RequestRecord]] = defaultdict(list)
        for r in done:
            spans[r.batch].append(r)
        return sum(
            max(r.t_done for r in b) - min(r.t_submit for r in b)  # type: ignore[operator]
            for b in spans.values()
        )
    if wall_mode == "span":
        return max(r.t_done for r in done) - min(r.t_sched for r in records)  # type: ignore[type-var]
    raise ValueError(f"unknown wall_mode {wall_mode!r}")


def summarize_requests(
    records: Iterable[RequestRecord],
    *,
    wall_mode: str,
    send_lag: bool = False,
    buckets: bool = True,
) -> Dict[str, float]:
    measured = _measured(records)
    out: Dict[str, float] = _counts(measured, "requests")
    ok = [r for r in measured if r.status == OK and r.t_done is not None]
    out.update(summarize((r.latency_ms for r in ok), "latency_ms"))
    out.update(summarize((r.latency_ms / 1e3 / r.duration_s for r in ok if r.duration_s), "rtf"))
    if send_lag:
        out.update(summarize(((r.t_submit - r.t_sched) * 1e3 for r in measured), "send_lag_ms"))
        out.update(summarize((r.service_ms for r in ok), "service_ms"))
    empty = _empty_rate([r.text for r in ok])
    if empty is not None:
        out["empty_rate"] = empty

    by_rep: Dict[int, List[RequestRecord]] = defaultdict(list)
    for r in measured:
        by_rep[r.rep].append(r)
    walls, audios, rtfxs, rates = [], [], [], []
    for rep in sorted(by_rep):
        recs = by_rep[rep]
        wall = phase_wall(recs, wall_mode)
        audio = sum(r.duration_s for r in recs if r.status == OK)
        n_ok = sum(1 for r in recs if r.status == OK)
        walls.append(wall)
        audios.append(audio)
        if wall > 0:
            rtfxs.append(audio / wall)
            rates.append(n_ok / wall)
    wall_total, audio_total = sum(walls), sum(audios)
    out["audio_s"] = audio_total
    out["wall_s"] = wall_total
    if wall_total > 0:
        out["rtfx"] = audio_total / wall_total
        out["throughput.req_per_s"] = len(ok) / wall_total
        out["throughput.audio_s_per_s"] = audio_total / wall_total
    if len(rtfxs) > 1:
        out.update(summarize_reps(rtfxs, "rtfx"))
        out.update(summarize_reps(rates, "throughput.req_per_s"))

    if wall_mode == "batches":
        spans: Dict[Any, List[RequestRecord]] = defaultdict(list)
        for r in ok:
            spans[(r.rep, r.batch)].append(r)
        out.update(
            summarize(
                (
                    (max(x.t_done for x in b) - min(x.t_submit for x in b)) * 1e3  # type: ignore
                    for b in spans.values()
                ),
                "batch_latency_ms",
            )
        )

    if buckets:
        by_bucket: Dict[str, List[RequestRecord]] = defaultdict(list)
        for r in ok:
            if r.bucket:
                by_bucket[r.bucket].append(r)
        if len(by_bucket) > 1:
            for name, recs in sorted(by_bucket.items()):
                lat = summarize((r.latency_ms for r in recs), f"bucket.{name}.latency_ms")
                out.update({k: v for k, v in lat.items() if not k.endswith((".std", ".min"))})
                rtf = [r.latency_ms / 1e3 / r.duration_s for r in recs if r.duration_s]
                if rtf:
                    out[f"bucket.{name}.rtf.mean"] = sum(rtf) / len(rtf)
    return out


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[\w']+")


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def emission_delays_ms(session: SessionRecord) -> List[float]:
    """Per final word: when it first appeared (as a stable prefix) minus when it ended.

    Needs the final transcript's word end times (``session.words``, seconds from
    the start of the audio).  A word counts as emitted at the first event whose
    normalised word sequence agrees with the final one up to and including it,
    so a partial that later revised the word does not count as emitting it.
    """
    if not session.words:
        return []
    final = [_words(x.get("word", "")) for x in session.words]
    flat_final: List[str] = []
    ends: List[float] = []
    for toks, w in zip(final, session.words):
        for t in toks:
            flat_final.append(t)
            ends.append(float(w.get("end", 0.0)))
    if not flat_final:
        return []
    first_seen: List[Optional[float]] = [None] * len(flat_final)
    for ev in sorted(session.events, key=lambda e: e["t"]):
        toks = _words(ev.get("text", ""))
        k = 0
        while k < len(toks) and k < len(flat_final) and toks[k] == flat_final[k]:
            if first_seen[k] is None:
                first_seen[k] = ev["t"]
            k += 1
    return [
        (seen - (session.t_audio0 + end)) * 1e3
        for seen, end in zip(first_seen, ends)
        if seen is not None
    ]


def summarize_sessions(
    sessions: Iterable[SessionRecord], *, late_tolerance_ms: float
) -> Dict[str, float]:
    measured = _measured(sessions)
    out: Dict[str, float] = _counts(measured, "sessions")
    ok = [s for s in measured if s.status == OK]
    first, final, final_srv, intervals, n_partials, lags, emission = [], [], [], [], [], [], []
    late = total_chunks = 0
    for s in ok:
        evs = sorted(s.events, key=lambda e: e["t"])
        text_evs = [e for e in evs if e.get("text", "").strip() and e["kind"] != "error"]
        if text_evs:
            first.append((text_evs[0]["t"] - s.t_audio0) * 1e3)
        finals = [e for e in evs if e["kind"] == "final"]
        last_sent = max(s.chunk_sent) if s.chunk_sent else s.t_audio0
        if finals:
            t_final = finals[-1]["t"]
            end_ref = s.t_audio0 + s.duration_s if s.paced else last_sent
            final.append((t_final - end_ref) * 1e3)
            final_srv.append((t_final - last_sent) * 1e3)
        partials = [e for e in evs if e["kind"] == "partial"]
        n_partials.append(len(partials))
        intervals.extend((b["t"] - a["t"]) * 1e3 for a, b in zip(partials, partials[1:]))
        if s.paced:
            for due, sent in zip(s.chunk_due, s.chunk_sent):
                lag = (sent - due) * 1e3
                lags.append(lag)
                total_chunks += 1
                late += lag > late_tolerance_ms
        if s.paced:
            # Word end times are audio time; only a paced session maps audio time
            # onto wall time.  Unpaced, the "delay" would be meaningless (negative).
            emission.extend(emission_delays_ms(s))
    out.update(summarize(first, "first_partial_ms"))
    out.update(summarize(final, "final_latency_ms"))
    out.update(summarize(final_srv, "final_after_last_chunk_ms"))
    out.update(summarize(intervals, "partial_interval_ms"))
    out.update(summarize(n_partials, "partials_per_session"))
    if lags:
        out.update(summarize(lags, "chunk_lag_ms"))
        out["late_chunk_pct"] = 100.0 * late / total_chunks
    if emission:
        out.update(summarize(emission, "emission_delay_ms"))
    empty = _empty_rate([s.text for s in ok])
    if empty is not None:
        out["empty_rate"] = empty

    audio = sum(s.duration_s for s in ok)
    out["audio_s"] = audio
    by_rep: Dict[int, List[SessionRecord]] = defaultdict(list)
    for s in measured:
        by_rep[s.rep].append(s)
    wall = 0.0
    for recs in by_rep.values():
        ends = [e["t"] for s in recs for e in s.events if e["kind"] in ("final", "error")]
        if ends:
            wall += max(ends) - min(s.t_audio0 for s in recs)
    out["wall_s"] = wall
    if wall > 0 and ok and not ok[0].paced:
        # Paced sessions are bounded by real time by construction; an aggregate
        # RTFx there would just restate the session count.
        out["rtfx"] = audio / wall
        out["throughput.audio_s_per_s"] = audio / wall
    return out
