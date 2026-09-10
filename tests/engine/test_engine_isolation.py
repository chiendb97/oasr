# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Failure isolation: one bad request must not take down its peers.

Two holes this pins, both from the architecture review's C6.

*The KV-pool invariant.*  ``max_num_blocks >= max_batch_size *
blocks_per_stream`` was documented in three places and checked in none.  With
eviction enabled (``num_left_chunks >= 0``) ``at_capacity()`` unconditionally
returns ``False`` — the oldest block is recycled, so a stream is never "full" —
which means there is no proactive capacity gate at all.  Violating the
invariant surfaced as ``BlockPool exhausted`` raised from *inside the encoder
forward*, where it takes out the tick rather than one stream.

*Everything else.*  Any exception in collation, the encoder forward, feature
extraction or decode escaped ``step()``, and the serving dispatcher turns a
failed step into an INTERNAL error for every in-flight request.  One
pathological input — a zero-length waveform, a NaN, an out-of-range vocab id —
was a multi-tenant outage.

These run on CPU with fakes: the isolation logic is control flow, and pinning
it should not need a GPU or a checkpoint.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from oasr.cache.types import CacheConfig
from oasr.engine.executor.offline import OfflineExecutor
from oasr.engine.executor.streaming import StreamingExecutor
from oasr.engine.metrics import build_metrics
from oasr.engine.request import Request, RequestOutput, RequestState

# ---------------------------------------------------------------------------
# The KV-pool invariant
# ---------------------------------------------------------------------------


class TestPoolInvariant:
    def test_undersized_pool_raises_at_construction(self):
        with pytest.raises(ValueError, match="cannot hold"):
            CacheConfig(
                num_left_chunks=16,
                chunk_size=16,
                block_size_frames=16,
                max_batch_size=32,
                max_num_blocks=64,
            )

    def test_the_message_carries_the_arithmetic(self):
        """A capacity error nobody can act on is a crash with extra steps."""
        with pytest.raises(ValueError) as exc:
            CacheConfig(
                num_left_chunks=16,
                chunk_size=16,
                block_size_frames=16,
                max_batch_size=32,
                max_num_blocks=64,
            )
        msg = str(exc.value)
        for fragment in ("64", "32", "16 blocks each", "512", "max_num_blocks"):
            assert fragment in msg, f"{fragment!r} missing from: {msg}"

    def test_sufficient_pool_is_accepted(self):
        cfg = CacheConfig(
            num_left_chunks=16,
            chunk_size=16,
            block_size_frames=16,
            max_batch_size=32,
            max_num_blocks=512,
        )
        assert cfg.max_num_blocks >= cfg.max_batch_size * cfg.blocks_per_stream

    def test_unlimited_history_is_bounded_by_construction(self):
        """``num_left_chunks < 0`` derives blocks_per_stream from the pool.

        The check must not fire there: the fair share *is* ``max_num_blocks //
        max_batch_size``, so the invariant is an identity rather than a
        constraint, and rejecting a small pool would break the default config.
        """
        cfg = CacheConfig(num_left_chunks=-1, max_batch_size=32, max_num_blocks=8)
        assert cfg.max_num_blocks <= cfg.max_batch_size * cfg.blocks_per_stream


# ---------------------------------------------------------------------------
# Offline: one bad request in a micro-batch
# ---------------------------------------------------------------------------


def _request(rid: str) -> Request:
    req = Request(request_id=rid, streaming=False)
    req.state = RequestState.RUNNING
    return req


class _Poisoned:
    """Model runner whose forward raises for one request id."""

    def __init__(self, bad_id: str, exc: Exception):
        self.bad_id = bad_id
        self.exc = exc
        self.chunks_seen: list[list[str]] = []

    def forward_offline(self, features, lengths):
        ids = list(features)  # the fake collate hands us the id list
        self.chunks_seen.append(ids)
        if self.bad_id in ids:
            raise self.exc
        return SimpleNamespace(ids=ids), lengths


def _offline_executor(runner) -> OfflineExecutor:
    """An ``OfflineExecutor`` with only what ``run()`` touches.

    ``__init__`` wants a scheduler, an input processor and a real device; the
    isolation path needs none of them.
    """
    ex = OfflineExecutor.__new__(OfflineExecutor)
    ex._scheduler = SimpleNamespace(
        split_offline_batch=lambda batch: ([batch], None),
    )
    ex._mr = runner
    ex._enable_packing = False
    ex._pending = {}
    ex._op = SimpleNamespace(
        strategy=SimpleNamespace(incremental=False, consumes="log_probs"),
        # ``requests`` rides along for the families that read per-request
        # options at decode time (word timings); this stub ignores it.
        decode_offline=lambda enc, lens, requests=None: [
            RequestOutput(request_id=i, text=f"ok-{i}", tokens=[[1]]) for i in enc.ids
        ],
        fill_nbest_texts=lambda req, out: None,
    )
    # The fake collate passes the ids straight through as "features".
    ex._collate = lambda chunk: ([r.request_id for r in chunk], torch.tensor([1] * len(chunk)))
    return ex


class TestOfflineIsolation:
    def test_one_bad_request_does_not_take_its_peers(self):
        runner = _Poisoned("bad", ValueError("out-of-range vocab id"))
        ex = _offline_executor(runner)
        batch = [_request("a"), _request("bad"), _request("c")]

        outs = {o.request_id: o for o in ex.run(batch)}

        assert set(outs) == {"a", "bad", "c"}
        assert outs["a"].text == "ok-a" and outs["c"].text == "ok-c"
        assert outs["bad"].finish_reason == "error"
        assert outs["bad"].error_stage == "offline_forward"
        assert outs["bad"].finished and outs["bad"].text == ""

    def test_the_failing_batch_is_retried_singly(self):
        """Isolation is what makes the peers survive — check the mechanism."""
        runner = _Poisoned("bad", ValueError("boom"))
        ex = _offline_executor(runner)
        ex.run([_request("a"), _request("bad"), _request("c")])

        # First the whole micro-batch, then one pass per member.
        assert runner.chunks_seen[0] == ["a", "bad", "c"]
        assert runner.chunks_seen[1:] == [["a"], ["bad"], ["c"]]

    def test_a_singleton_failure_is_not_retried(self):
        runner = _Poisoned("bad", ValueError("boom"))
        ex = _offline_executor(runner)
        outs = ex.run([_request("bad")])
        assert len(runner.chunks_seen) == 1, "a batch of one has nothing to isolate"
        assert outs[0].error_stage == "offline_forward"

    def test_oom_rejects_the_batch_without_retrying(self):
        """Re-running under memory pressure is how one big request cascades."""
        runner = _Poisoned("bad", torch.cuda.OutOfMemoryError("CUDA out of memory"))
        ex = _offline_executor(runner)
        outs = ex.run([_request("a"), _request("bad"), _request("c")])

        assert len(runner.chunks_seen) == 1, "OOM must not trigger a retry pass"
        assert {o.error_stage for o in outs} == {"offline_oom"}
        assert all(o.finish_reason == "error" for o in outs)

    def test_a_healthy_batch_is_untouched(self):
        runner = _Poisoned("nobody", ValueError("never raised"))
        ex = _offline_executor(runner)
        outs = ex.run([_request("a"), _request("b")])
        assert [o.text for o in outs] == ["ok-a", "ok-b"]
        assert all(o.finish_reason is None for o in outs)
        assert len(runner.chunks_seen) == 1

    def test_a_post_collate_failure_isolates_over_the_features(self, caplog):
        """After collation the waveforms are gone, so the retry cannot re-collate.

        ``InputProcessor.collate`` releases ``request.audio`` once the GPU
        feature tensor owns the batch.  Re-running the whole micro-batch per
        request therefore dies on ``NoneType.size`` — and *that* is what every
        request in the batch got told, including the healthy ones, while the
        real cause never reached a log line.  It shipped: a conv kernel failing
        on an over-wide batch (``max_batch_size >= ~220``) returned empty
        transcripts for the entire corpus under an ``AttributeError`` about
        waveforms.  Past collate the isolation pass runs over the features
        already built, which needs no waveform.
        """
        import logging

        runner = _Poisoned("bad", ValueError("Conv2DActivation kernel failed"))
        ex = _offline_executor(runner)
        base_collate = ex._collate

        def releasing_collate(chunk):
            # Exactly the real one's hazard: refuses a second pass, because the
            # waveforms it needs were handed to the GPU on the first.
            if any(r.audio is None for r in chunk):
                raise AttributeError("'NoneType' object has no attribute 'size'")
            out = base_collate(chunk)
            for r in chunk:
                r.audio = None
            return out

        batch = [_request("a"), _request("bad"), _request("c")]
        for req in batch:
            req.audio = torch.zeros(4)
        ex._collate = releasing_collate

        with caplog.at_level(logging.WARNING, logger="oasr.engine.executor.offline"):
            outs = {o.request_id: o for o in ex.run(batch)}

        # The peers survive — under the old retry they died on the re-collate.
        assert outs["a"].text == "ok-a" and outs["c"].text == "ok-c"
        assert outs["a"].finish_reason is None and outs["c"].finish_reason is None
        assert outs["bad"].finish_reason == "error"
        assert outs["bad"].error_stage == "offline_forward"
        # The isolation pass reran the rows, not the collation.
        assert runner.chunks_seen == [["a", "bad", "c"], ["a"], ["bad"], ["c"]]
        # And the log names the real cause rather than the released waveform.
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "Conv2DActivation kernel failed" in text
        assert "NoneType" not in text


# ---------------------------------------------------------------------------
# Streaming: a failed cohort must not drain the pool
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.freed: list[str] = []
        self.finished: list[str] = []

    def free_session(self, req):
        self.freed.append(("session", req.request_id))

    def free_stream(self, req):
        self.freed.append(("stream", req.request_id))

    def finish_request(self, rid):
        self.finished.append(rid)


def _streaming_executor(rec: _Recorder) -> StreamingExecutor:
    ex = StreamingExecutor.__new__(StreamingExecutor)
    ex._op = SimpleNamespace(free_session=rec.free_session)
    ex._mr = SimpleNamespace(free_stream=rec.free_stream)
    ex._scheduler = SimpleNamespace(finish_request=rec.finish_request)
    return ex


class TestStreamingIsolation:
    def test_failed_cohort_is_finalized_and_freed(self):
        rec = _Recorder()
        ex = _streaming_executor(rec)
        cohort = [_request("s1"), _request("s2")]

        outs = ex._fail_cohort(cohort, RuntimeError("BlockPool exhausted"), "streaming_forward")

        assert [o.request_id for o in outs] == ["s1", "s2"]
        assert all(o.finish_reason == "error" for o in outs)
        assert all(o.error_stage == "streaming_forward" for o in outs)
        assert all(o.finished for o in outs)
        assert all(r.state is RequestState.FINISHED for r in cohort)
        # Both caches released for both streams, and the scheduler told.
        assert set(rec.freed) == {
            ("session", "s1"),
            ("stream", "s1"),
            ("session", "s2"),
            ("stream", "s2"),
        }
        assert rec.finished == ["s1", "s2"]

    def test_a_release_that_itself_raises_does_not_abort_teardown(self):
        """Teardown runs after an unknown failure; it cannot assume clean state.

        Leaking a cache slot per failure exhausts the pool in a way that looks
        like a capacity bug rather than an error path, so every release is
        attempted regardless of the previous one.
        """
        rec = _Recorder()
        ex = _streaming_executor(rec)

        def angry_free_session(req):
            raise RuntimeError("session already gone")

        ex._op = SimpleNamespace(free_session=angry_free_session)
        cohort = [_request("s1")]

        outs = ex._fail_cohort(cohort, RuntimeError("boom"), "streaming_features")

        assert outs[0].error_stage == "streaming_features"
        assert ("stream", "s1") in rec.freed, "the second release must still run"
        assert rec.finished == ["s1"]


class TestStreamingStepDoesNotRaise:
    """The contract C6 is actually about: a failed forward must not escape step().

    ``_fail_cohort`` above is the mechanism; this is the property.  Before the
    guard, ``RuntimeError("BlockPool exhausted")`` propagated out of
    ``ASREngine.step()`` and the dispatcher fanned it out as INTERNAL to every
    in-flight request — three such ticks and the process drained.
    """

    @staticmethod
    def _executor(rec, *, forward_raises, lookahead=False):
        ex = StreamingExecutor.__new__(StreamingExecutor)
        ready = [_request("s1"), _request("s2")]
        for r in ready:
            r.stream_id = 0

        def schedule():
            return [], list(ready)

        def forward(reqs):
            if forward_raises:
                raise RuntimeError("BlockPool exhausted: requested 1 block but 0 are free")
            return {r.request_id: torch.zeros(1) for r in reqs}

        ex._scheduler = SimpleNamespace(
            schedule_streaming=schedule, finish_request=rec.finish_request
        )
        ex._inp = SimpleNamespace(extract_streaming_batch=lambda reqs, cuda_stream=None: None)
        ex._mr = SimpleNamespace(forward_streaming_step=forward, free_stream=rec.free_stream)
        ex._op = SimpleNamespace(
            decode_streaming_batch=lambda reqs, m: [],
            finalize_streaming=lambda req: RequestOutput(req.request_id, "", [[]]),
            fill_nbest_texts=lambda req, out: None,
            free_session=rec.free_session,
        )
        ex._config = SimpleNamespace(decoding_window=1)
        ex._feat_stream = None
        # Both step orders have their own forward/extract/decode sequencing and
        # therefore their own teardown ordering; the contract is the same.
        ex._lookahead = lookahead
        return ex, ready

    @pytest.mark.parametrize("lookahead", [False, True], ids=["serial", "pipelined"])
    def test_a_raising_forward_returns_error_outputs(self, monkeypatch, lookahead):
        rec = _Recorder()
        ex, ready = self._executor(rec, forward_raises=True, lookahead=lookahead)
        # Every stream has a full window and no pending audio.
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: True)
        monkeypatch.setattr(Request, "has_pending_audio", property(lambda self: False))

        outs = ex.step()  # must not raise

        assert {o.request_id for o in outs} == {"s1", "s2"}
        assert all(o.finish_reason == "error" for o in outs)
        assert all(o.error_stage == "streaming_forward" for o in outs)
        assert rec.finished == ["s1", "s2"]

    @pytest.mark.parametrize("lookahead", [False, True], ids=["serial", "pipelined"])
    def test_a_healthy_forward_is_unaffected(self, monkeypatch, lookahead):
        rec = _Recorder()
        ex, ready = self._executor(rec, forward_raises=False, lookahead=lookahead)
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: True)
        monkeypatch.setattr(Request, "has_pending_audio", property(lambda self: False))

        outs = ex.step()

        assert outs == []
        assert rec.finished == [], "no stream should be torn down on the happy path"


# ---------------------------------------------------------------------------
# Pipelined ticks: streaming feature lookahead and offline collate prefetch
#
# Both changes are reorderings, so what has to be pinned is the *order* — which
# is invisible to a transcript comparison but is the entire point.  Each test
# below fails if its stage is moved back to where it used to be.
# ---------------------------------------------------------------------------


class _OrderRecorder:
    """Records the sequence of executor stages as they are entered."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def mark(self, name):
        def _record(*args, **kwargs):
            self.seen.append(name)
            return None

        return _record


def _lookahead_executor(order: _OrderRecorder, *, lookahead: bool):
    """A ``StreamingExecutor`` wired to record stage order and nothing else."""
    ex = StreamingExecutor.__new__(StreamingExecutor)
    running = [_request("s1")]
    for r in running:
        r.stream_id = 0

    def extract(reqs, cuda_stream=None):
        order.seen.append("extract")

    def forward(reqs):
        order.seen.append("forward")
        return {r.request_id: torch.zeros(1) for r in reqs}

    def decode(reqs, m):
        order.seen.append("decode")
        return []

    ex._scheduler = SimpleNamespace(
        schedule_streaming=lambda: ([], list(running)),
        finish_request=lambda rid: None,
    )
    ex._inp = SimpleNamespace(extract_streaming_batch=extract)
    ex._mr = SimpleNamespace(forward_streaming_step=forward, free_stream=lambda r: None)
    ex._op = SimpleNamespace(
        decode_streaming_batch=decode,
        finalize_streaming=lambda req: RequestOutput(req.request_id, "", [[]]),
        fill_nbest_texts=lambda req, out: None,
        free_session=lambda r: None,
    )
    ex._config = SimpleNamespace(decoding_window=1)
    ex._feat_stream = None
    ex._lookahead = lookahead
    return ex, running


class TestStreamingFeatureLookahead:
    """The pack has to run *behind* the encoder, not in front of it.

    ``pad+pin`` — the per-stream concat + ``audio_scale`` + write into pinned
    staging — is host work that issues no GPU operation, and it profiled as the
    largest single block of GPU-idle in a streaming step precisely because it
    sat ahead of the forward.  Moving it between the forward and the decode is
    the fix, and the decode is the boundary that makes the placement matter: it
    ends in a device->host readback, so anything after it overlaps nothing.
    """

    @staticmethod
    def _run(lookahead, monkeypatch):
        order = _OrderRecorder()
        ex, _ = _lookahead_executor(order, lookahead=lookahead)
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: True)
        monkeypatch.setattr(Request, "has_pending_audio", property(lambda self: True))
        ex.step()
        return order.seen

    def test_pipelined_extracts_between_the_forward_and_the_decode(self, monkeypatch):
        assert self._run(True, monkeypatch) == ["forward", "extract", "decode"]

    def test_serial_extracts_before_the_forward(self, monkeypatch):
        assert self._run(False, monkeypatch) == ["extract", "forward", "decode"]


class TestStreamingLookaheadDrains:
    """A stream's last chunk must still be forwarded, one step later.

    Under lookahead the features a step extracts are consumed by the *next*
    step, so the finalisation check has to keep a stream alive on the strength
    of a ready encoder chunk alone — its audio deque is already empty.  Getting
    this wrong finalises the stream one chunk early and silently truncates every
    transcript's last word.
    """

    def test_a_stream_is_not_finalised_while_a_chunk_is_still_unforwarded(self, monkeypatch):
        order = _OrderRecorder()
        ex, running = _lookahead_executor(order, lookahead=True)
        req = running[0]
        req.audio_final = True
        # The state right after the extract that consumed the last chunk: no
        # audio left, one encoder window built and not yet forwarded.
        monkeypatch.setattr(Request, "has_pending_audio", property(lambda self: False))
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: True)

        outs = ex.step()

        assert outs == [], "a stream with an unforwarded window is not drained"
        assert req.state is not RequestState.FINISHED

        # Next step: the window has been forwarded, nothing is left.
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: False)
        outs = ex.step()
        assert [o.request_id for o in outs] == ["s1"]


def _prefetch_executor(order: _OrderRecorder, batches, *, prefetch: bool):
    """An ``OfflineExecutor`` wired for the pipelined tick, on CPU.

    ``batches`` is consumed one per ``schedule_offline`` call, so the fixture can
    drive several ticks and watch where each stage lands.  ``_prefetch`` selects
    the tick shape and ``_collate_stream`` stays ``None``, so the collate runs
    inline and ``_StagedBatch.ready`` is ``None`` — the ordering under test is
    the host's, and it is the same with or without the side stream.
    """
    ex = OfflineExecutor.__new__(OfflineExecutor)
    queue = list(batches)

    def schedule_offline(limit=None):
        order.seen.append("schedule")
        return queue.pop(0) if queue else []

    def collate(chunk):
        order.seen.append("collate")
        return [r.request_id for r in chunk], torch.tensor([1] * len(chunk))

    def forward_offline(features, lengths):
        order.seen.append("forward")
        return SimpleNamespace(ids=list(features)), lengths

    def decode_offline(enc, lens, requests=None):
        order.seen.append("decode")
        return [RequestOutput(request_id=i, text=f"ok-{i}", tokens=[[1]]) for i in enc.ids]

    ex._scheduler = SimpleNamespace(
        schedule_offline=schedule_offline,
        split_offline_batch=lambda batch: ([batch], None),
        num_waiting_offline=0,
    )
    ex._mr = SimpleNamespace(forward_offline=forward_offline)
    ex._enable_packing = False
    ex._pending = {}
    ex._op = SimpleNamespace(
        strategy=SimpleNamespace(
            incremental=False, consumes="log_probs", has_pending=lambda: False
        ),
        decode_offline=decode_offline,
        fill_nbest_texts=lambda req, out: None,
    )
    ex._collate = collate
    ex._prefetch = prefetch
    ex._collate_stream = None
    ex._collate_done = None
    ex._queued = deque()
    ex._staged = None
    ex._skipped_admits = 0
    ex._decode_admit_window_ms = 0.0
    ex._max_batch_size = 8
    ex._max_decode_slots = None
    ex._metrics = build_metrics(enabled=False)
    return ex


class TestOfflineCollatePrefetch:
    """The next batch has to be selected and collated *after* the forward.

    Batch selection is pure host work and the collate's GPU work cannot start
    until it finishes, so in the serial tick both sit ahead of the encoder with
    nothing queued for the GPU — ~4 ms of a ~23.5 ms step at ``max_batch_size``
    256.  Issuing them after the forward puts them in the window where the GPU
    is busy.  Issued *before* the forward, they are once again the thing the GPU
    is idle for, which is exactly the state being replaced.
    """

    @staticmethod
    def _batches():
        return [[_request("a1"), _request("a2")], [_request("b1")]]

    def test_pipelined_stages_the_next_batch_after_the_forward(self):
        order = _OrderRecorder()
        ex = _prefetch_executor(order, self._batches(), prefetch=True)

        first = ex.step()

        # Priming collates once in front of the GPU — then the second batch's
        # schedule + collate land between this batch's forward and its decode.
        assert order.seen == [
            "schedule",
            "collate",
            "forward",
            "schedule",
            "collate",
            "decode",
        ]
        assert [o.request_id for o in first] == ["a1", "a2"]

    def test_serial_collates_before_the_forward(self):
        order = _OrderRecorder()
        ex = _prefetch_executor(order, self._batches(), prefetch=False)

        ex.step()

        assert order.seen == ["schedule", "collate", "forward", "decode"]

    def test_a_staged_batch_keeps_the_engine_pending(self):
        """A drain loop asks ``has_pending``; a staged batch is invisible to the
        scheduler queue and is not parked in ``_pending``, so without it the
        loop exits one tick before running the batch it just collated."""
        order = _OrderRecorder()
        ex = _prefetch_executor(order, self._batches(), prefetch=True)

        ex.step()  # returns batch a, stages batch b

        assert ex._staged is not None
        assert ex.has_pending(), "the staged batch must keep the drain loop alive"
        assert ex.num_running() == 1

        second = ex.step()
        assert [o.request_id for o in second] == ["b1"]
        assert not ex.has_pending()

    def test_every_request_comes_back_exactly_once(self):
        """Across ticks, with the pipeline priming and draining."""
        order = _OrderRecorder()
        batches = [[_request(f"r{i}")] for i in range(5)]
        ex = _prefetch_executor(order, [list(b) for b in batches], prefetch=True)

        seen: list[str] = []
        for _ in range(8):
            seen.extend(o.request_id for o in ex.step())

        assert seen == [f"r{i}" for i in range(5)]


class TestOfflinePrefetchOrdering:
    """The partitioner's length sort must still be undone per micro-batch.

    ``split_offline_batch`` returns indices that are flat over the *whole*
    scheduled batch, and its chunks no longer finish in the same tick, so the
    restore has to work from each chunk's own slice.  Treating those indices as
    positions within the chunk is an ``IndexError`` waiting for the first batch
    the scheduler actually splits.
    """

    def test_scattered_indices_restore_by_rank(self):
        outs = [RequestOutput(request_id=r, text=r, tokens=[[1]]) for r in ("c", "a", "b")]
        # Length-sorted chunk whose members came 7th, 2nd and 5th in the batch.
        restored = OfflineExecutor._restore_order(outs, [7, 2, 5])
        assert [o.request_id for o in restored] == ["a", "b", "c"]

    def test_a_full_permutation_matches_the_serial_restore(self):
        outs = [RequestOutput(request_id=r, text=r, tokens=[[1]]) for r in ("c", "a", "b")]
        order = [2, 0, 1]
        restored = OfflineExecutor._restore_order(outs, order)
        expected: list = [None] * 3
        for pos, orig in enumerate(order):  # the pre-pipeline restore, verbatim
            expected[orig] = outs[pos]
        assert restored == expected
