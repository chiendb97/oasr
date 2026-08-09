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

from types import SimpleNamespace

import pytest
import torch

from oasr.cache.types import CacheConfig
from oasr.engine.executor.offline import OfflineExecutor
from oasr.engine.executor.streaming import StreamingExecutor
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
    def _executor(rec, *, forward_raises):
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
        return ex, ready

    def test_a_raising_forward_returns_error_outputs(self, monkeypatch):
        rec = _Recorder()
        ex, ready = self._executor(rec, forward_raises=True)
        # Every stream has a full window and no pending audio.
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: True)
        monkeypatch.setattr(Request, "has_pending_audio", property(lambda self: False))

        outs = ex.step()  # must not raise

        assert {o.request_id for o in outs} == {"s1", "s2"}
        assert all(o.finish_reason == "error" for o in outs)
        assert all(o.error_stage == "streaming_forward" for o in outs)
        assert rec.finished == ["s1", "s2"]

    def test_a_healthy_forward_is_unaffected(self, monkeypatch):
        rec = _Recorder()
        ex, ready = self._executor(rec, forward_raises=False)
        monkeypatch.setattr(Request, "has_ready_encoder_chunk", lambda self, w: True)
        monkeypatch.setattr(Request, "has_pending_audio", property(lambda self: False))

        outs = ex.step()

        assert outs == []
        assert rec.finished == [], "no stream should be torn down on the happy path"
