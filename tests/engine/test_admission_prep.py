# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``AdmissionPrep``: what ``overlap_admit`` holds between the caller and the scheduler.

Audit L8. Before it, a failed prep was logged and dropped, an abort could not
find a request that was still in prep, and the drain skipped the audio count.
The engine-level consequences are pinned in ``test_engine.py``. These tests pin
the holding rules, one request state at a time: queued, being prepared, and
prepared.
"""

from __future__ import annotations

import threading
import time

import pytest

from oasr.engine.admission_prep import AdmissionPrep
from oasr.engine.request import Request


def _req(rid):
    return Request(audio=None, request_id=rid, streaming=False)


def _wait_done(prep, n, timeout=5.0):
    deadline = time.monotonic() + timeout
    while prep._done.qsize() < n:
        assert time.monotonic() < deadline, "the prep thread did not finish"
        time.sleep(0.001)


@pytest.fixture
def started():
    """Yields a function that builds and starts an ``AdmissionPrep``, and stops them all."""
    preps = []

    def build(prepare):
        prep = AdmissionPrep(prepare)
        prep.start()
        preps.append(prep)
        return prep

    yield build
    for prep in preps:
        prep.stop()


def test_a_request_is_counted_from_submission_until_it_is_drained(started):
    """Including one whose prep failed: it still has an output to deliver."""

    def prepare(req):
        if req.request_id == "bad":
            raise ValueError("no audio")

    prep = started(prepare)
    reqs = [_req("a"), _req("bad"), _req("b")]
    prep.submit(reqs)
    assert prep.in_flight == 3
    _wait_done(prep, 3)
    assert prep.in_flight == 3
    drained = prep.drain()
    assert prep.in_flight == 0
    assert [r.request_id for r, _ in drained] == ["a", "bad", "b"]
    assert [r for r, _ in drained] == reqs
    errors = {r.request_id: e for r, e in drained}
    assert errors["a"] is None and errors["b"] is None
    assert isinstance(errors["bad"], ValueError)


def test_a_withdrawn_request_is_not_prepared_while_it_is_still_queued():
    calls = []
    prep = AdmissionPrep(lambda req: calls.append(req.request_id))
    prep.submit([_req("gone"), _req("kept")])
    assert prep.withdraw("gone") is True
    assert prep.in_flight == 1
    prep.start()
    try:
        _wait_done(prep, 1)
    finally:
        prep.stop()
    assert calls == ["kept"]
    assert [r.request_id for r, _ in prep.drain()] == ["kept"]


def test_a_request_withdrawn_while_it_is_being_prepared_is_discarded(started):
    entered, gate = threading.Event(), threading.Event()

    def prepare(req):
        entered.set()
        gate.wait(5.0)

    prep = started(prepare)
    prep.submit([_req("gone")])
    assert entered.wait(5.0)
    assert prep.withdraw("gone") is True
    assert prep.in_flight == 0
    gate.set()
    _wait_done(prep, 1)
    assert prep.drain() == []


@pytest.mark.parametrize("failed", [False, True], ids=["prepared", "failed"])
def test_a_request_withdrawn_after_its_prep_is_discarded(started, failed):
    """An aborted request gets no output, not even an error one."""

    def prepare(req):
        if failed:
            raise RuntimeError("boom")

    prep = started(prepare)
    prep.submit([_req("gone")])
    _wait_done(prep, 1)
    assert prep.withdraw("gone") is True
    assert prep.in_flight == 0
    assert prep.drain() == []


def test_withdrawing_an_id_that_is_not_held_changes_nothing(started):
    prep = started(lambda req: None)
    prep.submit([_req("a")])
    assert prep.withdraw("never-submitted") is False
    _wait_done(prep, 1)
    assert [r.request_id for r, _ in prep.drain()] == ["a"]
    assert prep.withdraw("a") is False, "a drained request is the executor's now"


def test_a_withdrawal_is_by_request_not_by_id(started):
    """A request reusing a withdrawn one's id is not mistaken for it."""
    entered, gate = threading.Event(), threading.Event()

    def prepare(req):
        entered.set()
        gate.wait(5.0)

    prep = started(prepare)
    first, second = _req("r"), _req("r")
    prep.submit([first])
    assert entered.wait(5.0)
    prep.withdraw("r")
    prep.submit([second])
    assert prep.in_flight == 1
    gate.set()
    _wait_done(prep, 2)
    drained = prep.drain()
    assert len(drained) == 1 and drained[0][0] is second
    assert prep.in_flight == 0


def test_withdraw_releases_every_held_request_with_the_id():
    prep = AdmissionPrep(lambda req: None)
    prep.submit([_req("r"), _req("r"), _req("other")])
    assert prep.withdraw("r") is True
    assert prep.in_flight == 1


def test_stop_ends_the_thread_and_is_idempotent():
    prep = AdmissionPrep(lambda req: None)
    prep.start()
    thread = prep._thread
    prep.stop()
    assert thread is not None and not thread.is_alive()
    prep.stop()
