# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Offline admission prep on a daemon thread (``EngineConfig.overlap_admit``).

The caller gets a request id back before ``prepare_offline`` (waveform
canonicalisation and the frame-count stamp) has run, and the prep thread hands
the request to :meth:`~oasr.engine.ASREngine.step`, which admits it under the
engine lock.  Between the two the request belongs to neither the caller nor the
scheduler, and that gap is what this class owns:

* **Accounting.**  A held request counts as waiting, so the dispatcher does not
  read the engine as idle while prep is in flight.
* **Failure.**  A ``prepare_offline`` that raises comes back from :meth:`drain`
  with its exception, so the engine can answer with an error output.  Logging
  it and dropping the request left the client waiting for its timeout.
* **Abort.**  :meth:`withdraw` releases a held request wherever it is: queued,
  being prepared, or prepared and waiting for the next step.  Only the
  scheduler was searched before, so an aborted request was admitted at the next
  step and decoded anyway.

Requests are tracked by identity, not by id, so a second request reusing an
aborted one's id is not mistaken for it.

The same holding is what offline ``vad.mode="segment"`` needs while its
detector runs, so :meth:`AdmissionPrep.batched` gives it a second thread whose
prep takes everything queued as one batch: one detector call for whatever
arrived since the last one.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .request import Request

logger = logging.getLogger(__name__)

#: A prepared request, and the exception its prep raised (``None`` on success).
Prepared = Tuple[Request, Optional[BaseException]]

#: Prepares a batch; returns each request's exception (``None`` on success), in order.
BatchPrepare = Callable[[List[Request]], Sequence[Optional[BaseException]]]


class AdmissionPrep:
    """Runs ``prepare`` on admitted requests off the step thread.

    ``prepare`` must touch only the request: the thread never takes the engine
    lock, so it keeps making progress while ``run()`` holds that lock.
    """

    def __init__(self, prepare: Callable[[Request], None]) -> None:
        self._setup(_one_at_a_time(prepare), max_batch=1, name="oasr-admit-prep")

    @classmethod
    def batched(cls, prepare: BatchPrepare, *, max_batch: int, name: str) -> "AdmissionPrep":
        """A prep thread that hands ``prepare`` everything queued, up to ``max_batch``.

        Nothing waits for a batch to fill: the thread takes what is queued when
        it is free, so a lone request is prepared at once and a burst that
        arrives while a batch is being prepared becomes the next batch.
        ``prepare`` may raise, which fails every request in that batch.
        """
        prep = cls.__new__(cls)
        prep._setup(prepare, max_batch=max_batch, name=name)
        return prep

    def _setup(self, prepare: BatchPrepare, *, max_batch: int, name: str) -> None:
        self._prepare_batch = prepare
        self._max_batch = max(1, int(max_batch))
        self._name = name
        # Lists rather than single requests, so a submission that is one batch
        # reaches the thread as one item and is not split by a wake-up between
        # two of its puts.
        self._todo: "queue.Queue[Optional[List[Request]]]" = queue.Queue()
        self._done: "queue.Queue[Prepared]" = queue.Queue()
        self._lock = threading.Lock()
        # Accepted and not yet released, by id.  A list because nothing makes
        # request ids unique, and identity decides which one was withdrawn.
        self._held: Dict[str, List[Request]] = {}
        self._count = 0
        self._thread: Optional[threading.Thread] = None
        #: Requests carried over from a queue item that did not fit the last batch.
        self._spill: List[Request] = []
        self._stopping = False

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._stopping = False
        self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the thread.  Requests still held are dropped with it."""
        t = self._thread
        if t is not None and t.is_alive():
            self._stopping = True
            self._todo.put(None)
            t.join(timeout=timeout)
        self._thread = None

    # -- the caller's side -----------------------------------------------------

    def submit(self, requests: Iterable[Request]) -> None:
        """Hold ``requests`` and queue them for prep, in order.

        Held before they are queued, so a request is counted, and can be
        withdrawn, from the moment its id is handed back to the caller.
        """
        reqs = self._hold(requests)
        if reqs:
            self._todo.put(reqs)

    def prepare_inline(self, requests: Iterable[Request]) -> None:
        """Hold ``requests`` and prepare them now, on the calling thread.

        For a caller with nothing better to do meanwhile -- an idle engine, whose
        next step would otherwise wait for the thread to wake, prepare and hand
        back.  Everything else is as if the thread had prepared them: they are
        held, counted and withdrawable until :meth:`drain` releases them, and a
        failure comes back from there.
        """
        reqs = self._hold(requests)
        if reqs:
            self._run_batch(reqs)

    def _hold(self, requests: Iterable[Request]) -> List[Request]:
        reqs = list(requests)
        with self._lock:
            for req in reqs:
                self._held.setdefault(req.request_id, []).append(req)
            self._count += len(reqs)
        return reqs

    def withdraw(self, request_id: str) -> bool:
        """Release every held request with this id; ``True`` if there was one.

        A withdrawn request is skipped by the thread if it has not been
        prepared yet, and discarded by :meth:`drain` if it has.
        """
        with self._lock:
            held = self._held.pop(request_id, None)
            if not held:
                return False
            self._count -= len(held)
        return True

    @property
    def in_flight(self) -> int:
        """Requests accepted and not yet released by :meth:`drain` or :meth:`withdraw`."""
        with self._lock:
            return self._count

    # -- the step's side -------------------------------------------------------

    def drain(self) -> List[Prepared]:
        """Release every request whose prep has finished, with its exception.

        The caller admits each successful one and reports each failed one.
        Withdrawn requests are not returned.  A request stays held, and so
        counted as waiting, until it comes out of here, so its error output
        is delivered by a step the dispatcher still runs.
        """
        out: List[Prepared] = []
        while True:
            try:
                req, exc = self._done.get_nowait()
            except queue.Empty:
                return out
            if self._release(req):
                out.append((req, exc))

    # -- internals -------------------------------------------------------------

    def _is_held(self, req: Request) -> bool:
        with self._lock:
            return any(r is req for r in self._held.get(req.request_id, ()))

    def _release(self, req: Request) -> bool:
        with self._lock:
            held = self._held.get(req.request_id)
            if not held:
                return False
            for i, r in enumerate(held):
                if r is req:
                    del held[i]
                    break
            else:
                return False
            if not held:
                del self._held[req.request_id]
            self._count -= 1
        return True

    def _next_batch(self) -> Optional[List[Request]]:
        """Block for work, then take whatever else is queued, up to ``max_batch``.

        ``None`` at the sentinel :meth:`stop` queues.  Requests withdrawn while
        queued are dropped here: they are not worth preparing.
        """
        if self._stopping:
            # What a batch left over is dropped with the thread; the queue is
            # read on to the stop sentinel, so a restarted thread starts clean.
            self._spill = []
        batch, self._spill = self._spill, []
        if not batch:
            item = self._todo.get()
            if item is None:
                return None
            batch = item
        while len(batch) < self._max_batch:
            try:
                item = self._todo.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._todo.put(None)  # stop after this batch
                break
            batch.extend(item)
        batch, self._spill = batch[: self._max_batch], batch[self._max_batch :]
        return [req for req in batch if self._is_held(req)]

    def _run_batch(self, batch: List[Request]) -> None:
        """Prepare one batch and queue each request's outcome for :meth:`drain`."""
        if not batch:
            return
        try:
            errors = list(self._prepare_batch(batch))
        except Exception as exc:
            errors = [exc] * len(batch)
        for req, err in zip(batch, errors):
            if err is not None:
                logger.warning("admission prep failed for request %s: %s", req.request_id, err)
            self._done.put((req, err))

    def _loop(self) -> None:
        while True:
            batch = self._next_batch()
            if batch is None:
                return
            self._run_batch(batch)


def _one_at_a_time(prepare: Callable[[Request], None]) -> BatchPrepare:
    """A per-request ``prepare`` as a batch one, each failure kept to its request."""

    def run(batch: List[Request]) -> List[Optional[BaseException]]:
        errors: List[Optional[BaseException]] = []
        for req in batch:
            try:
                prepare(req)
            except Exception as exc:
                errors.append(exc)
            else:
                errors.append(None)
        return errors

    return run


__all__ = ["AdmissionPrep", "BatchPrepare", "Prepared"]
