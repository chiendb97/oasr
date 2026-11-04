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
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .request import Request

logger = logging.getLogger(__name__)

#: A prepared request, and the exception its prep raised (``None`` on success).
Prepared = Tuple[Request, Optional[BaseException]]


class AdmissionPrep:
    """Runs ``prepare`` on admitted requests off the step thread.

    ``prepare`` must touch only the request: the thread never takes the engine
    lock, so it keeps making progress while ``run()`` holds that lock.
    """

    def __init__(self, prepare: Callable[[Request], None]) -> None:
        self._prepare = prepare
        self._todo: "queue.Queue[Optional[Request]]" = queue.Queue()
        self._done: "queue.Queue[Prepared]" = queue.Queue()
        self._lock = threading.Lock()
        # Accepted and not yet released, by id.  A list because nothing makes
        # request ids unique, and identity decides which one was withdrawn.
        self._held: Dict[str, List[Request]] = {}
        self._count = 0
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="oasr-admit-prep", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the thread.  Requests still held are dropped with it."""
        t = self._thread
        if t is not None and t.is_alive():
            self._todo.put(None)
            t.join(timeout=timeout)
        self._thread = None

    # -- the caller's side -----------------------------------------------------

    def submit(self, requests: Iterable[Request]) -> None:
        """Hold ``requests`` and queue them for prep, in order.

        Held before they are queued, so a request is counted, and can be
        withdrawn, from the moment its id is handed back to the caller.
        """
        reqs = list(requests)
        with self._lock:
            for req in reqs:
                self._held.setdefault(req.request_id, []).append(req)
            self._count += len(reqs)
        for req in reqs:
            self._todo.put(req)

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

    def _loop(self) -> None:
        while True:
            req = self._todo.get()
            if req is None:
                return
            if not self._is_held(req):
                continue  # withdrawn while queued: not worth preparing
            try:
                self._prepare(req)
            except Exception as exc:
                logger.warning("admission prep failed for request %s: %s", req.request_id, exc)
                self._done.put((req, exc))
            else:
                self._done.put((req, None))


__all__ = ["AdmissionPrep", "Prepared"]
