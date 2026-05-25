#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Stand-in engine worker for end-to-end testing.

Uses the real ``oasr.serving.engine_worker.EngineWorker`` loop but plugs in a
fake ``ASREngine`` that:

- accepts streaming and offline requests
- emits one partial after the first ``feed_chunk`` (text = "partial-i")
- emits a final after ``is_last=True`` or the first ``add_request`` for offline
  (text = "echo " + base64(audio[:8]))

No torch / no checkpoint required.  Reads CLI exactly like the real worker.
"""

from __future__ import annotations

import base64
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional

import numpy as np

# Lightweight stand-in for oasr.engine.RequestOutput — fields the worker reads.
@dataclass
class FakeRequestOutput:
    request_id: str
    text: str
    tokens: List[List[int]]
    scores: Optional[List[float]]
    finished: bool


@dataclass
class _StreamState:
    request_id: str
    partials_emitted: int = 0
    pending_audio: Deque[np.ndarray] = field(default_factory=deque)
    is_last_seen: bool = False
    final_emitted: bool = False
    audio_seed: bytes = b""


class FakeEngine:
    """Drop-in for ASREngine with the public API the worker uses."""

    def __init__(self):
        self._lock = threading.RLock()
        self._streams: Dict[str, _StreamState] = {}
        self._offline: Deque[str] = deque()
        self._offline_audio: Dict[str, bytes] = {}
        self._waiting = 0
        self._running = 0
        self._config = SimpleNamespace(
            ckpt_dir="/tmp/fake",
            device="cpu",
            dtype="float32",
            chunk_size=16,
            max_batch_size=4,
            decoder_type="ctc_prefix_beam",
            _model_config=SimpleNamespace(vocab_size=32),
        )

    @property
    def num_running(self) -> int:
        with self._lock:
            return self._running

    @property
    def num_waiting(self) -> int:
        with self._lock:
            return self._waiting

    def add_request(self, audio, request_id=None, sample_rate=16000, streaming=True, priority=0):
        with self._lock:
            if streaming:
                self._streams[request_id] = _StreamState(request_id=request_id)
                self._streams[request_id].pending_audio.append(audio)
                self._streams[request_id].is_last_seen = True
            else:
                buf = audio.tobytes() if isinstance(audio, np.ndarray) else bytes(audio)
                self._offline_audio[request_id] = buf
                self._offline.append(request_id)
                self._waiting += 1
        return request_id

    def add_streaming_request(self, request_id=None, sample_rate=16000, priority=0):
        with self._lock:
            self._streams[request_id] = _StreamState(request_id=request_id)
            self._running += 1
        return request_id

    def feed_chunk(self, request_id, chunk, is_last=False):
        with self._lock:
            st = self._streams.get(request_id)
            if st is None:
                raise KeyError(f"unknown request_id: {request_id}")
            if not st.audio_seed and isinstance(chunk, np.ndarray) and chunk.size > 0:
                st.audio_seed = chunk.tobytes()[:8]
            st.pending_audio.append(chunk)
            if is_last:
                st.is_last_seen = True

    def abort_request(self, request_id):
        with self._lock:
            if request_id in self._streams:
                del self._streams[request_id]
                self._running = max(0, self._running - 1)

    def step(self) -> List[FakeRequestOutput]:
        outs: List[FakeRequestOutput] = []
        with self._lock:
            # Offline: drain one request → final.
            while self._offline:
                rid = self._offline.popleft()
                self._waiting -= 1
                buf = self._offline_audio.pop(rid, b"")
                text = "offline-echo:" + base64.b64encode(buf[:8]).decode()
                outs.append(FakeRequestOutput(
                    request_id=rid, text=text, tokens=[[1]], scores=[-0.1], finished=True,
                ))

            # Streaming: one partial per pending chunk; final after is_last and
            # all chunks consumed.
            for rid, st in list(self._streams.items()):
                if st.final_emitted:
                    continue
                if st.pending_audio:
                    st.pending_audio.popleft()
                    st.partials_emitted += 1
                    outs.append(FakeRequestOutput(
                        request_id=rid,
                        text=f"partial-{st.partials_emitted}",
                        tokens=[[st.partials_emitted]], scores=[-0.05],
                        finished=False,
                    ))
                elif st.is_last_seen:
                    text = "streaming-echo:" + base64.b64encode(st.audio_seed).decode()
                    st.final_emitted = True
                    outs.append(FakeRequestOutput(
                        request_id=rid, text=text, tokens=[[42]], scores=[-0.02],
                        finished=True,
                    ))
                    del self._streams[rid]
                    self._running = max(0, self._running - 1)
        return outs


def main(argv):
    from oasr.serving.engine_worker import EngineWorker, WorkerOptions

    # Parse via the real WorkerOptions so the CLI surface matches.
    opts = WorkerOptions.from_argv(argv)
    worker = EngineWorker(opts, engine=FakeEngine())
    worker.announce_ready()
    return worker.run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
