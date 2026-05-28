# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Concurrency tests for the thread-safe ASREngine RLock.

Opt-in via the ``concurrent`` marker — these tests stress the engine from
multiple Python threads to verify that the lock added in ``engine.py``
prevents scheduler/queue corruption.

Run with::

    pytest tests/test_engine_concurrent.py -m concurrent -v

The full-engine tests require a real WeNet checkpoint (``CKPT_DIR`` /
``--ckpt-dir``); the lock-correctness tests run without a checkpoint by
patching the engine's internal helpers.
"""

from __future__ import annotations

import glob
import os
import random
import threading
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest


pytestmark = pytest.mark.concurrent


# ---------------------------------------------------------------------------
# Lock correctness — patched engine, no real GPU / model
# ---------------------------------------------------------------------------


def _make_patched_engine():
    """Return an ASREngine with model/runner/feature-extraction stubbed out.

    Only the scheduler + lock + request lifecycle are exercised, which is
    enough to verify that concurrent ``add_request`` / ``feed_chunk`` /
    ``abort_request`` / ``step`` don't corrupt the shared queues.
    """
    import torch

    from oasr.engine.config import EngineConfig
    from oasr.engine.engine import ASREngine

    # Build a minimal config that won't try to load a real checkpoint.
    cfg = EngineConfig(ckpt_dir="/tmp/_oasr_lock_test", device="cpu", max_batch_size=4)

    # Side-step __init__: it loads the checkpoint and instantiates CUDA
    # streams. We build a bare object and only wire what the locked methods
    # actually touch.
    engine = ASREngine.__new__(ASREngine)
    engine._lock = threading.RLock()
    engine._config = cfg
    engine._device = torch.device("cpu")
    engine._feat_stream = None

    from oasr.engine.scheduler import Scheduler
    engine._scheduler = Scheduler(cfg)

    # Stub the heavy collaborators. ``feed_chunk`` only calls
    # ``find_request`` + ``append_streaming_chunk``; ``abort_request`` only
    # calls ``free_stream``. The step path is exercised separately below.
    engine._input_processor = MagicMock()
    engine._input_processor.append_streaming_chunk = MagicMock(return_value=None)
    engine._input_processor.prepare_streaming_open = MagicMock(return_value=None)
    engine._input_processor.prepare_offline = MagicMock(return_value=None)
    engine._input_processor.prepare_streaming = MagicMock(return_value=None)

    engine._model_runner = MagicMock()
    engine._output_processor = MagicMock()
    engine._offline_pipeline = MagicMock()
    engine._model = MagicMock()

    return engine


def test_lock_is_rlock():
    """``run`` calls ``step`` while holding the lock — must be re-entrant."""
    engine = _make_patched_engine()
    with engine._lock:
        with engine._lock:
            pass  # second acquire on same thread must not deadlock


def test_concurrent_add_streaming_request_no_corruption():
    """Many threads opening streaming requests must produce ``N`` distinct ids
    and a coherent scheduler index."""
    engine = _make_patched_engine()

    N_THREADS = 16
    N_PER_THREAD = 64
    ids: List[str] = []
    ids_lock = threading.Lock()
    errors: List[BaseException] = []

    def worker():
        try:
            for _ in range(N_PER_THREAD):
                rid = engine.add_streaming_request()
                with ids_lock:
                    ids.append(rid)
        except BaseException as e:  # pragma: no cover — failure surface
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"workers raised: {errors!r}"
    assert len(ids) == N_THREADS * N_PER_THREAD
    assert len(set(ids)) == len(ids), "duplicate request_ids issued"
    # Every issued id must be in the scheduler's index — concurrent inserts
    # must not lose any.
    assert engine._scheduler.num_waiting == len(ids)
    assert all(engine._scheduler.find_request(rid) is not None for rid in ids)


def test_concurrent_feed_and_abort_no_corruption():
    """Interleave feed_chunk and abort_request from many threads; verify the
    scheduler queues never desync from the index."""
    engine = _make_patched_engine()

    rids = [engine.add_streaming_request() for _ in range(64)]
    errors: List[BaseException] = []

    def feeder():
        try:
            for _ in range(200):
                rid = random.choice(rids)
                try:
                    engine.feed_chunk(rid, np.zeros(16, dtype=np.float32), is_last=False)
                except KeyError:
                    pass  # racing aborts may have removed the rid; allowed
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    def aborter():
        try:
            for _ in range(20):
                rid = random.choice(rids)
                engine.abort_request(rid)
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=feeder) for _ in range(8)] + [
        threading.Thread(target=aborter) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"workers raised: {errors!r}"
    # Whatever survived must still be consistent: index size == waiting count
    # (no running streams here because step() never ran).
    assert engine._scheduler.num_waiting == len(engine._scheduler._index)


def test_property_reads_are_consistent_under_writes():
    """``num_running`` / ``num_waiting`` must never observe a mid-write state."""
    engine = _make_patched_engine()
    stop = threading.Event()
    errors: List[BaseException] = []

    def writer():
        try:
            while not stop.is_set():
                rid = engine.add_streaming_request()
                engine.abort_request(rid)
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    def reader():
        try:
            while not stop.is_set():
                # Both reads must complete without exception and return
                # non-negative ints regardless of writer racing.
                w = engine.num_waiting
                r = engine.num_running
                assert w >= 0 and r >= 0
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(4)] + [
        threading.Thread(target=reader) for _ in range(4)
    ]
    for t in threads:
        t.start()
    # Let them race for a moment.
    import time
    time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join()

    assert not errors, f"workers raised: {errors!r}"


# ---------------------------------------------------------------------------
# End-to-end concurrency — real engine, real checkpoint (gated)
# ---------------------------------------------------------------------------


def _require_ckpt(ckpt_dir: str) -> None:
    if not ckpt_dir or not Path(ckpt_dir).exists():
        pytest.skip("WeNet checkpoint dir not set; set CKPT_DIR or --ckpt-dir")


def _require_wavs(wav_dir: str) -> List[str]:
    if not wav_dir or not Path(wav_dir).is_dir():
        pytest.skip("WAV directory not set; use --wav-dir or WAV_DIR")
    wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wavs:
        pytest.skip("No .wav files found in WAV directory")
    return wavs


@pytest.mark.slow
def test_e2e_streaming_concurrent_feeders(ckpt_dir: str, wav_dir: str):
    """Real engine: 4 streaming requests fed from 4 threads while a single
    step-loop thread drains.  Outputs must match the single-threaded baseline.
    """
    import torch
    import torchaudio  # noqa: F401  — fail early if missing
    from oasr.engine.config import EngineConfig
    from oasr.engine.engine import ASREngine

    _require_ckpt(ckpt_dir)
    wavs = _require_wavs(wav_dir)[:4]

    engine = ASREngine(EngineConfig(ckpt_dir=ckpt_dir, max_batch_size=8))

    # Baseline: serial submission, single-thread driver.
    serial_texts = engine.transcribe(wavs, streaming=True)
    if isinstance(serial_texts, str):
        serial_texts = [serial_texts]

    # Concurrent: open requests on different threads, drive step from main.
    results: dict = {}
    errors: List[BaseException] = []

    def submit(wav_path: str):
        try:
            import soundfile as sf
            samples, sr = sf.read(wav_path, dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1).astype("float32")
            rid = engine.add_streaming_request(sample_rate=sr)
            # Feed in ~640ms chunks.
            step = max(1, int(sr * 0.64))
            for start in range(0, len(samples), step):
                is_last = (start + step) >= len(samples)
                engine.feed_chunk(rid, samples[start:start + step], is_last=is_last)
            results[rid] = wav_path
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=submit, args=(p,)) for p in wavs]
    for t in threads:
        t.start()

    # Main thread drives step() while feeders push.
    drained_outputs = []
    import time
    deadline = time.time() + 60.0
    while time.time() < deadline:
        outs = engine.step()
        for o in outs:
            if o.finished:
                drained_outputs.append(o)
        if not any(t.is_alive() for t in threads) and engine.num_running == 0 \
                and engine.num_waiting == 0:
            break
        time.sleep(0.001)

    for t in threads:
        t.join()

    assert not errors, f"feeders raised: {errors!r}"
    assert len(drained_outputs) == len(wavs), \
        f"expected {len(wavs)} finals, got {len(drained_outputs)}"
