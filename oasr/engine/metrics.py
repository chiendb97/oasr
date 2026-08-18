# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Engine-side metric collection, drained by the serving front-end.

The Prometheus exporter lives in Rust (``oasr-metrics``), but the numbers an
ASR operator actually alerts on — how wide the batches are, how much of one is
padding, how full the paged KV pool is, how long a request queued — are only
knowable here.  This module is the seam: Python accumulates, and the dispatcher
thread drains :meth:`EngineMetrics.snapshot` under the GIL it is already
holding, then replays the result into the exporter.

Three properties make that seam cheap and honest:

**Counters are absolute, never deltas.**  Python keeps monotonic totals and the
front-end replays them with ``Counter::absolute``.  A missed drain therefore
loses nothing and a double drain double-counts nothing, which a delta protocol
cannot promise across a dispatcher restart.

**Histograms hand over raw samples, and say when they could not.**  A bounded
buffer per series (:data:`MAX_SAMPLES_PER_SERIES`) keeps a stalled drain from
growing without limit, and what it drops is counted rather than silently
truncated.

**Stage timings are host time, and the metric name says so.**
``oasr_engine_stage_host_seconds`` is wall time on the calling thread.  CUDA is
asynchronous, so for a GPU stage that is *issue* time plus whatever
synchronisation the stage happens to absorb: the encoder forward returns while
the GPU is still working, and the decode that reads tokens back pays for both.
Read as "where does the step loop's wall clock go" — the right question for a
loop that is interpreter-bound at batch — it is exactly right.  Read as "where
does GPU time go" it is worse than nothing, which is why the name cannot be
mistaken for the latter and why real GPU attribution stays a job for Nsight
Systems over the NVTX ranges already in the tree (``OASR_NVTX=1``).

Set ``OASR_METRICS=0`` to bind every entry point to a no-op collector.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Per-series cap on buffered histogram samples between two drains.  Reached
#: only if the dispatcher stops draining; the overflow is counted into
#: :data:`SAMPLES_DROPPED` so a truncated distribution never passes for a
#: complete one.
MAX_SAMPLES_PER_SERIES = 4096

#: How often the device counters are sampled.  ``torch.cuda.utilization`` costs
#: ~17 us and ``mem_get_info`` ~7 us — trivial once a second, and 2-3% of a core
#: if a 1 kHz tick loop paid it every tick while holding the GIL.
GPU_SAMPLE_INTERVAL_S = 1.0

# --------------------------------------------------------------------------
# Metric names.  These must match the declarations in `oasr-metrics`, which is
# what maps them onto the exporter (and onto their buckets).  A name emitted
# here with no counterpart there is dropped with a warning rather than exported
# bucket-less, because a bucket-less histogram is silently downgraded to a
# rolling summary whose quantiles cannot be aggregated across replicas.
# --------------------------------------------------------------------------

STAGE_HOST_SECONDS = "oasr_engine_stage_host_seconds"
BATCH_SIZE = "oasr_engine_batch_size"
BATCH_PADDING_RATIO = "oasr_engine_batch_padding_ratio"
QUEUE_WAIT_SECONDS = "oasr_engine_queue_wait_seconds"

KV_BLOCKS_USED = "oasr_engine_kv_blocks_used"
KV_BLOCKS_CAPACITY = "oasr_engine_kv_blocks_capacity"
KV_EXHAUSTED = "oasr_engine_kv_exhausted_total"
DECODE_SLOTS_IN_USE = "oasr_engine_decode_slots_in_use"
DECODE_SLOTS_CAPACITY = "oasr_engine_decode_slots_capacity"
TOKENS_GENERATED = "oasr_engine_tokens_generated_total"
AUDIO_SECONDS = "oasr_engine_audio_seconds_total"
SAMPLES_DROPPED = "oasr_engine_metric_samples_dropped_total"

GPU_MEMORY_USED = "oasr_gpu_memory_used_bytes"
GPU_MEMORY_TOTAL = "oasr_gpu_memory_total_bytes"
GPU_UTILIZATION = "oasr_gpu_utilization_ratio"

#: Every value the ``stage`` label may take.
#:
#: Declared, not free-form, for the same reason the NVTX ranges next door are
#: *not* reusable as label values: ``nvtx_push(f"offline.micro_batch[B={n}]")``
#: is fine for a profiler timeline and would be a cardinality bomb in a time
#: series database.  :meth:`EngineMetrics.observe_stage` rejects anything not
#: listed here, so that mistake fails on its first call instead of after a
#: monitoring bill.
STAGES = frozenset(
    {
        "offline.collate",
        "offline.encode",
        "offline.prefill",
        "offline.decode",
        "offline.finalize",
        "offline.advance",
        "streaming.schedule",
        "streaming.allocate",
        "streaming.features",
        "streaming.encode",
        "streaming.decode",
        "streaming.finalize",
    }
)

#: Values of the ``mode`` label.
MODE_OFFLINE = "offline"
MODE_STREAMING = "streaming"


class EngineMetrics:
    """Accumulates engine-scope metrics for one :class:`~oasr.engine.ASREngine`.

    One instance per engine rather than a module global, so a process holding
    an ``EnginePool`` of several engines keeps their series apart — which is
    what the ``engine`` label on the exported metric is for.

    Thread safety: counters and gauges are plain dict writes and histogram
    appends are ``list.append``, both atomic under the GIL.  :meth:`snapshot`
    swaps the sample buffers under a lock so a concurrent append cannot land in
    a list that is being handed away.
    """

    __slots__ = (
        "_counters",
        "_gauges",
        "_hist",
        "_keyed_hist",
        "_lock",
        "_gpu_device",
        "_gpu_ok",
        "_gpu_next_sample",
    )

    def __init__(self, device: Any = None) -> None:
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._hist: Dict[str, List[float]] = {}
        self._keyed_hist: Dict[str, Dict[str, List[float]]] = {}
        self._lock = threading.Lock()
        self._gpu_device = device
        self._gpu_ok = True
        self._gpu_next_sample = 0.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether recording does anything.  ``False`` on :class:`NullMetrics`."""
        return True

    def incr(self, name: str, value: float = 1.0) -> None:
        """Add to a monotonic counter."""
        self._counters[name] = self._counters.get(name, 0.0) + value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge to its current value."""
        self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        """Record one sample into an unlabelled histogram."""
        buf = self._hist.get(name)
        if buf is None:
            buf = self._hist[name] = []
        if len(buf) >= MAX_SAMPLES_PER_SERIES:
            self._counters[SAMPLES_DROPPED] = self._counters.get(SAMPLES_DROPPED, 0.0) + 1.0
            return
        buf.append(value)

    def observe_keyed(self, name: str, key: str, value: float) -> None:
        """Record one sample into a histogram carrying one extra label.

        ``key`` is the label *value*; which label *key* it belongs to is
        declared once, on the Rust side (``oasr_metrics::keyed_label_for``), so
        the two ends cannot disagree about whether a value means a stage or a
        mode.
        """
        by_key = self._keyed_hist.get(name)
        if by_key is None:
            by_key = self._keyed_hist[name] = {}
        buf = by_key.get(key)
        if buf is None:
            buf = by_key[key] = []
        if len(buf) >= MAX_SAMPLES_PER_SERIES:
            self._counters[SAMPLES_DROPPED] = self._counters.get(SAMPLES_DROPPED, 0.0) + 1.0
            return
        buf.append(value)

    # ------------------------------------------------------------------
    # Typed helpers for the call sites
    # ------------------------------------------------------------------

    def observe_stage(self, stage: str, seconds: float) -> None:
        """Record host wall time for one engine stage.

        Deliberately takes an already-computed duration rather than being a
        context manager: the call sites sit inside ``try``/``except`` blocks
        that fail a batch, and a stage that raised has no meaningful duration
        to record.  Skipping it on the error path is the correct behaviour, and
        a bare pair of calls makes that the *default* behaviour rather than
        something a ``__exit__`` has to be talked out of.
        """
        if stage not in STAGES:
            raise ValueError(
                f"undeclared engine stage {stage!r}; add it to oasr.engine.metrics.STAGES. "
                "Stage names are a Prometheus label value, so they must be a fixed, "
                "small set — never interpolated per batch."
            )
        self.observe_keyed(STAGE_HOST_SECONDS, stage, seconds)

    def observe_batch(self, mode: str, rows: int) -> None:
        """Record the width of one executed batch."""
        self.observe_keyed(BATCH_SIZE, mode, float(rows))

    def observe_padding(self, lengths: List[int]) -> None:
        """Record the padding fraction of one padded batch.

        ``lengths`` are **host-side** row lengths.  Deriving this from the
        device ``lengths`` tensor instead would need a ``.sum().item()``, and a
        device-to-host read on the collate path is a synchronisation added to
        the hot loop by a metric — the observation paying for itself in wall
        clock, which is the one thing an observability change must not do.
        """
        n = len(lengths)
        if n == 0:
            return
        longest = max(lengths)
        if longest <= 0:
            return
        padded = longest * n
        self.observe(BATCH_PADDING_RATIO, 1.0 - (sum(lengths) / padded))

    def observe_queue_wait(self, mode: str, requests: Any) -> None:
        """Record how long each request in ``requests`` waited for admission."""
        now = time.monotonic()
        for req in requests:
            self.observe_keyed(QUEUE_WAIT_SECONDS, mode, now - req.arrival_time)

    def observe_gpu(self) -> None:
        """Sample device memory and utilization, at most every second.

        Rate-limited here rather than at the caller so every call site can
        simply ask on each tick.  A device that cannot be queried disables the
        probe permanently instead of paying for the failure once a second.
        """
        if not self._gpu_ok:
            return
        now = time.monotonic()
        if now < self._gpu_next_sample:
            return
        self._gpu_next_sample = now + GPU_SAMPLE_INTERVAL_S
        try:
            import torch

            dev = self._gpu_device
            if dev is None or getattr(dev, "type", None) != "cuda":
                self._gpu_ok = False
                return
            free, total = torch.cuda.mem_get_info(dev)
            self.set_gauge(GPU_MEMORY_USED, float(total - free))
            self.set_gauge(GPU_MEMORY_TOTAL, float(total))
            self.set_gauge(GPU_UTILIZATION, torch.cuda.utilization(dev) / 100.0)
        except Exception as exc:  # noqa: BLE001 — a metric must never fail a tick
            self._gpu_ok = False
            logger.debug("GPU metric sampling disabled: %s: %s", type(exc).__name__, exc)

    # ------------------------------------------------------------------
    # Draining
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return everything accumulated, clearing the histogram buffers.

        Counters and gauges are returned as **absolute** values and are not
        cleared; the front-end replays them with ``Counter::absolute`` /
        ``Gauge::set``, so the protocol is idempotent.
        """
        with self._lock:
            hist = self._hist
            keyed = self._keyed_hist
            self._hist = {}
            self._keyed_hist = {}
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "hist": hist,
                "keyed_hist": keyed,
            }


class NullMetrics(EngineMetrics):
    """No-op collector, selected by ``OASR_METRICS=0``.

    A subclass rather than a duck-typed stand-in so the two cannot drift: a
    method added to :class:`EngineMetrics` without an override here still
    works, it just records into buffers nothing drains.
    """

    __slots__ = ()

    @property
    def enabled(self) -> bool:
        return False

    def incr(self, name: str, value: float = 1.0) -> None:
        pass

    def set_gauge(self, name: str, value: float) -> None:
        pass

    def observe(self, name: str, value: float) -> None:
        pass

    def observe_keyed(self, name: str, key: str, value: float) -> None:
        pass

    def observe_stage(self, stage: str, seconds: float) -> None:
        pass

    def observe_batch(self, mode: str, rows: int) -> None:
        pass

    def observe_padding(self, lengths: List[int]) -> None:
        pass

    def observe_queue_wait(self, mode: str, requests: Any) -> None:
        pass

    def observe_gpu(self) -> None:
        pass

    def snapshot(self) -> Dict[str, Any]:
        return {"counters": {}, "gauges": {}, "hist": {}, "keyed_hist": {}}


def build_metrics(device: Any = None, enabled: Optional[bool] = None) -> EngineMetrics:
    """Build the collector for one engine.

    ``enabled`` defaults to the ``OASR_METRICS`` environment variable (on unless
    set to ``0``), following the same before-process-start convention as the
    other ``OASR_*`` switches.
    """
    if enabled is None:
        enabled = os.environ.get("OASR_METRICS", "1") != "0"
    return EngineMetrics(device) if enabled else NullMetrics(device)


__all__ = [
    "EngineMetrics",
    "NullMetrics",
    "build_metrics",
    "STAGES",
    "MODE_OFFLINE",
    "MODE_STREAMING",
    "MAX_SAMPLES_PER_SERIES",
    "GPU_SAMPLE_INTERVAL_S",
]
