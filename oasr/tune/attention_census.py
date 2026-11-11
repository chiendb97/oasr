# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Which attention shapes to tune: the ``fmha`` census.

The GEMM census (:mod:`oasr.tune.census`) probes the encoder at the engine's
own keys.  That cannot see attention's heaviest calls in an AR model: a
Whisper decoder step attends a 1500-frame encoder output once per layer per
token, and the shapes of those steps -- how many, over how long a cache --
depend on what was said.  So this census **replays real audio** through the
deployment's engine with every CUDA graph off (a graph replay never re-enters
the Python wrapper that records), and counts every
:func:`oasr.functionals.attention.fmha` call the lanes would be chosen for:

* offline: the manifest transcribed through the engine's own batching;
* streaming: the manifest streamed at a few concurrency levels (the traffic
  model's histogram, else ``1``, a quarter of ``max_batch_size``, and
  ``max_batch_size``), each weighted by its share of the histogram.

A point's weight is its calls per unit of traffic (an utterance, a stream)
times a crude roofline estimate -- for ordering signatures under a build
budget and for reporting a time-weighted headroom, not for any decision.
Without audio, the offline census falls back to the GEMM census's padded
encoder probe (:class:`oasr.tune.census.ShapeProbe`), which cannot see a decode.

Calls that named a lane (``backend=``), fp8 KV reads, fp32 and CPU calls are not
recorded: the measured selection never applies to them.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch

from oasr.jit.attention_select import AttnShape, batch_bucket, q_class, sig_key

logger = logging.getLogger("oasr.tune")

__all__ = [
    "AttnPoint",
    "AttnShapeSet",
    "AttentionShapeRecorder",
    "capture_attention_shapes",
    "attention_census",
    "estimate_ms",
    "is_attention_census",
]

#: The ``family`` field that marks a census file as an attention census.
CENSUS_FAMILY = "fmha"

_HALF = (torch.float16, torch.bfloat16)


@dataclass
class AttnPoint:
    """One recorded call shape and how much of the workload it is."""

    dtype: str
    head_dim: int
    num_heads: int
    num_kv_heads: int
    mask: str
    bias: bool
    block_size: int
    seqstart: bool
    batch: int
    t_q: int
    t_k: int
    bias_aligned: bool = False
    #: Whether the calls carried per-row key lengths (``cache_seqlens``): what
    #: the build's operands reproduce.  Not part of the signature.
    lens: bool = True
    window_left: int = -1
    window_right: int = -1
    #: Calls per unit of traffic (an utterance offline, a stream when streaming).
    calls: float = 0.0
    #: ``calls x estimate_ms``: the point's share of attention time, for ranking.
    weight: float = 0.0
    keys: List[str] = field(default_factory=list)

    @property
    def shape(self) -> AttnShape:
        return AttnShape(
            self.dtype,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            self.mask,
            self.bias,
            self.block_size,
            self.seqstart,
            self.batch,
            self.t_q,
            self.t_k,
            self.bias_aligned,
        )

    @property
    def sig(self) -> str:
        return sig_key(self.shape)


@dataclass
class AttnShapeSet:
    """An attention census: what ``oasr tune build --family fmha`` measures."""

    points: List[AttnPoint]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def signatures(self) -> Dict[str, List[AttnPoint]]:
        """Points by tuning-DB signature key, each list ascending in ``t_k``."""
        out: Dict[str, List[AttnPoint]] = defaultdict(list)
        for p in self.points:
            out[p.sig].append(p)
        for pts in out.values():
            pts.sort(key=lambda p: (p.t_k, p.batch, p.t_q))
        return dict(out)

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "family": CENSUS_FAMILY,
            "provenance": self.provenance,
            "points": [dataclasses.asdict(p) for p in self.points],
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "AttnShapeSet":
        if d.get("family") != CENSUS_FAMILY:
            raise ValueError(f"not an attention census (family {d.get('family')!r})")
        return cls(
            points=[AttnPoint(**p) for p in d["points"]],
            provenance=dict(d.get("provenance", {})),
        )

    def save(self, path: str) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "AttnShapeSet":
        with open(path) as f:
            return cls.from_json(json.load(f))


def is_attention_census(path: str) -> bool:
    """Whether the census file at *path* is an attention census (not a GEMM one)."""
    with open(path) as f:
        return bool(json.load(f).get("family") == CENSUS_FAMILY)


def estimate_ms(shape: AttnShape) -> float:
    """A crude roofline for ranking points before anything is measured."""
    rows = shape.batch * shape.num_heads * shape.t_q
    flops = 4.0 * rows * shape.t_k * shape.head_dim
    kv = 2.0 * shape.batch * shape.num_kv_heads * shape.t_k * shape.head_dim
    bytes_ = 2.0 * (2.0 * rows * shape.head_dim + kv + (rows * shape.t_k if shape.bias else 0))
    return max(flops / 100e12, bytes_ / 1.0e12) * 1e3 + 0.004


# =============================================================================
# The recorder
# =============================================================================


class AttentionShapeRecorder:
    """Thread-safe count of :func:`~oasr.functionals.attention.fmha` call shapes.

    Keyed ``(shape, has lengths, window_left, window_right)``: the selection
    signature plus what the build needs to reproduce the call's operands.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts: Counter = Counter()

    def record(self, shape: AttnShape, lens: bool = True, window: tuple = (-1, -1)) -> None:
        with self._lock:
            self.counts[(shape, bool(lens), int(window[0]), int(window[1]))] += 1

    def take(self) -> Counter:
        """The counts so far, and a fresh start."""
        with self._lock:
            out, self.counts = self.counts, Counter()
        return out


def _shape_of(args, kwargs) -> Optional[tuple]:
    """``(shape, has lengths, window)`` of one ``fmha`` call, as ``fmha`` itself reads it."""
    from oasr.functionals.attention import _paged_table_width

    if kwargs.get("backend") is not None:
        return None  # a named lane is required, never selected
    try:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
    except KeyError:
        return None
    if not q.is_cuda or q.dtype not in _HALF or k.dtype not in _HALF:
        return None
    B, H, T_q, D = (int(x) for x in q.shape)
    bias = kwargs.get("attn_bias")
    block_table = kwargs.get("block_table")
    if block_table is not None:
        bs = int(k.size(1))
        H_kv = int(k.size(2))
        T_k = _paged_table_width(bias, block_table, block_size=bs) * bs
    else:
        bs = 0
        H_kv = int(k.size(1))
        T_k = int(k.size(2))
    window = (int(kwargs.get("window_left", -1)), int(kwargs.get("window_right", -1)))
    windowed = window[0] >= 0 or window[1] >= 0
    mask = "window" if windowed else ("causal" if kwargs.get("causal", False) else "none")
    shape = AttnShape(
        "float16" if q.dtype is torch.float16 else "bfloat16",
        D,
        H,
        H_kv,
        mask,
        bias is not None,
        bs,
        kwargs.get("cache_seqstarts") is not None,
        B,
        T_q,
        T_k,
        bias is not None and bias.size(-1) % 2 == 0,
    )
    return shape, kwargs.get("cache_seqlens") is not None, window


def _make_wrapper(rec: AttentionShapeRecorder, fn):
    def wrapper(*args, **kwargs):
        try:
            seen = _shape_of(args, kwargs)
        except Exception:  # noqa: BLE001 -- recording must never break a forward
            seen = None
        if seen is not None:
            rec.record(*seen)
        return fn(*args, **kwargs)

    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


@contextlib.contextmanager
def capture_attention_shapes(
    recorder: Optional[AttentionShapeRecorder] = None,
) -> Iterator[AttentionShapeRecorder]:
    """Record every ``fmha`` call made inside the block.

    ``fmha`` is reached three ways: the package attribute (``oasr.fmha``), the
    defining module (a function-body ``from oasr.functionals.attention import
    fmha`` resolves it at call time), and :mod:`oasr.layers.attention.core`,
    which binds the name at import.  Each is patched with the same recorder and
    the *original* function, so a call is recorded once.
    """
    import oasr
    import oasr.functionals.attention as _fa
    import oasr.layers.attention.core as _core

    rec = recorder if recorder is not None else AttentionShapeRecorder()
    original = _fa.fmha
    while hasattr(original, "__wrapped__") and getattr(original, "__name__", "") == "wrapper":
        original = original.__wrapped__
    wrapped = _make_wrapper(rec, original)
    saved = []
    try:
        for owner in (oasr, _fa, _core):
            if getattr(owner, "fmha", None) is not None:
                saved.append((owner, owner.fmha))
                owner.fmha = wrapped  # type: ignore[attr-defined]
        yield rec
    finally:
        for owner, fn in reversed(saved):
            owner.fmha = fn  # type: ignore[attr-defined]


# =============================================================================
# The census
# =============================================================================


def _streaming_levels(cfg, traffic) -> Dict[int, float]:
    """Concurrency levels to stream at, with their traffic share."""
    cap = max(1, int(cfg.max_batch_size))
    if traffic is not None and traffic.streaming_widths:
        hist = {min(cap, max(1, int(w))): float(v) for w, v in traffic.streaming_widths.items()}
    else:
        hist = dict.fromkeys(sorted({1, max(1, cap // 4), cap}), 1.0)
    total = sum(hist.values()) or 1.0
    return {w: v / total for w, v in hist.items()}


def _accumulate(points: Dict[tuple, AttnPoint], counts: Counter, scale: float, label: str) -> None:
    for key, n in counts.items():
        shape, lens, wl, wr = key
        pt = points.get(key)
        if pt is None:
            pt = points[key] = AttnPoint(
                **dataclasses.asdict(shape), lens=lens, window_left=wl, window_right=wr
            )
        c = scale * n
        pt.calls += c
        pt.weight += c * estimate_ms(shape)
        if label not in pt.keys and len(pt.keys) < 8:
            pt.keys.append(label)


def attention_census(
    engine,
    deploy_cfg,
    traffic=None,
    *,
    waves: Optional[Sequence[torch.Tensor]] = None,
    batch_size: Optional[int] = None,
    probe_limit: Optional[int] = None,
) -> AttnShapeSet:
    """Build the :class:`AttnShapeSet` of *engine*'s deployment.

    *engine* is an ``ASREngine`` built from
    :func:`oasr.tune.census.census_engine_config` of *deploy_cfg* (graphs off);
    *waves* are real waveforms at the model's rate (a manifest), without which
    the offline census probes the encoder alone.
    """
    from oasr.testing.accuracy import transcribe

    streaming = deploy_cfg.service_mode == "streaming"
    bs = int(batch_size or deploy_cfg.max_batch_size)
    points: Dict[tuple, AttnPoint] = {}
    rec = AttentionShapeRecorder()
    provenance: Dict[str, Any] = {
        "family": CENSUS_FAMILY,
        "service_mode": deploy_cfg.service_mode,
        "max_batch_size": int(deploy_cfg.max_batch_size),
        "traffic": getattr(traffic, "source", "declared"),
    }
    with capture_attention_shapes(rec):
        if waves:
            if streaming:
                levels = _streaming_levels(deploy_cfg, traffic)
                provenance["streaming_levels"] = {str(w): round(p, 6) for w, p in levels.items()}
                for width, share in levels.items():
                    group = list(waves[: max(width, min(len(waves), 2 * width))])
                    transcribe(engine, group, width, streaming=True)
                    torch.cuda.synchronize()
                    _accumulate(points, rec.take(), share / len(group), f"streaming:x{width}")
                    logger.info("[census.fmha] streaming x%d: %d shapes", width, len(points))
            else:
                transcribe(engine, list(waves), bs, streaming=False)
                torch.cuda.synchronize()
                _accumulate(points, rec.take(), 1.0 / len(waves), "offline:manifest")
            provenance["utterances"] = len(waves)
        else:
            from oasr.tune.census import ShapeProbe, offline_keys, streaming_keys

            probe = ShapeProbe(engine)
            if streaming:
                keys = streaming_keys(deploy_cfg, traffic)
            else:
                keys = offline_keys(deploy_cfg, traffic)
            total = sum(keys.values()) or 1.0
            ordered = sorted(keys.items(), key=lambda kv: -kv[1])
            if probe_limit is not None:
                ordered = ordered[:probe_limit]
            for key, w in ordered:
                if streaming:
                    _, steps = probe.streaming(key.batch, 12)
                    scale = (w / total) / max(1, key.batch)
                else:
                    probe.offline(key)
                    scale = w / total
                _accumulate(points, rec.take(), scale, key.label())
            provenance["keys"] = {k.label(): round(v / total, 6) for k, v in ordered}
            provenance["encoder_only"] = True
    selected = sorted(
        points.values(), key=lambda p: (p.sig, p.t_k, p.batch, p.t_q)  # stable, reviewable order
    )
    provenance["signatures"] = len({p.sig for p in selected})
    provenance["q_classes"] = sorted({q_class(p.t_q) for p in selected})
    provenance["batch_buckets"] = sorted({batch_bucket(p.batch) for p in selected})
    return AttnShapeSet(points=selected, provenance=provenance)
