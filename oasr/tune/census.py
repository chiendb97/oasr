# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Which problem sizes to tune: a workload census from the deployment itself.

The engine already bounds most of the shape space before a request arrives:
offline encoder graphs are keyed ``(B_bucket, T_bucket)``, streaming ones by
cohort width, and a captured key replays the same shapes for the life of the
process.  So instead of guessing a grid (``--batches 1 8 64 --durations 4 16
64``), the census

1. **enumerates the engine keys** a configuration can produce
   (:func:`offline_keys`, :func:`streaming_keys`);
2. **weights them by traffic** (:class:`TrafficModel`) -- a replay of the
   length-sorted, bucket-filled batching over an utterance-duration sample for
   offline, a concurrency histogram for streaming;
3. **probes each key eagerly** under the shape recorder, with inputs padded
   exactly as the captured path pads them (:class:`ShapeProbe`) -- probing rather
   than deriving, because analytic derivation has twice been blind to real call
   sites (function-body imports, Zipformer's per-stack downsampling);
4. **weights each (signature, M) point** by ``P(key) x calls x estimated time``
   and marks every point a captured key reaches as **must-tune**: those shapes
   run for the life of the process, so they get exact entries, not
   interpolations (:func:`census`).

The result is a :class:`ShapeSet` -- a JSON file ``oasr tune build`` consumes.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("oasr.tune")

__all__ = [
    "EngineKey",
    "TrafficModel",
    "ShapePoint",
    "ShapeSet",
    "offline_keys",
    "streaming_keys",
    "streaming_run_widths",
    "ShapeProbe",
    "census",
    "census_engine_config",
]

#: A recorded GEMM-family call: ``(op, N, K, dtype, batch, M)``.
ShapeKey = Tuple[str, int, int, str, int, int]


@dataclass(frozen=True)
class EngineKey:
    """One shape the engine can run a forward at."""

    mode: str  # "offline" | "streaming"
    batch: int
    #: Offline: padded feature frames (the graph's T bucket).  Streaming: 0.
    frames: int = 0
    #: Whether production runs this key under a CUDA graph.
    captured: bool = False

    def label(self) -> str:
        return f"{self.mode}:B{self.batch}" + (f"xT{self.frames}" if self.frames else "")


# =============================================================================
# Traffic
# =============================================================================


@dataclass
class TrafficModel:
    """What the deployment is asked to do.

    ``durations_s`` is a sample of offline utterance durations (a manifest, or
    telemetry); ``streaming_widths`` a concurrency histogram ``{active streams:
    weight}``; ``stream_seconds`` the typical stream length.  Empty fields fall
    back to a declared, flat profile over what the configuration allows.
    """

    durations_s: List[float] = field(default_factory=list)
    streaming_widths: Dict[int, float] = field(default_factory=dict)
    stream_seconds: float = 8.0
    #: Where the numbers came from, for the ShapeSet's provenance.
    source: str = "declared"

    @classmethod
    def from_manifest(cls, path: str, audio_root: Optional[str] = None, **kw) -> "TrafficModel":
        """Durations from a JSONL manifest (``duration`` field, or the audio header)."""
        durations = []
        root = Path(audio_root) if audio_root else Path(path).parent
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "duration" in rec:
                    durations.append(float(rec["duration"]))
                    continue
                audio = rec.get("audio") or rec.get("audio_filepath") or rec.get("path")
                if audio is None:
                    continue
                d = _audio_seconds(Path(audio) if os.path.isabs(audio) else root / audio)
                if d is not None:
                    durations.append(d)
        return cls(durations_s=durations, source=f"manifest:{path}", **kw)

    def offline_durations(self, cfg) -> List[float]:
        if self.durations_s:
            return list(self.durations_s)
        # Declared: uniform over 1-30 s, or the fixed window.
        return [float(s) for s in range(1, 31)]


def _audio_seconds(path: Path) -> Optional[float]:
    try:
        import soundfile

        info = soundfile.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001
        pass
    try:
        import wave

        with wave.open(str(path)) as w:
            return float(w.getnframes()) / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return None


# =============================================================================
# Engine keys
# =============================================================================


def _feature_frames(seconds: float, fc) -> int:
    """Feature frames for *seconds* of audio (snip-edges geometry)."""
    n = int(round(seconds * int(fc.sample_rate)))
    return max(1, 1 + (n - int(fc.frame_length_samples)) // int(fc.frame_shift_samples))


def offline_keys(cfg, traffic: TrafficModel) -> Dict[EngineKey, float]:
    """Offline keys with their traffic weight (micro-batches per unit of traffic).

    Replays the batching the scheduler does before a batch reaches the forward:
    requests are length-sorted and filled into micro-batches of the configured
    widths (``preferred_batch_size``, else ``max_batch_size``); a micro-batch's
    key is ``(B_bucket, T_bucket)`` exactly as ``GraphedOfflineForward`` pads it.
    An approximation of arrival-time effects, and labelled as one.
    """
    from oasr.engine.offline_graph import resolve_batch_buckets

    fc = cfg.feature_config
    use_graphs = bool(cfg.use_cuda_graphs) and bool(getattr(cfg, "use_offline_cuda_graphs", True))
    buckets = resolve_batch_buckets(cfg)
    g = int(getattr(cfg, "offline_graph_frame_granularity", 64))
    max_frames = int(getattr(cfg, "offline_graph_max_frames", 4096))
    fixed = getattr(fc, "fixed_window_frames", None)
    widths = sorted({int(b) for b in (cfg.preferred_batch_size or [cfg.max_batch_size])})

    frames = sorted(
        (int(fixed) if fixed else _feature_frames(d, fc)) for d in traffic.offline_durations(cfg)
    )
    weights: Dict[EngineKey, float] = defaultdict(float)
    i = 0
    while i < len(frames):
        left = len(frames) - i
        b = max([w for w in widths if w <= left] or [left])
        group = frames[i : i + b]
        i += b
        # A fixed-window frontend pads *and trims* to one width and its encoder
        # discards the lengths, so its key is that width exactly -- rounding
        # 3000 up to 3008 is not waste but a different (rejected) input.
        t = int(fixed) if fixed else -(-max(group) // g) * g
        bb = next((x for x in buckets if x >= b), None)
        captured = use_graphs and bb is not None and t <= max_frames
        key = EngineKey("offline", bb if captured and bb is not None else b, t, captured)
        weights[key] += 1.0
    return dict(weights)


def streaming_run_widths(cfg, rungs: int = 2) -> List[int]:
    """The cohort widths a streaming forward runs at (the padding-lane ladder)."""
    cap = max(1, int(cfg.max_batch_size))
    ladder = getattr(cfg, "streaming_graph_batch_ladder", None)
    if ladder:
        return sorted({int(b) for b in ladder if 1 <= int(b) <= cap} | {cap})
    pad = getattr(cfg, "streaming_graph_pad_batch", None)
    if pad is None:
        pad = rungs > 1
    if not (pad and cfg.use_cuda_graphs):
        return list(range(1, cap + 1))
    widths, b = [], 1
    while b < cap:
        widths.append(b)
        b *= 2
    return widths + [cap]


def streaming_keys(cfg, traffic: TrafficModel, rungs: int = 2) -> Dict[EngineKey, float]:
    """Streaming keys (run widths) weighted by encoder steps per unit of traffic."""
    widths = streaming_run_widths(cfg, rungs)
    hist = traffic.streaming_widths or dict.fromkeys(range(1, int(cfg.max_batch_size) + 1), 1.0)
    captured = bool(cfg.use_cuda_graphs)
    out: Dict[EngineKey, float] = defaultdict(float)
    for active, w in hist.items():
        run = next((x for x in widths if x >= int(active)), widths[-1])
        out[EngineKey("streaming", run, 0, captured)] += float(w)
    return dict(out)


# =============================================================================
# The probe
# =============================================================================


def census_engine_config(cfg):
    """*cfg* with every CUDA graph off, so the shape recorder sees every call.

    A ``feature_config`` the caller did not set goes back to ``None``.
    ``EngineConfig.__post_init__`` fills in a default frontend and marks it
    implicit, but :func:`dataclasses.replace` hands that default to a fresh
    ``__post_init__``, where it counts as explicit and overrides the
    checkpoint's ``FeatureSpec``.  Whisper was then probed with 448 fbank frames
    in place of its 3000-frame window, and Nemotron/Paraformer with 80-dim
    features they cannot consume.
    """
    over = {
        "use_cuda_graphs": False,
        "use_offline_cuda_graphs": False,
        "use_feature_cuda_graphs": False,
        "use_ctc_cuda_graphs": False,
        "use_transducer_cuda_graphs": False,
    }
    if not getattr(cfg, "_feature_config_explicit", True):
        over["feature_config"] = None
    return dataclasses.replace(cfg, **over)


class ShapeProbe:
    """Records the GEMM-family calls one forward at an :class:`EngineKey` makes.

    Offline keys call the model directly with inputs padded to the key -- the
    routing ``ASREngine._prewarm_offline`` mirrors -- so the recorded M is the
    captured path's.  Streaming keys drive the real engine with exactly
    ``batch`` lockstep streams, graphs off.
    """

    def __init__(self, engine) -> None:
        self.engine = engine

    @torch.no_grad()
    def offline(self, key: EngineKey) -> Counter:
        from oasr.engine.request import Request
        from oasr.tune.capture import GemmShapeRecorder, capture_gemm_shapes

        eng = self.engine
        fc = eng._config.feature_config
        sr = int(fc.sample_rate)
        n = int(fc.frame_length_samples) + max(0, key.frames - 1) * int(fc.frame_shift_samples)
        g = torch.Generator().manual_seed(0)
        reqs = [
            Request(audio=0.01 * torch.randn(n, generator=g), streaming=False, sample_rate=sr)
            for _ in range(key.batch)
        ]
        feats, lengths = eng._input_processor.collate(reqs)
        if feats.size(1) < key.frames:
            padded = feats.new_zeros((feats.size(0), key.frames, feats.size(2)))
            padded[:, : feats.size(1)].copy_(feats)
            feats = padded
        consumes = eng._output_processor.strategy.consumes
        rec = GemmShapeRecorder()
        with capture_gemm_shapes(rec):
            if consumes == "hidden":
                eng._model.encode_offline(feats, lengths)
            elif consumes == "both":
                hidden, _ = eng._model.encode_offline(feats, lengths)
                eng._model.head(hidden)
            else:
                eng._model.forward_offline(feats, lengths)
        torch.cuda.synchronize()
        return _counts(rec)

    def streaming(self, width: int, chunks: int = 6) -> Tuple[Counter, int]:
        """``(calls over the run, encoder steps)`` for *width* lockstep streams."""
        from oasr.tune.capture import GemmShapeRecorder, capture_gemm_shapes

        eng = self.engine
        chunk = int(eng._input_processor.streaming_audio_chunk_samples)
        g = torch.Generator().manual_seed(0)
        wav = 0.01 * torch.randn(chunk * chunks, generator=g)
        rec = GemmShapeRecorder()
        steps = 0
        with capture_gemm_shapes(rec):
            rids = [eng.add_streaming_request() for _ in range(width)]
            for j in range(chunks):
                piece = wav[j * chunk : (j + 1) * chunk].contiguous()
                for rid in rids:
                    eng.feed_chunk(rid, piece, is_last=(j == chunks - 1))
            while eng.num_running or eng.num_waiting:
                eng.step()
                steps += 1
                if steps > 10 * chunks + 50:
                    break
        torch.cuda.synchronize()
        return _counts(rec), max(1, chunks)


def _counts(rec) -> Counter:
    out: Counter = Counter()
    for st in rec.aggregate():
        for m, c in st.m_counts.items():
            out[(st.op, st.N, st.K, st.dtype, st.batch, int(m))] += c
    return out


# =============================================================================
# The census
# =============================================================================


@dataclass
class ShapePoint:
    op: str
    N: int
    K: int
    dtype: str
    batch: int
    M: int
    #: Expected calls per unit of traffic.
    calls: float = 0.0
    #: Expected time per unit of traffic (calls x estimated time), for ranking.
    weight: float = 0.0
    #: Reached by a captured key: tuned exactly, never interpolated.
    must: bool = False
    #: Fraction of the calls that run eagerly (the issue-cost weight alpha).
    eager_fraction: float = 0.0
    keys: List[str] = field(default_factory=list)

    @property
    def sig(self) -> Tuple[str, int, int, str, int]:
        return (self.op, self.N, self.K, self.dtype, self.batch)


@dataclass
class ShapeSet:
    points: List[ShapePoint]
    #: Per-forward weight bytes of every distinct signature -- the L2 working set.
    working_set_bytes: int
    provenance: Dict[str, Any] = field(default_factory=dict)

    def signatures(self) -> Dict[Tuple, List[ShapePoint]]:
        out: Dict[Tuple, List[ShapePoint]] = defaultdict(list)
        for p in self.points:
            out[p.sig].append(p)
        for pts in out.values():
            pts.sort(key=lambda p: p.M)
        return dict(out)

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "working_set_bytes": self.working_set_bytes,
            "provenance": self.provenance,
            "points": [dataclasses.asdict(p) for p in self.points],
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "ShapeSet":
        return cls(
            points=[ShapePoint(**p) for p in d["points"]],  # type: ignore[arg-type]
            working_set_bytes=int(d.get("working_set_bytes", 0)),  # type: ignore[arg-type]
            provenance=dict(d.get("provenance", {})),  # type: ignore[arg-type]
        )

    def save(self, path: str) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "ShapeSet":
        with open(path) as f:
            return cls.from_json(json.load(f))


def _estimate_ms(op: str, M: int, N: int, K: int, batch: int, itemsize: int = 2) -> float:
    """A crude roofline, only for *ranking* points before anything is measured.

    ``oasr.tune.cost_model`` replaces it once calibrated; a point's weight is
    re-measured anyway, so the constants only have to order points sensibly.
    """
    flops = 2.0 * M * N * K * batch
    bytes_ = (M * K + N * K + M * N) * itemsize * batch
    return max(flops / 100e12, bytes_ / 1.0e12) * 1e3 + 0.004


def census(
    engine,
    deploy_cfg,
    traffic: TrafficModel,
    *,
    coverage: float = 0.97,
    streaming_chunks: int = 6,
    probe_limit: Optional[int] = None,
) -> ShapeSet:
    """Build the :class:`ShapeSet` for *deploy_cfg* under *traffic*.

    *engine* is an ``ASREngine`` built from :func:`census_engine_config` of the
    deployment's config (graphs off); *deploy_cfg* is the deployment's own
    config, which decides which keys are captured.
    """
    probe = ShapeProbe(engine)
    resolved = engine._config.feature_config
    if resolved is not None and resolved is not deploy_cfg.feature_config:
        # Key derivation turns seconds into frames with the frontend's framing,
        # which is the checkpoint's (resolved by the engine), not the default a
        # bare deployment config carries.
        deploy_cfg = dataclasses.replace(deploy_cfg, feature_config=resolved)
    streaming = deploy_cfg.service_mode == "streaming"
    if streaming:
        rungs = len(
            getattr(engine._model_runner.streaming_backend, "cache_bucket_ladder", ()) or ()
        )
        keys = streaming_keys(deploy_cfg, traffic, max(rungs, 1))
    else:
        keys = offline_keys(deploy_cfg, traffic)
    total = sum(keys.values()) or 1.0
    ordered = sorted(keys.items(), key=lambda kv: -kv[1])
    if probe_limit is not None:
        ordered = ordered[:probe_limit]

    points: Dict[ShapeKey, ShapePoint] = {}
    eager_calls: Dict[ShapeKey, float] = defaultdict(float)
    for key, w in ordered:
        p_key = w / total
        if streaming:
            counts, steps = probe.streaming(key.batch, streaming_chunks)
            per_step = {k: c / steps for k, c in counts.items()}
        else:
            per_step = dict(probe.offline(key))
        for sk, calls in per_step.items():
            op, N, K, dtype, batch, M = sk
            pt = points.get(sk)
            if pt is None:
                pt = points[sk] = ShapePoint(op, N, K, dtype, batch, M)
            c = p_key * calls
            pt.calls += c
            pt.weight += c * _estimate_ms(op, M, N, K, batch)
            pt.must = pt.must or (key.captured and p_key > 0)
            if not key.captured:
                eager_calls[sk] += c
            if len(pt.keys) < 8:
                pt.keys.append(key.label())
        logger.info("[census] %s (p=%.3f): %d shapes", key.label(), p_key, len(per_step))

    for sk, pt in points.items():
        pt.eager_fraction = eager_calls.get(sk, 0.0) / pt.calls if pt.calls else 0.0

    keep = {sk for sk, p in points.items() if p.must}
    keep |= _cover(points, lambda p: p.weight, coverage)
    keep |= _cover(points, lambda p: p.calls, coverage)
    selected = [points[sk] for sk in sorted(keep, key=lambda s: (s[0], s[1], s[2], s[5]))]

    ws = {}
    for p in points.values():
        ws[(p.op, p.N, p.K)] = p.N * p.K * 2 * (p.batch if p.op == "bmm" else 1)
    return ShapeSet(
        points=selected,
        working_set_bytes=int(sum(ws.values())),
        provenance={
            "service_mode": deploy_cfg.service_mode,
            "max_batch_size": int(deploy_cfg.max_batch_size),
            "traffic": traffic.source,
            "keys": {k.label(): round(v / total, 6) for k, v in ordered},
            "coverage": coverage,
            "points_seen": len(points),
        },
    )


def _cover(points: Dict, value, frac: float) -> set:
    items = sorted(points.items(), key=lambda kv: -value(kv[1]))
    total = sum(value(p) for _, p in items) or 1.0
    kept, acc = set(), 0.0
    for sk, p in items:
        kept.add(sk)
        acc += value(p)
        if acc / total >= frac:
            break
    return kept


def from_capture(path: str) -> ShapeSet:
    """A ShapeSet from a ``GemmShapeRecorder`` JSON (``OASR_CAPTURE_GEMM``) -- observed
    traffic, every call eager.  Merge with :func:`census` output for decode paths
    the probes do not reach."""
    from oasr.tune.capture import GemmShapeRecorder

    pts = []
    ws = {}
    for st in GemmShapeRecorder.load_json(path):
        ws[(st.op, st.N, st.K)] = st.N * st.K * 2 * (st.batch if st.op == "bmm" else 1)
        for m, c in st.m_counts.items():
            pts.append(
                ShapePoint(
                    st.op,
                    st.N,
                    st.K,
                    st.dtype,
                    st.batch,
                    int(m),
                    float(c),
                    float(c) * _estimate_ms(st.op, int(m), st.N, st.K, st.batch),
                    False,
                    1.0,
                    ["capture"],
                )
            )
    return ShapeSet(pts, int(sum(ws.values())), {"source": f"capture:{path}"})
