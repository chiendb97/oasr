# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Workloads: a deterministic, recorded selection of inputs from a prepared dataset.

A :class:`WorkloadSpec` (from a profile) plus a prepared dataset resolves to a
:class:`Workload` -- an explicit list of items, each with the exact span of
audio it uses.  The resolved list is written into the run, and its id is a hash
of *what* was selected (dataset manifest hash, item ids, offsets, durations,
order), not *where* the files live -- so two machines running the same profile
against the same corpus get the same workload id, and ``compare`` can refuse to
diff runs that measured different inputs.

Every random choice draws from ``random.Random(f"{seed}:{purpose}")``.  String
seeds are hashed with SHA-512 by CPython, independent of ``PYTHONHASHSEED``, so a
selection is reproducible across processes and versions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmarks.asrbench.datasets import PreparedDataset, load_prepared

INF = math.inf

#: ``(name, lo, hi)`` -- half-open ``[lo, hi)`` seconds.  15 and 30 s are not
#: arbitrary: 30 s is Whisper's fixed window (anything longer is multi-window),
#: and 15 s is roughly where a single CTC forward stops being "a typical
#: request".
DEFAULT_BUCKETS: List[Tuple[str, float, float]] = [
    ("very_short", 0.0, 2.0),
    ("short", 2.0, 5.0),
    ("typical", 5.0, 15.0),
    ("medium", 15.0, 30.0),
    ("long", 30.0, 120.0),
    ("longform", 120.0, INF),
]

SAMPLING = ("random", "stratified", "first", "all")
ORDERS = ("shuffled", "manifest", "duration_asc", "duration_desc")
COMPOSITIONS = ("uniform", "mixed", "as_is")
ARRIVALS = ("poisson", "constant")


def bucket_of(
    duration: float, buckets: Sequence[Tuple[str, float, float]] = DEFAULT_BUCKETS
) -> str:
    for name, lo, hi in buckets:
        if lo <= duration < hi:
            return name
    return "other"


def parse_buckets(spec: Any) -> List[Tuple[str, float, float]]:
    """``None`` / ``"default"`` / ``[[name, lo, hi], ...]`` -> bucket list."""
    if spec in (None, "default"):
        return list(DEFAULT_BUCKETS)
    out = []
    for entry in spec:
        name, lo, hi = entry
        out.append((str(name), float(lo), INF if hi in (None, "inf") else float(hi)))
    return out


def _rng(seed: int, purpose: str) -> random.Random:
    return random.Random(f"{seed}:{purpose}")


# ---------------------------------------------------------------------------
# Spec + resolved workload
# ---------------------------------------------------------------------------


@dataclass
class WorkloadSpec:
    name: str
    dataset: str
    num_items: Optional[int] = None
    duration_range: Tuple[float, float] = (0.0, INF)
    sampling: str = "random"
    buckets: Any = None
    #: Fixed-duration inputs: crop real speech to exactly these lengths.
    crop_s: Optional[List[float]] = None
    order: str = "shuffled"
    seed: int = 0
    #: Only items with a reference transcript (correctness workloads).
    require_text: bool = False
    #: A previously resolved workload file; overrides every selection field.
    manifest: Optional[str] = None

    @classmethod
    def from_dict(cls, name: str, d: Dict[str, Any]) -> "WorkloadSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"workload {name!r}: unknown key(s) {sorted(unknown)}")
        d = dict(d)
        if "duration_range" in d and d["duration_range"] is not None:
            lo, hi = d["duration_range"]
            d["duration_range"] = (float(lo or 0.0), INF if hi in (None, "inf") else float(hi))
        if d.get("crop_s") is not None:
            d["crop_s"] = [float(c) for c in d["crop_s"]]
        spec = cls(name=name, **{k: v for k, v in d.items() if k != "name"})
        if spec.sampling not in SAMPLING:
            raise ValueError(f"workload {name!r}: sampling must be one of {SAMPLING}")
        if spec.order not in ORDERS:
            raise ValueError(f"workload {name!r}: order must be one of {ORDERS}")
        return spec

    def to_json(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        lo, hi = self.duration_range
        d["duration_range"] = [lo, None if hi == INF else hi]
        return d


@dataclass
class WorkItem:
    """One input: a span ``[offset_s, offset_s + duration_s)`` of one audio file."""

    uid: str
    source_id: str
    audio_path: str
    duration_s: float
    offset_s: float = 0.0
    text: Optional[str] = None
    bucket: str = ""

    @property
    def cropped(self) -> bool:
        """A fixed-duration crop (``uid`` carries ``@offset+duration``)."""
        return "@" in self.uid

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Workload:
    spec: WorkloadSpec
    items: List[WorkItem]
    dataset: Dict[str, Any] = field(default_factory=dict)  # the source card
    #: Per bucket: ``{"requested": n, "selected": m, "available": k}``.
    selection: Dict[str, Dict[str, int]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return workload_id(self.dataset.get("manifest_sha256", ""), self.items)

    @property
    def total_audio_s(self) -> float:
        return float(sum(i.duration_s for i in self.items))

    @property
    def synthetic(self) -> bool:
        return bool(self.dataset.get("synthetic"))

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "spec": self.spec.to_json(),
            "dataset": self.dataset,
            "selection": self.selection,
            "warnings": self.warnings,
            "num_items": len(self.items),
            "total_audio_s": round(self.total_audio_s, 3),
            "duration_stats": duration_summary([i.duration_s for i in self.items]),
            "items": [i.to_json() for i in self.items],
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Workload":
        spec = WorkloadSpec.from_dict(d["spec"]["name"], dict(d["spec"]))
        items = [WorkItem(**i) for i in d["items"]]
        return cls(
            spec=spec,
            items=items,
            dataset=d.get("dataset", {}),
            selection=d.get("selection", {}),
            warnings=list(d.get("warnings", [])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1) + "\n", encoding="utf-8")


def workload_id(manifest_sha: str, items: Sequence[WorkItem]) -> str:
    h = hashlib.sha256(manifest_sha.encode())
    for it in items:
        h.update(f"|{it.source_id}@{it.offset_s:.3f}+{it.duration_s:.3f}".encode())
    return h.hexdigest()[:16]


def duration_summary(durations: Sequence[float]) -> Dict[str, float]:
    from benchmarks.asrbench.metrics import percentile

    if not durations:
        return {}
    return {
        "min": round(min(durations), 3),
        "p50": round(percentile(durations, 50), 3),
        "mean": round(sum(durations) / len(durations), 3),
        "max": round(max(durations), 3),
        "total": round(sum(durations), 3),
    }


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve(spec: WorkloadSpec, data_root: Optional[Path] = None) -> Workload:
    """Select the items *spec* describes from its prepared dataset."""
    if spec.manifest:
        loaded = Workload.from_json(json.loads(Path(spec.manifest).read_text(encoding="utf-8")))
        loaded.spec.name = spec.name
        return loaded
    ds = load_prepared(spec.dataset, data_root)
    return select(spec, ds)


def select(spec: WorkloadSpec, ds: PreparedDataset) -> Workload:
    buckets = parse_buckets(spec.buckets)
    lo, hi = spec.duration_range
    pool = [
        it
        for it in sorted(ds.items, key=lambda i: i.id)
        if (spec.crop_s or lo <= it.duration < hi) and (not spec.require_text or it.text)
    ]
    warnings: List[str] = []
    selection: Dict[str, Dict[str, int]] = {}
    chosen: List[WorkItem] = []

    def as_work(m, bucket: str, offset: float = 0.0, dur: Optional[float] = None) -> WorkItem:
        d = m.duration if dur is None else dur
        cropped = dur is not None
        return WorkItem(
            uid=f"{m.id}@{offset:.2f}+{d:g}" if cropped else m.id,
            source_id=m.id,
            audio_path=str(ds.audio_path(m)),
            duration_s=float(d),
            offset_s=float(offset),
            text=None if cropped else m.text,
            bucket=bucket,
        )

    if spec.crop_s:
        per = _split_quota(spec.num_items or len(spec.crop_s), len(spec.crop_s))
        for crop, quota in zip(spec.crop_s, per):
            name = f"fixed_{crop:g}s"
            cands = [m for m in pool if m.duration >= crop]
            rng = _rng(spec.seed, f"crop:{crop}")
            # A long recording can yield several crops at different offsets, so a
            # handful of long-form sources still fills a large fixed-duration set.
            order = _shuffled(cands, spec.seed, f"crop-order:{crop}")
            picks = [order[k % len(order)] for k in range(quota)] if order else []
            selection[name] = {"requested": quota, "selected": len(picks), "available": len(cands)}
            if not picks:
                warnings.append(f"{name}: no source item is >= {crop:g}s")
            seen = set()
            for m in picks:
                for _ in range(100):
                    offset = round(rng.uniform(0.0, m.duration - crop), 2)
                    if (m.id, offset) not in seen:
                        break
                seen.add((m.id, offset))
                chosen.append(as_work(m, name, offset, crop))
    elif spec.sampling == "stratified":
        by_bucket: Dict[str, list] = {}
        for m in pool:
            by_bucket.setdefault(bucket_of(m.duration, buckets), []).append(m)
        names = [b[0] for b in buckets if by_bucket.get(b[0])]
        if not names:
            raise ValueError(f"workload {spec.name!r}: no items in {spec.duration_range}")
        want = spec.num_items or len(pool)
        quotas = dict(zip(names, _split_quota(want, len(names))))
        shuffled = {n: _shuffled(by_bucket[n], spec.seed, f"bucket:{n}") for n in names}
        taken = {n: min(quotas[n], len(shuffled[n])) for n in names}
        # Hand an exhausted bucket's shortfall to buckets with spare items, so the
        # workload still has num_items -- and say so in the selection record.
        short = want - sum(taken.values())
        while short > 0:
            spare = [n for n in names if taken[n] < len(shuffled[n])]
            if not spare:
                break
            for n in spare:
                if short == 0:
                    break
                taken[n] += 1
                short -= 1
        for n in names:
            selection[n] = {
                "requested": quotas[n],
                "selected": taken[n],
                "available": len(shuffled[n]),
            }
            if taken[n] < quotas[n]:
                warnings.append(f"bucket {n}: only {taken[n]}/{quotas[n]} item(s) available")
            chosen.extend(as_work(m, n) for m in shuffled[n][: taken[n]])
        missing = [b[0] for b in buckets if b[0] not in names and _overlaps(b, lo, hi)]
        if missing:
            warnings.append(f"no items in bucket(s) {missing} within {spec.duration_range}")
    else:
        if spec.sampling == "random":
            picks = _shuffled(pool, spec.seed, "random")[: spec.num_items or len(pool)]
        elif spec.sampling == "first":
            picks = pool[: spec.num_items or len(pool)]
        else:  # all
            picks = pool
        if spec.num_items and len(picks) < spec.num_items:
            warnings.append(f"only {len(picks)}/{spec.num_items} item(s) match the filters")
        chosen = [as_work(m, bucket_of(m.duration, buckets)) for m in picks]
        for w in chosen:
            s = selection.setdefault(w.bucket, {"requested": 0, "selected": 0, "available": 0})
            s["selected"] += 1
        for m in pool:
            b = bucket_of(m.duration, buckets)
            if b in selection:
                selection[b]["available"] += 1

    if not chosen:
        raise ValueError(f"workload {spec.name!r}: the selection is empty ({warnings})")
    return Workload(
        spec=spec,
        items=order_items(chosen, spec.order, spec.seed),
        dataset=ds.card.to_json(),
        selection=selection,
        warnings=warnings,
    )


def _overlaps(bucket: Tuple[str, float, float], lo: float, hi: float) -> bool:
    return bucket[1] < hi and bucket[2] > lo


def _split_quota(total: int, parts: int) -> List[int]:
    base, extra = divmod(int(total), int(parts))
    return [base + (1 if k < extra else 0) for k in range(parts)]


def _shuffled(items: Sequence[Any], seed: int, purpose: str) -> List[Any]:
    out = list(items)
    _rng(seed, purpose).shuffle(out)
    return out


def order_items(items: Sequence[WorkItem], order: str, seed: int) -> List[WorkItem]:
    if order == "shuffled":
        return _shuffled(sorted(items, key=lambda i: i.uid), seed, "order")
    if order == "duration_asc":
        return sorted(items, key=lambda i: (i.duration_s, i.uid))
    if order == "duration_desc":
        return sorted(items, key=lambda i: (-i.duration_s, i.uid))
    return list(items)


# ---------------------------------------------------------------------------
# Batches, schedules, audio
# ---------------------------------------------------------------------------


def make_batches(
    items: Sequence[WorkItem],
    batch_size: int,
    composition: str = "mixed",
    num_batches: Optional[int] = None,
    seed: int = 0,
) -> List[List[WorkItem]]:
    """Full batches of exactly *batch_size* items.

    ``uniform`` groups items of similar duration (minimal padding), ``mixed``
    draws a seeded shuffle (realistic mixtures), ``as_is`` keeps workload order.
    A workload smaller than ``batch_size * num_batches`` is cycled to fill --
    a partial trailing batch would silently be a different batch size.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if composition not in COMPOSITIONS:
        raise ValueError(f"composition must be one of {COMPOSITIONS}")
    n = num_batches or max(1, math.ceil(len(items) / batch_size))
    need = n * batch_size
    pool = [items[k % len(items)] for k in range(need)]
    if composition == "uniform":
        pool = sorted(pool, key=lambda i: (i.duration_s, i.uid))
    elif composition == "mixed":
        pool = _shuffled(pool, seed, f"batches:{batch_size}")
    batches = [pool[k : k + batch_size] for k in range(0, need, batch_size)]
    if composition == "uniform":
        batches = _shuffled(batches, seed, f"batch-order:{batch_size}")
    return batches


def padding_ratio(batches: Sequence[Sequence[WorkItem]]) -> float:
    """Padded audio over real audio, minus one -- what a padded batch wastes."""
    real = sum(i.duration_s for b in batches for i in b)
    padded = sum(len(b) * max(i.duration_s for i in b) for b in batches if b)
    return padded / real - 1.0 if real > 0 else 0.0


def arrival_offsets(
    rate: float,
    *,
    process: str = "poisson",
    count: Optional[int] = None,
    duration_s: Optional[float] = None,
    seed: int = 0,
) -> List[float]:
    """Scheduled send offsets (seconds from load start) of an open-loop arrival process.

    Poisson inter-arrivals are the standard model of independent clients (and
    MLPerf's Server scenario); constant spacing isolates the server from
    burstiness.  Bounded by *count* or *duration_s*, whichever comes first.
    """
    if rate <= 0:
        raise ValueError("rate must be > 0")
    if process not in ARRIVALS:
        raise ValueError(f"arrival process must be one of {ARRIVALS}")
    if count is None and duration_s is None:
        raise ValueError("bound the schedule by count or duration_s")
    rng = _rng(seed, f"arrivals:{process}:{rate}")
    out: List[float] = []
    t = 0.0
    while True:
        if count is not None and len(out) >= count:
            break
        if duration_s is not None and t >= duration_s:
            break
        out.append(t)
        # Constant spacing is i / rate, never an accumulated sum: ten additions
        # of 0.1 land at 0.9999..., which admits an 11th arrival into a 1 s window.
        t = t + rng.expovariate(rate) if process == "poisson" else len(out) / rate
    return out


def load_audio(items: Sequence[WorkItem], sample_rate: int) -> Dict[str, np.ndarray]:
    """Decode every distinct item once, before any measurement."""
    from benchmarks.asrbench.audio import load

    out: Dict[str, np.ndarray] = {}
    for it in items:
        if it.uid not in out:
            out[it.uid] = load(
                it.audio_path,
                sample_rate,
                offset_s=it.offset_s,
                duration_s=it.duration_s if it.cropped else None,
            )
    return out
