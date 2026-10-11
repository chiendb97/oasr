# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Workload selection, batches and arrival schedules: deterministic and recorded."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.asrbench.datasets import DatasetCard, ManifestItem, PreparedDataset
from benchmarks.asrbench.workloads import (
    DEFAULT_BUCKETS,
    WorkItem,
    Workload,
    WorkloadSpec,
    arrival_offsets,
    bucket_of,
    load_audio,
    make_batches,
    order_items,
    padding_ratio,
    resolve,
    select,
)


def _dataset(durations, root="/data/a", with_text=True) -> PreparedDataset:
    items = [
        ManifestItem(
            id=f"u{k:03d}",
            audio=f"u{k:03d}.flac",
            duration=d,
            sample_rate=16000,
            text=f"text {k}" if with_text and k % 5 else None,
            group=f"g{k // 4}",
        )
        for k, d in enumerate(durations)
    ]
    card = DatasetCard(name="fake", subset="s", description="", license="", manifest_sha256="abc")
    return PreparedDataset(root=Path(root), card=card, items=items)


#: 10 items per bucket, except medium (2) and long (0).
DURATIONS = (
    [1.0 + 0.05 * k for k in range(10)]
    + [3.0 + 0.1 * k for k in range(10)]
    + [8.0 + 0.3 * k for k in range(10)]
    + [20.0, 25.0]
    + [200.0]
)


@pytest.mark.parametrize(
    "d,name",
    [(0.0, "very_short"), (1.999, "very_short"), (2.0, "short"), (14.99, "typical"),
     (15.0, "medium"), (30.0, "long"), (119.9, "long"), (120.0, "longform"), (1e6, "longform")],
)  # fmt: skip
def test_bucket_edges_are_half_open(d, name):
    assert bucket_of(d) == name
    assert [b[0] for b in DEFAULT_BUCKETS][-1] == "longform"


def test_stratified_is_deterministic_balanced_and_records_shortfall():
    ds = _dataset(DURATIONS)
    spec = WorkloadSpec(
        name="w",
        dataset="fake/s",
        num_items=16,
        sampling="stratified",
        seed=7,
        duration_range=(0, 30),
    )
    a, b = select(spec, ds), select(spec, ds)
    assert [i.uid for i in a.items] == [i.uid for i in b.items]
    assert len(a.items) == 16
    # 4 buckets in range; medium has only 2 -> its shortfall goes to the others.
    assert a.selection["medium"] == {"requested": 4, "selected": 2, "available": 2}
    assert sum(s["selected"] for s in a.selection.values()) == 16
    assert any("medium" in w for w in a.warnings)
    other = select(WorkloadSpec(**{**spec.__dict__, "seed": 8}), ds)
    assert {i.uid for i in other.items} != {i.uid for i in a.items}


def test_random_first_all_and_filters():
    ds = _dataset(DURATIONS)
    first = select(
        WorkloadSpec(name="w", dataset="fake/s", num_items=3, sampling="first", order="manifest"),
        ds,
    )
    assert [i.uid for i in first.items] == ["u000", "u001", "u002"]
    rnd = select(
        WorkloadSpec(
            name="w", dataset="fake/s", num_items=5, sampling="random", duration_range=(2.0, 15.0)
        ),
        ds,
    )
    assert all(2.0 <= i.duration_s < 15.0 for i in rnd.items) and len(rnd.items) == 5
    everything = select(
        WorkloadSpec(name="w", dataset="fake/s", sampling="all", require_text=True), ds
    )
    assert all(i.text for i in everything.items)
    with pytest.raises(ValueError, match="empty"):
        select(
            WorkloadSpec(name="w", dataset="fake/s", sampling="all", duration_range=(500, 600)), ds
        )


def test_fixed_duration_crops_are_exact_unique_and_reuse_long_sources():
    ds = _dataset(DURATIONS)
    spec = WorkloadSpec(name="w", dataset="fake/s", crop_s=[2.0, 100.0], num_items=6, seed=1)
    w = select(spec, ds)
    by_bucket = {}
    for it in w.items:
        by_bucket.setdefault(it.bucket, []).append(it)
        assert it.cropped and it.text is None
        src = next(m for m in ds.items if m.id == it.source_id)
        assert 0.0 <= it.offset_s <= src.duration - it.duration_s + 1e-9
    assert {b: len(v) for b, v in by_bucket.items()} == {"fixed_2s": 3, "fixed_100s": 3}
    # Only one source is >= 100 s: three crops of it, at different offsets.
    assert len({i.source_id for i in by_bucket["fixed_100s"]}) == 1
    assert len({i.uid for i in w.items}) == len(w.items)
    assert all(i.duration_s == 100.0 for i in by_bucket["fixed_100s"])


def test_workload_id_identifies_inputs_not_paths():
    spec = WorkloadSpec(name="w", dataset="fake/s", num_items=5, sampling="random", seed=3)
    a = select(spec, _dataset(DURATIONS, root="/mnt/x"))
    b = select(spec, _dataset(DURATIONS, root="/home/y"))
    assert a.id == b.id
    c = select(WorkloadSpec(**{**spec.__dict__, "seed": 4}), _dataset(DURATIONS))
    assert c.id != a.id


def test_order():
    items = [
        WorkItem(uid=f"i{k}", source_id=f"i{k}", audio_path="", duration_s=d)
        for k, d in enumerate([5, 1, 3])
    ]
    assert [i.duration_s for i in order_items(items, "duration_asc", 0)] == [1, 3, 5]
    assert [i.duration_s for i in order_items(items, "duration_desc", 0)] == [5, 3, 1]
    assert order_items(items, "shuffled", 9) == order_items(list(reversed(items)), "shuffled", 9)


def test_spec_validation_and_json_round_trip(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        WorkloadSpec.from_dict("w", {"dataset": "x", "bogus": 1})
    with pytest.raises(ValueError, match="sampling"):
        WorkloadSpec.from_dict("w", {"dataset": "x", "sampling": "magic"})
    spec = WorkloadSpec.from_dict(
        "w", {"dataset": "fake/s", "duration_range": [1, None], "num_items": 4}
    )
    assert spec.duration_range == (1.0, math.inf)
    w = select(spec, _dataset(DURATIONS))
    w.save(tmp_path / "w.json")
    back = Workload.from_json(json.loads((tmp_path / "w.json").read_text()))
    assert back.id == w.id and [i.uid for i in back.items] == [i.uid for i in w.items]
    # A saved workload pins the inputs: resolving it again ignores the selection fields.
    pinned = resolve(
        WorkloadSpec(name="again", dataset="ignored", manifest=str(tmp_path / "w.json"))
    )
    assert pinned.id == w.id and pinned.spec.name == "again"


def _items(durations):
    return [
        WorkItem(uid=f"i{k}", source_id=f"i{k}", audio_path="", duration_s=d)
        for k, d in enumerate(durations)
    ]


def test_batches_are_full_and_cycled():
    items = _items([1, 2, 3, 4, 5])
    batches = make_batches(items, 4, "mixed", num_batches=3, seed=0)
    assert [len(b) for b in batches] == [4, 4, 4]
    assert sorted(i.uid for b in batches for i in b).count("i0") >= 2  # cycled to fill
    with pytest.raises(ValueError):
        make_batches(items, 0)


def test_uniform_batches_pad_less_than_mixed():
    items = _items([1.0, 1.1, 1.2, 1.3, 9.0, 9.1, 9.2, 9.3] * 4)
    uniform = make_batches(items, 4, "uniform", seed=0)
    mixed = make_batches(items, 4, "mixed", seed=0)
    assert padding_ratio(uniform) < 0.1 < padding_ratio(mixed)
    assert padding_ratio([[*_items([2.0, 1.0])]]) == pytest.approx(4.0 / 3.0 - 1.0)


def test_arrival_schedules():
    const = arrival_offsets(10.0, process="constant", count=5)
    assert const == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    by_time = arrival_offsets(10.0, process="constant", duration_s=1.0)
    assert len(by_time) == 10 and by_time[-1] < 1.0
    p = arrival_offsets(50.0, process="poisson", count=5000, seed=1)
    assert p == arrival_offsets(50.0, process="poisson", count=5000, seed=1)
    assert p != arrival_offsets(50.0, process="poisson", count=5000, seed=2)
    rate = (len(p) - 1) / p[-1]
    assert rate == pytest.approx(50.0, rel=0.05)
    gaps = [b - a for a, b in zip(p, p[1:])]
    cv = (sum((g - 1 / 50) ** 2 for g in gaps) / len(gaps)) ** 0.5 * 50
    assert cv == pytest.approx(1.0, abs=0.08)  # exponential gaps: CV = 1
    with pytest.raises(ValueError):
        arrival_offsets(0.0, count=1)
    with pytest.raises(ValueError):
        arrival_offsets(1.0)


def test_load_audio_honours_crops(data_root):
    spec = WorkloadSpec(name="w", dataset="synthetic/default", crop_s=[1.25], num_items=2, seed=0)
    w = resolve(spec, data_root)
    audio = load_audio(w.items, 16000)
    assert all(len(audio[i.uid]) == 20000 for i in w.items)
