# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Dataset adapters: preparation, integrity, determinism -- all offline."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from asrbench_fixtures import LIBRI_UTTS, make_librispeech_tree

from benchmarks.asrbench.audio import read_info, write
from benchmarks.asrbench.datasets import (
    DatasetUnavailable,
    ManifestItem,
    PreparedDataset,
    get_adapter,
    librispeech as libri_mod,
    load_prepared,
)
from benchmarks.asrbench.datasets.base import safe_extract
from benchmarks.asrbench.datasets.longform import plan
from benchmarks.asrbench.datasets.synthetic import signal

# ---------------------------------------------------------------------------
# LibriSpeech
# ---------------------------------------------------------------------------


def test_librispeech_from_source_dir(tmp_path):
    src = make_librispeech_tree(tmp_path / "src")
    ds = get_adapter("librispeech").prepare(tmp_path / "data", "test-clean", source_dir=src)
    assert ds.card.num_items == len(LIBRI_UTTS)
    assert ds.card.has_transcripts and not ds.card.synthetic
    by_id = {i.id: i for i in ds.items}
    item = by_id["26-495-0001"]
    assert item.text == "THERE WAS NO POSSIBILITY OF TAKING A WALK THAT DAY"
    assert item.speaker == "26" and item.group == "26-495"
    assert item.duration == pytest.approx(12.0, abs=1e-3)
    assert ds.card.total_duration_s == pytest.approx(sum(u[3] for u in LIBRI_UTTS), abs=1e-2)
    assert ds.verify(deep=True) == []
    # The prepared set re-opens to the same content.
    again = load_prepared("librispeech/test-clean", tmp_path / "data")
    assert again.card.manifest_sha256 == ds.card.manifest_sha256


def test_librispeech_never_downloads_without_opt_in(tmp_path):
    with pytest.raises(DatasetUnavailable, match="--download"):
        get_adapter("librispeech").prepare(tmp_path, "test-clean")


def _tarball(src_root: Path, dest: Path) -> Path:
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(src_root / "LibriSpeech", arcname="LibriSpeech")
    return dest


def test_librispeech_download_is_checksum_verified(tmp_path, monkeypatch):
    src = make_librispeech_tree(tmp_path / "src")
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    archive = _tarball(src, mirror / "test-clean.tar.gz")
    good = hashlib.md5(archive.read_bytes()).hexdigest()

    monkeypatch.setitem(libri_mod.MD5, "test-clean", "0" * 32)
    with pytest.raises(DatasetUnavailable, match="mismatch"):
        get_adapter("librispeech").prepare(
            tmp_path / "bad", "test-clean", download=True, options={"mirrors": [mirror.as_uri()]}
        )

    monkeypatch.setitem(libri_mod.MD5, "test-clean", good)
    ds = get_adapter("librispeech").prepare(
        tmp_path / "data", "test-clean", download=True, options={"mirrors": [mirror.as_uri()]}
    )
    assert ds.card.num_items == len(LIBRI_UTTS)
    assert ds.card.revision == f"md5:{good}"
    assert ds.verify() == []
    assert not (
        tmp_path / "data" / "librispeech" / "test-clean" / "_download" / "test-clean.tar.gz"
    ).exists()


def test_safe_extract_refuses_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tar:
        data = b"x"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(DatasetUnavailable, match="escapes"):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


# ---------------------------------------------------------------------------
# LJSpeech
# ---------------------------------------------------------------------------


def _lj_tree(root: Path, metadata: str) -> Path:
    for uid, sec in (("LJ001-0001", 1.5), ("LJ001-0002", 2.0), ("LJ002-0001", 1.0)):
        write(root / "wavs" / f"{uid}.wav", signal(sec, 22050, seed=1), 22050)
    (root / "metadata.csv").write_text(metadata, encoding="utf-8")
    return root


def test_ljspeech_release_format(tmp_path):
    root = _lj_tree(
        tmp_path / "LJSpeech-1.1",
        "LJ001-0001|Printing (in the only sense)|Printing (in the only sense)\n"
        "LJ001-0002|in 1878|in eighteen seventy-eight\n"
        "LJ002-0001|missing audio is fine||\n"
        "LJ009-0009|not on disk|not on disk\n",
    )
    ds = get_adapter("ljspeech").prepare(tmp_path / "data", source_dir=root)
    by_id = {i.id: i for i in ds.items}
    assert set(by_id) == {"LJ001-0001", "LJ001-0002", "LJ002-0001"}
    # Normalised column preferred; brackets unwrapped, words kept.
    assert by_id["LJ001-0002"].text == "in eighteen seventy-eight"
    assert by_id["LJ001-0001"].text == "Printing in the only sense"
    assert by_id["LJ001-0001"].group == "LJ001"
    assert by_id["LJ001-0001"].sample_rate == 22050
    # Indexed in place: absolute paths, no copy.
    assert Path(by_id["LJ001-0001"].audio).is_absolute()


def test_ljspeech_headed_csv_and_env_discovery(tmp_path, monkeypatch):
    root = _lj_tree(
        tmp_path / "lj",
        ",id,sentence,file_name\n0,LJ001-0001,Hello there,LJ001-0001.wav\n"
        '1,LJ001-0002,"Hi, again",LJ001-0002.wav\n',
    )
    monkeypatch.delenv("LJSPEECH_DIR", raising=False)
    monkeypatch.setenv("AUDIO_DIR", str(root / "wavs"))  # what .env points at
    ds = get_adapter("ljspeech").prepare(tmp_path / "data")
    assert {i.id: i.text for i in ds.items} == {
        "LJ001-0001": "Hello there",
        "LJ001-0002": "Hi, again",
    }


# ---------------------------------------------------------------------------
# Local manifest
# ---------------------------------------------------------------------------


def _manifest(path: Path, lines) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), "utf-8")
    return path


@pytest.fixture
def no_audio_env(monkeypatch):
    for var in ("WAV_DIR", "AUDIO_DIR"):  # the sourced .env must not decide the root
        monkeypatch.delenv(var, raising=False)


def test_local_manifest_indexes_in_place(tmp_path, no_audio_env):
    src = tmp_path / "mini" / "segments"
    for uid, sec in (("bai01__01", 1.5), ("bai01__02", 2.0)):
        write(src / f"{uid}.wav", signal(sec, 16000, seed=2), 16000)
    elsewhere = tmp_path / "other" / "x.wav"
    write(elsewhere, signal(0.5, 8000, seed=3), 8000)
    path = _manifest(
        src / "manifest.jsonl",
        [
            {"id": "bai01__01", "audio": "bai01__01.wav", "text": "Cho tam giác A Bê Xê."},
            {"id": "bai01__02", "audio": "bai01__02.wav", "text": "đo góc"},
            {"audio": str(elsewhere)},  # absolute, no id, no text
        ],
    )
    adapter = get_adapter("manifest")
    ds = adapter.prepare(
        tmp_path / "data", source_dir=src, options={"group_sep": "__", "language": "vi"}
    )
    assert ds.card.ref == "manifest/segments"  # manifest.jsonl -> its directory's name
    by_id = {i.id: i for i in ds.items}
    assert list(by_id) == ["bai01__01", "bai01__02", "x"]
    assert by_id["bai01__01"].text == "Cho tam giác A Bê Xê." and by_id["x"].text is None
    assert by_id["bai01__02"].duration == pytest.approx(2.0)
    assert by_id["x"].sample_rate == 8000
    assert by_id["bai01__01"].group == "bai01" and by_id["x"].group == "x"
    assert all(Path(i.audio).is_absolute() for i in ds.items)  # in place, no copy
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert ds.card.revision == f"sha256:{digest[:16]}" and ds.card.language == "vi"
    assert ds.card.source == [str(path.resolve()), str(src.resolve())]
    assert load_prepared("manifest/segments", tmp_path / "data").verify(deep=True) == []
    # Named explicitly, from the file itself.
    named = adapter.prepare(tmp_path / "data", "mini20", source_dir=path)
    assert named.card.ref == "manifest/mini20" and [i.id for i in named.items] == list(by_id)
    # Groups are part of the inputs' identity: without group_sep they are absent.
    assert named.items[0].group is None
    assert named.card.manifest_sha256 != ds.card.manifest_sha256


def test_local_manifest_audio_root_is_all_or_nothing(tmp_path, monkeypatch, no_audio_env):
    audio = tmp_path / "wavs"
    for uid in ("a", "b"):
        write(audio / f"{uid}.wav", signal(1.0, 16000, seed=4), 16000)
    path = _manifest(
        tmp_path / "manifests" / "set.jsonl",
        [{"id": "a", "audio": "a.wav", "text": "x"}, {"id": "b", "audio": "b.wav", "text": "y"}],
    )
    adapter = get_adapter("manifest")
    with pytest.raises(DatasetUnavailable, match="audio_root"):
        adapter.prepare(tmp_path / "data", source_dir=path)
    explicit = adapter.prepare(tmp_path / "data", source_dir=path, options={"audio_root": audio})
    assert explicit.card.ref == "manifest/set" and len(explicit.items) == 2
    monkeypatch.setenv("WAV_DIR", str(audio))  # what validate --audio-root defaults to
    assert len(adapter.prepare(tmp_path / "data", source_dir=path).items) == 2
    # A root holding only some of the files is never taken: that would benchmark a subset.
    (audio / "b.wav").unlink()
    with pytest.raises(DatasetUnavailable, match="1/2 missing"):
        adapter.prepare(tmp_path / "data", source_dir=path)


@pytest.mark.parametrize(
    "lines,kwargs,exc,match",
    [
        ([{"id": "a", "audio": "a.wav"}] * 2, {}, ValueError, "duplicate id"),
        ([{"id": "a", "text": "no audio"}], {}, ValueError, "'audio' key"),
        ([], {}, DatasetUnavailable, "no items"),
        ([{"audio": "/nonexistent/a.wav"}], {}, DatasetUnavailable, "absolute audio"),
        ([{"audio": "a.wav"}], {"options": {"bogus": 1}}, ValueError, "unknown option"),
        ([{"audio": "a.wav"}], {"subset": "a/b"}, ValueError, "plain name"),
    ],
)  # fmt: skip
def test_local_manifest_rejects_bad_input(tmp_path, no_audio_env, lines, kwargs, exc, match):
    write(tmp_path / "a.wav", signal(0.5, 16000, seed=5), 16000)
    path = _manifest(tmp_path / "m.jsonl", lines)
    with pytest.raises(exc, match=match):
        get_adapter("manifest").prepare(tmp_path / "data", source_dir=path, **kwargs)
    with pytest.raises(DatasetUnavailable, match="--source-dir"):
        get_adapter("manifest").prepare(tmp_path / "data")


# ---------------------------------------------------------------------------
# Synthetic / long-form / Earnings-21
# ---------------------------------------------------------------------------


def test_synthetic_is_flagged_and_exact(tmp_path):
    ds = get_adapter("synthetic").prepare(tmp_path, options={"durations": [1.0, 2.5]})
    assert ds.card.synthetic and not ds.card.has_transcripts
    assert [round(i.duration, 3) for i in ds.items] == [1.0, 2.5]
    assert all(i.text is None for i in ds.items)


def _items(spec):
    return [
        ManifestItem(
            id=f"{g}-{k:03d}", audio="x", duration=d, sample_rate=16000, text=f"t{k}", group=g
        )
        for g, durations in spec.items()
        for k, d in enumerate(durations)
    ]


def test_longform_plan_is_deterministic_contiguous_and_uncropped():
    items = _items({"a": [4.0] * 10, "b": [5.0] * 6, "c": [3.0] * 20})
    p1 = plan(items, [12.0, 30.0], per_target=2, seed=3)
    assert p1 == plan(items, [12.0, 30.0], per_target=2, seed=3)
    assert p1 != plan(items, [12.0, 30.0], per_target=2, seed=4)
    ordered = sorted(items, key=lambda i: (i.group, i.id))
    for rec_id, members in p1:
        target = float(rec_id.split("s_")[0][4:])
        total = sum(m.duration for m in members)
        assert total >= target
        assert total - members[-1].duration < target  # stops at the first boundary past it
        idx = [ordered.index(m) for m in members]
        assert idx == list(range(idx[0], idx[0] + len(idx)))  # consecutive utterances


def test_longform_plan_without_cross_group_stays_in_group():
    items = _items({"a": [4.0] * 3, "b": [4.0] * 10})
    for _, members in plan(items, [20.0], per_target=1, seed=0, cross_group=False):
        assert {m.group for m in members} == {"b"}
    with pytest.raises(DatasetUnavailable):
        plan(items, [100.0], per_target=1, cross_group=False)


def test_longform_prepare_concatenates_audio_and_text(data_root, tmp_path):
    ds = get_adapter("longform").prepare(
        data_root,
        "lf-test",
        options={"source": "librispeech/test-clean", "targets": [20.0], "per_target": 1},
    )
    (item,) = ds.items
    frames, sr = read_info(ds.audio_path(item))
    assert frames / sr == pytest.approx(item.duration, abs=1e-3)
    members = item.extra["members"]
    src = {i.id: i for i in load_prepared("librispeech/test-clean", data_root).items}
    assert item.duration == pytest.approx(sum(src[m].duration for m in members), abs=1e-2)
    assert item.text == " ".join(src[m].text for m in members)
    assert ds.card.revision.startswith("librispeech/test-clean@manifest:")


def test_earnings21_from_local_shard(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import soundfile as sf

    from benchmarks.asrbench.datasets.earnings21 import shard_name

    buf = io.BytesIO()
    sf.write(buf, signal(3.0, 8000, seed=0), 8000, format="FLAC")
    table = pa.table(
        {
            "audio": [{"bytes": buf.getvalue(), "path": "4320211.flac"}],
            "audio_length": [3.0],
            "company_name": ["Acme"],
            "text": ["good morning everyone"],
        }
    )
    shards = tmp_path / "shards"
    shards.mkdir()
    pq.write_table(table, shards / shard_name(8))
    ds = get_adapter("earnings21").prepare(
        tmp_path / "data", source_dir=shards, options={"shards": [8]}
    )
    (item,) = ds.items
    assert item.id == "4320211" and item.sample_rate == 16000
    assert item.duration == pytest.approx(3.0, abs=1e-3)  # resampled 8 kHz -> 16 kHz
    assert item.extra["company_name"] == "Acme"
    assert ds.verify(deep=True) == []


def test_earnings21_without_shard_explains(tmp_path):
    with pytest.raises(DatasetUnavailable, match="--download"):
        get_adapter("earnings21").prepare(tmp_path)


# ---------------------------------------------------------------------------
# Integrity / registry
# ---------------------------------------------------------------------------


def test_verify_detects_tampering(tmp_path):
    ds = get_adapter("synthetic").prepare(tmp_path, options={"durations": [1.0, 2.0]})
    manifest = ds.root / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["duration"] = 9.0
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tampered = PreparedDataset.load(ds.root)
    assert any("manifest_sha256" in p for p in tampered.verify())
    assert any("header says" in p for p in tampered.verify(deep=True))
    Path(tampered.audio_path(tampered.items[1])).unlink()
    assert any("missing" in p for p in tampered.verify())


def test_load_prepared_names_the_fix(tmp_path):
    with pytest.raises(DatasetUnavailable, match="prepare librispeech"):
        load_prepared("librispeech/test-other", tmp_path)
    with pytest.raises(ValueError, match="unknown dataset"):
        get_adapter("no-such-corpus")


def test_audio_load_crop_and_resample(tmp_path):
    from benchmarks.asrbench.audio import load

    x = np.arange(22050 * 2, dtype=np.float32) / (22050 * 4)
    write(tmp_path / "a.wav", x, 22050)
    y = load(tmp_path / "a.wav", 16000, offset_s=0.5, duration_s=1.0)
    assert y.dtype == np.float32 and len(y) == 16000
