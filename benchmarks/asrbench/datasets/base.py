# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The dataset contract: an adapter prepares a corpus into a manifest + a card.

A prepared dataset is a directory::

    <data_root>/<name>/<subset>/
        dataset.json      the card: source, revision, license, checksums, stats
        manifest.jsonl    one line per item: {id, audio, duration, sample_rate, text, ...}
        audio/            materialised audio, when the adapter had to write any

The manifest is a superset of the ``{id, audio, text}`` format that
``oasr.testing.accuracy.load_manifest`` and the accuracy harness read, so a
prepared LibriSpeech or LJSpeech manifest also drives the existing WER gate.
``duration`` is stored, not recomputed: a workload sampler selecting by duration
must not have to open thousands of files, and recording it is what lets
``verify`` notice a corpus that changed underneath a recorded run.

``audio`` is relative to the dataset directory unless absolute (an adapter
that indexes a corpus in place, like LJSpeech, points at it rather than
copying gigabytes).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from benchmarks.asrbench import SUITE_VERSION
from benchmarks.asrbench.metrics import percentile

MANIFEST = "manifest.jsonl"
CARD = "dataset.json"

#: Where prepared datasets live unless ``--data-dir`` says otherwise.
DATA_DIR_ENV = "ASRBENCH_DATA_DIR"


def default_data_root() -> Path:
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "oasr" / "asrbench"


class DatasetUnavailable(RuntimeError):
    """The corpus is not on disk and was not (or could not be) fetched.

    Raised with acquisition instructions rather than a traceback: a missing
    corpus is an expected state, not a bug.
    """


# ---------------------------------------------------------------------------
# Manifest + card
# ---------------------------------------------------------------------------


@dataclass
class ManifestItem:
    """One utterance or recording."""

    id: str
    audio: str
    duration: float
    sample_rate: int
    text: Optional[str] = None
    speaker: Optional[str] = None
    #: Recording-level grouping (a LibriSpeech chapter, an LJSpeech book).
    #: Consecutive items of one group are contiguous speech, which is what the
    #: long-form builder concatenates.
    group: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "audio": self.audio,
            "duration": round(float(self.duration), 6),
            "sample_rate": int(self.sample_rate),
            "text": self.text,
        }
        if self.speaker is not None:
            d["speaker"] = self.speaker
        if self.group is not None:
            d["group"] = self.group
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "ManifestItem":
        return cls(
            id=str(d["id"]),
            audio=str(d["audio"]),
            duration=float(d["duration"]),
            sample_rate=int(d.get("sample_rate", 16000)),
            text=d.get("text"),
            speaker=d.get("speaker"),
            group=d.get("group"),
            extra=dict(d.get("extra") or {}),
        )


@dataclass
class DatasetCard:
    """Provenance and shape of one prepared dataset (``dataset.json``)."""

    name: str
    subset: str
    description: str
    license: str
    language: str = "en"
    version: str = ""
    #: What pins the bytes: an archive checksum, a dataset-repo commit, or the
    #: source card's manifest hash for a derived dataset.
    revision: str = ""
    source: List[str] = field(default_factory=list)
    has_transcripts: bool = True
    synthetic: bool = False
    #: Why this dataset is in the suite -- which workloads it serves.
    purpose: str = ""
    prepared_utc: str = ""
    suite_version: str = SUITE_VERSION
    num_items: int = 0
    total_duration_s: float = 0.0
    duration_stats: Dict[str, float] = field(default_factory=dict)
    sample_rate: int = 16000
    manifest_sha256: str = ""
    checksums: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def ref(self) -> str:
        return f"{self.name}/{self.subset}"

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "DatasetCard":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def duration_stats(durations: Sequence[float]) -> Dict[str, float]:
    if not durations:
        return {}
    return {
        "min": min(durations),
        "p10": percentile(durations, 10),
        "p50": percentile(durations, 50),
        "p90": percentile(durations, 90),
        "max": max(durations),
        "mean": sum(durations) / len(durations),
    }


def manifest_sha256(items: Iterable[ManifestItem]) -> str:
    """Hash of the canonical manifest text -- identifies the item set exactly."""
    h = hashlib.sha256()
    for item in items:
        h.update(json.dumps(item.to_json(), sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class PreparedDataset:
    root: Path
    card: DatasetCard
    items: List[ManifestItem]

    def audio_path(self, item: ManifestItem) -> Path:
        p = Path(item.audio)
        return p if p.is_absolute() else self.root / p

    @classmethod
    def load(cls, root: Path) -> "PreparedDataset":
        root = Path(root)
        card_path = root / CARD
        if not card_path.is_file():
            raise DatasetUnavailable(f"{root} is not a prepared dataset (no {CARD})")
        card = DatasetCard.from_json(json.loads(card_path.read_text(encoding="utf-8")))
        items = read_manifest(root / MANIFEST)
        return cls(root=root, card=card, items=items)

    def verify(self, deep: bool = False) -> List[str]:
        """Integrity problems, empty when the dataset is intact.

        Shallow: the manifest hash matches the card and every audio file exists.
        Deep: additionally every file's header duration matches the manifest.
        """
        problems: List[str] = []
        if manifest_sha256(self.items) != self.card.manifest_sha256:
            problems.append("manifest.jsonl does not match the card's manifest_sha256")
        missing = [i.id for i in self.items if not self.audio_path(i).is_file()]
        if missing:
            problems.append(f"{len(missing)} audio file(s) missing, e.g. {missing[0]}")
        if deep and not missing:
            from benchmarks.asrbench.audio import read_info

            for item in self.items:
                frames, sr = read_info(self.audio_path(item))
                if abs(frames / sr - item.duration) > 0.01:
                    problems.append(
                        f"{item.id}: header says {frames / sr:.3f}s, manifest {item.duration:.3f}s"
                    )
        return problems


def read_manifest(path: Path) -> List[ManifestItem]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(ManifestItem.from_json(json.loads(line)))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: bad manifest line ({exc})") from None
    return items


def write_prepared(root: Path, card: DatasetCard, items: List[ManifestItem]) -> PreparedDataset:
    """Write the manifest and the card (stats and hash filled in here)."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / MANIFEST, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_json(), ensure_ascii=False) + "\n")
    durations = [i.duration for i in items]
    card.num_items = len(items)
    card.total_duration_s = round(sum(durations), 3)
    card.duration_stats = {k: round(v, 3) for k, v in duration_stats(durations).items()}
    card.manifest_sha256 = manifest_sha256(items)
    card.prepared_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    card.suite_version = SUITE_VERSION
    (root / CARD).write_text(json.dumps(card.to_json(), indent=2) + "\n", encoding="utf-8")
    return PreparedDataset(root=root, card=card, items=items)


# ---------------------------------------------------------------------------
# Adapter contract
# ---------------------------------------------------------------------------


class DatasetAdapter(ABC):
    """Turns one corpus into a :class:`PreparedDataset`."""

    name: str = ""
    description: str = ""
    license: str = ""
    #: Which workloads this dataset exists to serve.
    purpose: str = ""
    subsets: Sequence[str] = ()
    default_subset: str = ""
    has_transcripts: bool = True
    #: Human-readable size of what ``--download`` fetches.
    download_size: str = ""

    def dataset_dir(self, data_root: Path, subset: str) -> Path:
        return Path(data_root) / self.name / subset

    def is_prepared(self, data_root: Path, subset: Optional[str] = None) -> bool:
        return (self.dataset_dir(data_root, subset or self.default_subset) / CARD).is_file()

    @abstractmethod
    def prepare(
        self,
        data_root: Path,
        subset: Optional[str] = None,
        *,
        source_dir: Optional[Path] = None,
        download: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> PreparedDataset:
        """Index (and, with ``download=True``, fetch) the corpus.

        Must raise :class:`DatasetUnavailable` -- never download -- when the
        corpus is absent and ``download`` is false: a large fetch is an explicit
        opt-in.
        """

    @abstractmethod
    def acquisition(self) -> str:
        """How to obtain the corpus, for the error message and ``datasets``."""

    def resolve_subset(self, subset: Optional[str]) -> str:
        subset = subset or self.default_subset
        if self.subsets and subset not in self.subsets:
            raise ValueError(
                f"{self.name}: unknown subset {subset!r}; choose from {list(self.subsets)}"
            )
        return subset


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


def file_digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(
    urls: Sequence[str],
    dest: Path,
    *,
    md5: Optional[str] = None,
    sha256: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    quiet: bool = False,
) -> Path:
    """Fetch the first reachable URL in *urls* to *dest*, verifying the digest.

    Writes to ``dest.part`` and renames on success, so an interrupted fetch
    never leaves a file that looks complete.  An existing *dest* whose digest
    matches is reused without touching the network.  Proxies come from the
    environment (``urllib`` prefers the lower-case variables).
    """
    dest = Path(dest)
    algo, expected = ("sha256", sha256) if sha256 else ("md5", md5)
    if dest.is_file() and (expected is None or file_digest(dest, algo) == expected):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    for url in urls:
        part = dest.with_name(dest.name + ".part")
        try:
            req = urllib.request.Request(url, headers=dict(headers or {}))
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done, last = 0, 0.0
                for block in iter(lambda: resp.read(1 << 20), b""):
                    out.write(block)
                    done += len(block)
                    now = time.monotonic()
                    if not quiet and now - last > 2.0:
                        pct = f" ({100.0 * done / total:.0f}%)" if total else ""
                        print(f"[asrbench] {dest.name}: {done / 2**20:.0f} MiB{pct}", flush=True)
                        last = now
        except Exception as exc:  # noqa: BLE001 -- try the next mirror
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            part.unlink(missing_ok=True)
            continue
        if expected is not None:
            got = file_digest(part, algo)
            if got != expected:
                part.unlink(missing_ok=True)
                errors.append(f"{url}: {algo} mismatch (got {got}, expected {expected})")
                continue
        part.replace(dest)
        return dest
    raise DatasetUnavailable(f"could not fetch {dest.name}:\n  " + "\n  ".join(errors))


def safe_extract(archive: Path, dest: Path, members_prefix: Optional[str] = None) -> None:
    """Extract a tarball, refusing members that escape *dest* or are links."""
    dest = Path(dest).resolve()
    with tarfile.open(archive) as tar:
        chosen = []
        for m in tar.getmembers():
            if members_prefix and not m.name.startswith(members_prefix):
                continue
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest) + os.sep) and target != dest:
                raise DatasetUnavailable(f"{archive.name}: member {m.name!r} escapes {dest}")
            if m.issym() or m.islnk() or m.isdev():
                continue
            chosen.append(m)
        if sys.version_info >= (3, 12):
            tar.extractall(dest, members=chosen, filter="data")
        else:  # pragma: no cover - 3.10/3.11; members were vetted above
            tar.extractall(dest, members=chosen)


def remove_tree(path: Path) -> None:
    if Path(path).exists():
        shutil.rmtree(path)
