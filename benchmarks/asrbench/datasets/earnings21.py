# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Earnings-21 -- real long-form recordings (whole earnings calls, ~53 min on average).

The long-form and long-session streaming workloads need *continuous* real
recordings with natural pauses and speaker turns, which no utterance corpus
provides.  Earnings-21 (Rev, 2021; 44 calls, ~39 h) is the most permissively
licensed of the standard long-form sets (CC BY-SA 4.0 -- TED-LIUM is
CC BY-NC-ND, SPGISpeech is non-commercial and gated) and is one of the long-form
tracks of the Open ASR Leaderboard.

It is fetched from the leaderboard's Parquet repackaging pinned to one commit,
one shard at a time: a shard holds a handful of whole calls (238-652 MB each),
so a benchmark that needs a few hours of long-form audio downloads one shard,
not the 4.4 GB set.  Each shard's sha256 is the Git-LFS object id at that
commit, recorded below and verified after download.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from benchmarks.asrbench.audio import DEFAULT_SAMPLE_RATE, resample, write
from benchmarks.asrbench.datasets.base import (
    DatasetAdapter,
    DatasetCard,
    DatasetUnavailable,
    ManifestItem,
    PreparedDataset,
    download as fetch_url,
    write_prepared,
)

REPO = "hf-audio/asr-leaderboard-longform"
REVISION = "d6797370d3189c618e722721ab5b6c9be78c022c"

#: shard index -> (bytes, sha256) at :data:`REVISION`.
SHARDS: Dict[int, tuple] = {
    0: (544211032, "b65aa9c994dc74c537186ebddbe5467014eba5acded87ca196c03f95ae03aab8"),
    1: (652202114, "f04af6776a6c133fb880d48ea78778b187b1f7545061ea7baaeffcb1d8fbeea2"),
    2: (649281171, "516dbbbc98fad48e7cdda046fdd506d90e8627aeb2ce45daa92c8a2dd7be23db"),
    3: (504297060, "10013067b8b86f511beca828457cff9a250fd5c898ad86a5075f1f2d0b044434"),
    4: (361215516, "ca9897afbfabfc101398ab1dfbb86f07381d7fe01bdf34c0d7cbc4d4368f9ff9"),
    5: (399679050, "222c61627a534201d5adb795eeb3005dc2866d8d6d1d9332e5694f5622812823"),
    6: (358924463, "d62c2596792f7600e6bdc6abf77218729735e56383bbe908bf8d65706a680242"),
    7: (341841085, "f094f5b1ec11b3867c1c5cb5012bf4ece444a50bd3fd347597d70fa665ec2bca"),
    8: (238636133, "333303373972579cd91188ebcfdbb374f2a552c96ddc0b7d5229f26655b2104d"),
    9: (327251636, "b33df9b1231318a5057d26a7763d3d127517006774edec39c85842828bb5dfba"),
}

#: The smallest shard -- the default when none is named.
DEFAULT_SHARDS = (8,)


def shard_name(index: int) -> str:
    return f"test-{index:05d}-of-00010.parquet"


def shard_url(index: int, endpoint: str = "https://huggingface.co") -> str:
    return f"{endpoint}/datasets/{REPO}/resolve/{REVISION}/earnings21/{shard_name(index)}"


class Earnings21(DatasetAdapter):
    name = "earnings21"
    description = "Earnings-21 English earnings calls, whole recordings (44 calls, ~39 h)"
    license = "CC BY-SA 4.0"
    purpose = "long-form offline workloads and long streaming sessions (real continuous audio)"
    subsets = ("test",)
    default_subset = "test"
    download_size = "238-652 MB per shard (default: 1 shard, 238 MB)"

    def acquisition(self) -> str:
        return (
            "python -m benchmarks.asrbench prepare earnings21 --download [--shards 8]\n"
            f"  (Parquet shards of {REPO}@{REVISION[:10]}, sha256-verified;\n"
            "   needs pyarrow; honours HF_ENDPOINT / HF_TOKEN)"
        )

    def prepare(
        self,
        data_root: Path,
        subset: Optional[str] = None,
        *,
        source_dir: Optional[Path] = None,
        download: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> PreparedDataset:
        subset = self.resolve_subset(subset)
        options = dict(options or {})
        shards: Sequence[int] = tuple(int(s) for s in options.get("shards") or DEFAULT_SHARDS)
        unknown = [s for s in shards if s not in SHARDS]
        if unknown:
            raise ValueError(f"earnings21: no shard(s) {unknown}; valid 0-9")
        dest = self.dataset_dir(data_root, subset)
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        headers = {}
        if os.environ.get("HF_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"

        parquet_files: List[Path] = []
        for idx in shards:
            local = (
                Path(source_dir) / shard_name(idx)
                if source_dir
                else dest / "_download" / shard_name(idx)
            )
            if not local.is_file():
                if source_dir is not None or not download:
                    raise DatasetUnavailable(
                        f"earnings21 shard {idx} not found at {local}.\n{self.acquisition()}"
                    )
                fetch_url([shard_url(idx, endpoint)], local, sha256=SHARDS[idx][1], headers=headers)
            parquet_files.append(local)

        items = _materialise(
            parquet_files, dest, int(options.get("sample_rate", DEFAULT_SAMPLE_RATE))
        )
        if not options.get("keep_archive", False) and source_dir is None:
            for p in parquet_files:
                p.unlink(missing_ok=True)
        card = DatasetCard(
            name=self.name,
            subset=subset,
            description=self.description,
            license=self.license,
            version="Earnings-21 (Rev), leaderboard long-form repackaging",
            revision=f"{REPO}@{REVISION}",
            source=[shard_url(i) for i in shards],
            has_transcripts=True,
            purpose=self.purpose,
            sample_rate=int(options.get("sample_rate", DEFAULT_SAMPLE_RATE)),
            checksums={shard_name(i): f"sha256:{SHARDS[i][1]}" for i in shards},
            options={"shards": list(shards)},
            notes="whole calls, re-encoded to 16 kHz mono FLAC; transcripts are verbatim",
        )
        return write_prepared(dest, card, items)


def _materialise(parquet_files: Sequence[Path], dest: Path, sample_rate: int) -> List[ManifestItem]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise DatasetUnavailable("earnings21 needs pyarrow (`pip install pyarrow`)") from exc
    import soundfile as sf

    items: List[ManifestItem] = []
    for path in parquet_files:
        table = pq.read_table(path)
        cols = set(table.column_names)
        for row in table.to_pylist():
            audio = row["audio"]
            data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            mono = np.ascontiguousarray(resample(mono, int(sr), sample_rate))
            stem = Path(audio.get("path") or "").stem
            uid = stem or f"{row.get('company_name', 'call')}_{len(items)}"
            rel = f"audio/{uid}.flac"
            write(dest / rel, mono, sample_rate)
            extra = {
                k: row[k]
                for k in ("company_name", "sector", "financial_quarter", "unique_speakers")
                if k in cols and row.get(k) is not None
            }
            items.append(
                ManifestItem(
                    id=uid,
                    audio=rel,
                    duration=len(mono) / sample_rate,
                    sample_rate=sample_rate,
                    text=row.get("text"),
                    group=uid,
                    extra=extra,
                )
            )
    items.sort(key=lambda i: i.id)
    return items
