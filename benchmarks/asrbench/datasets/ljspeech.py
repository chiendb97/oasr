# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""LJSpeech 1.1 -- the zero-download smoke source.

Indexed in place: the corpus is usually already on disk (the repo's CI WER gate
reads it through ``$WAV_DIR``), so preparation writes a manifest pointing at it
rather than copying 24 hours of audio.  It is a narrow workload -- one speaker,
1-10 s clips -- which is exactly what a smoke run wants and why it is not the
primary set.  Public domain.

Accepted layouts: the release directory (``metadata.csv`` + ``wavs/``), or a
``wavs/`` directory whose parent holds ``metadata.csv`` -- which is what
``$AUDIO_DIR`` / ``$WAV_DIR`` point at in ``.env``.  Any sample rate is fine (the
release is 22.05 kHz; resampled copies are common); audio is resampled at load
time, before timing.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.asrbench.audio import read_info
from benchmarks.asrbench.datasets.base import (
    DatasetAdapter,
    DatasetCard,
    DatasetUnavailable,
    ManifestItem,
    PreparedDataset,
    file_digest,
    write_prepared,
)

#: Bracket characters, unwrapped (kept words) -- same rule and reason as
#: ``benchmarks/accuracy/accuracy.py:build_manifest``: Whisper's English
#: normaliser deletes bracketed spans, but in LJSpeech they are read aloud.
_BRACKETS = re.compile(r"[()\[\]]")

#: Environment variables consulted, in order, when no ``--source-dir`` is given.
SOURCE_ENVS = ("LJSPEECH_DIR", "AUDIO_DIR", "WAV_DIR")


class LJSpeech(DatasetAdapter):
    name = "ljspeech"
    description = "LJSpeech 1.1 single-speaker read English, 1-10 s clips"
    license = "Public domain"
    purpose = "smoke profile (already on disk here), short-utterance latency, WER guardrail"
    subsets = ("all",)
    default_subset = "all"
    download_size = "2.6 GB (not fetched automatically)"

    def acquisition(self) -> str:
        return (
            "Download https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 (2.6 GB),\n"
            "extract it, then:\n"
            "python -m benchmarks.asrbench prepare ljspeech --source-dir /path/to/LJSpeech-1.1\n"
            f"(or export one of {', '.join(SOURCE_ENVS)} pointing at it or at its wavs/)"
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
        root = _find_root(source_dir)
        if root is None:
            raise DatasetUnavailable(
                "LJSpeech not found"
                + (f" under {source_dir}" if source_dir else "")
                + f".\n{self.acquisition()}"
            )
        metadata = root / "metadata.csv"
        wavs = root / "wavs"
        items: List[ManifestItem] = []
        for uid, text in read_metadata(metadata):
            text = " ".join(_BRACKETS.sub(" ", text).split())
            wav = wavs / f"{uid}.wav"
            if not wav.is_file():
                continue
            frames, sr = read_info(wav)
            items.append(
                ManifestItem(
                    id=uid,
                    audio=str(wav.resolve()),
                    duration=frames / sr,
                    sample_rate=sr,
                    text=text,
                    speaker="LJ",
                    # LJ001..LJ050 are the 50 source books; consecutive clips of a
                    # book are contiguous reading.
                    group=uid.split("-")[0],
                )
            )
        if not items:
            raise DatasetUnavailable(f"{root}: metadata.csv lists no clip present under wavs/")
        card = DatasetCard(
            name=self.name,
            subset=subset,
            description=self.description,
            license=self.license,
            version="LJSpeech-1.1",
            revision=f"metadata.csv sha256:{file_digest(metadata, 'sha256')[:16]}",
            source=[str(root.resolve())],
            has_transcripts=True,
            purpose=self.purpose,
            sample_rate=items[0].sample_rate,
            checksums={"metadata.csv": "sha256:" + file_digest(metadata, "sha256")},
            notes="indexed in place; text is the normalized column with brackets unwrapped",
        )
        return write_prepared(self.dataset_dir(data_root, subset), card, items)


def read_metadata(path: Path) -> List[tuple]:
    """``[(id, text)]`` from the release's ``id|text|normalized`` file, or from a
    headed CSV re-export (Kaggle's has ``id,sentence,...``).  The release's
    normalised column is preferred where present, as the accuracy harness does."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    if "|" in lines[0]:
        out = []
        for line in lines:
            parts = line.split("|")
            if len(parts) >= 2 and parts[0]:
                out.append((parts[0], parts[2] if len(parts) > 2 and parts[2] else parts[1]))
        return out
    rows = list(csv.DictReader(lines))
    if not rows:
        return []
    cols = {c.lower(): c for c in rows[0] if c}
    id_col = next((cols[c] for c in ("id", "utt_id", "utterance_id") if c in cols), None)
    text_col = next(
        (
            cols[c]
            for c in ("normalized_text", "normalized", "sentence", "text", "transcription")
            if c in cols
        ),
        None,
    )
    if id_col is None or text_col is None:
        raise DatasetUnavailable(f"{path}: no id/text columns among {list(cols)}")
    return [(r[id_col].strip(), (r[text_col] or "").strip()) for r in rows if r.get(id_col)]


def _find_root(source_dir: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if source_dir is not None:
        candidates.append(Path(source_dir))
    else:
        candidates.extend(Path(os.environ[e]) for e in SOURCE_ENVS if os.environ.get(e))
    for cand in candidates:
        for root in (cand, cand.parent):
            if (root / "metadata.csv").is_file() and (root / "wavs").is_dir():
                return root
    return None
