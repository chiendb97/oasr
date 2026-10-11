# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Any local ``{id, audio, text}`` JSONL -- bring your own corpus.

The repo's manifest format (``benchmarks/manifests/ljspeech_200.jsonl``) indexed
in place: nothing is copied or downloaded, durations are read from the audio
headers, and the manifest file's sha256 pins the revision.  The subset is a name
you choose, so one adapter serves any number of local sets::

    python -m benchmarks.asrbench prepare manifest --subset mini20 \\
        --source-dir /data/asr_benchmark_mini20/segments/manifest.jsonl
    python -m benchmarks.asrbench run --profile smoke --dataset manifest/mini20 ...

Relative ``audio`` paths resolve against ``-o audio_root=DIR``, or else the
first of (the manifest's directory, ``$WAV_DIR``, ``$AUDIO_DIR``) under which
every one of them exists; the root chosen is recorded on the card.  Optional
per-line ``speaker`` and ``group`` keys are kept; ``-o group_sep=__`` derives the
group from the id instead (``lesson__07`` -> ``lesson``), which is what the
long-form builder needs to concatenate contiguous speech.
"""

from __future__ import annotations

import json
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

#: Environment variables tried, after the manifest's own directory, for
#: relative audio paths -- the same ones ``validate --audio-root`` defaults to.
AUDIO_ENVS = ("WAV_DIR", "AUDIO_DIR")

OPTIONS = ("audio_root", "group_sep", "language", "license")

_SUBSET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class LocalManifest(DatasetAdapter):
    name = "manifest"
    description = "a local {id, audio, text} JSONL manifest, indexed in place"
    license = "as supplied"
    purpose = "your own corpus: any workload, with the same protocols as the built-in sets"
    subsets = ()  # free-form: the name you give it
    default_subset = ""
    download_size = "none (local files only)"

    def acquisition(self) -> str:
        return (
            "Point it at a JSONL manifest with one {id, audio, text} object per line:\n"
            "python -m benchmarks.asrbench prepare manifest --subset NAME "
            "--source-dir /path/to/manifest.jsonl [-o audio_root=DIR]"
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
        options = dict(options or {})
        unknown = sorted(set(options) - set(OPTIONS))
        if unknown:
            raise ValueError(f"manifest: unknown option(s) {unknown}; known: {list(OPTIONS)}")
        if source_dir is None:
            raise DatasetUnavailable(f"manifest: --source-dir is required.\n{self.acquisition()}")
        path = Path(source_dir)
        if path.is_dir():
            path = path / "manifest.jsonl"
        if not path.is_file():
            raise DatasetUnavailable(f"manifest: no such file {path}")
        path = path.resolve()
        subset = subset or default_subset_name(path)
        if not _SUBSET.match(subset):
            raise ValueError(
                f"manifest: subset {subset!r} must be a plain name ([A-Za-z0-9._-], "
                "no slash); pass --subset"
            )

        entries = read_entries(path)
        root = audio_root(path, entries, options.get("audio_root"))
        sep = options.get("group_sep")
        items: List[ManifestItem] = []
        for e in entries:
            audio = Path(e["audio"])
            audio = audio if audio.is_absolute() else root / audio
            frames, sr = read_info(audio)
            uid = e["id"]
            group = e.get("group")
            if sep:
                group = uid.rsplit(str(sep), 1)[0] if str(sep) in uid else uid
            items.append(
                ManifestItem(
                    id=uid,
                    audio=str(audio.resolve()),
                    duration=frames / sr,
                    sample_rate=sr,
                    text=e.get("text"),
                    speaker=None if e.get("speaker") is None else str(e["speaker"]),
                    group=None if group is None else str(group),
                )
            )
        digest = file_digest(path, "sha256")
        card = DatasetCard(
            name=self.name,
            subset=subset,
            description=f"local manifest {path}",
            license=str(options.get("license") or "unspecified (local data)"),
            language=str(options.get("language") or "unspecified"),
            revision=f"sha256:{digest[:16]}",
            source=[str(path), str(root)],
            has_transcripts=any(i.text for i in items),
            purpose=self.purpose,
            sample_rate=items[0].sample_rate,
            checksums={path.name: f"sha256:{digest}"},
            options={k: str(options[k]) for k in OPTIONS if k in options},
            notes="indexed in place; durations from the audio headers",
        )
        return write_prepared(self.dataset_dir(data_root, subset), card, items)


def default_subset_name(path: Path) -> str:
    """``mini20.jsonl`` -> ``mini20``; a file called ``manifest.jsonl`` takes its
    directory's name instead."""
    return path.parent.name if path.stem == "manifest" else path.stem


def read_entries(path: Path) -> List[Dict[str, Any]]:
    """The manifest's objects, each with a string ``id`` and an ``audio``."""
    out: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            e = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: not JSON ({exc})") from None
        if not isinstance(e, dict) or not e.get("audio"):
            raise ValueError(f"{path}:{lineno}: each line needs an 'audio' key")
        e["id"] = str(e.get("id") or Path(e["audio"]).stem)
        if e["id"] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {e['id']!r} (line {seen[e['id']]})")
        seen[e["id"]] = lineno
        out.append(e)
    if not out:
        raise DatasetUnavailable(f"{path}: the manifest lists no items")
    return out


def audio_root(path: Path, entries: List[Dict[str, Any]], explicit: Optional[str]) -> Path:
    """The directory relative ``audio`` paths resolve against.

    An explicit root must hold every file.  Otherwise the first candidate that
    holds all of them wins -- never a partial match, which would silently
    benchmark a subset.
    """
    relative = [Path(e["audio"]) for e in entries if not Path(e["audio"]).is_absolute()]
    absolute_missing = [
        e["audio"]
        for e in entries
        if Path(e["audio"]).is_absolute() and not Path(e["audio"]).is_file()
    ]
    if absolute_missing:
        raise DatasetUnavailable(
            f"manifest: {len(absolute_missing)} absolute audio path(s) missing, "
            f"e.g. {absolute_missing[0]}"
        )
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = [path.parent] + [Path(os.environ[v]) for v in AUDIO_ENVS if os.environ.get(v)]
    tried = []
    for cand in dict.fromkeys(c.resolve() for c in candidates):
        missing = [a for a in relative if not (cand / a).is_file()]
        if not missing:
            return cand
        tried.append(f"{cand} ({len(missing)}/{len(relative)} missing, e.g. {missing[0]})")
    raise DatasetUnavailable(
        "manifest: no audio root holds every file; tried\n  "
        + "\n  ".join(tried)
        + "\npass -o audio_root=DIR"
    )
