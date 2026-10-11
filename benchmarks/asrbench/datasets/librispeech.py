# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""LibriSpeech (OpenSLR SLR12) -- the primary utterance-level workload source.

Why it is the primary set, for *performance* rather than accuracy reasons:

* duration spread -- test-clean runs 1.3 s to 35 s (mean ~7.4 s), which covers
  the very-short through medium buckets and straddles Whisper's 30 s window;
* volume -- 2,620 utterances / 5.4 h per test split, enough to hold a latency
  distribution and to keep a concurrency test fed without repeating inputs;
* reproducibility -- fixed splits behind md5-published archives on several
  mirrors, and per-chapter grouping that lets consecutive utterances be
  concatenated into contiguous long-form speech;
* compatibility -- 16 kHz mono FLAC, the rate every shipped model expects, and
  the training domain of the repo's English checkpoints, so the WER guardrail
  on it is a meaningful "is this still working" signal.

CC BY 4.0.  One split is a single 330-350 MB archive; nothing larger is fetched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.asrbench.audio import read_info
from benchmarks.asrbench.datasets.base import (
    DatasetAdapter,
    DatasetCard,
    DatasetUnavailable,
    ManifestItem,
    PreparedDataset,
    download as fetch_url,
    safe_extract,
    write_prepared,
)

#: md5 of each archive, from https://www.openslr.org/resources/12/md5sum.txt.
MD5 = {
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other": "fb5a50374b501bb3bac4815ee91d3135",
}

MIRRORS = (
    "https://www.openslr.org/resources/12",
    "https://openslr.trmal.net/resources/12",
    "https://openslr.elda.org/resources/12",
    "https://openslr.magicdatatech.com/resources/12",
)


class LibriSpeech(DatasetAdapter):
    name = "librispeech"
    description = "LibriSpeech read English audiobooks, 16 kHz FLAC (OpenSLR SLR12)"
    license = "CC BY 4.0"
    purpose = (
        "primary utterance workloads (very-short..medium buckets), concurrency and "
        "streaming sessions, long-form concatenation source, WER guardrail"
    )
    subsets = tuple(MD5)
    default_subset = "test-clean"
    download_size = "~330-350 MB per split"

    def acquisition(self) -> str:
        return (
            "python -m benchmarks.asrbench prepare librispeech --subset test-clean --download\n"
            "  (fetches test-clean.tar.gz, 346 MB, md5-verified, from openslr.org or a mirror)\n"
            "or, with the archive already extracted somewhere:\n"
            "python -m benchmarks.asrbench prepare librispeech --source-dir /path/to/LibriSpeech"
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
        dest = self.dataset_dir(data_root, subset)
        options = dict(options or {})
        mirrors = list(options.get("mirrors") or MIRRORS)

        split_dir: Optional[Path] = None
        checksums: Dict[str, str] = {}
        sources: List[str] = []
        if source_dir is not None:
            split_dir = _find_split(Path(source_dir), subset)
            if split_dir is None:
                raise DatasetUnavailable(
                    f"no LibriSpeech {subset} under {source_dir} (looked for "
                    f"LibriSpeech/{subset}, {subset}/, or speaker dirs with *.trans.txt)"
                )
            sources.append(str(Path(source_dir).resolve()))
        else:
            extracted = dest / "LibriSpeech" / subset
            if extracted.is_dir() and any(extracted.rglob("*.trans.txt")):
                split_dir = extracted
            elif not download:
                raise DatasetUnavailable(
                    f"LibriSpeech {subset} is not prepared.\n{self.acquisition()}"
                )
            else:
                archive = dest / "_download" / f"{subset}.tar.gz"
                urls = [f"{m}/{subset}.tar.gz" for m in mirrors]
                fetch_url(urls, archive, md5=MD5[subset])
                safe_extract(archive, dest, members_prefix=f"LibriSpeech/{subset}/")
                split_dir = extracted
                sources.extend(urls[:1])
                if not options.get("keep_archive", False):
                    archive.unlink(missing_ok=True)
            checksums[f"{subset}.tar.gz"] = f"md5:{MD5[subset]}"

        items = _index(split_dir, dest)
        if not items:
            raise DatasetUnavailable(f"{split_dir}: no transcribed utterances found")
        card = DatasetCard(
            name=self.name,
            subset=subset,
            description=self.description,
            license=self.license,
            version="LibriSpeech ASR corpus (OpenSLR SLR12)",
            revision=f"md5:{MD5[subset]}" if source_dir is None else "local-source",
            source=sources or [f"{MIRRORS[0]}/{subset}.tar.gz"],
            has_transcripts=True,
            purpose=self.purpose,
            sample_rate=16000,
            checksums=checksums,
            notes="text is the original upper-case transcript; normalise before scoring",
        )
        return write_prepared(dest, card, items)


def _find_split(root: Path, subset: str) -> Optional[Path]:
    for cand in (root / "LibriSpeech" / subset, root / subset, root):
        if cand.is_dir() and any(cand.glob("*/*/*.trans.txt")):
            return cand
    return None


def _index(split_dir: Path, dest: Path) -> List[ManifestItem]:
    items: List[ManifestItem] = []
    for trans in sorted(split_dir.glob("*/*/*.trans.txt")):
        spk, chap = trans.parent.parent.name, trans.parent.name
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, _, text = line.strip().partition(" ")
            if not uid:
                continue
            flac = trans.parent / f"{uid}.flac"
            if not flac.is_file():
                continue
            frames, sr = read_info(flac)
            try:
                audio = str(flac.relative_to(dest))
            except ValueError:
                audio = str(flac.resolve())
            items.append(
                ManifestItem(
                    id=uid,
                    audio=audio,
                    duration=frames / sr,
                    sample_rate=sr,
                    text=text.strip(),
                    speaker=spk,
                    group=f"{spk}-{chap}",
                )
            )
    return items
