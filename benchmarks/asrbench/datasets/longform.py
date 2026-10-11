# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Long-form recordings built from consecutive utterances of a prepared dataset.

LibriSpeech and LJSpeech utterances were cut out of continuous chapter
recordings; concatenating consecutive utterances of one chapter in order puts
that contiguous reading back together (the "LibriSpeech-Long" construction).
This gives medium (30-120 s) and long (minutes) inputs that are real speech,
need no download beyond the source, carry an exact transcript, and come in
controlled target lengths -- which a real long-form corpus cannot offer.  Use
Earnings-21 when the recording itself must be real (natural pauses, crosstalk).

Recordings are never cropped to the target: the builder stops at the first
utterance boundary at or past it, so the transcript stays exact.  When a group
runs out before the target, it continues into the next group (a speaker change,
recorded in ``extra.groups``) unless ``cross_group`` is false.  ``gap_s`` of
digital silence may be inserted between utterances; the default is none.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from benchmarks.asrbench.audio import load, write
from benchmarks.asrbench.datasets.base import (
    DatasetAdapter,
    DatasetCard,
    DatasetUnavailable,
    ManifestItem,
    PreparedDataset,
    default_data_root,
    write_prepared,
)

DEFAULT_TARGETS_S = (60.0, 300.0, 900.0)


class LongForm(DatasetAdapter):
    name = "longform"
    description = "Contiguous long-form recordings concatenated from a prepared utterance set"
    license = "inherits the source dataset's license"
    purpose = "medium/long duration buckets, long-form offline and long streaming sessions"
    default_subset = ""  # derived: named after its source
    has_transcripts = True
    download_size = "none (derived from a prepared dataset)"

    def acquisition(self) -> str:
        return (
            "python -m benchmarks.asrbench prepare longform --source librispeech/test-clean "
            "[--targets 60,300,900 --per-target 2]\n"
            "  (the source must be prepared first)"
        )

    def resolve_subset(self, subset: Optional[str]) -> str:
        if not subset:
            raise ValueError("longform needs a subset name (defaults to the source's name)")
        return subset

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
        from benchmarks.asrbench.datasets import load_prepared  # registry imports this module

        source_ref = str(options.get("source") or "librispeech/test-clean")
        try:
            if source_dir is not None:
                source = PreparedDataset.load(Path(source_dir))
            else:
                source = load_prepared(source_ref, data_root or default_data_root())
        except DatasetUnavailable as exc:
            raise DatasetUnavailable(
                f"longform source {source_ref!r} is not prepared -- prepare it first.\n{exc}"
            ) from None
        subset = subset or f"{source.card.name}-{source.card.subset}"
        targets = [float(t) for t in options.get("targets") or DEFAULT_TARGETS_S]
        per_target = int(options.get("per_target", 2))
        gap_s = float(options.get("gap_s", 0.0))
        cross_group = bool(options.get("cross_group", True))
        seed = int(options.get("seed", 0))
        sample_rate = int(options.get("sample_rate", source.card.sample_rate or 16000))

        dest = self.dataset_dir(data_root, subset)
        plans = plan(source.items, targets, per_target, seed=seed, cross_group=cross_group)
        items: List[ManifestItem] = []
        for item_id, members in plans:
            pieces: List[np.ndarray] = []
            gap = np.zeros(int(round(gap_s * sample_rate)), dtype=np.float32)
            for k, m in enumerate(members):
                if k and gap.size:
                    pieces.append(gap)
                pieces.append(load(source.audio_path(m), sample_rate))
            wave = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
            rel = f"audio/{item_id}.flac"
            write(dest / rel, wave, sample_rate)
            texts = [m.text for m in members if m.text]
            items.append(
                ManifestItem(
                    id=item_id,
                    audio=rel,
                    duration=len(wave) / sample_rate,
                    sample_rate=sample_rate,
                    text=" ".join(texts) if len(texts) == len(members) else None,
                    speaker=members[0].speaker,
                    group=members[0].group,
                    extra={
                        "members": [m.id for m in members],
                        "groups": sorted({m.group or "" for m in members}),
                    },
                )
            )
        card = DatasetCard(
            name=self.name,
            subset=subset,
            description=f"{self.description} ({source.card.ref})",
            license=source.card.license,
            version=f"derived from {source.card.ref}",
            revision=f"{source.card.ref}@manifest:{source.card.manifest_sha256[:16]}",
            source=[str(source.root)],
            has_transcripts=all(i.text is not None for i in items),
            synthetic=source.card.synthetic,
            purpose=self.purpose,
            sample_rate=sample_rate,
            options={
                "source": source.card.ref,
                "targets_s": targets,
                "per_target": per_target,
                "gap_s": gap_s,
                "cross_group": cross_group,
                "seed": seed,
            },
            notes="concatenation of consecutive utterances; never cropped",
        )
        return write_prepared(dest, card, items)


def plan(
    items: Sequence[ManifestItem],
    targets: Sequence[float],
    per_target: int,
    *,
    seed: int = 0,
    cross_group: bool = True,
) -> List[tuple]:
    """``[(item_id, [members])]`` -- which utterances form each recording.

    Pure (no audio touched), so the selection is unit-testable and the same
    seed always produces the same recordings.  Utterances are ordered by
    ``(group, id)``; a recording starts at a seeded group boundary and takes
    consecutive utterances until it reaches the target.
    """
    ordered = sorted(items, key=lambda i: (i.group or "", i.id))
    if not ordered:
        raise DatasetUnavailable("longform: the source dataset has no items")
    starts = [k for k, it in enumerate(ordered) if k == 0 or it.group != ordered[k - 1].group]
    rng = random.Random(f"longform:{seed}")
    out: List[tuple] = []
    for target in sorted(targets):
        order = list(starts)
        rng.shuffle(order)
        made = 0
        for start in order:
            if made >= per_target:
                break
            members: List[ManifestItem] = []
            total = 0.0
            k = start
            while k < len(ordered) and total < target:
                if members and not cross_group and ordered[k].group != members[0].group:
                    break
                members.append(ordered[k])
                total += ordered[k].duration
                k += 1
            if total < target:
                continue  # this start cannot reach the target; try another
            out.append((f"long{int(target)}s_{made:02d}_{members[0].id}", members))
            made += 1
        if made < per_target:
            raise DatasetUnavailable(
                f"longform: only {made}/{per_target} recordings of {target:.0f}s fit in the "
                "source (enable cross_group, lower the target, or use a larger source)"
            )
    return out
