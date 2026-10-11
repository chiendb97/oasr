# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Generated signals with known durations -- for testing the suite, never for measuring.

Exists so the infrastructure (sampling, scheduling, records, reports) can be
exercised in CI with no corpus and no network.  The card is marked
``synthetic: true``; every run that uses it carries that flag, and ``report``
prints a banner saying the numbers are not a performance measurement.  An ASR
model's cost depends on what it decodes, so a tone is not a stand-in for speech.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from benchmarks.asrbench.audio import write
from benchmarks.asrbench.datasets.base import (
    DatasetAdapter,
    DatasetCard,
    ManifestItem,
    PreparedDataset,
    write_prepared,
)

DEFAULT_DURATIONS_S = (0.8, 1.5, 3.0, 4.5, 7.0, 10.0, 12.0, 18.0, 25.0, 40.0, 90.0)


def signal(seconds: float, sample_rate: int, seed: int) -> np.ndarray:
    """A syllable-rate amplitude-modulated harmonic tone over low noise."""
    n = int(round(seconds * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(seed)
    f0 = 110.0 + 40.0 * rng.random()
    env = 0.5 + 0.5 * np.sin(2 * math.pi * 4.0 * t)
    x = sum(np.sin(2 * math.pi * f0 * h * t) / h for h in (1, 2, 3))
    x = 0.25 * env * x + 0.01 * rng.standard_normal(n)
    return x.astype(np.float32)


class Synthetic(DatasetAdapter):
    name = "synthetic"
    description = "Generated AM tones of fixed durations (infrastructure tests only)"
    license = "n/a (generated)"
    purpose = "CI and unit tests of the suite itself; not a performance input"
    default_subset = "default"
    has_transcripts = False
    download_size = "none"

    def acquisition(self) -> str:
        return "python -m benchmarks.asrbench prepare synthetic"

    def resolve_subset(self, subset: Optional[str]) -> str:
        return subset or self.default_subset

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
        durations = [float(d) for d in options.get("durations") or DEFAULT_DURATIONS_S]
        repeat = int(options.get("repeat", 1))
        sample_rate = int(options.get("sample_rate", 16000))
        dest = self.dataset_dir(data_root, subset)
        items: List[ManifestItem] = []
        k = 0
        for r in range(repeat):
            for d in durations:
                uid = f"syn{k:04d}_{d:g}s"
                rel = f"audio/{uid}.wav"
                x = signal(d, sample_rate, seed=k)
                write(dest / rel, x, sample_rate)
                items.append(
                    ManifestItem(
                        id=uid,
                        audio=rel,
                        duration=len(x) / sample_rate,
                        sample_rate=sample_rate,
                        text=None,
                        group=f"g{r}",
                    )
                )
                k += 1
        card = DatasetCard(
            name=self.name,
            subset=subset,
            description=self.description,
            license=self.license,
            version="generated",
            revision=f"durations={durations};repeat={repeat};sr={sample_rate}",
            has_transcripts=False,
            synthetic=True,
            purpose=self.purpose,
            sample_rate=sample_rate,
            options={"durations": durations, "repeat": repeat},
        )
        return write_prepared(dest, card, items)
