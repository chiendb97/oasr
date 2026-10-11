# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared builders for the asrbench suite's own tests (used by conftest and tests).

A uniquely named module rather than helpers in ``conftest``: ``tests/`` already
has a ``conftest``, and two modules imported under one name is a collision
waiting to happen.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from helpers import REPO_ROOT

# `benchmarks` is not an installed package; CI runs with PYTHONPATH=<repo>, and
# this makes a bare `pytest` from any directory behave the same.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.asrbench.audio import write  # noqa: E402
from benchmarks.asrbench.datasets.synthetic import signal  # noqa: E402

#: (speaker, chapter, utterance, seconds, text) for the fake LibriSpeech tree.
LIBRI_UTTS = [
    ("19", "198", "0000", 1.2, "NORTHANGER ABBEY"),
    ("19", "198", "0001", 3.4, "IT WAS A FINE MORNING"),
    ("19", "198", "0002", 6.5, "THE SUN SHONE ON THE ABBEY WALLS"),
    ("19", "198", "0003", 9.0, "SHE WALKED ALONG THE GRAVEL PATH SLOWLY"),
    ("26", "495", "0000", 2.5, "CHAPTER ONE"),
    ("26", "495", "0001", 12.0, "THERE WAS NO POSSIBILITY OF TAKING A WALK THAT DAY"),
    ("26", "495", "0002", 17.5, "WE HAD BEEN WANDERING INDEED IN THE LEAFLESS SHRUBBERY AN HOUR"),
]

SYNTHETIC_DURATIONS = [0.8, 1.5, 3.0, 4.5, 7.0, 10.0, 18.0, 25.0, 40.0]


def make_librispeech_tree(root: Path, subset: str = "test-clean") -> Path:
    """A minimal extracted-archive layout: ``LibriSpeech/<subset>/<spk>/<chap>/``."""
    split = root / "LibriSpeech" / subset
    by_chapter: Dict[tuple, List[str]] = {}
    for k, (spk, chap, utt, sec, text) in enumerate(LIBRI_UTTS):
        uid = f"{spk}-{chap}-{utt}"
        write(split / spk / chap / f"{uid}.flac", signal(sec, 16000, seed=k), 16000)
        by_chapter.setdefault((spk, chap), []).append(f"{uid} {text}")
    for (spk, chap), lines in by_chapter.items():
        (split / spk / chap / f"{spk}-{chap}.trans.txt").write_text("\n".join(lines) + "\n")
    return root


def tiny_profile(**overrides: Any) -> Dict[str, Any]:
    """A fast all-kinds profile over the synthetic set (mock-backend timings)."""
    doc: Dict[str, Any] = {
        "name": "tiny",
        "defaults": {"primary": "synthetic/default", "seed": 0, "timeout_s": 5},
        "workloads": {
            "utts": {
                "dataset": "${primary}",
                "sampling": "stratified",
                "num_items": 6,
                "duration_range": [0, 12],
            },
            "short": {
                "dataset": "${primary}",
                "sampling": "first",
                "num_items": 2,
                "duration_range": [0, 2],
            },
            "long": {"dataset": "${primary}", "sampling": "all", "duration_range": [30, None]},
        },
        "benchmarks": [
            {"name": "cold_start", "kind": "cold_start", "workload": "utts", "warm_repeats": 1},
            {"name": "single", "kind": "single", "workload": "utts", "repetitions": 2},
            {
                "name": "batch",
                "kind": "batch",
                "workload": "utts",
                "batch_sizes": [1, 2],
                "compositions": ["uniform", "mixed"],
                "num_batches": 2,
            },
            {"name": "throughput", "kind": "throughput", "workload": "utts"},
            {
                "name": "load_closed",
                "kind": "load",
                "workload": "utts",
                "mode": "closed",
                "concurrency": [1, 3],
                "num_requests": 6,
            },
            {
                "name": "load_open",
                "kind": "load",
                "workload": "utts",
                "mode": "open",
                "rate_fractions": [0.5],
                "capacity_from": "throughput",
                "num_requests": 6,
            },
            {"name": "longform", "kind": "longform", "workload": "long", "warmup": 0},
            {
                "name": "streaming",
                "kind": "streaming",
                "workload": "short",
                "chunk_ms": [200],
                "sessions": [2],
                "pacing": "realtime",
                "word_timestamps": True,
            },
        ],
    }
    doc.update(overrides)
    return doc


def run_tiny(
    data_root: Path,
    out: Path,
    options: Optional[Dict[str, Any]] = None,
    *,
    tag: str = "",
    monitor_interval_s: float = 0.05,
    profile: Optional[Dict[str, Any]] = None,
    backend: str = "mock",
) -> Path:
    """Run *profile* (default :func:`tiny_profile`) and return the run directory."""
    from benchmarks.asrbench.profiles import Profile
    from benchmarks.asrbench.runner import run_profile

    resolved = Profile.from_dict(profile or tiny_profile()).resolved()
    return asyncio.run(
        run_profile(
            resolved,
            backend,
            options or {},
            data_root=data_root,
            results_root=out,
            monitor_interval_s=monitor_interval_s,
            tag=tag,
        )
    )
