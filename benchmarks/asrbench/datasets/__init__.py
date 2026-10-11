# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Dataset registry.  Add a corpus: subclass ``DatasetAdapter``, register it here."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Type

from benchmarks.asrbench.datasets.base import (
    DatasetAdapter,
    DatasetCard,
    DatasetUnavailable,
    ManifestItem,
    PreparedDataset,
    default_data_root,
)
from benchmarks.asrbench.datasets.earnings21 import Earnings21
from benchmarks.asrbench.datasets.librispeech import LibriSpeech
from benchmarks.asrbench.datasets.ljspeech import LJSpeech
from benchmarks.asrbench.datasets.longform import LongForm
from benchmarks.asrbench.datasets.manifest import LocalManifest
from benchmarks.asrbench.datasets.synthetic import Synthetic

DATASETS: Dict[str, Type[DatasetAdapter]] = {
    cls.name: cls for cls in (LibriSpeech, LJSpeech, Earnings21, LongForm, LocalManifest, Synthetic)
}

__all__ = [
    "DATASETS",
    "DatasetAdapter",
    "DatasetCard",
    "DatasetUnavailable",
    "ManifestItem",
    "PreparedDataset",
    "default_data_root",
    "get_adapter",
    "load_prepared",
    "parse_ref",
]


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return DATASETS[name]()
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; registered: {sorted(DATASETS)}") from None


def parse_ref(ref: str) -> tuple:
    """``"librispeech/test-clean"`` -> ``("librispeech", "test-clean")``; subset may be empty."""
    name, _, subset = ref.partition("/")
    return name, subset


def load_prepared(ref: str, data_root: Optional[Path] = None) -> PreparedDataset:
    """Open a prepared dataset by ``name[/subset]``, explaining how to prepare it if absent."""
    data_root = Path(data_root) if data_root else default_data_root()
    name, subset = parse_ref(ref)
    adapter = get_adapter(name)
    subset = subset or adapter.default_subset
    root = adapter.dataset_dir(data_root, subset)
    try:
        return PreparedDataset.load(root)
    except DatasetUnavailable:
        raise DatasetUnavailable(
            f"dataset {name}/{subset} is not prepared under {data_root}.\n"
            f"{adapter.acquisition()}"
        ) from None
