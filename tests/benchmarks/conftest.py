# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the asrbench suite's own tests -- CPU-only and offline.

Generated audio, a fake LibriSpeech tree and runs produced by the mock backend:
the subject is the measurement infrastructure, so nothing here needs a GPU, a
corpus or a network.  Builders live in ``asrbench_fixtures``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from asrbench_fixtures import SYNTHETIC_DURATIONS, make_librispeech_tree, run_tiny

from benchmarks.asrbench.datasets import get_adapter


@pytest.fixture(autouse=True)
def isolated_results_root(tmp_path, monkeypatch) -> Path:
    """No test writes into the developer's real results directory."""
    root = tmp_path / "asrbench-results-root"
    monkeypatch.setenv("ASRBENCH_RESULTS_DIR", str(root))
    return root


@pytest.fixture(scope="session")
def data_root(tmp_path_factory) -> Path:
    """A data root holding a prepared synthetic set and a prepared fake LibriSpeech."""
    root = tmp_path_factory.mktemp("asrbench-data")
    get_adapter("synthetic").prepare(root, options={"durations": SYNTHETIC_DURATIONS})
    src = make_librispeech_tree(tmp_path_factory.mktemp("libri-src"))
    get_adapter("librispeech").prepare(root, "test-clean", source_dir=src)
    return root


@pytest.fixture(scope="session")
def mock_run(data_root, tmp_path_factory) -> Path:
    """One complete mock run, shared by the report / regression / CLI tests."""
    return run_tiny(data_root, tmp_path_factory.mktemp("asrbench-results"), tag="baseline")
