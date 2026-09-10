# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified ``from_pretrained`` loader (local dir + HF Hub id)."""

from __future__ import annotations

import sys
import types


def test_resolve_local_dir(tmp_path):
    from oasr.models.loaders import _resolve_to_local_dir

    assert _resolve_to_local_dir(tmp_path) == str(tmp_path)


def test_from_pretrained_local_passthrough(tmp_path, monkeypatch):
    import oasr.models.loaders as L

    seen = {}

    def fake_build(local_dir, checkpoint_name, device=None, dtype=None, architecture=None):
        seen["args"] = (str(local_dir), checkpoint_name, device, dtype)
        return ("MODEL", "CONFIG")

    monkeypatch.setattr(L, "build_model_from_checkpoint", fake_build)
    out = L.from_pretrained(tmp_path, checkpoint_name="final.pt", device="cpu")
    assert out == ("MODEL", "CONFIG")
    assert seen["args"][0] == str(tmp_path)
    assert seen["args"][1] == "final.pt"


def test_from_pretrained_hf_download(tmp_path, monkeypatch):
    """A non-local id is resolved via huggingface_hub.snapshot_download."""
    captured = {}

    hub = types.ModuleType("huggingface_hub")

    def fake_snapshot(repo_id, revision=None, cache_dir=None, allow_patterns=None):
        captured.update(repo_id=repo_id, revision=revision)
        return str(tmp_path)

    hub.snapshot_download = fake_snapshot
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    import oasr.models.loaders as L

    monkeypatch.setattr(
        L,
        "build_model_from_checkpoint",
        lambda d, n, device=None, dtype=None, architecture=None: ("M", "C"),
    )
    out = L.from_pretrained("some-org/some-asr-model", revision="v1")
    assert out == ("M", "C")
    assert captured == {"repo_id": "some-org/some-asr-model", "revision": "v1"}


def test_top_level_and_classmethod_exports():
    import oasr
    from oasr.models import from_pretrained as module_fp
    from oasr.models.base import BaseAsrModel

    assert oasr.from_pretrained is module_fp
    assert callable(module_fp)
    # Classmethod exists and is bound to the class (auto-detect loader).
    assert hasattr(BaseAsrModel, "from_pretrained")
