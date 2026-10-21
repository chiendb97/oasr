# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr/quantization/calibration``: what calibration feeds the model.

The frontend is the dangerous part.  ``audio_scale`` differs per framework
(WeNet 32768, icefall 1.0) and a wrong one degrades recognition without an
error, so calibration statistics taken through a re-derived frontend would be
statistics of a different signal than serving sees.  The check here is the
strong one: for the same waveform, the calibration features are **the engine's
features**, bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path

import assets
import numpy as np
import pytest
import torch
from helpers import make_engine

from oasr.quantization.calibration import CheckpointFrontend, load_waveform, select_manifest


def _dev_clean() -> Path:
    return Path(assets.require("LIBRISPEECH_DEV_DIR"))


class TestSelection:
    def test_selection_is_deterministic_and_spread(self):
        root = _dev_clean()
        a = select_manifest(root / "manifest.jsonl", n=128)
        b = select_manifest(root / "manifest.jsonl", n=128)
        assert [e["id"] for e in a.entries] == [e["id"] for e in b.entries]
        assert len(a.entries) == 128
        speakers = {e["id"].split("-")[0] for e in a.entries}
        assert len(speakers) == 40, "dev-clean has 40 speakers; a strided pick reaches them all"
        assert a.provenance["utterances"] == 128 and len(a.provenance["manifest_sha256"]) == 64

    def test_small_manifests_are_taken_whole(self, tmp_path):
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(
            "\n".join(json.dumps({"id": f"u{i}", "audio": f"{i}.wav"}) for i in range(5))
        )
        chosen = select_manifest(manifest, n=128)
        assert [e["id"] for e in chosen.entries] == [f"u{i}" for i in range(5)]
        assert chosen.audio_root == tmp_path

    def test_audio_at_another_rate_is_refused(self, tmp_path):
        sf = pytest.importorskip("soundfile")
        path = tmp_path / "x.wav"
        sf.write(str(path), np.zeros(800, dtype=np.float32), 8000)
        with pytest.raises(ValueError, match="8000 Hz audio for a 16000 Hz model"):
            load_waveform(path, 16000)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine frontend runs on CUDA")
def test_calibration_features_are_the_engines_features():
    ckpt = assets.require("CKPT_DIR")
    root = _dev_clean()
    entry = select_manifest(root / "manifest.jsonl", n=3).entries[1]
    engine = make_engine(ckpt, dtype=torch.float16)
    try:
        wav = load_waveform(root / entry["audio"], engine.sample_rate)
        processor = engine._input_processor  # noqa: SLF001 - the oracle is its exact path
        padded = processor._padded_waveform_batch([wav])  # noqa: SLF001
        lengths = torch.tensor([wav.numel()], dtype=torch.int64)
        want, want_lengths = processor._fbank_batch(padded, lengths)  # noqa: SLF001

        from oasr.models.registry import load_checkpoint_bundle

        _arch, bundle = load_checkpoint_bundle(ckpt, map_location="cpu")
        frontend = CheckpointFrontend(bundle.features, torch.device("cuda"))
        got, got_lengths = frontend(wav)
    finally:
        del engine
        torch.cuda.empty_cache()
    assert torch.equal(got_lengths.cpu(), want_lengths.cpu())
    assert torch.equal(got.to(want.dtype), want)
