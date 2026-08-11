# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end accuracy regression gate.

Every other test in this repo compares tensors.  That is necessary and it is
not sufficient: a parity oracle feeds identical features to both sides, so a
bug in *how audio becomes features* cancels on both and the suite stays green.
The ``audio_scale`` defect did exactly that — it shipped, survived the whole
parity suite, and was caught only when somebody exported one more env var and
read a transcript.  This file is the check that would have caught it on the
first run: an empty transcript is 100% WER and a dropped leading token is a
visible delta.

Recorded rates live in ``ci/wer-reference.json``; a measured rate above
``recorded + tolerance`` fails.  Re-record deliberately, in the same commit as
the change that moves it, with the reason in the commit message — a reference
quietly bumped to make CI green is worse than no reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import assets
import pytest
import torch

from oasr.testing.accuracy import load_audio, load_manifest, transcribe
from oasr.testing.wer import compute, normalizer

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = REPO_ROOT / "ci" / "wer-reference.json"
REFERENCE = json.loads(REFERENCE_PATH.read_text())
ENTRIES = REFERENCE["entries"]


def _manifest_path() -> Path:
    return REPO_ROOT / REFERENCE["manifest"]


class TestManifest:
    """The manifest itself, without a GPU or a checkpoint."""

    def test_parses_and_has_the_recorded_length(self):
        entries = load_manifest(_manifest_path(), check_audio=False)
        assert len(entries) == REFERENCE["utterances"]
        assert all(e.uid and e.text.strip() for e in entries)

    def test_reference_units_match_the_manifest(self):
        """Pins the *reference side* of every recorded rate.

        A WER is `errors / ref_units`.  If the manifest changes — utterances
        added, text renormalized — every recorded rate silently refers to a
        different denominator and the gate compares numbers that were never
        comparable.  Checking the denominator here means that shows up as a
        failure in this file rather than as a mysterious pass elsewhere.
        """
        entries = load_manifest(_manifest_path(), check_audio=False)
        norm = normalizer("english")
        ref_units = sum(len(norm(e.text).split()) for e in entries)
        expected = {e["ref_units"] for e in ENTRIES}
        assert expected == {ref_units}, (
            f"manifest now has {ref_units} reference words but ci/wer-reference.json "
            f"records {expected}; re-record the rates rather than editing the count"
        )

    def test_no_bracketed_spans_survive(self):
        """The English normalizer *deletes* `(...)` and `[...]`.

        In LJSpeech the parenthetical is read aloud, so leaving brackets in the
        manifest drops those words from the reference while the hypothesis keeps
        them — scoring a correct transcription as a run of insertions.  That was
        ~40 spurious insertions per model and +0.85 WER before it was found.
        `bench_accuracy.py --build-manifest` unwraps them; this keeps it that way.
        """
        entries = load_manifest(_manifest_path(), check_audio=False)
        offenders = [e.uid for e in entries if set("()[]") & set(e.text)]
        assert not offenders, (
            f"{len(offenders)} manifest entries contain brackets, e.g. {offenders[:3]}; "
            "rebuild with `bench_accuracy.py --build-manifest` (it unwraps them)"
        )


@pytest.mark.cuda
@pytest.mark.parametrize("spec", ENTRIES, ids=[e["key"] for e in ENTRIES])
def test_wer_has_not_regressed(spec, device):
    """Measure the manifest and compare against the recorded rate."""
    assets.require(spec["asset"], REFERENCE["audio_asset"])
    ckpt = assets.resolve(spec["asset"])
    audio_root = Path(assets.resolve(REFERENCE["audio_asset"]))

    from oasr.engine import ASREngine, EngineConfig

    entries = load_manifest(_manifest_path(), audio_root)
    # ``service_mode`` defaults to offline, which is what every entry recorded
    # before streaming was gated.  An entry that names ``"streaming"`` measures the
    # chunk-by-chunk path instead — the same manifest and the same denominator, so
    # the two rates are directly comparable, which is the point: a streaming
    # regression that a parity test cannot see (a cache that resets, a flush that
    # truncates) shows up here as deletions.
    mode = spec.get("service_mode", "offline")
    cfg_kwargs = {
        "ckpt_dir": ckpt,
        "service_mode": mode,
        "dtype": getattr(torch, spec["dtype"]),
        "max_batch_size": spec["max_batch_size"],
    }
    # ``load_architecture`` is the *loader* selector, distinct from
    # ``architecture`` above, which is only the human label in the failure
    # message.  Set it for an explicit-only converter: an icefall pruned-RNNT dir
    # sniffs as ``zipformer``, so without it this entry would silently measure a
    # different branch of the same checkpoint.
    if spec.get("load_architecture"):
        cfg_kwargs["architecture"] = spec["load_architecture"]
    if spec.get("checkpoint_name"):
        cfg_kwargs["checkpoint_name"] = spec["checkpoint_name"]
    if spec["decode_method"]:
        cfg_kwargs["decode_method"] = spec["decode_method"]
    if spec.get("chunk_size"):
        cfg_kwargs["chunk_size"] = spec["chunk_size"]

    engine = ASREngine(EngineConfig(**cfg_kwargs))
    try:
        waves, _ = load_audio(entries, engine.sample_rate)
        hyps, _ = transcribe(engine, waves, spec["max_batch_size"], streaming=mode == "streaming")
    finally:
        del engine
        torch.cuda.empty_cache()

    result = compute(
        [e.text for e in entries],
        hyps,
        unit="word" if spec["metric"] == "wer" else "char",
        normalizer=normalizer(spec["normalizer"]),
        uids=[e.uid for e in entries],
    )

    # Denominator first: a mismatch means the comparison itself is invalid, and
    # saying so beats reporting a rate against the wrong reference length.
    assert result.counts.ref_len == spec["ref_units"], (
        f"{spec['key']}: reference length changed "
        f"({result.counts.ref_len} vs recorded {spec['ref_units']})"
    )

    ceiling = spec["error_rate_pct"] + REFERENCE["tolerance_pct"]
    detail = "\n".join("  " + line for line in result.worst(5))
    assert result.percent <= ceiling, (
        f"{spec['key']} ({spec['architecture']}) regressed: "
        f"{result.percent:.2f}% > {ceiling:.2f}% "
        f"(recorded {spec['error_rate_pct']:.2f}% + {REFERENCE['tolerance_pct']} tolerance)\n"
        f"{result.summary()}\n"
        f"worst utterances:\n{detail}\n"
        f"If this is a deliberate accuracy change, re-record with:\n"
        f"  {REFERENCE['recorded_with'].replace('$ASSET', str(ckpt))}"
    )
