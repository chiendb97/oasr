# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The WER guardrail: scoring, normalisation, drift against a reference, thresholds."""

from __future__ import annotations

import json

import pytest

from benchmarks.asrbench.correctness import (
    Thresholds,
    evaluate,
    load_reference,
    manifest_workload,
    run_validation,
)
from benchmarks.asrbench.workloads import WorkItem


def items(texts):
    return [
        WorkItem(uid=f"u{k}", source_id=f"u{k}", audio_path="", duration_s=1.0, text=t)
        for k, t in enumerate(texts)
    ]


REFS = ["the cat sat", "on the mat", "hello world"]


def test_wer_counts_and_empty_rate():
    r = evaluate(items(REFS), ["the cat sat", "on a mat", ""], [None] * 3,
                 normalizer="none", reference=None, thresholds=Thresholds())  # fmt: skip
    assert (r["substitutions"], r["deletions"], r["insertions"]) == (1, 2, 0)
    assert r["ref_words"] == 8
    assert r["wer_pct"] == pytest.approx(100 * 3 / 8)
    assert r["empty_rate"] == pytest.approx(1 / 3)
    # default thresholds: empty transcripts are not tolerated
    assert r["status"] == "fail"
    assert {c["check"]: c["ok"] for c in r["checks"]} == {
        "failed_requests": True,
        "empty_rate": False,
    }


def test_failed_requests_fail_the_check_and_are_not_scored():
    r = evaluate(items(REFS), ["the cat sat", None, "hello world"], [None, "boom", None],
                 normalizer="none", reference=None, thresholds=Thresholds())  # fmt: skip
    assert r["failed"] == 1 and r["wer_pct"] == 0.0 and r["status"] == "fail"
    assert r["errors"] == ["boom"]


def test_normalizer_is_applied_to_both_sides():
    pytest.importorskip("transformers")
    r = evaluate(items(["The Cat, sat!"]), ["the cat sat"], [None],
                 normalizer="basic", reference=None, thresholds=Thresholds())  # fmt: skip
    assert r["wer_pct"] == 0.0


def test_drift_against_a_reference_run():
    reference = {"u0": "the cat sat", "u1": "on a mat", "u2": "hello world"}
    r = evaluate(
        items(REFS), ["the cat sat", "on the mat", "hello word"], [None] * 3,
        normalizer="none", reference=reference,
        thresholds=Thresholds(max_changed_pct=50.0, max_drift_wer_pct=10.0, max_wer_delta_pct=0.5),
    )  # fmt: skip
    ref = r["reference"]
    assert ref["matched"] == 3
    assert ref["changed_pct"] == pytest.approx(100 * 2 / 3)
    assert ref["drift_wer_pct"] == pytest.approx(100 * 2 / 8)
    assert ref["reference_wer_pct"] == pytest.approx(100 * 1 / 8)
    assert ref["wer_delta_pct"] == pytest.approx(0.0)  # one fixed, one broken
    failed = {c["check"] for c in r["checks"] if not c["ok"]}
    assert failed == {"changed_pct", "drift_wer_pct"}


def test_manifest_workload_and_reference_loading(data_root, tmp_path):
    from benchmarks.asrbench.datasets import load_prepared

    ds = load_prepared("librispeech/test-clean", data_root)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"id": i.id, "audio": str(ds.audio_path(i).name), "text": i.text}) + "\n"
            for i in ds.items[:2]
        )
    )
    audio_root = ds.audio_path(ds.items[0]).parent
    w = manifest_workload(manifest, audio_root, None)
    assert [i.uid for i in w.items] == [i.id for i in ds.items[:2]]
    assert w.items[0].duration_s == pytest.approx(ds.items[0].duration, abs=1e-3)

    preds = tmp_path / "v" / "predictions.jsonl"
    preds.parent.mkdir()
    preds.write_text('{"id": "a", "hyp": "x"}\n{"id": "b", "hyp": null}\n')
    assert load_reference(preds.parent) == {"a": "x"}


def test_run_validation_end_to_end_with_the_mock(data_root, tmp_path):
    report, code = run_validation(
        "mock", {}, dataset="librispeech/test-clean", num_items=5, normalizer="none",
        data_root=data_root, out_root=tmp_path, thresholds=Thresholds(max_wer_pct=0.0),
    )  # fmt: skip
    assert code == 0 and report["wer_pct"] == 0.0  # the mock echoes the reference
    first = report["output_dir"]
    report2, code2 = run_validation(
        "mock", {"drop_word_every": 3}, dataset="librispeech/test-clean", num_items=5,
        normalizer="none", data_root=data_root, out_root=tmp_path,
        reference=tmp_path / first.split("/")[-1],
        thresholds=Thresholds(max_wer_pct=5.0, max_changed_pct=0.0),
    )  # fmt: skip
    assert code2 == 1 and report2["wer_pct"] > 5.0
    assert report2["reference"]["changed_pct"] > 0
    rows = [json.loads(x) for x in open(f"{report2['output_dir']}/predictions.jsonl")]
    assert len(rows) == 5 and all(r["ref"] for r in rows)
