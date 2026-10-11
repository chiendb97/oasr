# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression comparison: directions, floors, noise, comparability, exit codes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks.asrbench.regression import (
    EXIT_INCOMPARABLE,
    EXIT_OK,
    EXIT_REGRESSION,
    Comparable,
    Rule,
    Thresholds,
    compare,
    compare_paths,
    compare_summaries,
    load_thresholds,
    render_text,
)
from benchmarks.asrbench.results import load_run

RULES = Thresholds(
    rules=[
        Rule("latency_ms.p50", max_regression_pct=10, min_abs=1.0),
        Rule("latency_ms.p99", max_regression_pct=10),
        Rule("rtfx", max_regression_pct=10, unless={"mode": "open"}),
        Rule("error_rate", max_regression_abs=0.01),
    ],
    noise_sigmas=2.0,
)


def verdicts(base, cand, point=None):
    return {v["metric"]: v["verdict"] for v in compare_summaries(base, cand, RULES, point)}


def test_direction_aware_verdicts():
    base = {"latency_ms.p50": 100.0, "latency_ms.n": 500, "rtfx": 50.0, "error_rate": 0.0}
    assert verdicts(base, {**base, "latency_ms.p50": 115.0}) == {
        "latency_ms.p50": "regression", "rtfx": "same", "error_rate": "same",
    }  # fmt: skip
    assert verdicts(base, {**base, "latency_ms.p50": 80.0})["latency_ms.p50"] == "improvement"
    assert verdicts(base, {**base, "rtfx": 40.0})["rtfx"] == "regression"  # lower is worse here
    assert verdicts(base, {**base, "rtfx": 60.0})["rtfx"] == "improvement"
    assert verdicts(base, {**base, "error_rate": 0.02})["error_rate"] == "regression"


def test_absolute_floor_suppresses_tiny_moves():
    base = {"latency_ms.p50": 2.0, "latency_ms.n": 500}
    assert (
        verdicts(base, {**base, "latency_ms.p50": 2.8})["latency_ms.p50"] == "same"
    )  # +40%, < 1 ms
    assert verdicts(base, {**base, "latency_ms.p50": 3.5})["latency_ms.p50"] == "regression"


def test_noise_widens_the_limit():
    base = {"rtfx": 100.0, "rtfx.rep_std": 6.0}
    cand = {"rtfx": 85.0, "rtfx.rep_std": 6.0}  # -15%, but 2 sigma combined is ~17
    (v,) = compare_summaries(base, cand, RULES)
    assert v["verdict"] == "same" and v["noise"] == pytest.approx(2 * (72**0.5))
    assert verdicts({"rtfx": 100.0}, {"rtfx": 85.0})["rtfx"] == "regression"


def test_unresolvable_percentiles_are_not_judged():
    base = {"latency_ms.p99": 10.0, "latency_ms.n": 20}
    (v,) = compare_summaries(base, {"latency_ms.p99": 50.0, "latency_ms.n": 20}, RULES)
    assert v["verdict"] == "insufficient_samples"


def test_rules_can_exclude_points():
    base, cand = {"rtfx": 100.0}, {"rtfx": 50.0}
    assert verdicts(base, cand, {"mode": "open", "rate": 5}) == {}
    assert verdicts(base, cand, {"mode": "closed"}) == {"rtfx": "regression"}


def _cmp(summary, gpu="GPU-A", options=None, mode=None, kind="run", served=None):
    instances = {m: {"info": {"model": served}} for m in ("offline", "streaming")} if served else {}
    return Comparable(
        path=Path("/x"),
        label="x",
        kind=kind,
        meta={
            "environment": {"gpus": [{"name": gpu}]},
            "backend": {"name": "b", "options": options or {}, "instances": instances},
        },
        results={"k": (summary, {"streaming_mode": mode, "point": {}})},
    )


def test_comparability_gates_and_exit_codes():
    s = {"latency_ms.p50": 10.0, "latency_ms.n": 100}
    worse = {"latency_ms.p50": 20.0, "latency_ms.n": 100}
    assert compare(_cmp(s), _cmp(s), RULES)[1] == EXIT_OK
    summary, code = compare(_cmp(s), _cmp(worse), RULES)
    assert code == EXIT_REGRESSION and summary["regressions"][0]["metric"] == "latency_ms.p50"
    _, code = compare(_cmp(s), _cmp(s, gpu="GPU-B"), RULES)
    assert code == EXIT_INCOMPARABLE
    assert compare(_cmp(s), _cmp(s, gpu="GPU-B"), RULES, allow_hardware_change=True)[1] == EXIT_OK
    _, code = compare(_cmp(s), _cmp(s, options={"dtype": "float32"}), RULES)
    assert code == EXIT_INCOMPARABLE
    assert (
        compare(_cmp(s), _cmp(s, options={"x": 1}), RULES, allow_config_change=True)[1] == EXIT_OK
    )
    summary, code = compare(_cmp(s, mode="native"), _cmp(s, mode="buffered"), RULES)
    assert code == EXIT_INCOMPARABLE and summary["skipped"]  # nothing comparable remains


def test_attached_server_is_identified_by_what_it_reports_not_its_pid():
    s = {"latency_ms.p50": 10.0, "latency_ms.n": 100}
    url = {"base_url": "http://127.0.0.1:8080"}
    model = {"id": "ckpt", "dtype": "torch.float16", "max_batch_size": 32}
    a = _cmp(s, options={**url, "server_pid": 111}, served=model)
    restarted = _cmp(s, options={**url, "server_pid": 222}, served=model)
    assert compare(a, restarted, RULES)[1] == EXIT_OK
    reconfigured = _cmp(
        s, options={**url, "server_pid": 222}, served={**model, "max_batch_size": 64}
    )
    summary, code = compare(a, reconfigured, RULES)
    assert code == EXIT_INCOMPARABLE and "max_batch_size" in summary["problems"][0]


def test_default_thresholds_load_and_yaml_override(tmp_path):
    t = load_thresholds(None)
    assert any(r.metric == "latency_ms.p50" for r in t.rules)
    f = tmp_path / "t.yaml"
    f.write_text("noise_sigmas: 3\nrules:\n  - {metric: 'rtfx', max_regression_pct: 1}\n")
    t = load_thresholds(f)
    assert t.noise_sigmas == 3 and t.rules[0].max_regression_pct == 1


def test_run_against_itself_passes_and_a_doctored_run_regresses(mock_run, tmp_path):
    summary, code = compare_paths(mock_run, mock_run)
    assert code == EXIT_OK and summary["matched"] == len(load_run(mock_run).results)
    assert "PASS" in render_text(summary)

    doctored = tmp_path / "doctored"
    shutil.copytree(mock_run, doctored)
    doc = json.loads((doctored / "run.json").read_text())
    for r in doc["results"]:
        if r["benchmark"] == "throughput":
            r["summary"]["rtfx"] *= 0.5
    (doctored / "run.json").write_text(json.dumps(doc))
    summary, code = compare_paths(mock_run, doctored)
    assert code == EXIT_REGRESSION
    assert [v["metric"] for v in summary["regressions"]] == ["rtfx"]
    assert "REGRESSIONS" in render_text(summary)


def test_validation_reports_compare(tmp_path):
    def write(d, wer):
        d.mkdir()
        (d / "validation.json").write_text(
            json.dumps(
                {
                    "wer_pct": wer,
                    "empty_rate": 0.0,
                    "workload": {"id": "w1"},
                    "backend": {"name": "b", "options": {}},
                    "environment": {"gpus": []},
                }
            )
        )
        return d

    a, b = write(tmp_path / "a", 4.0), write(tmp_path / "b", 4.5)
    summary, code = compare_paths(a, b)
    assert code == EXIT_REGRESSION and summary["regressions"][0]["metric"] == "wer_pct"
    assert compare_paths(a, write(tmp_path / "c", 4.2))[1] == EXIT_OK


def test_unreadable_input_is_incomparable(tmp_path):
    summary, code = compare_paths(tmp_path, tmp_path)
    assert code == EXIT_INCOMPARABLE and summary["status"] == "error"
