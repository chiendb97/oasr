# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The command line: the documented workflow, end to end, on the mock backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from benchmarks.asrbench.cli import main
from benchmarks.asrbench.profiles import Profile, builtin_profiles


def test_builtin_profiles_parse_and_resolve():
    names = builtin_profiles()
    assert {"smoke", "standard", "full", "selftest"} <= set(names)
    for name in names:
        p = Profile.load(name).resolved(primary="ljspeech/all")
        assert p.benchmarks
        for spec in p.workloads.values():
            assert "${" not in spec["dataset"], (name, spec)
        p.workload_spec(next(iter(p.workloads)))  # validates the workload schema
    std = Profile.load("standard").resolved(primary="ljspeech/all")
    assert std.workloads["long"]["dataset"] == "longform/ljspeech-all"


def test_profile_overrides():
    p = Profile.load("smoke").resolved(
        sweeps={"batch_sizes": [2, 8], "rates": [3.0], "repetitions": 4},
        workload_overrides={"num_items": 5},
        sets=["defaults.timeout_s=7", "streaming.chunk_ms=[160]"],
        skip=["cold_start"],
    )
    by = {b["name"]: b for b in p.benchmarks}
    assert by["batch"]["batch_sizes"] == [2, 8]
    assert by["load_open"]["rates"] == [3.0] and by["load_closed"]["mode"] == "open"
    assert all(b["repetitions"] == 4 for b in p.benchmarks)
    assert all(w["num_items"] == 5 for w in p.workloads.values())
    assert p.defaults["timeout_s"] == 7 and by["streaming"]["chunk_ms"] == [160]
    assert "cold_start" not in by
    with pytest.raises(ValueError):
        Profile.load("smoke").resolved(sets=["nosuch.x=1"])
    with pytest.raises(ValueError):
        Profile.load("smoke").resolved(only=["nothing-matches"])


def test_prepare_without_opt_in_fails_cleanly(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "prepare", "librispeech"]) == 2
    assert "--download" in capsys.readouterr().err


def test_default_results_root_is_outside_the_checkout(tmp_path, monkeypatch):
    from benchmarks.asrbench.results import default_results_root

    monkeypatch.delenv("ASRBENCH_RESULTS_DIR")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert default_results_root() == tmp_path / "xdg" / "oasr" / "asrbench" / "results"
    monkeypatch.setenv("XDG_DATA_HOME", "relative/dir")  # ignored, as the XDG spec says
    expected = Path.home() / ".local" / "share" / "oasr" / "asrbench" / "results"
    assert default_results_root() == expected
    monkeypatch.setenv("ASRBENCH_RESULTS_DIR", str(tmp_path / "mine"))
    assert default_results_root() == tmp_path / "mine"


def test_workflow(tmp_path, capsys):
    data, out = tmp_path / "data", tmp_path / "out"
    assert main(["--data-dir", str(data), "prepare", "synthetic"]) == 0
    capsys.readouterr()
    assert main(["--data-dir", str(data), "datasets", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert any(d["name"] == "synthetic" and d["prepared"] for d in listing)

    args = ["--data-dir", str(data), "run", "--profile", "selftest", "--backend", "mock",
            "--out", str(out), "--no-monitor", "--only", "single,batch,throughput", "--num-items", "4"]  # fmt: skip
    assert main(args + ["--tag", "a"]) == 0
    assert main(args + ["--tag", "b", "-o", "base_ms=40"]) == 0
    runs = sorted(p for p in out.iterdir() if p.is_dir())
    assert len(runs) == 2
    a, b = runs
    assert main(["recompute", str(a)]) == 0
    assert main(["report", str(a), str(b), "--out", str(tmp_path / "rep")]) == 0
    assert (tmp_path / "rep" / "report.md").is_file()
    assert main(["compare", str(a), str(a)]) == 0
    assert main(["compare", str(a), str(b)]) == 2  # different backend config
    assert (
        main(
            ["compare", str(a), str(b), "--allow-config-change", "--json", str(tmp_path / "c.json")]
        )
        == 1
    )
    assert json.loads((tmp_path / "c.json").read_text())["status"] == "regression"
    assert main(["save-baseline", str(a), "--name", "nightly", "--out", str(out)]) == 0
    assert (out / "baselines" / "nightly" / "run.json").is_file()


def test_a_failed_point_fails_run_and_compare(tmp_path, capsys, monkeypatch):
    """A point that raised is recorded and the run carries on -- but exits 1.

    ``run`` used to exit 0 here and ``compare`` to skip the point, so
    ``run && compare`` passed a candidate whose throughput benchmark crashed.
    """
    from benchmarks.asrbench.results import load_run
    from benchmarks.asrbench.runners.offline import Throughput

    data, out = tmp_path / "data", tmp_path / "out"
    assert main(["--data-dir", str(data), "prepare", "synthetic"]) == 0
    args = ["--data-dir", str(data), "run", "--profile", "selftest", "--backend", "mock",
            "--out", str(out), "--no-monitor", "--num-items", "2", "--only"]  # fmt: skip
    assert main(args + ["single,throughput", "--tag", "good"]) == 0
    assert main(args + ["single", "--tag", "subset"]) == 0

    async def crash(self, ctx, point, result):
        raise RuntimeError("CUDA error: an illegal memory access")

    monkeypatch.setattr(Throughput, "measure", crash)
    capsys.readouterr()
    assert main(args + ["single,throughput", "--tag", "crashed"]) == 1
    captured = capsys.readouterr()
    # By tag: directory names sort by second, then by a random run id.
    runs = {load_run(p).meta["tag"]: p for p in out.iterdir() if p.is_dir()}
    good, subset, crashed = runs["good"], runs["subset"], runs["crashed"]
    assert captured.out.splitlines()[-1] == str(crashed)  # still printed, to inspect or compare
    assert "FAILED throughput" in captured.err and "illegal memory access" in captured.err

    assert main(["compare", str(good), str(crashed)]) == 1
    assert "FAILED POINTS" in capsys.readouterr().out
    # A deliberate subset is a lost result unless the caller says so.
    assert main(["compare", str(good), str(subset)]) == 1
    assert main(["compare", str(good), str(subset), "--allow-missing"]) == 0
    assert main(["compare", str(subset), str(good)]) == 0


def test_runs_are_found_by_id_under_the_results_root(tmp_path, capsys, monkeypatch):
    data, root = tmp_path / "data", tmp_path / "results"
    monkeypatch.setenv("ASRBENCH_RESULTS_DIR", str(root))
    assert main(["--data-dir", str(data), "prepare", "synthetic"]) == 0
    args = ["--data-dir", str(data), "run", "--profile", "selftest", "--backend", "mock",
            "--no-monitor", "--only", "single", "--num-items", "2"]  # fmt: skip
    assert main(args + ["--tag", "first"]) == 0
    assert main(args + ["--tag", "second"]) == 0
    capsys.readouterr()
    assert main(["runs", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [r["tag"] for r in listed] == ["second", "first"]  # newest first
    assert all(r["finished"] and r["ok"] == r["results"] > 0 for r in listed)
    new, old = (r["run_id"] for r in listed)
    assert main(["compare", old, new]) == 0
    assert main(["report", old, new, "--out", str(tmp_path / "rep")]) == 0
    assert main(["save-baseline", old, "--name", "nightly"]) == 0
    assert main(["compare", "nightly", new]) == 0  # a baseline name resolves too
    assert main(["runs"]) == 0
    text = capsys.readouterr().out
    assert str(root) in text and old in text and "second" in text
    assert main(["compare", "nosuchid", new]) == 2  # unresolved: reported as missing


def test_profiles_and_backends_listing(capsys):
    assert main(["profiles"]) == 0
    assert "standard" in capsys.readouterr().out
    assert main(["profiles", "smoke", "--batch-sizes", "16"]) == 0
    shown = yaml.safe_load(capsys.readouterr().out)
    assert {b["name"]: b for b in shown["benchmarks"]}["batch"]["batch_sizes"] == [16]
    assert main(["backends", "-v"]) == 0
    out = capsys.readouterr().out
    assert "oasr_engine" in out and "-o ckpt_dir=" in out
