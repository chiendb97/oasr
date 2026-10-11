# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Report generation from stored runs -- never from inference."""

from __future__ import annotations

import csv

import pytest

from benchmarks.asrbench.report import render_markdown, write_csv, write_report
from benchmarks.asrbench.results import load_run


def test_single_run_markdown_has_every_kind_and_the_simulation_banner(mock_run):
    md = render_markdown([load_run(mock_run)])
    assert "**Simulated:**" in md
    for section in ("## cold_start", "## single", "## batch", "## throughput", "## load",
                    "## longform", "## streaming", "## Resources", "## Definitions"):  # fmt: skip
        assert section in md, section
    assert "Latency by duration bucket" in md
    assert "baseline" in md  # the tag is the label


def test_multi_run_report_shows_deltas_and_csv_rows(mock_run, tmp_path):
    out = write_report([mock_run, mock_run], tmp_path / "rep")
    md = out.read_text()
    assert "(+0%)" in md or "(-0%)" in md
    rows = list(csv.DictReader(open(tmp_path / "rep" / "summary.csv")))
    run = load_run(mock_run)
    assert len(rows) == 2 * len(run.results)
    assert {"run", "benchmark", "kind", "status", "latency_ms.p50", "point.batch_size"} <= set(
        rows[0]
    )


def test_unsupported_results_are_listed(mock_run, tmp_path):
    import json
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(mock_run, copy)
    doc = json.loads((copy / "run.json").read_text())
    doc["results"][1]["status"] = "unsupported"
    doc["results"][1]["reason"] = "no native batch API"
    (copy / "run.json").write_text(json.dumps(doc))
    md = render_markdown([load_run(copy)])
    assert "## Not measured" in md and "no native batch API" in md


def test_plots_when_matplotlib_is_available(mock_run, tmp_path):
    pytest.importorskip("matplotlib")
    out = write_report([mock_run], tmp_path / "rep", plots=True)
    pngs = sorted(p.name for p in (tmp_path / "rep" / "plots").glob("*.png"))
    assert "throughput_by_batch.png" in pngs and "latency_by_duration.png" in pngs
    assert "## Charts" in out.read_text()


def test_csv_writer_handles_runs(mock_run, tmp_path):
    write_csv([load_run(mock_run)], tmp_path / "s.csv")
    assert (tmp_path / "s.csv").read_text().startswith("run,")
