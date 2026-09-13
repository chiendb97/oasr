# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Thin shims for the entry points the docs and muscle memory still name.

``bench_engine.py``, ``bench_service.py`` and ``bench_accuracy.py`` are cited
throughout ``AGENTS.md`` and ``docs/benchmarks.md``; ``oasr_benchmark.py`` is
cited by the ``/benchmark-kernel`` skill.  Each forwards into ``run.py`` with
its family already selected, so an old command line keeps working.

The other 27 ``bench_*.py`` wrappers are gone: they were either pure delegation
or a second implementation of a benchmark the family module already provided,
with their own default shapes and no path to a CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.run import main as _main  # noqa: E402


def forward(family: str | None = None) -> int:
    """Run ``run.py``, injecting ``--family`` when the caller did not give one."""
    argv = sys.argv[1:]
    if family and not any(
        a in ("--family", "--routine") or a.startswith(("--family=", "--routine=")) for a in argv
    ):
        sys.argv = [sys.argv[0], "--family", family, *argv]
    return _main()
