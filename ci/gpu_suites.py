#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The GPU test suite split, defined once for every backend that runs it.

Two things execute the GPU suite — the self-hosted runner
(`.github/workflows/test-gpu.yml`) and Modal (`ci/modal_app.py`) — and a split
maintained twice drifts.  Both read the families from here.

The split exists so one failing area does not mask the rest, and so the box
dropping off the bus costs one suite instead of the sweep.

    python ci/gpu_suites.py --list                # names
    python ci/gpu_suites.py --paths engine        # that family's test paths
    python ci/gpu_suites.py --github-matrix       # JSON for `fromJSON()`
    python ci/gpu_suites.py --check               # every test file is covered

``--check`` is the part worth keeping honest: a new `tests/**/test_*.py` that
sits outside every family directory would never run on the split matrix, and
the failure mode is silence.  It also enforces the one layout rule pytest
imposes on us -- `tests/` has no `__init__.py`, so modules are keyed by
basename and two `test_registry.py` in different folders is an import error,
not a merge.  It runs in `lint.yml`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

#: family -> test directories.  Each family is one directory under ``tests/``,
#: so the split is the layout rather than a list that has to be kept in step
#: with it.  Keep the groupings coarse; the point is isolation of blast radius,
#: not a taxonomy.
SUITES: dict[str, list[str]] = {
    # Kernels and the JIT/tuning machinery that selects them.
    "kernels": ["tests/kernels"],
    # CPU and GPU decoders, word timings, and the per-family decode options.
    "decoders": ["tests/decoders"],
    # Feature frontends and the kernels behind them.
    "features": ["tests/features"],
    # The engine and everything it owns: scheduler, executors, caches, CUDA
    # graphs, VAD, metrics, and the client that reaches it.
    "engine": ["tests/engine"],
    # Its own family so a WER regression is attributable at a glance rather than
    # buried in a model-family failure -- and because it is the one suite whose
    # failure means "the output got worse", not "a tensor moved".
    "accuracy": ["tests/accuracy"],
    # Architectures, the registry/contract ratchets and checkpoint conversion.
    "models": ["tests/models"],
}

#: Files deliberately outside the family split because they are reached through
#: an opt-in marker instead (`-m concurrent`), in its own job.
MARKER_ONLY: set[str] = {"tests/engine/test_concurrent.py"}

#: Markers the per-family jobs deselect; the opt-in job runs them separately.
DEFAULT_MARKER_EXPR = "not slow and not concurrent"


def paths_for(name: str) -> list[str]:
    try:
        return SUITES[name]
    except KeyError:
        raise SystemExit(f"unknown suite {name!r}; known: {', '.join(SUITES)}") from None


def github_matrix() -> str:
    """`include:` entries for a GitHub Actions matrix."""
    return json.dumps([{"name": n, "paths": " ".join(p)} for n, p in SUITES.items()])


def check() -> int:
    """Every tests/**/test_*.py lives under exactly one family, with a unique name.

    Two invariants, both of which fail silently otherwise:

    * a file outside every family directory never runs on the split matrix;
    * two test modules sharing a basename are an "import file mismatch" under
      pytest's prepend import mode, because ``tests/`` has no ``__init__.py``.
    """
    on_disk = sorted(p.relative_to(REPO_ROOT).as_posix() for p in TESTS_DIR.rglob("test_*.py"))
    families: dict[str, list[str]] = {}
    for family, roots in SUITES.items():
        for root in roots:
            for rel in on_disk:
                if rel == root or rel.startswith(root + "/"):
                    families.setdefault(rel, []).append(family)

    problems = []
    homeless = [p for p in on_disk if p not in families and p not in MARKER_ONLY]
    if homeless:
        problems.append(
            "outside every family directory (they would never run on the split "
            "matrix):\n" + "".join(f"    {p}\n" for p in homeless)
        )
    dupes = sorted(p for p, fams in families.items() if len(fams) > 1)
    if dupes:
        problems.append(
            "in more than one suite:\n"
            + "".join(f"    {p}: {', '.join(families[p])}\n" for p in dupes)
        )
    missing_roots = sorted(
        root for roots in SUITES.values() for root in roots if not (REPO_ROOT / root).is_dir()
    )
    if missing_roots:
        problems.append(
            "family directory not on disk:\n" + "".join(f"    {r}\n" for r in missing_roots)
        )
    stale_marker = sorted(m for m in MARKER_ONLY if not (REPO_ROOT / m).is_file())
    if stale_marker:
        problems.append(
            "MARKER_ONLY entry not on disk:\n" + "".join(f"    {p}\n" for p in stale_marker)
        )
    by_name: dict[str, list[str]] = {}
    for rel in on_disk:
        by_name.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)
    clashes = sorted(n for n, paths in by_name.items() if len(paths) > 1)
    if clashes:
        problems.append(
            "basename used by more than one test module (pytest imports test "
            "modules by basename; rename one):\n"
            + "".join(f"    {n}: {', '.join(by_name[n])}\n" for n in clashes)
        )

    if problems:
        print("ci/gpu_suites.py is out of sync with tests/:\n", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1
    print(
        f"OK: {len(on_disk)} test file(s) across {len(SUITES)} suites "
        f"+ {len(MARKER_ONLY)} marker-only"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="print suite names")
    g.add_argument("--paths", metavar="NAME", help="print one suite's test paths")
    g.add_argument("--github-matrix", action="store_true", help="print matrix JSON")
    g.add_argument("--check", action="store_true", help="verify every test file is covered")
    args = ap.parse_args()

    if args.list:
        print("\n".join(SUITES))
    elif args.paths:
        print(" ".join(paths_for(args.paths)))
    elif args.github_matrix:
        print(github_matrix())
    elif args.check:
        return check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
