#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-file mypy error ratchet.

``mypy oasr/`` reports several hundred errors today, almost all of them
untyped-torch noise rather than defects, so gating CI on "zero errors" would
mean the job is red forever and nobody reads it.  Gating on "no file got
worse" is a check that passes today and still fails on a real regression.

The baseline is a ``{path: count}`` map.  A run fails when any file's count
goes **up** or a file that had none acquires some; a file whose count goes
*down* is reported as slack to reclaim, not an error, so a cleanup commit
does not have to touch this file to stay green (but should).

Usage::

    python scripts/mypy_ratchet.py            # check against the baseline
    python scripts/mypy_ratchet.py --update   # rewrite the baseline
    python scripts/mypy_ratchet.py --paths oasr/engine   # narrow the scan

Exit code is 0 when nothing regressed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = REPO_ROOT / "ci" / "mypy-baseline.json"
DEFAULT_PATHS = ("oasr",)

# ``path:line: error: message  [code]`` — notes and summary lines are ignored.
_ERROR_RE = re.compile(r"^(?P<path>[^:]+):\d+:(?:\d+:)? error: ")


def run_mypy(paths: tuple[str, ...]) -> tuple[Counter, str]:
    """Run mypy over *paths* and return per-file error counts plus raw output."""
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", *paths, "--cache-dir=/dev/null", "--no-color-output"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    # mypy exits 2 on a crash / bad invocation; 0 and 1 both mean it ran.
    if proc.returncode not in (0, 1):
        print(output, file=sys.stderr)
        raise SystemExit(f"mypy failed to run (exit {proc.returncode})")
    counts: Counter = Counter()
    for line in output.splitlines():
        m = _ERROR_RE.match(line)
        if m:
            counts[m.group("path")] += 1
    return counts, output


def load_baseline() -> dict[str, int]:
    if not BASELINE.exists():
        raise SystemExit(f"no baseline at {BASELINE}; create it with --update")
    data = json.loads(BASELINE.read_text())
    return {k: int(v) for k, v in data["files"].items()}


def write_baseline(counts: Counter, paths: tuple[str, ...]) -> None:
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": (
            "Per-file mypy error counts. Regenerate with "
            "`python scripts/mypy_ratchet.py --update`. Counts may only go down."
        ),
        "paths": list(paths),
        "total": sum(counts.values()),
        "files": dict(sorted(counts.items())),
    }
    BASELINE.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="rewrite the baseline from this run")
    ap.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS), help="paths to check")
    ap.add_argument("--show-output", action="store_true", help="print mypy's raw output")
    args = ap.parse_args()

    paths = tuple(args.paths)
    counts, output = run_mypy(paths)
    if args.show_output:
        print(output)

    if args.update:
        write_baseline(counts, paths)
        print(f"baseline updated: {sum(counts.values())} errors across {len(counts)} files")
        return 0

    baseline = load_baseline()
    worse = sorted((p, baseline.get(p, 0), n) for p, n in counts.items() if n > baseline.get(p, 0))
    better = sorted(
        (p, baseline[p], counts.get(p, 0)) for p in baseline if counts.get(p, 0) < baseline[p]
    )

    total, base_total = sum(counts.values()), sum(baseline.values())
    print(f"mypy: {total} errors (baseline {base_total})")

    if better:
        reclaimed = sum(was - now for _, was, now in better)
        print(f"\n{len(better)} file(s) improved by {reclaimed} error(s) — refresh the baseline:")
        for p, was, now in better[:20]:
            print(f"  {p}: {was} -> {now}")
        print("  run: python scripts/mypy_ratchet.py --update")

    if worse:
        print(f"\nFAIL: {len(worse)} file(s) regressed:", file=sys.stderr)
        for p, was, now in worse:
            print(f"  {p}: {was} -> {now}", file=sys.stderr)
        print(
            "\nFix the new errors, or — if they are unavoidable torch/DSL noise — "
            "narrow them with a targeted `# type: ignore[code]` rather than "
            "raising the baseline.",
            file=sys.stderr,
        )
        if not args.show_output:
            print("\nmypy output:\n" + output, file=sys.stderr)
        return 1

    print("OK: no file regressed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
