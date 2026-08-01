#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Fail if a source file in the worktree is excluded by .gitignore.

`.gitignore` had a bare ``checkpoints/`` entry, which matches a directory of
that name at *any* depth — so the whole ``oasr/checkpoints/`` package (bundle
loader, native format, the ``oasr-convert`` CLI) was never committed.  The
local worktree had it, every clone did not, and nothing noticed until CI ran
``pytest`` on a fresh checkout and died at ``from oasr.checkpoints import ...``.

This check only works **locally**: on CI the files are already absent, so there
is nothing left to find.  It runs as a pre-commit hook.

    python scripts/check_no_ignored_sources.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that hold source.  Third-party trees are excluded outright.
SOURCE_ROOTS = ("oasr", "tests", "csrc", "include", "benchmarks", "scripts", "rust", "ci")
SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".rs",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".cc",
    ".cpp",
    ".jinja",
    ".proto",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".inc",
}
SKIP_PARTS = {"__pycache__", "target", "third_party", "3rdparty", "build", ".mypy_cache"}


def candidate_files() -> list[str]:
    out = []
    for root in SOURCE_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in SOURCE_SUFFIXES:
                continue
            if SKIP_PARTS & set(p.relative_to(REPO_ROOT).parts):
                continue
            out.append(str(p.relative_to(REPO_ROOT)))
    return sorted(out)


def main() -> int:
    files = candidate_files()
    if not files:
        return 0
    # `git check-ignore -v --stdin` prints one line per *ignored* path,
    # formatted `<source>:<line>:<pattern>\t<path>`; exit 1 means none matched.
    proc = subprocess.run(
        ["git", "check-ignore", "-v", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(files),
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"git check-ignore failed (exit {proc.returncode})")

    hits = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not hits:
        return 0

    print(f"ERROR: {len(hits)} source file(s) are excluded by .gitignore:\n", file=sys.stderr)
    for ln in hits:
        rule, _, path = ln.partition("\t")
        print(f"  {path}\n      matched by {rule}", file=sys.stderr)
    print(
        "\nA clone will not contain these files.  Anchor the offending pattern to "
        "the repo root (`/checkpoints/`, not `checkpoints/`) so it stops matching "
        "at every depth, or add a negation for this path.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
