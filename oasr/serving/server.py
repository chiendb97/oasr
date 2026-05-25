# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""``oasr-server`` console-script shim.

The real serving binary is implemented in Rust under ``rust/``.  This shim
keeps the historical ``oasr-server`` entry point in ``pyproject.toml`` /
``setup.py`` valid: it replaces the Python interpreter with the Rust binary
via ``os.execvp`` so command-line flags pass through unchanged.

Resolution order:
1. ``$OASR_RS_BIN`` if set and executable.
2. ``oasr-server`` on ``$PATH``.
3. ``./rust/target/release/oasr-server`` relative to the installed package.
4. Otherwise: print a build hint and exit 1.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _candidate_paths() -> list[str]:
    cands: list[str] = []
    env = os.environ.get("OASR_RS_BIN")
    if env:
        cands.append(env)
    path_bin = shutil.which("oasr-server")
    if path_bin:
        cands.append(path_bin)
    # In an editable install, oasr/__init__.py lives at <repo>/oasr/, so the
    # release binary sits at <repo>/rust/target/release/oasr-server.
    here = Path(__file__).resolve()
    for parent in (here.parents[2], here.parents[3] if len(here.parents) > 3 else here.parents[2]):
        bin_path = parent / "rust" / "target" / "release" / "oasr-server"
        if bin_path.is_file():
            cands.append(str(bin_path))
    return cands


def main() -> int:
    for cand in _candidate_paths():
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            os.execvp(cand, [cand, *sys.argv[1:]])
    sys.stderr.write(
        "oasr-server: Rust binary not found.\n"
        "  Build with:  cd rust && cargo build --release\n"
        "  Or set OASR_RS_BIN=/path/to/oasr-server.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
