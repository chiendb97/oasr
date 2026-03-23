# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Path constants for JIT compilation."""

import os
import pathlib

# Root of the OASR package (python/oasr/)
_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent.parent

# Source directories (relative to project root)
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent

OASR_CSRC_DIR = _PROJECT_ROOT / "csrc"
OASR_INCLUDE_DIR = _PROJECT_ROOT / "include"

# Generated source directory for Jinja2 templates
OASR_GEN_SRC_DIR = pathlib.Path(
    os.environ.get("OASR_GEN_SRC_DIR", pathlib.Path.home() /
                   ".cache" / "oasr" / "generated")
)

# JIT cache directory
OASR_JIT_CACHE_DIR = pathlib.Path(
    os.environ.get("OASR_JIT_CACHE_DIR", pathlib.Path.home() /
                   ".cache" / "oasr" / "jit")
)


def _find_cutlass_include_dirs():
    """Find CUTLASS include directories for JIT compilation.

    Returns a list of include directories (main + tools/util).

    Search order:
    1. CUTLASS_DIR environment variable
    2. third_party/cutlass/ in the project tree
    3. FlashInfer's bundled CUTLASS copy
    """
    def _collect_dirs(cutlass_root):
        """Given a CUTLASS root dir, return all relevant include dirs."""
        dirs = []
        main_inc = cutlass_root / "include"
        if (main_inc / "cutlass" / "cutlass.h").exists():
            dirs.append(str(main_inc))
        # CUTLASS utility headers live under tools/util/include/
        util_inc = cutlass_root / "tools" / "util" / "include"
        if util_inc.exists():
            dirs.append(str(util_inc))
        return dirs

    # 1. Environment variable
    env_dir = os.environ.get("CUTLASS_DIR")
    if env_dir:
        dirs = _collect_dirs(pathlib.Path(env_dir))
        if dirs:
            return dirs

    # 2. Project third_party/
    project_cutlass = _PROJECT_ROOT / "third_party" / "cutlass"
    dirs = _collect_dirs(project_cutlass)
    if dirs:
        return dirs

    # 3. FlashInfer's bundled copy
    try:
        import flashinfer
        fi_dir = pathlib.Path(flashinfer.__file__).parent / "data" / "cutlass"
        dirs = _collect_dirs(fi_dir)
        if dirs:
            return dirs
    except ImportError:
        pass

    return []


OASR_CUTLASS_INCLUDE_DIRS = _find_cutlass_include_dirs()
