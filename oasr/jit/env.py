# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Path constants for JIT compilation."""

import os
import pathlib

# Root of the OASR package (oasr/)
_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent.parent

# Source directories (relative to project root)
_PROJECT_ROOT = _PACKAGE_DIR.parent

OASR_CSRC_DIR = _PROJECT_ROOT / "csrc"
OASR_INCLUDE_DIR = _PROJECT_ROOT / "include"
OASR_TEMPLATE_DIR = _PROJECT_ROOT / "csrc" / "templates"

# Generated source directory for Jinja2 templates
OASR_GEN_SRC_DIR = pathlib.Path(
    os.environ.get("OASR_GEN_SRC_DIR", pathlib.Path.home() / ".cache" / "oasr" / "generated")
)

# JIT build directory (Ninja builds, compiled .so artifacts)
OASR_JIT_DIR = pathlib.Path(
    os.environ.get(
        "OASR_JIT_DIR",
        os.environ.get("OASR_JIT_CACHE_DIR", pathlib.Path.home() / ".cache" / "oasr" / "jit"),
    )
)

# Backward-compatible alias
OASR_JIT_CACHE_DIR = OASR_JIT_DIR


def _find_cutlass_include_dirs():
    """Find CUTLASS include directories for JIT compilation.

    Returns a list of include directories (main + tools/util) from the
    ``3rdparty/cutlass`` git submodule -- the single source of CUTLASS for this
    project.  CMake never fetches or links it: every CUTLASS-dependent kernel
    is JIT-compiled, so these dirs reach nvcc through
    ``cpp_ext.build_common_cflags`` at runtime instead.  The submodule's
    version is part of the JIT cache key; see :func:`cutlass_version_stamp`.
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

    project_cutlass = _PROJECT_ROOT / "3rdparty" / "cutlass"
    dirs = _collect_dirs(project_cutlass)

    return dirs


OASR_CUTLASS_INCLUDE_DIRS = _find_cutlass_include_dirs()


def cutlass_version_stamp():
    """Identity of the CUTLASS headers the JIT will compile against.

    Returned as ``(include_dir, version.h bytes)`` pairs so it can go straight
    into a cache key.  ``JitSpec._content_hash`` folds this in: the hash covers
    OASR's own sources and headers, but a JIT module that includes
    ``cutlass/gemm/...`` also depends on *these* headers, and ``build_and_load``
    short-circuits on an existing ``.so`` without consulting ninja.  Bumping
    the submodule (4.4.2 -> 4.6.1, say) would otherwise keep loading binaries
    built against the old CUTLASS, silently.

    Only ``version.h`` is read, not the whole tree -- hashing several thousand
    third-party headers on every JIT call is not worth it, and a release bump
    always moves this file.  A *local edit* to a vendored CUTLASS header at a
    fixed version still needs the cache cleared by hand.
    """
    stamp = []
    for inc in OASR_CUTLASS_INCLUDE_DIRS:
        version_h = pathlib.Path(inc) / "cutlass" / "version.h"
        if version_h.is_file():
            stamp.append((inc, version_h.read_bytes()))
        else:
            stamp.append((inc, b""))
    return stamp
