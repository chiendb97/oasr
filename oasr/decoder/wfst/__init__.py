# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""In-tree GPU WFST graph tooling: hlg.img format + k2 export + lattice assembly.

Migrated from the standalone ``wfst`` project.  :mod:`graph_image` is pure numpy;
:func:`export_hlg` and :func:`build_lattice` are imported lazily and use ``k2`` only
inside the functions, so ``import oasr.decoder.wfst`` stays k2-free for decoding a
prebuilt ``.img`` graph.
"""

from oasr.decoder.wfst.graph_image import GraphImage, build_image, read_image, write_image

__all__ = [
    "GraphImage",
    "build_image",
    "read_image",
    "write_image",
    "export_hlg",
    "build_lattice",
]

# k2/torch-heavy entry points, resolved on first access so the numpy-only graph
# tooling imports without pulling k2.
_LAZY = {"export_hlg": "graph_export", "build_lattice": "lattice"}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is not None:
        import importlib

        return getattr(importlib.import_module(f"oasr.decoder.wfst.{mod}"), name)
    raise AttributeError(f"module 'oasr.decoder.wfst' has no attribute {name!r}")
