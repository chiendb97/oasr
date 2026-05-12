# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CuteDSL kernels for OASR.

Subpackages and helper modules. The attention forward kernel
(``oasr.kernels.cute.attention``) is the headline consumer; the helper
modules in this directory factor out the pieces that other CuteDSL
kernels (or a future backward pass) can reuse.
"""

# Submodule re-exports. Importing the submodules eagerly would pull the
# CuteDSL compiler infra into module-load time -- which is fine on a GPU
# host but kills cold-start latency for tooling. Keep these as lazy
# attribute-style imports so callers pay only for what they touch.

__all__ = [
    "ampere_helpers",
    "block_info",
    "copy_utils",
    "layout_utils",
    "mask",
    "named_barrier",
    "pack_gqa",
    "paged_kv",
    "seqlen_info",
    "softmax",
    "tile_scheduler",
    "utils",
]
