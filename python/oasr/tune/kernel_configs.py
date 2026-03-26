# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible re-exports from JIT layer.

Tile configurations have been moved to ``oasr.jit.gemm`` and ``oasr.jit.conv``
(FlashInfer-style).  Import from there instead.
"""

from oasr.jit.gemm import (
    TileConfig,
    get_unique_compile_configs,
    GEMM_DEFAULT,
    GEMM_TILE_CONFIGS,
    BMM_TILE_CONFIGS,
    GROUP_GEMM_TILE_CONFIGS,
)
from oasr.jit.conv import (
    CONV2D_DEFAULT,
    CONV2D_TILE_CONFIGS,
)

__all__ = [
    "TileConfig",
    "get_unique_compile_configs",
    "GEMM_DEFAULT",
    "GEMM_TILE_CONFIGS",
    "BMM_TILE_CONFIGS",
    "GROUP_GEMM_TILE_CONFIGS",
    "CONV2D_DEFAULT",
    "CONV2D_TILE_CONFIGS",
]
