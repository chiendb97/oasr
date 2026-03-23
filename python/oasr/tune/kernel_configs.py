# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Valid kernel configurations for autotuning.

Each configuration defines tile sizes, warp shapes, pipeline stages, and
other tunable parameters.  The autotuner JIT-compiles a separate shared
library for each tile configuration and profiles them at runtime.

Naming convention:
    tile_{M}x{N}x{K}_warp_{WM}x{WN}x{WK}_stages_{S}

Constraints (CUTLASS 2.x TensorOp, SM80+, MMA 16×8×16):
    - ThreadBlock.M divisible by Warp.M
    - ThreadBlock.N divisible by Warp.N
    - ThreadBlock.K divisible by Warp.K (and by MMA.K=16)
    - Warp.M divisible by MMA.M=16
    - Warp.N divisible by MMA.N=8
    - Total warps = (TB.M/W.M) × (TB.N/W.N), typically 4 or 8
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class TileConfig:
    """A CUTLASS tile configuration for GEMM or Conv2D."""

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    stages: int
    split_k: int = 1  # Only meaningful for GEMM (runtime param, not compiled)

    @property
    def name(self) -> str:
        """Unique identifier for this config."""
        parts = [f"t{self.tile_m}x{self.tile_n}x{self.tile_k}"]
        parts.append(f"w{self.warp_m}x{self.warp_n}x{self.warp_k}")
        parts.append(f"s{self.stages}")
        if self.split_k > 1:
            parts.append(f"sk{self.split_k}")
        return "_".join(parts)

    @property
    def num_warps(self) -> int:
        return (self.tile_m // self.warp_m) * (self.tile_n // self.warp_n)

    def cuda_flags(self, prefix: str) -> List[str]:
        """Return ``-D`` flags for NVCC compilation.

        Args:
            prefix: ``"OASR_GEMM"`` or ``"OASR_CONV2D"``.
        """
        return [
            f"-D{prefix}_TILE_M={self.tile_m}",
            f"-D{prefix}_TILE_N={self.tile_n}",
            f"-D{prefix}_TILE_K={self.tile_k}",
            f"-D{prefix}_WARP_M={self.warp_m}",
            f"-D{prefix}_WARP_N={self.warp_n}",
            f"-D{prefix}_WARP_K={self.warp_k}",
            f"-D{prefix}_STAGES={self.stages}",
        ]

    def to_tactic_config(self) -> Tuple[Tuple[str, int], ...]:
        """Convert to a ``Tactic.config`` tuple."""
        items = [
            ("tile_m", self.tile_m),
            ("tile_n", self.tile_n),
            ("tile_k", self.tile_k),
            ("warp_m", self.warp_m),
            ("warp_n", self.warp_n),
            ("warp_k", self.warp_k),
            ("stages", self.stages),
        ]
        if self.split_k > 1:
            items.append(("split_k", self.split_k))
        return tuple(items)


# =========================================================================
# GEMM tile configurations (SM80+ TensorOp 16x8x16)
# =========================================================================

# The default config matches ArchTraits<80>::Gemm and is always the fallback.
GEMM_DEFAULT = TileConfig(
    tile_m=128, tile_n=128, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=2
)

GEMM_TILE_CONFIGS: List[TileConfig] = [
    # Default (matches ArchTraits<80>)
    GEMM_DEFAULT,
    # More pipeline stages
    TileConfig(tile_m=128, tile_n=128, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=3),
    TileConfig(tile_m=128, tile_n=128, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=4),
    # Larger tiles — good for large M, N
    TileConfig(tile_m=256, tile_n=128, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=3),
    TileConfig(tile_m=128, tile_n=256, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=3),
    # Smaller tiles — good for small M or N
    TileConfig(tile_m=64, tile_n=64, tile_k=32, warp_m=32, warp_n=32, warp_k=32, stages=4),
    TileConfig(tile_m=128, tile_n=64, tile_k=32, warp_m=64, warp_n=32, warp_k=32, stages=3),
    TileConfig(tile_m=64, tile_n=128, tile_k=32, warp_m=32, warp_n=64, warp_k=32, stages=3),
    # Larger K tile — good for K-bound problems
    TileConfig(tile_m=128, tile_n=128, tile_k=64, warp_m=64, warp_n=64, warp_k=64, stages=3),
    # Split-K variants (runtime param, same compiled module as default)
    TileConfig(tile_m=128, tile_n=128, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=2,
               split_k=2),
    TileConfig(tile_m=128, tile_n=128, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=2,
               split_k=4),
]

# =========================================================================
# BMM tile configurations (shares GEMM configs, no split-K)
# =========================================================================

BMM_TILE_CONFIGS: List[TileConfig] = [
    cfg for cfg in GEMM_TILE_CONFIGS if cfg.split_k == 1
]

# =========================================================================
# Grouped GEMM tile configurations (SM80+ only, no split-K)
# =========================================================================

GROUP_GEMM_TILE_CONFIGS: List[TileConfig] = [
    cfg for cfg in GEMM_TILE_CONFIGS if cfg.split_k == 1
]

# =========================================================================
# Conv2D tile configurations (SM80+ TensorOp 16x8x16)
# =========================================================================

CONV2D_DEFAULT = TileConfig(
    tile_m=128, tile_n=128, tile_k=64, warp_m=64, warp_n=64, warp_k=64, stages=3
)

CONV2D_TILE_CONFIGS: List[TileConfig] = [
    # Default (matches ArchTraits<80>::Conv2d)
    CONV2D_DEFAULT,
    # More pipeline stages
    TileConfig(tile_m=128, tile_n=128, tile_k=64, warp_m=64, warp_n=64, warp_k=64, stages=4),
    # Different tile shapes
    TileConfig(tile_m=128, tile_n=128, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=3),
    TileConfig(tile_m=256, tile_n=128, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=3),
    TileConfig(tile_m=128, tile_n=256, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=3),
    TileConfig(tile_m=64, tile_n=64, tile_k=64, warp_m=32, warp_n=32, warp_k=64, stages=4),
]


def get_unique_compile_configs(configs: List[TileConfig]) -> Dict[str, TileConfig]:
    """Deduplicate configs that compile to the same module.

    Split-K is a runtime parameter, so configs that differ only in split_k
    share the same compiled module. Returns a dict of unique compile keys
    to representative configs (with split_k=1).
    """
    seen: Dict[str, TileConfig] = {}
    for cfg in configs:
        # Compile key excludes split_k (runtime param)
        compile_cfg = TileConfig(
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            stages=cfg.stages, split_k=1,
        )
        key = compile_cfg.name
        if key not in seen:
            seen[key] = compile_cfg
    return seen
