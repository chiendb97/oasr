# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for GEMM kernels (FlashInfer-style).

Tile configurations are defined here in the JIT layer, and ALL variants are
compiled into a single shared library per kernel family.  The autotuner
selects which pre-compiled variant to call — no JIT during tuning.

This matches FlashInfer's pattern where ``cta_m_n_k_list`` is defined inside
the JIT generator function and all variants are bundled into one module via
a single ``gen_jit_spec()`` call.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .core import gen_jit_spec, _get_target_sm, JitSpec
from . import env


# =============================================================================
# Tile configuration (moved from tune/kernel_configs.py)
# =============================================================================


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
    def compile_name(self) -> str:
        """Config name excluding runtime params (split_k)."""
        parts = [f"t{self.tile_m}x{self.tile_n}x{self.tile_k}"]
        parts.append(f"w{self.warp_m}x{self.warp_n}x{self.warp_k}")
        parts.append(f"s{self.stages}")
        return "_".join(parts)

    @property
    def num_warps(self) -> int:
        return (self.tile_m // self.warp_m) * (self.tile_n // self.warp_n)

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


def get_unique_compile_configs(configs: List[TileConfig]) -> Dict[str, TileConfig]:
    """Deduplicate configs that compile to the same variant.

    Split-K is a runtime parameter, so configs that differ only in split_k
    share the same compiled variant.  Returns a dict of unique compile keys
    to representative configs (with split_k=1).
    """
    seen: Dict[str, TileConfig] = {}
    for cfg in configs:
        compile_cfg = TileConfig(
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            stages=cfg.stages, split_k=1,
        )
        key = compile_cfg.compile_name
        if key not in seen:
            seen[key] = compile_cfg
    return seen


# =============================================================================
# GEMM tile configurations (SM80+ TensorOp 16x8x16)
# =============================================================================

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
    # Split-K variants (runtime param, same compiled variant as default)
    TileConfig(tile_m=128, tile_n=128, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=2,
               split_k=2),
    TileConfig(tile_m=128, tile_n=128, tile_k=32, warp_m=64, warp_n=64, warp_k=32, stages=2,
               split_k=4),
]

# BMM tile configurations (shares GEMM configs, no split-K)
BMM_TILE_CONFIGS: List[TileConfig] = [
    cfg for cfg in GEMM_TILE_CONFIGS if cfg.split_k == 1
]

# Grouped GEMM tile configurations (SM80+ only, no split-K)
GROUP_GEMM_TILE_CONFIGS: List[TileConfig] = [
    cfg for cfg in GEMM_TILE_CONFIGS if cfg.split_k == 1
]


# =============================================================================
# Helper: render all tile variants for a given template
# =============================================================================


def _render_all_variants(
    template_name: str,
    family: str,
    configs: List[TileConfig],
    *,
    with_activation: bool = False,
) -> List:
    """Render Jinja templates for all unique tile configs.

    Each unique compile config produces one ``.cu`` file with uniquely-named
    exported functions (e.g., ``gemm_t128x128x32_w64x64x32_s2``).

    Args:
        template_name: Jinja template file name.
        family: Kernel family name (``"gemm"``, ``"bmm"``, ``"group_gemm"``).
        configs: List of tile configurations.
        with_activation: Whether to include fused activation variants (GEMM only).

    Returns:
        List of Path objects for the rendered ``.cu`` files.
    """
    from .templates import render_template
    from .cubin_loader import write_if_different

    sm = _get_target_sm()
    unique_configs = get_unique_compile_configs(configs)
    source_paths = []

    for config_name, cfg in unique_configs.items():
        func_name = f"{family}_{config_name}"
        variant_file_name = f"{family}_sm{sm}_{config_name}"

        rendered = render_template(
            template_name,
            op_name=variant_file_name,
            func_name=func_name,
            tile_m=cfg.tile_m,
            tile_n=cfg.tile_n,
            tile_k=cfg.tile_k,
            warp_m=cfg.warp_m,
            warp_n=cfg.warp_n,
            warp_k=cfg.warp_k,
            stages=cfg.stages,
            sm_version=sm,
            with_activation=with_activation,
        )
        gen_path = env.OASR_GEN_SRC_DIR / family / f"{variant_file_name}.cu"
        write_if_different(gen_path, rendered)
        source_paths.append(gen_path)

    return source_paths


# =============================================================================
# Module generators — ALL variants compiled into ONE .so per family
# =============================================================================


def gen_gemm_module() -> JitSpec:
    """Generate JIT spec for GEMM with ALL tile variants in one module.

    Each variant exports ``gemm_{config_name}`` and ``gemm_{config_name}_activation``
    as TVM-FFI functions.  The autotuner selects which to call; the default path
    uses ``GEMM_DEFAULT``.
    """
    source_paths = _render_all_variants(
        "gemm_cutlass_template.cu.jinja",
        "gemm",
        GEMM_TILE_CONFIGS,
        with_activation=True,
    )
    return gen_jit_spec("gemm", source_paths)


def gen_bmm_module() -> JitSpec:
    """Generate JIT spec for BMM with ALL tile variants in one module.

    Each variant exports ``bmm_{config_name}`` as a TVM-FFI function.
    """
    source_paths = _render_all_variants(
        "bmm_cutlass_template.cu.jinja",
        "bmm",
        BMM_TILE_CONFIGS,
    )
    return gen_jit_spec("bmm", source_paths)


def gen_group_gemm_module() -> JitSpec:
    """Generate JIT spec for grouped GEMM with ALL tile variants in one module.

    Each variant exports ``group_gemm_{config_name}`` as a TVM-FFI function.
    """
    source_paths = _render_all_variants(
        "group_gemm_cutlass_template.cu.jinja",
        "group_gemm",
        GROUP_GEMM_TILE_CONFIGS,
    )
    return gen_jit_spec("group_gemm", source_paths)


# =============================================================================
# Default function name helpers
# =============================================================================


def gemm_func_name(cfg: TileConfig) -> str:
    """Return the TVM-FFI export name for a GEMM variant."""
    return f"gemm_{cfg.compile_name}"


def gemm_activation_func_name(cfg: TileConfig) -> str:
    """Return the TVM-FFI export name for a GEMM+activation variant."""
    return f"gemm_{cfg.compile_name}_activation"


def bmm_func_name(cfg: TileConfig) -> str:
    """Return the TVM-FFI export name for a BMM variant."""
    return f"bmm_{cfg.compile_name}"


def group_gemm_func_name(cfg: TileConfig) -> str:
    """Return the TVM-FFI export name for a grouped GEMM variant."""
    return f"group_gemm_{cfg.compile_name}"
