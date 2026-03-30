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
from typing import Dict, List, Tuple, Union
from .core import gen_jit_spec, _get_target_sm, JitSpec
from . import env


# =============================================================================
# Tile configuration (moved from tune/kernel_configs.py)
# =============================================================================


@dataclass(frozen=True)
class TileShape:
    """A CUTLASS tile configuration for GEMM or Conv2D."""
    BK: int
    BN: int
    BM: int
    WK: int
    WN: int
    WM: int


class TileShapeSm90:
    """A CUTLASS tile configuration for GEMM or Conv2D."""
    BM: int
    BN: int
    BK: int


class ClusterShape:
    """A CUTLASS cluster configuration for GEMM or Conv2D."""
    CM: int
    CN: int
    CK: int


class CutlassGemmConfig:
    """A CUTLASS GEMM configuration."""
    BM: int
    BN: int
    BK: int
    WM: int
    WN: int
    WK: int
    kStages: int
    kSmVersion: int

    @property
    def name(self) -> str:
        """Unique identifier for this config."""
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.BM}x{self.BN}x{self.BK}")
        parts.append(f"w{self.WM}x{self.WN}x{self.WK}")
        parts.append(f"s{self.kStages}")
        return "_".join(parts)

    @property
    def compile_name(self) -> str:
        """Config name excluding runtime params (split_k)."""
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.BM}x{self.BN}x{self.BK}")
        parts.append(f"w{self.WM}x{self.WN}x{self.WK}")
        parts.append(f"s{self.kStages}")
        return "_".join(parts)

    @property
    def num_warps(self) -> int:
        return (self.BM // self.WM) * (self.BN // self.WN)

    def to_tactic_config(self) -> Tuple[Tuple[str, int], ...]:
        """Convert to a ``Tactic.config`` tuple."""
        items = [
            ("BM", self.BM),
            ("BN", self.BN),
            ("BK", self.BK),
            ("WM", self.WM),
            ("WN", self.WN),
            ("WK", self.WK),
            ("kStages", self.kStages),
        ]
        return tuple(items)


class CutlassGemmConfigSm90:
    """A CUTLASS GEMM configuration for SM90+."""
    BM: int
    BN: int
    BK: int
    CM: int
    CN: int
    CK: int
    kSMs: int
    kStages: int
    kSmVersion: int

    @property
    def name(self) -> str:
        """Unique identifier for this config."""
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.BM}x{self.BN}x{self.BK}")
        parts.append(f"c{self.CM}x{self.CN}x{self.CK}")
        parts.append(f"k{self.kSMs}")
        parts.append(f"s{self.kStages}")
        return "_".join(parts)

    @property
    def compile_name(self) -> str:
        """Config name excluding runtime params (split_k)."""
        parts = [f"sm{self.kSmVersion}"]
        parts.append(f"b{self.BM}x{self.BN}x{self.BK}")
        parts.append(f"c{self.CM}x{self.CN}x{self.CK}")
        parts.append(f"k{self.kSMs}")
        parts.append(f"s{self.kStages}")
        return "_".join(parts)

    @property
    def num_warps(self) -> int:
        return self.kSMs * (self.CM // self.CK) * (self.CN // self.CK)

    def to_tactic_config(self) -> Tuple[Tuple[str, int], ...]:
        """Convert to a ``Tactic.config`` tuple."""
        items = [
            ("BM", self.BM),
            ("BN", self.BN),
            ("BK", self.BK),
            ("CM", self.CM),
            ("CN", self.CN),
            ("CK", self.CK),
            ("kSMs", self.kSMs),
            ("kStages", self.kStages),
        ]
        return tuple(items)


def get_unique_compile_configs(sm: int,
                               tile_shape_configs: List[TileShape],
                               tile_shape_configs_sm90: List[TileShapeSm90],
                               cluster_shape_configs_sm90: List[ClusterShape]) -> Dict[str, Union[CutlassGemmConfig, CutlassGemmConfigSm90]]:
    """Deduplicate configs that compile to the same variant.

    Split-K is a runtime parameter, so configs that differ only in split_k
    share the same compiled variant.  Returns a dict of unique compile keys
    to representative configs (with split_k=1).
    """
    seen: Dict[str, Union[CutlassGemmConfig, CutlassGemmConfigSm90]] = {}

    if sm in [75, 80, 86, 89]:
        for cfg in tile_shape_configs:
            compile_cfg = CutlassGemmConfig(
                BM=cfg.BM, BN=cfg.BN, BK=cfg.BK, WM=cfg.WM, WN=cfg.WN, WK=cfg.WK, kStages=3, kSmVersion=sm)
            key = compile_cfg.compile_name
            if key not in seen:
                seen[key] = compile_cfg
    else:
        for tile_shape_cfg in tile_shape_configs_sm90:
            for cluster_shape_cfg in cluster_shape_configs_sm90:
                compile_cfg = CutlassGemmConfigSm90(
                    BM=tile_shape_cfg.BM,
                    BN=tile_shape_cfg.BN,
                    BK=tile_shape_cfg.BK,
                    CM=cluster_shape_cfg.CM,
                    CN=cluster_shape_cfg.CN,
                    CK=cluster_shape_cfg.CK,
                    kSMs=1,
                    kStages=3,
                    kSmVersion=sm
                )
            key = compile_cfg.compile_name
            if key not in seen:
                seen[key] = compile_cfg
    return seen


# =============================================================================
# GEMM tile configurations (SM80+ TensorOp 16x8x16)
# =============================================================================


TileShapeConfigs: List[TileShape] = [
    TileShape(BM=16, BN=128, BK=64, WM=16, WN=32, WK=64),
    TileShape(BM=32, BN=128, BK=64, WM=32, WN=32, WK=64),
    TileShape(BM=64, BN=128, BK=64, WM=32, WN=64, WK=64),
    TileShape(BM=64, BN=64, BK=128, WM=32, WN=64, WK=64),
    TileShape(BM=64, BN=128, BK=64, WM=64, WN=32, WK=64),
    TileShape(BM=128, BN=64, BK=64, WM=64, WN=32, WK=64),
    TileShape(BM=128, BN=128, BK=64, WM=64, WN=32, WK=64),
    TileShape(BM=128, BN=128, BK=64, WM=64, WN=64, WK=64),
    TileShape(BM=128, BN=128, BK=64, WM=128, WN=32, WK=64),
    TileShape(BM=128, BN=256, BK=64, WM=64, WN=64, WK=64),
    TileShape(BM=256, BN=128, BK=64, WM=64, WN=64, WK=64),
    TileShape(BM=128, BN=64, BK=128, WM=64, WN=32, WK=128),
    TileShape(BM=16, BN=256, BK=64, WM=16, WN=64, WK=64),
    TileShape(BM=16, BN=256, BK=128, WM=16, WN=64, WK=128),
]

TileShapeConfigsSm90: List[TileShapeSm90] = [
    TileShapeSm90(BM=64, BN=16, BK=128),
    TileShapeSm90(BM=64, BN=32, BK=128),
    TileShapeSm90(BM=64, BN=64, BK=128),
    TileShapeSm90(BM=64, BN=128, BK=128),
    TileShapeSm90(BM=64, BN=256, BK=128),
    TileShapeSm90(BM=64, BN=512, BK=128),
    TileShapeSm90(BM=128, BN=16, BK=128),
    TileShapeSm90(BM=128, BN=32, BK=128),
    TileShapeSm90(BM=128, BN=64, BK=128),
    TileShapeSm90(BM=128, BN=128, BK=128),
    TileShapeSm90(BM=128, BN=256, BK=128),
    # SM103
    TileShapeSm90(BM=128, BN=128, BK=768),
    TileShapeSm90(BM=128, BN=192, BK=768),
    TileShapeSm90(BM=128, BN=256, BK=768),
]

ClusterShapeConfigsSm90: List[ClusterShape] = [
    ClusterShape(CM=1, CN=1, CK=1),
    ClusterShape(CM=2, CN=1, CK=1),
    ClusterShape(CM=1, CN=2, CK=1),
    ClusterShape(CM=2, CN=2, CK=1),
    ClusterShape(CM=1, CN=4, CK=1),
    ClusterShape(CM=4, CN=2, CK=1),
    ClusterShape(CM=2, CN=4, CK=1),
    ClusterShape(CM=4, CN=4, CK=1),
    ClusterShape(CM=1, CN=8, CK=1),
    ClusterShape(CM=8, CN=1, CK=1),
    ClusterShape(CM=4, CN=1, CK=1),
]


# =============================================================================
# Helper: render all tile variants for a given template
# =============================================================================


def _render_all_variants(
    template_name: str,
    template_sm90_name: str,
    family: str,
    tile_shape_configs: List[TileShape],
    tile_shape_configs_sm90: List[TileShapeSm90],
    cluster_shape_configs_sm90: List[ClusterShape],
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
    unique_configs = get_unique_compile_configs(
        sm, tile_shape_configs, tile_shape_configs_sm90, cluster_shape_configs_sm90)
    source_paths = []

    for config_name, cfg in unique_configs.items():
        func_name = f"{family}_{config_name}"
        variant_file_name = f"{family}_sm{sm}_{config_name}"

        rendered = render_template(
            template_name if sm in [75, 80, 86, 89] else template_sm90_name,
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
        "gemm_cutlass_template_sm90.cu.jinja",
        "gemm",
        TileShapeConfigs,
        TileShapeConfigsSm90,
        ClusterShapeConfigsSm90,
        with_activation=True,
    )
    return gen_jit_spec("gemm", source_paths)


def gen_bmm_module() -> JitSpec:
    """Generate JIT spec for BMM with ALL tile variants in one module.

    Each variant exports ``bmm_{config_name}`` as a TVM-FFI function.
    """
    source_paths = _render_all_variants(
        "bmm_cutlass_template.cu.jinja",
        "bmm_cutlass_template_sm90.cu.jinja",
        "bmm",
        TileShapeConfigs,
        TileShapeConfigsSm90,
        ClusterShapeConfigsSm90,
    )
    return gen_jit_spec("bmm", source_paths)


def gen_group_gemm_module() -> JitSpec:
    """Generate JIT spec for grouped GEMM with ALL tile variants in one module.

    Each variant exports ``group_gemm_{config_name}`` as a TVM-FFI function.
    """
    source_paths = _render_all_variants(
        "group_gemm_cutlass_template.cu.jinja",
        "group_gemm_cutlass_template_sm90.cu.jinja",
        "group_gemm",
        TileShapeConfigs,
        TileShapeConfigsSm90,
        ClusterShapeConfigsSm90,
    )
    return gen_jit_spec("group_gemm", source_paths)


# =============================================================================
# Default function name helpers
# =============================================================================


def gemm_func_name(cfg: CutlassGemmConfig) -> str:
    """Return the TVM-FFI export name for a GEMM variant."""
    return f"gemm_{cfg.compile_name}"


def gemm_activation_func_name(cfg: CutlassGemmConfig) -> str:
    """Return the TVM-FFI export name for a GEMM+activation variant."""
    return f"gemm_{cfg.compile_name}_activation"


def bmm_func_name(cfg: CutlassGemmConfig) -> str:
    """Return the TVM-FFI export name for a BMM variant."""
    return f"bmm_{cfg.compile_name}"


def group_gemm_func_name(cfg: CutlassGemmConfig) -> str:
    """Return the TVM-FFI export name for a grouped GEMM variant."""
    return f"group_gemm_{cfg.compile_name}"
