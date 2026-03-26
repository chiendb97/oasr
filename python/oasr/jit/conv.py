# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for convolution kernels (FlashInfer-style).

Conv1D uses static source files.  Conv2D tile configurations are defined here
and ALL variants are compiled into a single shared library, matching the
FlashInfer pattern used by jit/gemm.py.
"""

from typing import List

from .core import gen_jit_spec, _get_target_sm, JitSpec
from .gemm import TileConfig, get_unique_compile_configs
from . import env


# =============================================================================
# Conv2D tile configurations (SM80+ TensorOp 16x8x16)
# =============================================================================

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


# =============================================================================
# Conv1D module (static sources, no variants)
# =============================================================================


def gen_conv_module() -> JitSpec:
    """Generate JIT spec for Conv1D kernels."""
    return gen_jit_spec(
        "conv",
        [
            env.OASR_CSRC_DIR / "conv.cu",
            env.OASR_CSRC_DIR / "conv_jit_binding.cu",
        ],
    )


# =============================================================================
# Conv2D module — ALL variants compiled into ONE .so
# =============================================================================


def _render_all_conv2d_variants() -> List:
    """Render Jinja templates for all unique Conv2D tile configs."""
    from .templates import render_template
    from .cubin_loader import write_if_different

    sm = _get_target_sm()
    unique_configs = get_unique_compile_configs(CONV2D_TILE_CONFIGS)
    source_paths = []

    for config_name, cfg in unique_configs.items():
        func_name = f"conv2d_{config_name}"
        variant_file_name = f"conv2d_sm{sm}_{config_name}"

        rendered = render_template(
            "conv2d_cutlass_template.cu.jinja",
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
            with_activation=True,
        )
        gen_path = env.OASR_GEN_SRC_DIR / "conv2d" / f"{variant_file_name}.cu"
        write_if_different(gen_path, rendered)
        source_paths.append(gen_path)

    return source_paths


def gen_conv2d_module() -> JitSpec:
    """Generate JIT spec for Conv2D with ALL tile variants in one module.

    Each variant exports ``conv2d_{config_name}`` and
    ``conv2d_{config_name}_activation`` as TVM-FFI functions.
    """
    source_paths = _render_all_conv2d_variants()
    return gen_jit_spec("conv2d", source_paths)


# =============================================================================
# cuDNN Conv2D (unchanged)
# =============================================================================


def gen_cudnn_conv2d_module() -> JitSpec:
    """Generate JIT spec for cuDNN Conv2D kernels (small IC path)."""
    return gen_jit_spec(
        "cudnn_conv2d",
        [env.OASR_CSRC_DIR / "cudnn_conv2d_kernel_launcher.cu"],
        extra_ldflags=["-lcudnn"],
    )


# =============================================================================
# Function name helpers
# =============================================================================


def conv2d_func_name(cfg: TileConfig) -> str:
    """Return the TVM-FFI export name for a Conv2D variant."""
    return f"conv2d_{cfg.compile_name}"


def conv2d_activation_func_name(cfg: TileConfig) -> str:
    """Return the TVM-FFI export name for a Conv2D+activation variant."""
    return f"conv2d_{cfg.compile_name}_activation"
