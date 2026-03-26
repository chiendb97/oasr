# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for GEMM kernels.

Default modules use static source files (csrc/gemm.cu + csrc/gemm_jit_binding.cu).
Variant modules (for autotuning) use Jinja templates to generate source files
with baked-in tile configurations.
"""

from .core import gen_jit_spec, gen_jinja_jit_spec, _get_target_sm, JitSpec
from . import env


# =============================================================================
# Default modules (static sources, use DefaultGemmConfig for the detected SM)
# =============================================================================

def gen_gemm_module() -> JitSpec:
    """Generate JIT spec for GEMM kernels with default config."""
    return gen_jit_spec(
        "gemm",
        [
            env.OASR_CSRC_DIR / "gemm.cu",
            env.OASR_CSRC_DIR / "gemm_jit_binding.cu",
        ],
    )


def gen_bmm_module() -> JitSpec:
    """Generate JIT spec for batched GEMM kernels with default config."""
    return gen_jit_spec(
        "bmm",
        [
            env.OASR_CSRC_DIR / "bmm.cu",
            env.OASR_CSRC_DIR / "bmm_jit_binding.cu",
        ],
    )


def gen_group_gemm_module() -> JitSpec:
    """Generate JIT spec for grouped GEMM kernels with default config."""
    return gen_jit_spec(
        "group_gemm",
        [
            env.OASR_CSRC_DIR / "group_gemm.cu",
            env.OASR_CSRC_DIR / "group_gemm_jit_binding.cu",
        ],
    )


# =============================================================================
# Variant modules (Jinja-generated, for autotuning)
# =============================================================================

def gen_gemm_module_variant(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    stages: int,
) -> JitSpec:
    """Generate JIT spec for GEMM with a custom tile configuration.

    Uses Jinja template to generate a .cu file with the config baked in,
    producing a single-config instantiation for fast compilation.
    """
    sm = _get_target_sm()
    variant_name = (
        f"gemm_sm{sm}_t{tile_m}x{tile_n}x{tile_k}_w{warp_m}x{warp_n}x{warp_k}_s{stages}"
    )
    return gen_jinja_jit_spec(
        name=variant_name,
        template_name="gemm_cutlass_template.cu.jinja",
        template_vars={
            "op_name": variant_name,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "warp_m": warp_m,
            "warp_n": warp_n,
            "warp_k": warp_k,
            "stages": stages,
            "sm_version": sm,
            "with_activation": True,
        },
    )


def gen_bmm_module_variant(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    stages: int,
) -> JitSpec:
    """Generate JIT spec for BMM with a custom tile configuration."""
    sm = _get_target_sm()
    variant_name = (
        f"bmm_sm{sm}_t{tile_m}x{tile_n}x{tile_k}_w{warp_m}x{warp_n}x{warp_k}_s{stages}"
    )
    return gen_jinja_jit_spec(
        name=variant_name,
        template_name="bmm_cutlass_template.cu.jinja",
        template_vars={
            "op_name": variant_name,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "warp_m": warp_m,
            "warp_n": warp_n,
            "warp_k": warp_k,
            "stages": stages,
            "sm_version": sm,
        },
    )


def gen_group_gemm_module_variant(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    stages: int,
) -> JitSpec:
    """Generate JIT spec for grouped GEMM with a custom tile configuration."""
    sm = _get_target_sm()
    variant_name = (
        f"group_gemm_sm{sm}_t{tile_m}x{tile_n}x{tile_k}_w{warp_m}x{warp_n}x{warp_k}_s{stages}"
    )
    return gen_jinja_jit_spec(
        name=variant_name,
        template_name="group_gemm_cutlass_template.cu.jinja",
        template_vars={
            "op_name": variant_name,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "warp_m": warp_m,
            "warp_n": warp_n,
            "warp_k": warp_k,
            "stages": stages,
            "sm_version": sm,
        },
    )
