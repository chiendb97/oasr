# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for GEMM kernels."""

from typing import List, Optional

from .core import gen_jit_spec, JitSpec
from . import env


def gen_gemm_module() -> JitSpec:
    """Generate JIT spec for GEMM kernels."""
    return gen_jit_spec(
        "gemm",
        [
            env.OASR_CSRC_DIR / "gemm.cu",
            env.OASR_CSRC_DIR / "gemm_jit_binding.cu",
        ],
    )


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

    Each unique set of tile parameters produces a separate compiled module,
    allowing the autotuner to profile different configurations.
    """
    variant_name = f"gemm_t{tile_m}x{tile_n}x{tile_k}_w{warp_m}x{warp_n}x{warp_k}_s{stages}"
    tile_flags = [
        f"-DOASR_GEMM_TILE_M={tile_m}",
        f"-DOASR_GEMM_TILE_N={tile_n}",
        f"-DOASR_GEMM_TILE_K={tile_k}",
        f"-DOASR_GEMM_WARP_M={warp_m}",
        f"-DOASR_GEMM_WARP_N={warp_n}",
        f"-DOASR_GEMM_WARP_K={warp_k}",
        f"-DOASR_GEMM_STAGES={stages}",
    ]
    return gen_jit_spec(
        variant_name,
        [
            env.OASR_CSRC_DIR / "gemm.cu",
            env.OASR_CSRC_DIR / "gemm_jit_binding.cu",
        ],
        extra_cuda_cflags=tile_flags,
    )


def gen_bmm_module() -> JitSpec:
    """Generate JIT spec for batched GEMM kernels."""
    return gen_jit_spec(
        "bmm",
        [
            env.OASR_CSRC_DIR / "bmm.cu",
            env.OASR_CSRC_DIR / "bmm_jit_binding.cu",
        ],
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
    variant_name = f"bmm_t{tile_m}x{tile_n}x{tile_k}_w{warp_m}x{warp_n}x{warp_k}_s{stages}"
    tile_flags = [
        f"-DOASR_GEMM_TILE_M={tile_m}",
        f"-DOASR_GEMM_TILE_N={tile_n}",
        f"-DOASR_GEMM_TILE_K={tile_k}",
        f"-DOASR_GEMM_WARP_M={warp_m}",
        f"-DOASR_GEMM_WARP_N={warp_n}",
        f"-DOASR_GEMM_WARP_K={warp_k}",
        f"-DOASR_GEMM_STAGES={stages}",
    ]
    return gen_jit_spec(
        variant_name,
        [
            env.OASR_CSRC_DIR / "bmm.cu",
            env.OASR_CSRC_DIR / "bmm_jit_binding.cu",
        ],
        extra_cuda_cflags=tile_flags,
    )


def gen_group_gemm_module() -> JitSpec:
    """Generate JIT spec for grouped GEMM kernels."""
    return gen_jit_spec(
        "group_gemm",
        [
            env.OASR_CSRC_DIR / "group_gemm.cu",
            env.OASR_CSRC_DIR / "group_gemm_jit_binding.cu",
        ],
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
    variant_name = (
        f"group_gemm_t{tile_m}x{tile_n}x{tile_k}_w{warp_m}x{warp_n}x{warp_k}_s{stages}"
    )
    tile_flags = [
        f"-DOASR_GEMM_TILE_M={tile_m}",
        f"-DOASR_GEMM_TILE_N={tile_n}",
        f"-DOASR_GEMM_TILE_K={tile_k}",
        f"-DOASR_GEMM_WARP_M={warp_m}",
        f"-DOASR_GEMM_WARP_N={warp_n}",
        f"-DOASR_GEMM_WARP_K={warp_k}",
        f"-DOASR_GEMM_STAGES={stages}",
    ]
    return gen_jit_spec(
        variant_name,
        [
            env.OASR_CSRC_DIR / "group_gemm.cu",
            env.OASR_CSRC_DIR / "group_gemm_jit_binding.cu",
        ],
        extra_cuda_cflags=tile_flags,
    )
