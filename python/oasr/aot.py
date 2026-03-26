# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""AOT (ahead-of-time) kernel registration.

Provides gen_all_modules() and register_default_modules() for pre-compiling
all OASR kernels, analogous to FlashInfer's aot.py.
"""

from typing import List


def gen_all_modules() -> List:
    """Generate JIT specs for all OASR kernel families (default configs).

    Returns:
        List of JitSpec objects for all kernel modules.
    """
    from oasr.jit.activation import gen_activation_module
    from oasr.jit.norm import gen_norm_module
    from oasr.jit.conv import gen_conv_module, gen_conv2d_module, gen_cudnn_conv2d_module
    from oasr.jit.gemm import gen_gemm_module, gen_bmm_module, gen_group_gemm_module

    return [
        gen_activation_module(),
        gen_norm_module(),
        gen_conv_module(),
        gen_conv2d_module(),
        gen_cudnn_conv2d_module(),
        gen_gemm_module(),
        gen_bmm_module(),
        gen_group_gemm_module(),
    ]


def gen_all_gemm_variants() -> List:
    """Generate JIT specs for all GEMM tile configuration variants.

    Used by the autotuner to pre-compile all candidate tile configs.
    Each unique compile config produces a separate Jinja-generated module.
    """
    from oasr.jit.gemm import (
        gen_gemm_module_variant,
        gen_bmm_module_variant,
        gen_group_gemm_module_variant,
    )
    from oasr.tune.kernel_configs import (
        GEMM_TILE_CONFIGS,
        BMM_TILE_CONFIGS,
        GROUP_GEMM_TILE_CONFIGS,
        get_unique_compile_configs,
    )

    specs = []

    for cfg in get_unique_compile_configs(GEMM_TILE_CONFIGS).values():
        specs.append(gen_gemm_module_variant(
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            stages=cfg.stages,
        ))

    for cfg in get_unique_compile_configs(BMM_TILE_CONFIGS).values():
        specs.append(gen_bmm_module_variant(
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            stages=cfg.stages,
        ))

    for cfg in get_unique_compile_configs(GROUP_GEMM_TILE_CONFIGS).values():
        specs.append(gen_group_gemm_module_variant(
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            stages=cfg.stages,
        ))

    return specs


def register_default_modules() -> int:
    """Pre-compile and load all default kernel modules.

    Returns:
        Number of modules compiled.
    """
    specs = gen_all_modules()
    for spec in specs:
        spec.build_and_load()
    return len(specs)
