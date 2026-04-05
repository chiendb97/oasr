# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""AOT (ahead-of-time) kernel registration.

Provides gen_all_modules() and register_default_modules() for pre-compiling
all OASR kernels, analogous to FlashInfer's aot.py.

Since tile variants are now compiled into the same module as the default
(FlashInfer-style), gen_all_modules() already includes all variants.
"""

from typing import List


def gen_all_modules() -> List:
    """Generate JIT specs for all OASR kernel families.

    Each GEMM/Conv2D module already contains ALL tile variants compiled into
    a single ``.so``, so no separate ``gen_all_gemm_variants()`` is needed.

    Returns:
        List of JitSpec objects for all kernel modules.
    """
    from oasr.jit.activation import gen_activation_module
    from oasr.jit.norm import gen_norm_module
    from oasr.jit.conv import gen_conv_module, gen_conv2d_module, gen_cudnn_conv2d_module
    from oasr.jit.gemm import gen_gemm_module, gen_bmm_module, gen_group_gemm_module
    from oasr.jit.ctc_decoder import gen_ctc_decoder_module

    return [
        gen_activation_module(),
        gen_norm_module(),
        gen_conv_module(),
        gen_conv2d_module(),
        gen_cudnn_conv2d_module(),
        gen_gemm_module(),
        gen_bmm_module(),
        gen_group_gemm_module(),
        gen_ctc_decoder_module(),
    ]


def register_default_modules() -> int:
    """Pre-compile and load all default kernel modules.

    Returns:
        Number of modules compiled.
    """
    specs = gen_all_modules()
    for spec in specs:
        spec.build_and_load()
    return len(specs)
