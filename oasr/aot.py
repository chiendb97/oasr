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
    Fused attention is the exception to "one module per family": its cells are
    keyed by ``(dtype, head_dim)`` because those change the shared-memory
    layouts, so :func:`oasr.jit.fmha.gen_fmha_modules` contributes one spec per
    cell -- and none at all on an architecture it does not serve.

    Returns:
        List of JitSpec objects for all kernel modules.
    """
    from oasr.jit.activation import gen_activation_module
    from oasr.jit.conv import (
        gen_conv2d_module,
        gen_conv_module,
        gen_cudnn_conv2d_module,
        gen_grouped_conv2d_module,
    )
    from oasr.jit.ctc_decoder import gen_ctc_decoder_module
    from oasr.jit.features import gen_features_module
    from oasr.jit.fft import gen_fft_module
    from oasr.jit.fmha import gen_fmha_modules
    from oasr.jit.gemm import (
        gen_bmm_module,
        gen_gemm_log_softmax_module,
        gen_gemm_module,
        gen_group_gemm_module,
    )
    from oasr.jit.norm import gen_norm_module
    from oasr.jit.pooling import gen_pooling_module
    from oasr.jit.recurrent import gen_recurrent_module
    from oasr.jit.softmax import gen_softmax_module
    from oasr.jit.topk import gen_topk_module

    # Fused attention is the one family whose module count depends on the
    # *shapes* shipped models use, not on the kernel: one `.so` per
    # (dtype, head_dim) cell, each holding all 12 feature variants.  It returns
    # an empty list on an architecture the C++ lane is not compiled for, which
    # is why it splices rather than appends.
    return [
        gen_activation_module(),
        gen_norm_module(),
        gen_pooling_module(),
        gen_recurrent_module(),
        gen_conv_module(),
        gen_conv2d_module(),
        gen_cudnn_conv2d_module(),
        gen_grouped_conv2d_module(),
        gen_gemm_module(),
        gen_bmm_module(),
        gen_group_gemm_module(),
        gen_gemm_log_softmax_module(),
        gen_ctc_decoder_module(),
        gen_softmax_module(),
        gen_topk_module(),
        gen_fft_module(),
        gen_features_module(),
    ] + gen_fmha_modules()


def register_default_modules() -> int:
    """Pre-compile and load all default kernel modules.

    Returns:
        Number of modules compiled.
    """
    specs = gen_all_modules()
    for spec in specs:
        spec.build_and_load()
    return len(specs)
