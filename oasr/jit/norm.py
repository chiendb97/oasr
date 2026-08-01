# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for normalization kernels."""

from . import env
from .core import JitSpec, gen_jit_spec


def gen_norm_module() -> JitSpec:
    """Generate JIT spec for normalization kernels."""
    return gen_jit_spec(
        "norm",
        [
            env.OASR_CSRC_DIR / "norm.cu",
            env.OASR_CSRC_DIR / "norm_jit_binding.cu",
        ],
    )
