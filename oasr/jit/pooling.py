# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for pooling kernels."""

from . import env
from .core import JitSpec, gen_jit_spec


def gen_pooling_module() -> JitSpec:
    """Generate the pooling JIT module (AvgPool1D + MaxPool1D)."""
    return gen_jit_spec(
        "pooling",
        [
            env.OASR_CSRC_DIR / "pooling.cu",
            env.OASR_CSRC_DIR / "pooling_jit_binding.cu",
        ],
    )
