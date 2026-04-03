# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for activation kernels."""

from .core import gen_jit_spec, JitSpec
from . import env


def gen_activation_module() -> JitSpec:
    """Generate JIT spec for activation kernels (GLU, Swish)."""
    return gen_jit_spec(
        "activation",
        [
            env.OASR_CSRC_DIR / "activation.cu",
            env.OASR_CSRC_DIR / "activation_jit_binding.cu",
        ],
    )
