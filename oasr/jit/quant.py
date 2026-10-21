# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for activation quantization kernels."""

from . import env
from .core import JitSpec, gen_jit_spec


def gen_quant_module() -> JitSpec:
    """Generate JIT spec for activation quantization kernels."""
    return gen_jit_spec(
        "quant",
        [
            env.OASR_CSRC_DIR / "quant.cu",
            env.OASR_CSRC_DIR / "quant_jit_binding.cu",
        ],
    )
