# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for softmax kernel."""

from . import env
from .core import JitSpec, gen_jit_spec


def gen_softmax_module() -> JitSpec:
    """Generate JIT spec for softmax kernel."""
    return gen_jit_spec(
        "softmax",
        [
            env.OASR_CSRC_DIR / "softmax.cu",
            env.OASR_CSRC_DIR / "softmax_jit_binding.cu",
        ],
    )
