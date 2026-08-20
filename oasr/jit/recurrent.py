# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for fused recurrent kernels."""

from . import env
from .core import JitSpec, gen_jit_spec


def gen_recurrent_module() -> JitSpec:
    """Generate the LSTM/RNN JIT module."""
    return gen_jit_spec(
        "recurrent",
        [
            env.OASR_CSRC_DIR / "recurrent.cu",
            env.OASR_CSRC_DIR / "recurrent_jit_binding.cu",
        ],
    )
