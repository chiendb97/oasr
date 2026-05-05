# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for the FFT kernel family."""

from .core import gen_jit_spec, JitSpec
from . import env


def gen_fft_module() -> JitSpec:
    """Generate JIT spec for the rfft / rfft_power kernels."""
    return gen_jit_spec(
        "fft",
        [
            env.OASR_CSRC_DIR / "fft.cu",
            env.OASR_CSRC_DIR / "fft_jit_binding.cu",
        ],
    )
