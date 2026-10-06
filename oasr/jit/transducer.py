# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for the transducer decode kernels."""

from . import env
from .core import JitSpec, gen_jit_spec


def gen_transducer_module() -> JitSpec:
    """Generate the JIT spec for the fused stateless-transducer greedy decode."""
    return gen_jit_spec(
        "transducer",
        [
            env.OASR_CSRC_DIR / "transducer.cu",
            env.OASR_CSRC_DIR / "transducer_jit_binding.cu",
        ],
    )
