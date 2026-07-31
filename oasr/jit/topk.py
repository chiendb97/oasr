# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for top-k kernel."""

from . import env
from .core import JitSpec, gen_jit_spec


def gen_topk_module() -> JitSpec:
    """Generate JIT spec for top-k kernel."""
    return gen_jit_spec(
        "topk",
        [
            env.OASR_CSRC_DIR / "topk.cu",
            env.OASR_CSRC_DIR / "topk_jit_binding.cu",
        ],
    )
