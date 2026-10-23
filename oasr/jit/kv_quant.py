# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for the FP8 KV-cache quantize-store kernel."""

from . import env
from .core import JitSpec, gen_jit_spec


def gen_kv_quant_module() -> JitSpec:
    """Generate the JIT spec for ``kv_quant_store``."""
    return gen_jit_spec(
        "kv_quant",
        [
            env.OASR_CSRC_DIR / "kv_quant.cu",
            env.OASR_CSRC_DIR / "kv_quant_jit_binding.cu",
        ],
    )
