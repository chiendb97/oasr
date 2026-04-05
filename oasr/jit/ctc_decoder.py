# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for GPU CTC prefix beam search decoder kernel."""

from .core import gen_jit_spec, JitSpec
from . import env


def gen_ctc_decoder_module() -> JitSpec:
    """Generate JIT spec for GPU CTC prefix beam search decoder."""
    return gen_jit_spec(
        "ctc_decoder",
        [
            env.OASR_CSRC_DIR / "ctc_decoder.cu",
            env.OASR_CSRC_DIR / "ctc_decoder_jit_binding.cu",
        ],
    )
