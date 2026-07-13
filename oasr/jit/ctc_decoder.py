# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for GPU CTC prefix beam search decoder kernel."""

from . import env
from .core import JitSpec, gen_jit_spec


def gen_ctc_decoder_module(use_fused: bool = True) -> JitSpec:
    """Generate JIT spec for GPU CTC prefix beam search decoder.

    ``use_fused=False`` builds a separate module with the fused single-kernel
    beam-search step disabled (``-DOASR_CTC_DISABLE_FUSED``), forcing the
    legacy multi-kernel pipeline for every beam size.  Used for A/B parity
    testing and as an emergency rollback (``OASR_CTC_FUSED=0``).
    """
    sources = [
        env.OASR_CSRC_DIR / "decoder" / "ctc" / "ctc_decoder.cu",
        env.OASR_CSRC_DIR / "decoder" / "ctc" / "ctc_decoder_jit_binding.cu",
    ]
    if not use_fused:
        return gen_jit_spec(
            "ctc_decoder_legacy",
            sources,
            extra_cuda_cflags=["-DOASR_CTC_DISABLE_FUSED=1"],
        )
    return gen_jit_spec("ctc_decoder", sources)
