# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for the FBANK / MFCC feature-extraction kernels."""

from .core import gen_jit_spec, JitSpec
from . import env


def gen_features_module() -> JitSpec:
    """Generate JIT spec for fbank_preprocess / mel_log / dct_lifter."""
    return gen_jit_spec(
        "features",
        [
            env.OASR_CSRC_DIR / "features.cu",
            env.OASR_CSRC_DIR / "features_jit_binding.cu",
        ],
    )
