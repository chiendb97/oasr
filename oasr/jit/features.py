# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""JIT generator for the FBANK / MFCC feature-extraction kernels."""

from . import env
from .core import JitSpec, gen_jit_spec


def gen_features_module() -> JitSpec:
    """Generate the feature framing, projection, normalization and LFR module."""
    return gen_jit_spec(
        "features",
        [
            env.OASR_CSRC_DIR / "features.cu",
            env.OASR_CSRC_DIR / "features_jit_binding.cu",
        ],
    )
