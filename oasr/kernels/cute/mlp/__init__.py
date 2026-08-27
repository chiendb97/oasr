# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""CuTeDSL fused gated-MLP kernels.

One kernel for the first two thirds of a SwiGLU/GeGLU block::

    out = activation(x @ w_gateᵀ + b_gate) * (x @ w_upᵀ + b_up)
"""

from oasr.kernels.cute.mlp.gated import GatedMlpCute

__all__ = ["GatedMlpCute"]
