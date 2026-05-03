# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for softmax operation."""

import functools
from typing import Optional

import torch

from oasr.api_logging import oasr_api


@functools.cache
def _get_softmax_module():
    from oasr.jit.softmax import gen_softmax_module

    return gen_softmax_module().build_and_load()


@oasr_api
def softmax(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply softmax along the last dimension.

    Args:
        input: Input tensor of any shape [..., num_cols] (CUDA).
        out: Optional pre-allocated output tensor (same shape as input).

    Returns:
        Softmax probabilities with the same shape as input.
    """
    if out is None:
        out = torch.empty_like(input)
    _get_softmax_module().softmax(out, input)
    return out
