# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for top-k operation."""

import functools
from typing import Optional, Tuple

import torch

from oasr.api_logging import oasr_api


@functools.cache
def _get_topk_module():
    from oasr.jit.topk import gen_topk_module

    return gen_topk_module().build_and_load()


@oasr_api
def topk(
    input: torch.Tensor,
    k: int,
    out_values: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the k largest values and their indices along the last dimension.

    Selects the top-k elements along the last dimension, returning them in
    descending order. This is equivalent to ``torch.topk(input, k, dim=-1)``,
    but uses a custom CUDA kernel optimised for the vocab/channel sizes common
    in ASR (up to ~8192 elements per row).

    Args:
        input: CUDA tensor of shape ``[..., num_cols]``.
        k: Number of top elements. Must satisfy ``1 <= k <= num_cols``.
        out_values: Optional pre-allocated output for values, shape ``[..., k]``,
            same dtype as *input*.
        out_indices: Optional pre-allocated output for indices, shape ``[..., k]``,
            dtype ``torch.int32``.

    Returns:
        A ``(values, indices)`` tuple, both of shape ``[..., k]``.
        *values* has the same dtype as *input*; *indices* has dtype
        ``torch.int32``.

    Note:
        Only the last dimension is supported (``dim=-1``).
        Only largest=True is supported.
        ``num_cols`` is limited to approximately 8192 for float32 inputs
        (larger for float16/bfloat16) due to shared-memory constraints.
    """
    out_shape = input.shape[:-1] + (k,)
    if out_values is None:
        out_values = torch.empty(out_shape, device=input.device, dtype=input.dtype)
    if out_indices is None:
        out_indices = torch.empty(out_shape, device=input.device, dtype=torch.int32)

    _get_topk_module().topk(out_values, out_indices, input, k)
    return out_values, out_indices
