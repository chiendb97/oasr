# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for GEMM operations."""

import functools
from typing import Optional

import torch

from oasr.api_logging import oasr_api


@functools.cache
def _get_gemm_module():
    from oasr.jit.gemm import gen_gemm_module

    return gen_gemm_module().build_and_load()


@functools.cache
def _get_bmm_module():
    from oasr.jit.gemm import gen_bmm_module

    return gen_bmm_module().build_and_load()


@functools.cache
def _get_group_gemm_module():
    from oasr.jit.gemm import gen_group_gemm_module

    return gen_group_gemm_module().build_and_load()


def _default_gemm_fn():
    from oasr.jit.gemm import GEMM_DEFAULT, gemm_func_name

    return getattr(_get_gemm_module(), gemm_func_name(GEMM_DEFAULT))


def _default_gemm_activation_fn():
    from oasr.jit.gemm import GEMM_DEFAULT, gemm_activation_func_name

    return getattr(_get_gemm_module(), gemm_activation_func_name(GEMM_DEFAULT))


def _default_bmm_fn():
    from oasr.jit.gemm import GEMM_DEFAULT, bmm_func_name

    return getattr(_get_bmm_module(), bmm_func_name(GEMM_DEFAULT))


def _default_group_gemm_fn():
    from oasr.jit.gemm import GEMM_DEFAULT, group_gemm_func_name

    return getattr(_get_group_gemm_module(), group_gemm_func_name(GEMM_DEFAULT))


@oasr_api
def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """General matrix multiplication: D = A @ B + C.

    Args:
        A: Input tensor [M, K] or [batch, M', K].
        B: Input tensor [N, K].
        C: Optional bias tensor [M, N].
        out: Optional pre-allocated output tensor.

    Returns:
        Output tensor [M, N].
    """
    K = A.shape[-1]
    N = B.shape[0]
    if out is None:
        out_shape = list(A.shape[:-1]) + [N]
        out = torch.empty(out_shape, device=A.device, dtype=A.dtype)

    from oasr.tune import is_tuning_enabled

    if is_tuning_enabled():
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        M = out.reshape(-1, N).shape[0]
        get_tuner().dispatch(
            op_key=OpKey("gemm", "gemm"),
            shape_sig=(M, N, K),
            dtype=A.dtype,
            device=A.device,
            runner_args=(out.reshape(-1, N), A.reshape(-1, K), B, C),
        )
        return out

    _default_gemm_fn()(out.reshape(-1, N), A.reshape(-1, K), B, C, 1)
    return out


@oasr_api
def bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched matrix multiplication: D[b] = A[b] @ B[b].

    Args:
        A: Input [batch_size, M, K].
        B: Input [batch_size, N, K].
        out: Optional pre-allocated output tensor.

    Returns:
        Output [batch_size, M, N].
    """
    batch_size, M, K = A.shape
    N = B.shape[1]
    if out is None:
        out = torch.empty(batch_size, M, N, device=A.device, dtype=A.dtype)

    from oasr.tune import is_tuning_enabled

    if is_tuning_enabled():
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        get_tuner().dispatch(
            op_key=OpKey("gemm", "bmm"),
            shape_sig=(batch_size, M, N, K),
            dtype=A.dtype,
            device=A.device,
            runner_args=(out, A, B),
        )
        return out

    _default_bmm_fn()(out, A, B)
    return out


@oasr_api
def group_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    offset: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Grouped GEMM with variable problem sizes.

    Args:
        A: Input tensor [L, K].
        B: Input tensor [B, N, K].
        offset: Offset tensor [B] (cumulative row counts).
        out: Optional pre-allocated output tensor.

    Returns:
        Output tensor [L, N].
    """
    L, K = A.shape
    B_count = B.shape[0]
    N = B.shape[1]
    if out is None:
        out = torch.empty(L, N, device=A.device, dtype=A.dtype)

    from oasr.tune import is_tuning_enabled

    if is_tuning_enabled():
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        get_tuner().dispatch(
            op_key=OpKey("gemm", "group_gemm"),
            shape_sig=(L, B_count, N, K),
            dtype=A.dtype,
            device=A.device,
            runner_args=(out, A, B, offset),
        )
        return out

    _default_group_gemm_fn()(out, A, B, offset)
    return out


@oasr_api
def gemm_activation(
    A: torch.Tensor,
    B: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    activation_type: int = 2,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GEMM with fused activation: D = activation(A @ B + C).

    Args:
        A: Input tensor [M, K] or [batch, M', K].
        B: Input tensor [N, K].
        C: Optional bias tensor.
        activation_type: Activation type (0=RELU, 1=GELU, 2=SWISH).
        out: Optional pre-allocated output tensor.

    Returns:
        Output tensor [M, N].
    """
    K = A.shape[-1]
    N = B.shape[0]
    if out is None:
        out_shape = list(A.shape[:-1]) + [N]
        out = torch.empty(out_shape, device=A.device, dtype=A.dtype)

    from oasr.tune import is_tuning_enabled

    if is_tuning_enabled():
        from oasr.tune import get_tuner
        from oasr.tune.autotuner import OpKey

        M = out.reshape(-1, N).shape[0]
        get_tuner().dispatch(
            op_key=OpKey("gemm", "gemm_activation"),
            shape_sig=(M, N, K),
            dtype=A.dtype,
            device=A.device,
            runner_args=(out.reshape(-1, N), A.reshape(-1, K),
                         B, C, activation_type),
        )
        return out

    _default_gemm_activation_fn()(out.reshape(-1, N),
                                   A.reshape(-1, K), B, C, activation_type, 1)
    return out
