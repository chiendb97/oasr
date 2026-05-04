# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional API for batched real-to-complex FFT."""

import functools
from typing import Optional

import torch

from oasr.api_logging import oasr_api


@functools.cache
def _get_fft_module():
    from oasr.jit.fft import gen_fft_module

    return gen_fft_module().build_and_load()


def _validate_n_fft(n_fft: int) -> None:
    if n_fft < 8 or n_fft > 2048 or (n_fft & (n_fft - 1)) != 0:
        raise ValueError(
            f"n_fft must be a power of two in [8, 2048], got {n_fft}"
        )


def _resolve_n_fft(input: torch.Tensor, n: Optional[int]) -> int:
    if n is None:
        return int(input.shape[-1])
    return int(n)


def _prepare_input(input: torch.Tensor, n_fft: int) -> torch.Tensor:
    """Pad / truncate the last dim of *input* to length *n_fft* (matches torch.fft.rfft)."""
    last = input.shape[-1]
    if last == n_fft:
        return input.contiguous()
    if last < n_fft:
        return torch.nn.functional.pad(input, (0, n_fft - last)).contiguous()
    return input[..., :n_fft].contiguous()


@oasr_api
def rfft(
    input: torch.Tensor,
    n: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched real-to-complex FFT along the last dimension.

    Mirrors :func:`torch.fft.rfft` with ``norm='backward'`` for real, float32
    inputs whose FFT length is a power of two in [8, 2048].

    Args:
        input: Real tensor of shape ``(..., L)`` on CUDA, dtype float32.
        n: FFT length. If ``L != n``, the input is zero-padded or truncated
            to ``n`` along the last dimension. Defaults to ``L``.
        out: Optional pre-allocated complex64 output of shape ``(..., n//2+1)``.

    Returns:
        Complex64 tensor of shape ``(..., n//2+1)``.
    """
    if input.dtype != torch.float32:
        raise ValueError(f"rfft requires float32 input, got {input.dtype}")

    n_fft = _resolve_n_fft(input, n)
    _validate_n_fft(n_fft)

    x = _prepare_input(input, n_fft)
    out_shape = x.shape[:-1] + (n_fft // 2 + 1,)

    if out is None:
        out = torch.empty(out_shape, device=x.device, dtype=torch.complex64)
    elif out.shape != out_shape or out.dtype != torch.complex64:
        raise ValueError(
            f"out must have shape {tuple(out_shape)} and dtype complex64, "
            f"got shape {tuple(out.shape)} dtype {out.dtype}"
        )

    out_real = torch.view_as_real(out)  # (..., n//2+1, 2) float32 view
    _get_fft_module().rfft(out_real, x, n_fft)
    return out


@oasr_api
def rfft_power(
    input: torch.Tensor,
    n: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched real-FFT power spectrum ``|X[k]|^2`` along the last dimension.

    Equivalent to ``torch.fft.rfft(input, n=n).abs().pow(2)`` but fused into a
    single kernel -- the natural front-end of the FBANK / MFCC pipelines.

    Args:
        input: Real tensor of shape ``(..., L)`` on CUDA, dtype float32.
        n: FFT length (power of two in [8, 2048]). Defaults to ``L``.
        out: Optional pre-allocated float32 output of shape ``(..., n//2+1)``.

    Returns:
        Float32 tensor of shape ``(..., n//2+1)`` holding the power spectrum.
    """
    if input.dtype != torch.float32:
        raise ValueError(f"rfft_power requires float32 input, got {input.dtype}")

    n_fft = _resolve_n_fft(input, n)
    _validate_n_fft(n_fft)

    x = _prepare_input(input, n_fft)
    out_shape = x.shape[:-1] + (n_fft // 2 + 1,)

    if out is None:
        out = torch.empty(out_shape, device=x.device, dtype=torch.float32)
    elif out.shape != out_shape or out.dtype != torch.float32:
        raise ValueError(
            f"out must have shape {tuple(out_shape)} and dtype float32, "
            f"got shape {tuple(out.shape)} dtype {out.dtype}"
        )

    _get_fft_module().rfft_power(out, x, n_fft)
    return out
