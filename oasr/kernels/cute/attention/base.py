# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Abstract base for OASR fused attention forward kernels (CuteDSL).

Each per-arch backend (SM80, SM90, SM100, SM120) subclasses this and fills in:
* CTA tile sizes (``M_BLOCK``, ``N_BLOCK``)
* Smem layout / swizzle choice
* MMA atom selection
* The actual ``cute.kernel``-decorated device function

The base class only defines the contract. The dispatcher in
``oasr/jit/attention.py`` instantiates the correct subclass for the active GPU.
"""

from __future__ import annotations

from typing import Any, Optional, Type

# CuteDSL imports are deferred so this module can be imported on machines
# without a working CuteDSL install (e.g., for documentation builds).


class FmhaBase:
    """Abstract spec for a fused multi-head attention forward kernel.

    Subclasses are expected to provide:

    * ``arch`` -- integer compute capability (e.g., 120 for SM120).
    * ``__init__`` capturing the kernel-shape constexprs:
      ``head_dim``, ``dtype``, ``num_heads``, ``num_kv_heads``, ``has_bias``,
      ``paged``, ``block_size``.
    * ``can_implement(...)`` -- shape/feature feasibility check (smem cap, etc.)
    * ``__call__(self, mQ, mK, mV, mO, mBias, mCacheSeqlens, mBlockTable,
                 softmax_scale, stream)`` decorated with ``@cute.jit``,
      which configures layouts/atoms and launches the device kernel.

    Tensor-shape contract (head-first, head-major within batch):

    * ``mQ``  : ``(B, H,    T_q, D)`` query.
    * ``mK``  : ``(B, H_kv, T_k, D)`` key  (contiguous mode), or
                ``(num_blocks, block_size, H_kv, D)`` (paged mode).
    * ``mV``  : same shape as ``mK``.
    * ``mO``  : ``(B, H,    T_q, D)`` output (pre-allocated by caller).
    * ``mBias`` : optional ``(B, H, T_q, T_k_max)`` additive bias, dtype = mQ.dtype.
                  ``None`` if ``has_bias`` is False. In paged mode
                  ``T_k_max`` is the logical kv extent (typically
                  ``cache_seqlens.max() + T_q``) -- bias is a dense
                  matrix over the gathered/contiguous-T_k axis even when
                  K/V live in the paged pool.
    * ``mCacheSeqlens`` : optional ``(B,)`` int32, per-stream valid kv length.
                          ``None`` => attend over the full ``T_k``.
    * ``mBlockTable`` : ``(B, max_blocks_per_seq)`` int32 in paged mode;
                       zero-rank dummy tensor when ``paged`` is False.
    * ``softmax_scale`` : f32 scalar = 1/sqrt(D) (or whatever the caller chose).

    Math performed:

        S = (Q @ K^T) * softmax_scale          # f32 accumulator
        S += mBias                              # if mBias is not None
        S = mask_length(S, mCacheSeqlens)       # if mCacheSeqlens is not None
        P = softmax(S)                          # online (per-row max + sum)
        O = P @ V                               # f32 accumulator -> dtype on store

    GQA broadcasts: when ``num_kv_heads < num_heads``, head ``h`` of Q reads
    ``mK/mV`` head ``h // (num_heads // num_kv_heads)``. The kernel handles
    this internally; the caller does NOT need to ``repeat_interleave``.
    """

    arch: int = -1

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: Any,                       # cutlass.Float16 or cutlass.BFloat16
        num_heads: int,
        num_kv_heads: int,
        has_bias: bool = False,
        paged: bool = False,
        block_size: int = 0,
        causal: bool = False,
        window_size_left: int = -1,
        window_size_right: int = -1,
        varlen: bool = False,
    ):
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        if paged and block_size <= 0:
            raise ValueError(f"paged=True requires block_size > 0, got {block_size}")
        if paged and varlen:
            # Varlen + paged is FA's append-KV path. We do not support that
            # combination yet -- paged streaming already encodes per-stream
            # length via cache_seqlens.
            raise ValueError("paged=True is mutually exclusive with varlen=True")
        self._head_dim = head_dim
        self._dtype = dtype
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._gqa_ratio = num_heads // num_kv_heads
        self._has_bias = has_bias
        self._paged = paged
        self._block_size = block_size
        self._causal = causal
        self._window_size_left = window_size_left
        self._window_size_right = window_size_right
        self._varlen = varlen

    # --- Public introspection ------------------------------------------------

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def has_bias(self) -> bool:
        return self._has_bias

    @property
    def paged(self) -> bool:
        return self._paged

    @property
    def causal(self) -> bool:
        return self._causal

    @property
    def varlen(self) -> bool:
        return self._varlen

    # --- Subclass contract ---------------------------------------------------

    @staticmethod
    def can_implement(**kwargs) -> bool:
        """Return whether this backend can implement the given kernel shape.

        Must be overridden. Should check head_dim divisibility, smem capacity,
        thread count constraints, and any arch-specific feature gates.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "FmhaBase is abstract; instantiate a per-arch subclass."
        )


def pick_arch_cls(major: int, minor: int) -> Type[FmhaBase]:
    """Return the FmhaBase subclass for the given compute capability.

    Mapping:
      * sm_80 / sm_86 / sm_89 (Ampere + Ada) -> :class:`FmhaSm80`.
      * sm_120 (consumer Blackwell, RTX 50xx) -> :class:`FmhaSm120`, which
        is itself a thin subclass over :class:`FmhaSm80`.
      * Anything else (sm_70 / sm_75 / sm_90 / sm_100 / ...) raises --
        the caller in ``oasr.jit.attention`` falls back to PyTorch SDPA
        in ``auto`` mode.
    """
    sm = major * 10 + minor
    # Local imports keep CuteDSL out of base.py's import path.
    if sm == 120:
        from .fmha_sm120 import FmhaSm120
        return FmhaSm120
    if sm in (80, 86, 89):
        from .fmha_sm80 import FmhaSm80
        return FmhaSm80
    raise NotImplementedError(
        f"oasr CuteDSL attention has no kernel for sm{sm} yet "
        f"(supported: sm_80 / sm_86 / sm_89 / sm_120). "
        f"Set OASR_ATTN_BACKEND=auto to fall back to SDPA."
    )
