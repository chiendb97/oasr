# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""FP8 (e4m3) KV-cache quantization.

An autoregressive decoder's KV cache can be stored as ``float8_e4m3fn`` with one
fp32 scale per **(token, kv-head)** vector, chosen dynamically from that vector
when the token is written (``EngineConfig.quantization.kv_cache = "fp8_e4m3"``).
No calibration, and row-local: a token quantizes identically whatever else
shares its batch.  The scale is the smallest power of two that brings the
vector's amax inside +-448, which makes dequantization exact -- see
``include/oasr/kv_quant/kv_quant.cuh`` for the pinned numerics, mirrored bit for
bit by :func:`reference_quantize_kv`.

Reading the cache is the attention kernel's job: :func:`oasr.fmha` takes fp8
``k``/``v`` plus ``k_scale``/``v_scale`` and dequantizes inside the tile loop
(the C++ lane, ``oasr/jit/fmha.py``), so a decode step never materialises an
fp16 copy of the cache.  :class:`QuantizedKv` is how a cache hands such a pair
to :class:`oasr.layers.Attention` in place of a plain tensor.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from oasr.api_logging import oasr_api
from oasr.functionals.quant import round_to_quant

__all__ = [
    "FP8_KV_DTYPE",
    "KV_CACHE_DTYPES",
    "QuantizedKv",
    "kv_cache_bytes_per_token",
    "kv_cache_dtype",
    "quantize_kv",
    "quantize_kv_store",
    "reference_quantize_kv",
]

#: Storage dtype of a quantized KV cache.
FP8_KV_DTYPE = torch.float8_e4m3fn

#: ``QuantizationConfig.kv_cache`` values that quantize the cache, and their storage.
KV_CACHE_DTYPES = {"fp8_e4m3": FP8_KV_DTYPE}

#: Bytes of one scale (fp32, per (token, kv-head)).
_SCALE_BYTES = 4


def kv_cache_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    """The storage dtype a ``kv_cache`` setting names, or ``None`` for the model's own."""
    if name is None or name == "auto":
        return None
    try:
        return KV_CACHE_DTYPES[name]
    except KeyError:
        raise ValueError(
            f"unknown kv_cache {name!r}; expected 'auto' or one of {sorted(KV_CACHE_DTYPES)}"
        ) from None


def kv_cache_bytes_per_token(
    *, num_layers: int, n_kv_head: int, head_dim: int, dtype: torch.dtype, kv_cache: Optional[str]
) -> int:
    """K and V bytes one position costs across every layer, scales included."""
    if kv_cache_dtype(kv_cache) is None:
        itemsize = torch.empty((), dtype=dtype).element_size()
        return int(2 * num_layers * n_kv_head * head_dim * itemsize)
    return int(2 * num_layers * n_kv_head * (head_dim + _SCALE_BYTES))


@dataclass(frozen=True)
class QuantizedKv:
    """An fp8 K or V tensor and its per-(token, head) fp32 scales.

    ``scale.shape == data.shape[:-1]``: dense ``(B, H, T, D)`` / ``(B, H, T)``, or
    a paged pool ``(blocks, page, H, D)`` / ``(blocks, page, H)``.  Carries the
    few tensor methods the decoders' bookkeeping applies to a cached K/V
    (``shape``, ``index_select``) so a quantized cache moves through
    ``select`` / ``can_merge`` like a plain one.
    """

    data: torch.Tensor
    scale: torch.Tensor

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def device(self) -> torch.device:
        return self.data.device

    def size(self, dim: Optional[int] = None):
        return self.data.size() if dim is None else self.data.size(dim)

    def index_select(self, dim: int, index: torch.Tensor) -> "QuantizedKv":
        if dim < 0 or dim >= self.scale.dim():
            raise ValueError("QuantizedKv.index_select indexes a leading (non-head-dim) axis")
        return QuantizedKv(self.data.index_select(dim, index), self.scale.index_select(dim, index))

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        """``data * scale`` in *dtype* -- exact for a power-of-two scale."""
        return (self.data.float() * self.scale.unsqueeze(-1)).to(dtype)

    @staticmethod
    def cat(parts, dim: int = 0) -> "QuantizedKv":
        return QuantizedKv(
            torch.cat([p.data for p in parts], dim=dim),
            torch.cat([p.scale for p in parts], dim=dim),
        )


# ---------------------------------------------------------------------------
# The torch mirror
# ---------------------------------------------------------------------------


def _scale_exponent(amax: torch.Tensor) -> torch.Tensor:
    """``k`` with ``448 * 2^k >= amax`` minimal, clamped to ``>= -126``."""
    mantissa, exponent = torch.frexp(amax)
    k = torch.where(mantissa <= 0.875, exponent - 9, exponent - 8)
    return k.clamp(min=-126)


def _exp2_int(k: torch.Tensor) -> torch.Tensor:
    """``2^k`` built from its IEEE bits, exact for ``k`` in ``[-126, 127]``."""
    return ((k + 127) << 23).to(torch.int32).view(torch.float32)


def reference_quantize_kv(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(fp8 values, fp32 scales)`` for ``x [..., D]``, bit-identical to the kernel."""
    xf = x.float()
    k = _scale_exponent(xf.abs().amax(dim=-1)).to(torch.int32)
    scale = _exp2_int(k)
    inv = _exp2_int(-k)
    q = round_to_quant(xf * inv.unsqueeze(-1), FP8_KV_DTYPE)
    return q, scale


# ---------------------------------------------------------------------------
# Kernel path
# ---------------------------------------------------------------------------


@functools.cache
def _get_kv_quant_module():
    from oasr.jit.kv_quant import gen_kv_quant_module

    return gen_kv_quant_module().build_and_load()


def _kernel_eligible(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)


def _reference_store(
    x: torch.Tensor, out: torch.Tensor, scale: torch.Tensor, pos: Optional[torch.Tensor]
) -> None:
    B, _H, T, _D = x.shape
    q, s = reference_quantize_kv(x)
    if pos is None:
        slots = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    else:
        slots = pos.view(B, T).to(x.device, torch.long)
    rows = (
        torch.arange(B, device=x.device).unsqueeze(1).expand(B, T)
        if out.size(0) == B
        else torch.zeros(B, T, dtype=torch.long, device=x.device)
    )
    out.permute(0, 2, 1, 3)[rows, slots] = q.permute(0, 2, 1, 3).to(out.device)
    scale.permute(0, 2, 1)[rows, slots] = s.permute(0, 2, 1).to(scale.device)


@oasr_api
def quantize_kv_store(
    k: torch.Tensor,
    k_out: torch.Tensor,
    k_scale: torch.Tensor,
    v: Optional[torch.Tensor] = None,
    v_out: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    pos: Optional[torch.Tensor] = None,
) -> None:
    """Quantize ``k`` (and ``v``) into fp8 caches in place, one launch for both.

    Args:
        k, v: ``(B, H, T, D)`` fp16/bf16 (any strides with a unit last stride).
        k_out, v_out: ``float8_e4m3fn`` caches viewed ``(B or 1, H, slots, D)``;
            a leading 1 is a pool every row shares, addressed by ``pos``.
        k_scale, v_scale: fp32 ``(B or 1, H, slots)``.
        pos: int64 ``(B * T,)`` slot each new token lands in; ``None`` means
            token ``t`` lands at slot ``t`` of its own row.

    CUDA fp16/bf16 runs the kernel; anything else (CPU, fp32 -- the parity
    oracles) runs :func:`reference_quantize_kv`, which is bit-identical.
    """
    if (v is None) != (v_out is None) or (v is None) != (v_scale is None):
        raise ValueError("v, v_out and v_scale come together")
    if not _kernel_eligible(k):
        _reference_store(k, k_out, k_scale, pos)
        if v is not None:
            assert v_out is not None and v_scale is not None
            _reference_store(v, v_out, v_scale, pos)
        return
    if pos is not None and pos.dtype != torch.int64:
        pos = pos.to(torch.int64)
    _get_kv_quant_module().kv_quant_store(k_out, k_scale, v_out, v_scale, k, v, pos)


@oasr_api
def quantize_kv(x: torch.Tensor) -> QuantizedKv:
    """Quantize a whole ``(B, H, T, D)`` tensor into a fresh dense :class:`QuantizedKv`.

    For a cache written once and only read after -- a dense AED's
    cross-attention K/V.
    """
    B, H, T, D = x.shape
    data = torch.empty(B, H, T, D, dtype=FP8_KV_DTYPE, device=x.device)
    scale = torch.empty(B, H, T, dtype=torch.float32, device=x.device)
    quantize_kv_store(x, data, scale)
    return QuantizedKv(data, scale)
