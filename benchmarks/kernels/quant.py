# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-token 8-bit activation quantization -- ``oasr/functionals/quant.py``.

The first half of every w8a8 layer's forward (``include/oasr/quant/``): one
amax per row, one reciprocal, one rounding per element.  It reads fp16/bf16 and
writes int8 codes plus one fp32 scale per row, so it is a streaming kernel and
its work model is bytes, not FLOPs.

The torch arm is :func:`~oasr.functionals.quant.reference_quantize_per_token`,
the kernel's bit-exact twin, so the refcheck is exact (tolerance 0) and compares
*both* outputs.  The scale is half the contract: a kernel that wrote the right
codes under the wrong row's scale would pass a check that read only the codes.

Rows are given different magnitudes on purpose -- a per-*token* scale is the
thing under test, and unit-variance rows would all quantize alike.

``--fp8`` writes e4m3 codes instead of int8 (the same kernels, ``Q = 448``); both
arms then return the codes' bytes, which the exact refcheck compares.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List

import torch

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import dtype_size
from oasr.functionals.quant import reference_quantize_per_token

SUBROUTINES = ["quantize_per_token"]

#: The kernel takes half-precision activations only.
_SERVED = (torch.float16, torch.bfloat16)

#: Conformer widths (d_model 256 / 512, FFN 2048) at a streaming step, a
#: micro-batch and a 64-utterance offline batch.
DEFAULT_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "quantize_per_token": [
        {"M": m, "K": k, "smooth": False}
        for m, k in [
            (16, 256),
            (2048, 256),
            (16384, 256),
            (16384, 512),
            (16384, 2048),
        ]
    ]
    + [{"M": 16384, "K": 256, "smooth": True}],
}

TOLERANCES = {"quantize_per_token": (0.0, 0.0)}

#: Codes and scales, both exact.
COMPARE_ALL_OUTPUTS = frozenset({"quantize_per_token"})


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--M", type=int, default=None, help="Rows (tokens)")
    parser.add_argument("--K", type=int, default=None, help="Row width (hidden size)")
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply a SmoothQuant runtime smoother (fp32 [K]) before the amax",
    )
    parser.add_argument(
        "--fp8", action="store_true", help="Quantize to fp8 e4m3 instead of int8 (sm89+ GEMMs)"
    )


def resolve_configs(args: argparse.Namespace, subroutine: str) -> List[Dict[str, Any]]:
    M, K = getattr(args, "M", None), getattr(args, "K", None)
    smooth = bool(getattr(args, "smooth", False))
    fp8 = bool(getattr(args, "fp8", False))
    if (M is None) != (K is None):
        # A half-specified shape used to fall back to the default sweep and be
        # reported as the shape that was asked for.
        raise SystemExit("--M and --K go together for quantize_per_token")
    if M is not None:
        configs = [{"M": M, "K": K, "smooth": smooth}]
    elif smooth:
        configs = [{**cfg, "smooth": True} for cfg in DEFAULT_CONFIGS[subroutine]]
    else:
        configs = DEFAULT_CONFIGS[subroutine]
    return [{**cfg, "fp8": True} for cfg in configs] if fp8 else configs


def build_fns(
    subroutine: str, cfg: Dict[str, Any], dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    if dtype not in _SERVED:
        raise ValueError(f"quantize_per_token takes fp16/bf16 activations, not {dtype}")
    M, K = cfg["M"], cfg["K"]
    gain = torch.rand(M, 1, device="cuda", dtype=torch.float32) * 4.0 + 0.25
    x = (torch.randn(M, K, device="cuda", dtype=torch.float32) * gain).to(dtype)
    smooth_inv = torch.rand(K, device="cuda", dtype=torch.float32) + 0.5 if cfg["smooth"] else None
    # Preallocated, as a quantized layer's forward would like to: an allocation
    # is a measurable share of a call this small.
    if cfg.get("fp8"):
        fp8 = torch.float8_e4m3fn
        q8 = torch.empty(M, K, device="cuda", dtype=fp8)
        scale8 = torch.empty(M, device="cuda", dtype=torch.float32)

        def cuda_fp8():
            codes, s = oasr.quantize_per_token(x, smooth_inv, out=q8, scale=scale8)
            return codes.view(torch.uint8), s

        def torch_fp8():
            codes, s = reference_quantize_per_token(x, smooth_inv, dtype=fp8)
            return codes.view(torch.uint8), s

        return {"cuda": cuda_fp8, "torch": torch_fp8}
    q = torch.empty(M, K, device="cuda", dtype=torch.int8)
    scale = torch.empty(M, device="cuda", dtype=torch.float32)
    return {
        "cuda": lambda: oasr.quantize_per_token(x, smooth_inv, out=q, scale=scale),
        "torch": lambda: reference_quantize_per_token(x, smooth_inv),
    }


def describe(subroutine: str, cfg: Dict[str, Any], dtype: torch.dtype) -> Work:
    M, K = cfg["M"], cfg["K"]
    # Read the activation, write the int8 codes and one fp32 scale per row,
    # plus the smoother when there is one.
    nbytes = M * K * dtype_size(dtype) + M * K + 4 * M + (4 * K if cfg["smooth"] else 0)
    return Work(
        shape=f"({M}, {K})"
        + (" +smooth" if cfg["smooth"] else "")
        + (" e4m3" if cfg.get("fp8") else ""),
        params=params_of(cfg),
        bytes=nbytes,
    )
