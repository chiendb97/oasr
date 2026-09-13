# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused multi-head attention -- ``oasr/functionals/attention.py``.

The CuTeDSL kernel against PyTorch SDPA, across the four mask combinations
(none / bias / seqlens / both) and the two paged variants.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import fmha_flops

SUBROUTINES = [
    "fmha_offline",
    "fmha_bias",
    "fmha_seqlens",
    "fmha_bias_seqlens",
    "fmha_paged",
    "fmha_paged_bias",
]

#: Streaming-chunk shapes (Conformer), then offline shapes that put the online
#: softmax on its multi-tile path.
_BASE = [
    {"B": 1, "H": 4, "H_kv": 4, "T_q": 8, "T_k": 16, "D": 64},
    {"B": 4, "H": 4, "H_kv": 4, "T_q": 8, "T_k": 64, "D": 64},
    {"B": 1, "H": 4, "H_kv": 4, "T_q": 16, "T_k": 32, "D": 64},
    {"B": 2, "H": 8, "H_kv": 8, "T_q": 8, "T_k": 128, "D": 64},
    {"B": 2, "H": 8, "H_kv": 1, "T_q": 8, "T_k": 64, "D": 64},  # MQA
    {"B": 2, "H": 8, "H_kv": 2, "T_q": 8, "T_k": 64, "D": 64},  # GQA
    {"B": 1, "H": 4, "H_kv": 4, "T_q": 64, "T_k": 256, "D": 64},
    {"B": 1, "H": 8, "H_kv": 8, "T_q": 128, "T_k": 512, "D": 64},
    {"B": 1, "H": 8, "H_kv": 8, "T_q": 256, "T_k": 1024, "D": 64},
    {"B": 4, "H": 8, "H_kv": 8, "T_q": 256, "T_k": 256, "D": 64},
]

#: K/V in a (num_blocks, block_size, H_kv, D) pool behind per-stream block
#: tables.  head_dim must be a multiple of 32 for the paged kernel.
_PAGED = [
    {"B": 1, "H": 4, "H_kv": 4, "T_q": 8, "T_k": 64, "D": 64, "block_size": 16},
    {"B": 4, "H": 4, "H_kv": 4, "T_q": 8, "T_k": 64, "D": 64, "block_size": 16},
    {"B": 2, "H": 8, "H_kv": 2, "T_q": 16, "T_k": 256, "D": 64, "block_size": 16},
    {"B": 1, "H": 8, "H_kv": 8, "T_q": 64, "T_k": 512, "D": 64, "block_size": 16},
    {"B": 1, "H": 8, "H_kv": 8, "T_q": 128, "T_k": 1024, "D": 64, "block_size": 32},
    {"B": 4, "H": 8, "H_kv": 2, "T_q": 32, "T_k": 512, "D": 64, "block_size": 16},
]

DEFAULT_CONFIGS: Dict[str, list] = {
    sub: [dict(c) for c in (_PAGED if sub.startswith("fmha_paged") else _BASE)]
    for sub in SUBROUTINES
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--B", type=int, default=None, help="Batch size")
    parser.add_argument("--H", type=int, default=None, help="Query heads")
    parser.add_argument(
        "--H_kv", type=int, default=None, help="K/V heads (defaults to H; H %% H_kv must be 0)"
    )
    parser.add_argument("--T_q", type=int, default=None, help="Query length")
    parser.add_argument("--T_k", type=int, default=None, help="Key/value length")
    parser.add_argument("--D", type=int, default=None, help="Head dimension")


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    keys = ("B", "H", "T_q", "T_k", "D")
    vals = {k: getattr(args, k) for k in keys}
    if all(v is not None for v in vals.values()):
        configs = [{**vals, "H_kv": args.H_kv or vals["H"]}]
        if subroutine.startswith("fmha_paged"):
            configs[0].setdefault("block_size", 16)
    else:
        configs = [dict(c) for c in DEFAULT_CONFIGS[subroutine]]

    if subroutine.startswith("fmha_paged"):
        # Round T_k up to a whole number of blocks here rather than inside the
        # setup: the FLOP count must be the padded length the kernel actually
        # walks, and describe() sees the config, not the tensors.
        for cfg in configs:
            block = int(cfg.get("block_size", 16))
            cfg["T_k"] = ((cfg["T_k"] + block - 1) // block) * block
    return configs


def _sdpa_mask(q, k, bias, seqlens) -> Optional[torch.Tensor]:
    masks = []
    if bias is not None:
        masks.append(bias)
    if seqlens is not None:
        arange = torch.arange(k.size(2), device=seqlens.device)
        keep = arange.unsqueeze(0) < seqlens.unsqueeze(1)
        pad = torch.where(keep, 0.0, float("-inf")).to(q.dtype)
        masks.append(pad.unsqueeze(1).unsqueeze(1))
    if not masks:
        return None
    total = masks[0]
    for m in masks[1:]:
        total = total + m
    return total


def _expand_kv(k, v, heads):
    if k.size(1) == heads:
        return k, v
    repeat = heads // k.size(1)
    return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)


def _dense_fns(cfg, dtype, with_bias, with_seqlens):
    B, H, H_kv = cfg["B"], cfg["H"], cfg["H_kv"]
    T_q, T_k, D = cfg["T_q"], cfg["T_k"], cfg["D"]
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, H, T_q, D, dtype=dtype, device="cuda", generator=g)
    k = torch.randn(B, H_kv, T_k, D, dtype=dtype, device="cuda", generator=g)
    v = torch.randn(B, H_kv, T_k, D, dtype=dtype, device="cuda", generator=g)
    bias = (
        torch.randn(B, H, T_q, T_k, dtype=dtype, device="cuda", generator=g) * 0.1
        if with_bias
        else None
    )
    seqlens = None
    if with_seqlens:
        # Half short, half full: exercises the per-stream length mask.
        base = max(1, T_k // 4)
        seqlens = torch.tensor(
            [base if i < B // 2 else T_k for i in range(B)],
            dtype=torch.int32,
            device="cuda",
        )
    scale = 1.0 / (D**0.5)
    out = torch.empty_like(q)
    k_e, v_e = _expand_kv(k, v, H)
    mask = _sdpa_mask(q, k, bias, seqlens)

    return {
        "cutlass": lambda: oasr.fmha(
            q, k, v, softmax_scale=scale, attn_bias=bias, cache_seqlens=seqlens, out=out
        ),
        "torch": lambda: F.scaled_dot_product_attention(q, k_e, v_e, attn_mask=mask, scale=scale),
    }


def _paged_fns(cfg, dtype, with_bias):
    B, H, H_kv = cfg["B"], cfg["H"], cfg["H_kv"]
    T_q, T_k, D = cfg["T_q"], cfg["T_k"], cfg["D"]
    block = int(cfg["block_size"])
    blocks_per_seq = T_k // block

    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, H, T_q, D, dtype=dtype, device="cuda", generator=g)
    pool_blocks = max(B * blocks_per_seq + 4, 32)
    k_pool = torch.randn(pool_blocks, block, H_kv, D, dtype=dtype, device="cuda", generator=g)
    v_pool = torch.randn(pool_blocks, block, H_kv, D, dtype=dtype, device="cuda", generator=g)
    ids = torch.randperm(pool_blocks)[: B * blocks_per_seq]
    block_table = ids.reshape(B, blocks_per_seq).to(dtype=torch.int32, device="cuda")
    # Vary the lengths across streams so the per-stream mask is exercised.
    seqlens = torch.tensor(
        [max(1, T_k - 8 - 2 * b) for b in range(B)], dtype=torch.int32, device="cuda"
    )
    bias = (
        torch.randn(B, H, T_q, T_k, dtype=dtype, device="cuda", generator=g) * 0.1
        if with_bias
        else None
    )
    scale = 1.0 / (D**0.5)
    out = torch.empty_like(q)

    # Reference: gather the pages, then SDPA -- the fallback path's shape.
    idx = block_table.long()
    k_full = k_pool[idx].reshape(B, -1, H_kv, D).permute(0, 2, 1, 3)
    v_full = v_pool[idx].reshape(B, -1, H_kv, D).permute(0, 2, 1, 3)
    k_full, v_full = _expand_kv(k_full, v_full, H)
    mask = _sdpa_mask(q, k_full, bias, seqlens)

    return {
        "cutlass": lambda: oasr.fmha(
            q,
            k_pool,
            v_pool,
            softmax_scale=scale,
            attn_bias=bias,
            cache_seqlens=seqlens,
            block_table=block_table,
            out=out,
        ),
        "torch": lambda: F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=mask, scale=scale
        ),
    }


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    if subroutine.startswith("fmha_paged"):
        return _paged_fns(cfg, dtype, with_bias=subroutine.endswith("_bias"))
    return _dense_fns(
        cfg,
        dtype,
        with_bias="bias" in subroutine,
        with_seqlens="seqlens" in subroutine,
    )


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    shape = (
        f"(B={cfg['B']}, H={cfg['H']}, H_kv={cfg['H_kv']}, "
        f"T_q={cfg['T_q']}, T_k={cfg['T_k']}, D={cfg['D']})"
    )
    return Work(
        shape=shape,
        params=params_of(cfg),
        # Not causal: these kernels attend over the full key extent, and a
        # length mask shortens the rows without halving the rectangle.
        flops=fmha_flops(cfg["B"], cfg["H"], cfg["T_q"], cfg["T_k"], cfg["D"]),
    )
