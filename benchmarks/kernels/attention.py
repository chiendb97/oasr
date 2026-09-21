# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused multi-head attention -- ``oasr/functionals/attention.py``.

Three arms: the C++ CUTLASS/CuTe kernel (``cxx``), the CuTeDSL one (``cute``)
and PyTorch SDPA (``torch``).  The first two are the A/B that decides whether
``OASR_ATTN_BACKEND=auto`` should change which lane it prefers, so the
measurement protocol here is part of the claim, not decoration:

``INTERLEAVE = True``
    A single-order A/B on this family once read **0.876x** purely from
    allocator warmth -- the second arm inherits a warm pool.  Interleaving is
    the only way the ratio means anything.

``--layout head_split`` *by default*
    Real call sites hand over a permuted view of one packed qkv projection, not
    three fresh contiguous tensors.  Allocating contiguous inputs is exactly
    the mistake that overstated an earlier FMHA result by 2x, because it hides
    the canonical-stride copy the CuTeDSL lane has to make and the C++ lane
    does not.  ``--layout contiguous`` is still there -- the *difference*
    between the two is lever L1, measured directly.

``bytes_by_backend``
    The ``torch`` arm materialises a ``(B, H, T_q, T_k)`` mask that the fused
    arms never write.  One byte count for all three would credit SDPA with
    traffic it did not avoid.

``TOLERANCES = 2e-2``
    The driver's default is 1e-2, which is *tighter* than the parity suite's
    own bound for the same kernels -- an online softmax and a materialised one
    associate the row sum differently and legitimately differ by more than
    that at fp16.
"""

from __future__ import annotations

import argparse
import functools
from typing import Any, Callable, Dict

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
    "fmha_causal",
    "fmha_window",
    "fmha_causal_window",
    "fmha_decode",
    "fmha_headdim",
    "fmha_varlen",
    "fmha_paged",
    "fmha_paged_bias",
    "fmha_paged_decode",
]

#: Every arm is timed against every other in the same round.  See the module
#: docstring -- this family produced a 0.876x artefact without it.
INTERLEAVE = True

#: Which subroutines charge half the rectangle (`fmha_flops(causal=True)`).
CAUSAL_SUBROUTINES = frozenset({"fmha_causal", "fmha_causal_window"})

#: Looser than the driver default for the reason in the module docstring.
TOLERANCES = dict.fromkeys(SUBROUTINES, (2e-2, 2e-2))

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
    # The shape the `fmha-causal-short` routing rule was calibrated on.
    {"B": 16, "H": 4, "H_kv": 4, "T_q": 20, "T_k": 400, "D": 64},
    # Whisper's cross-attention tower: the row `fmha-unmasked` routes away.
    {"B": 1, "H": 8, "H_kv": 8, "T_q": 1500, "T_k": 1500, "D": 64},
]

#: Square, so causal actually halves the work.
_CAUSAL = [
    {"B": 1, "H": 8, "H_kv": 8, "T_q": 256, "T_k": 256, "D": 64},
    {"B": 1, "H": 8, "H_kv": 8, "T_q": 1024, "T_k": 1024, "D": 64},
    {"B": 4, "H": 8, "H_kv": 8, "T_q": 512, "T_k": 512, "D": 64},
    {"B": 16, "H": 4, "H_kv": 4, "T_q": 20, "T_k": 20, "D": 64},
    {"B": 2, "H": 16, "H_kv": 2, "T_q": 512, "T_k": 512, "D": 128},  # Qwen2 prefill
]

#: Nemotron's streaming window is 56 keys to the left of each row.
_WINDOW = [
    {"B": 1, "H": 8, "H_kv": 8, "T_q": 512, "T_k": 512, "D": 64, "w_left": 56, "w_right": 0},
    {"B": 4, "H": 8, "H_kv": 8, "T_q": 256, "T_k": 256, "D": 64, "w_left": 56, "w_right": 0},
    {"B": 1, "H": 8, "H_kv": 8, "T_q": 1024, "T_k": 1024, "D": 64, "w_left": 128, "w_right": 0},
    {"B": 2, "H": 8, "H_kv": 8, "T_q": 512, "T_k": 512, "D": 64, "w_left": 64, "w_right": 64},
]

#: `T_q == 1`: one query row against a long cache.  The split-KV shapes.
_DECODE = [
    {"B": 1, "H": 8, "H_kv": 8, "T_q": 1, "T_k": 1024, "D": 64},
    {"B": 4, "H": 8, "H_kv": 2, "T_q": 1, "T_k": 2048, "D": 128},
    {"B": 8, "H": 16, "H_kv": 2, "T_q": 1, "T_k": 4096, "D": 128},
    {"B": 1, "H": 32, "H_kv": 4, "T_q": 1, "T_k": 8192, "D": 128},
    {"B": 32, "H": 8, "H_kv": 8, "T_q": 1, "T_k": 512, "D": 64},
]

#: The head-dim sweep: `_BASE` is all D=64, which is one smem layout out of five.
_HEADDIM = [
    {"B": 2, "H": 8, "H_kv": 8, "T_q": 256, "T_k": 256, "D": d}
    for d in (32, 64, 72, 96, 128, 192, 256)
]

#: Sequence-packed segments -- the shape `packing="varlen"` produces.
_VARLEN = [
    {"B": 8, "H": 4, "H_kv": 4, "T_q": 200, "T_k": 200, "D": 64, "jitter": 0.5},
    {"B": 32, "H": 8, "H_kv": 8, "T_q": 128, "T_k": 128, "D": 64, "jitter": 0.75},
    {"B": 4, "H": 8, "H_kv": 8, "T_q": 512, "T_k": 512, "D": 64, "jitter": 0.25},
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

_PAGED_DECODE = [
    {"B": 4, "H": 8, "H_kv": 2, "T_q": 1, "T_k": 2048, "D": 128, "block_size": 32},
    {"B": 8, "H": 16, "H_kv": 2, "T_q": 1, "T_k": 1024, "D": 128, "block_size": 16},
    {"B": 1, "H": 32, "H_kv": 4, "T_q": 1, "T_k": 4096, "D": 128, "block_size": 32},
]

_CONFIG_SETS = {
    "fmha_causal": _CAUSAL,
    "fmha_causal_window": _WINDOW,
    "fmha_window": _WINDOW,
    "fmha_decode": _DECODE,
    "fmha_headdim": _HEADDIM,
    "fmha_varlen": _VARLEN,
    "fmha_paged": _PAGED,
    "fmha_paged_bias": _PAGED,
    "fmha_paged_decode": _PAGED_DECODE,
}

DEFAULT_CONFIGS: Dict[str, list] = {
    sub: [dict(c) for c in _CONFIG_SETS.get(sub, _BASE)] for sub in SUBROUTINES
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
    parser.add_argument(
        "--layout",
        choices=("head_split", "contiguous"),
        default="head_split",
        help=(
            "How q/k/v are laid out. 'head_split' (default) is a permuted view of "
            "one packed projection, which is what every real call site passes; "
            "'contiguous' allocates three fresh tensors. The difference between "
            "the two is the canonical-stride copy lever."
        ),
    )
    parser.add_argument(
        "--softmax-scale",
        type=float,
        default=None,
        help=(
            "Override 1/sqrt(D). Paraformer's SANM pre-scales q and passes 1.0, "
            "which is a different compile key on the CuTeDSL lane."
        ),
    )


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    keys = ("B", "H", "T_q", "T_k", "D")
    vals = {k: getattr(args, k) for k in keys}
    if all(v is not None for v in vals.values()):
        configs = [{**vals, "H_kv": args.H_kv or vals["H"]}]
        if subroutine.startswith("fmha_paged"):
            configs[0].setdefault("block_size", 16)
        if "window" in subroutine:
            configs[0].setdefault("w_left", 56)
            configs[0].setdefault("w_right", 0)
        if subroutine == "fmha_varlen":
            configs[0].setdefault("jitter", 0.5)
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


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def _qkv(cfg, dtype, layout: str, g):
    """q/k/v in the requested layout.

    ``head_split`` returns permuted views of one ``(B, T, 3, H, D)`` buffer.
    That is not a stylistic choice: it is the stride pattern an attention layer
    produces, and it is the one the CuTeDSL lane has to copy.
    """
    B, H, H_kv = cfg["B"], cfg["H"], cfg["H_kv"]
    T_q, T_k, D = cfg["T_q"], cfg["T_k"], cfg["D"]
    dev = "cuda"
    if layout == "contiguous" or H_kv != H or T_q != T_k:
        q = torch.randn(B, H, T_q, D, dtype=dtype, device=dev, generator=g)
        k = torch.randn(B, H_kv, T_k, D, dtype=dtype, device=dev, generator=g)
        v = torch.randn(B, H_kv, T_k, D, dtype=dtype, device=dev, generator=g)
        return q, k, v
    packed = torch.randn(B, T_q, 3, H, D, dtype=dtype, device=dev, generator=g)
    return tuple(packed[:, :, i].permute(0, 2, 1, 3) for i in range(3))


def _sdpa_mask(q, k, bias, seqlens, *, causal=False, w_left=-1, w_right=-1):
    T_q, T_k = q.size(2), k.size(2)
    masks = []
    if bias is not None:
        masks.append(bias)
    if seqlens is not None:
        arange = torch.arange(T_k, device=seqlens.device)
        keep = arange.unsqueeze(0) < seqlens.unsqueeze(1)
        pad = torch.where(keep, 0.0, float("-inf")).to(q.dtype)
        masks.append(pad.unsqueeze(1).unsqueeze(1))
    if causal or w_left >= 0 or w_right >= 0:
        rows = torch.arange(T_q, device=q.device).view(T_q, 1)
        cols = torch.arange(T_k, device=q.device).view(1, T_k)
        bad = torch.zeros(T_q, T_k, dtype=torch.bool, device=q.device)
        if causal or w_right >= 0:
            bad |= cols > rows + (0 if causal else w_right)
        if w_left >= 0:
            bad |= cols < rows - w_left
        geom = torch.zeros(1, 1, T_q, T_k, dtype=q.dtype, device=q.device)
        masks.append(geom.masked_fill_(bad.view(1, 1, T_q, T_k), float("-inf")))
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


#: Subroutines the CuTeDSL lane structurally cannot serve, so no ``cute`` arm is
#: offered for them.  Its ``local`` axis is the per-*stream*
#: ``[seqstart_k, seqlen_k)`` pair, not a per-row sliding window, and there is
#: no argument to pass one.  Offering an arm that raises is worse than offering
#: none: the driver drops the whole config, so the rows that *can* be measured
#: go missing too -- which is what happened the first time this ran and is why
#: the window families produced zero rows.
_NO_CUTE_ARM = frozenset({"fmha_window", "fmha_causal_window"})


def _fused_arms(call, *, subroutine: str = "") -> Dict[str, Callable[[], Any]]:
    """``cxx`` and ``cute`` off one call, bound per backend.

    Bound with ``functools.partial`` rather than by flipping
    ``OASR_ATTN_BACKEND``: ``set_backend_mode`` clears three compile caches, so
    an interleaved A/B driven through the global would re-invoke
    ``cutlass.cute.compile()`` on every round.
    """
    arms: Dict[str, Callable[[], Any]] = {"cxx": functools.partial(call, backend="cxx")}
    if subroutine not in _NO_CUTE_ARM:
        arms["cute"] = functools.partial(call, backend="cute")
    return arms


def _dense_fns(cfg, dtype, args, *, subroutine, with_bias, with_seqlens, causal, w_left, w_right):
    B, H = cfg["B"], cfg["H"]
    T_q, T_k, D = cfg["T_q"], cfg["T_k"], cfg["D"]
    g = torch.Generator(device="cuda").manual_seed(0)
    # `_qkv` reads `H_kv` off the config itself -- a GQA shape cannot use the
    # head-split view, because its k/v have fewer heads than q.
    q, k, v = _qkv(cfg, dtype, getattr(args, "layout", "head_split"), g)
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
    scale = getattr(args, "softmax_scale", None) or 1.0 / (D**0.5)
    out = torch.empty(B, H, T_q, D, dtype=dtype, device="cuda")
    k_e, v_e = _expand_kv(k, v, H)
    mask = _sdpa_mask(q, k, bias, seqlens, causal=causal, w_left=w_left, w_right=w_right)

    def call(*, backend):
        return oasr.fmha(
            q,
            k,
            v,
            softmax_scale=scale,
            attn_bias=bias,
            cache_seqlens=seqlens,
            causal=causal,
            window_left=w_left,
            window_right=w_right,
            out=out,
            backend=backend,
        )

    fns = _fused_arms(call, subroutine=subroutine)
    fns["torch"] = lambda: F.scaled_dot_product_attention(q, k_e, v_e, attn_mask=mask, scale=scale)
    return fns


def _varlen_fns(cfg, dtype, args):
    """One pack of ``B`` segments whose lengths vary by ``jitter``.

    Equal-length segments are the case where a per-segment stride bug cannot
    show, so the default jitter is non-zero.
    """
    B, H, H_kv = cfg["B"], cfg["H"], cfg["H_kv"]
    T, D = cfg["T_q"], cfg["D"]
    jitter = float(cfg.get("jitter", 0.5))
    g = torch.Generator(device="cuda").manual_seed(0)
    lens = [max(1, int(T * (1.0 - jitter * i / max(1, B - 1)))) for i in range(B)]
    total = sum(lens)
    q = torch.randn(total, H, D, dtype=dtype, device="cuda", generator=g)
    k = torch.randn(total, H_kv, D, dtype=dtype, device="cuda", generator=g)
    v = torch.randn(total, H_kv, D, dtype=dtype, device="cuda", generator=g)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")
    scale = getattr(args, "softmax_scale", None) or 1.0 / (D**0.5)
    out = torch.empty_like(q)

    def call(*, backend):
        return oasr.functionals.attention.fmha_varlen(
            q,
            k,
            v,
            softmax_scale=scale,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=max(lens),
            max_seqlen_k=max(lens),
            out=out,
            backend=backend,
        )

    fns = _fused_arms(call)
    fns["torch"] = functools.partial(call, backend="sdpa")
    return fns


def _paged_fns(cfg, dtype, args, *, with_bias):
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
    scale = getattr(args, "softmax_scale", None) or 1.0 / (D**0.5)
    out = torch.empty_like(q)

    # Reference: gather the pages, then SDPA -- the fallback path's shape.
    idx = block_table.long()
    k_full = k_pool[idx].reshape(B, -1, H_kv, D).permute(0, 2, 1, 3)
    v_full = v_pool[idx].reshape(B, -1, H_kv, D).permute(0, 2, 1, 3)
    k_full, v_full = _expand_kv(k_full, v_full, H)
    mask = _sdpa_mask(q, k_full, bias, seqlens)

    def call(*, backend):
        return oasr.fmha(
            q,
            k_pool,
            v_pool,
            softmax_scale=scale,
            attn_bias=bias,
            cache_seqlens=seqlens,
            block_table=block_table,
            out=out,
            backend=backend,
        )

    fns = _fused_arms(call)
    fns["torch"] = lambda: F.scaled_dot_product_attention(
        q, k_full, v_full, attn_mask=mask, scale=scale
    )
    return fns


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    if subroutine == "fmha_varlen":
        return _varlen_fns(cfg, dtype, args)
    if subroutine.startswith("fmha_paged"):
        return _paged_fns(cfg, dtype, args, with_bias=subroutine.endswith("_bias"))
    causal = subroutine == "fmha_causal"
    w_left = int(cfg.get("w_left", -1)) if "window" in subroutine else -1
    w_right = int(cfg.get("w_right", -1)) if "window" in subroutine else -1
    return _dense_fns(
        cfg,
        dtype,
        args,
        subroutine=subroutine,
        with_bias="bias" in subroutine,
        with_seqlens="seqlens" in subroutine,
        causal=causal,
        w_left=w_left,
        w_right=w_right,
    )


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    shape = (
        f"(B={cfg['B']}, H={cfg['H']}, H_kv={cfg['H_kv']}, "
        f"T_q={cfg['T_q']}, T_k={cfg['T_k']}, D={cfg['D']})"
    )
    causal = subroutine in CAUSAL_SUBROUTINES
    esize = torch.finfo(dtype).bits // 8
    # The `torch` arm materialises a (B, H, T_q, T_k) mask wherever there is
    # anything to mask; the fused arms express the same restriction as three
    # integers.  Charging one number to both would credit SDPA with traffic it
    # did not avoid.
    masked = subroutine not in ("fmha_offline", "fmha_headdim", "fmha_varlen")
    mask_bytes = cfg["B"] * cfg["H"] * cfg["T_q"] * cfg["T_k"] * esize if masked else 0
    qkv_bytes = (
        cfg["B"] * cfg["H"] * cfg["T_q"] * cfg["D"] * 2
        + 2 * cfg["B"] * cfg["H_kv"] * cfg["T_k"] * cfg["D"]
    ) * esize
    return Work(
        shape=shape,
        params=params_of(cfg),
        flops=fmha_flops(cfg["B"], cfg["H"], cfg["T_q"], cfg["T_k"], cfg["D"], causal=causal),
        bytes=qkv_bytes,
        bytes_by_backend={
            "cxx": qkv_bytes,
            "cute": qkv_bytes,
            "torch": qkv_bytes + 2 * mask_bytes,  # written, then read back
        },
    )
