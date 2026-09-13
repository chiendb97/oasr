# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""GEMM and its fused epilogues -- ``oasr/functionals/gemm.py``.

``gemm_log_softmax`` lives here rather than in a module of its own.  It is a
GEMM with a log-softmax epilogue, exactly as ``gemm_activation`` is a GEMM with
an activation epilogue, and the two were being measured by two harnesses that
shared their backends, their flags and their work model.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import bmm_flops, gemm_flops, group_gemm_flops

SUBROUTINES = [
    "gemm",
    "bmm",
    "bmm_strided",
    "group_gemm",
    "gemm_activation",
    "gemm_log_softmax",
]

DEFAULT_CONFIGS: Dict[str, list] = {
    "gemm": [
        {"M": 8000, "N": 256, "K": 256},
        {"M": 16000, "N": 256, "K": 256},
        {"M": 16000, "N": 2048, "K": 256},
        {"M": 16000, "N": 256, "K": 2048},
        {"M": 16000, "N": 512, "K": 512},
        {"M": 16000, "N": 2048, "K": 512},
        {"M": 16000, "N": 512, "K": 2048},
        {"M": 32000, "N": 256, "K": 256},
        {"M": 32000, "N": 512, "K": 512},
    ],
    "bmm": [
        {"B": 256, "M": 200, "N": 200, "K": 64},
        {"B": 512, "M": 400, "N": 400, "K": 64},
        {"B": 512, "M": 200, "N": 200, "K": 64},
        {"B": 64, "M": 200, "N": 200, "K": 64},
    ],
    # The general BMM lane (KG5): strided 4-D operands with small or unaligned
    # N and K.  Sizes are Zipformer-large's own attention products at 10 s of
    # audio -- T is the encoder frame count of one downsampling stack, so the
    # four rows are the four stacks.
    "bmm_strided": [
        {"product": p, "T": t, "heads": h, "hidden": hid, "batch": 1}
        for t, h, hid in [(496, 4, 144), (248, 4, 192), (124, 4, 384), (62, 8, 576)]
        for p in ("qk", "pos", "value", "nonlin")
    ],
    "group_gemm": [
        {"num_groups": 32, "M": 256, "N": 64, "K": 64},
        {"num_groups": 32, "M": 256, "N": 128, "K": 64},
        {"num_groups": 64, "M": 16000, "N": 256, "K": 2048},
        {"num_groups": 64, "M": 16000, "N": 512, "K": 2048},
    ],
    "gemm_activation": [
        {"M": 16000, "N": 256, "K": 256},
        {"M": 16000, "N": 512, "K": 512},
        {"M": 16000, "N": 2048, "K": 256},
    ],
    # CTC head shapes: M = effective_batch * frames, K = encoder hidden,
    # N = vocab.  CUTLASS fp16/bf16 needs N % 8 == 0, and the converter pads the
    # real vocab the same way (WeNet's 4233 -> 4240).
    "gemm_log_softmax": [
        {"M": 16 * 250, "N": 4240, "K": 256},
        {"M": 16 * 250, "N": 4240, "K": 512},
        {"M": 32 * 500, "N": 5000, "K": 512},
        {"M": 32 * 1000, "N": 8000, "K": 512},
        {"M": 64 * 250, "N": 16000, "K": 512},
        {"M": 64 * 500, "N": 32000, "K": 512},
        # Single-frame streaming step
        {"M": 1, "N": 4240, "K": 256},
        {"M": 1, "N": 4240, "K": 512},
    ],
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--M", type=int, default=None, help="M dimension (rows)")
    parser.add_argument("--N", type=int, default=None, help="N dimension")
    parser.add_argument("--K", type=int, default=None, help="K dimension")
    parser.add_argument(
        "--batch-count",
        type=int,
        default=None,
        help="Batch count (bmm) or group count (group_gemm)",
    )
    parser.add_argument(
        "--product",
        choices=["qk", "pos", "value", "nonlin"],
        default=None,
        help="Zipformer attention product (bmm_strided); --M is read as T",
    )


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    M, N, K, count = args.M, args.N, args.K, args.batch_count
    if subroutine == "bmm" and all(v is not None for v in (M, N, K)) and count is not None:
        return [{"B": count, "M": M, "N": N, "K": K}]
    if subroutine == "bmm_strided":
        if M is None:
            return DEFAULT_CONFIGS["bmm_strided"]
        # M is read as T; the product's own shape follows from it, so N and K
        # are not free parameters here.
        return [
            {
                "product": args.product or "qk",
                "T": M,
                "heads": count or 4,
                "hidden": N or 144,
                "batch": 1,
            }
        ]
    if subroutine == "group_gemm" and all(v is not None for v in (M, N, K)):
        return [{"num_groups": count or 32, "M": M, "N": N, "K": K}]
    if all(v is not None for v in (M, N, K)):
        return [{"M": M, "N": N, "K": K}]
    return DEFAULT_CONFIGS[subroutine]


def _strided_problem(product: str, T: int, heads: int, hidden: int) -> Tuple[int, int, int]:
    """``(M, N, K)`` for one Zipformer attention product."""
    if product == "qk":
        return T, T, 32
    if product == "pos":
        return T, 2 * T - 1, 4
    if product == "value":
        return T, 12, T
    if product == "nonlin":
        return T, hidden, T
    raise ValueError(f"unknown strided bmm product: {product}")


def _strided_batch(cfg: dict) -> int:
    heads = cfg["heads"] if cfg["product"] in ("qk", "pos", "value") else 1
    return heads * cfg["batch"]


def _group_problem_sizes(cfg: dict) -> List[Tuple[int, int, int]]:
    """The group's per-problem shapes.

    Seeded, so ``describe`` and ``build_fns`` agree on the row counts -- which
    is what lets the FLOP count be the sum of the problems actually solved
    rather than ``num_groups * M * N * K``, an approximation that understated
    the work by ~25% because the row counts average 1.25x M.
    """
    num_groups, M, N, K = cfg["num_groups"], cfg["M"], cfg["N"], cfg["K"]
    torch.manual_seed(0)
    low = max(16, M // 2)
    high = max(low + 1, M * 2)
    Ms = torch.randint(low=low, high=high, size=(num_groups,), device="cuda").tolist()
    return [(m, N, K) for m in Ms]


def _bmm_strided_fns(cfg: dict, dtype: torch.dtype) -> Dict[str, Callable]:
    """The general BMM lane against ``torch.matmul`` on identical operand memory.

    The arms share their buffers: ``oasr.bmm`` takes the logical ``[..., N, K]``
    operand and ``torch.matmul`` takes its transposed *view*, which is what the
    model expression built before KG5.  Neither copies, so the comparison is
    kernel against kernel.  Outputs are preallocated because at these sizes an
    allocation is a measurable share of the call.
    """
    product, T, heads, hidden, batch = (
        cfg["product"],
        cfg["T"],
        cfg["heads"],
        cfg["hidden"],
        cfg["batch"],
    )
    M, N, K = _strided_problem(product, T, heads, hidden)
    if product == "qk":
        x = torch.randn(T, batch, heads, 96, device="cuda", dtype=dtype)
        A = x[..., 0:32].permute(2, 1, 0, 3)
        B_nk = x[..., 32:64].permute(2, 1, 0, 3)
        B_torch = x[..., 32:64].permute(2, 1, 3, 0)
        out_heads = heads
    elif product == "pos":
        A = torch.randn(T, batch, heads, 4, device="cuda", dtype=dtype).permute(2, 1, 0, 3)
        pos = torch.randn(1, N, heads, 4, device="cuda", dtype=dtype)
        B_nk = pos.permute(2, 0, 1, 3)
        B_torch = pos.permute(2, 0, 3, 1)
        out_heads = heads
    else:
        out_heads = heads if product == "value" else 1
        A = torch.randn(out_heads, batch, M, K, device="cuda", dtype=dtype)
        v = torch.randn(T, batch, out_heads, N, device="cuda", dtype=dtype)
        B_nk = v.permute(2, 1, 3, 0)
        B_torch = v.permute(2, 1, 0, 3)
    out = torch.empty(out_heads, batch, M, N, device="cuda", dtype=dtype)
    # Both arms write this one buffer, which is why the refcheck in
    # benchmarks/core/driver.py clones the reference before running the other
    # arm -- without that it compared the buffer against itself and could not fail.
    return {
        "cutlass": lambda: oasr.bmm(A, B_nk, out=out),
        "torch": lambda: torch.matmul(A, B_torch, out=out),
    }


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    if subroutine == "bmm_strided":
        return _bmm_strided_fns(cfg, dtype)

    if subroutine == "bmm":
        A = torch.randn(cfg["B"], cfg["M"], cfg["K"], device="cuda", dtype=dtype)
        Bmat = torch.randn(cfg["B"], cfg["N"], cfg["K"], device="cuda", dtype=dtype)
        Bt = Bmat.transpose(1, 2).contiguous()
        return {"cutlass": lambda: oasr.bmm(A, Bmat), "torch": lambda: torch.bmm(A, Bt)}

    if subroutine == "group_gemm":
        sizes = _group_problem_sizes(cfg)
        _, N, K = sizes[0]
        Ms = [m for m, _, _ in sizes]
        A = torch.randn(sum(Ms), K, device="cuda", dtype=dtype)
        B = torch.randn(len(sizes), N, K, device="cuda", dtype=dtype)
        Bt = B.transpose(1, 2).contiguous()
        offset = torch.cumsum(
            torch.tensor(Ms, dtype=torch.int32, device="cuda"), dim=0, dtype=torch.int32
        )

        def torch_fn():
            try:
                return F.grouped_mm(A, Bt, offs=offset)
            except Exception:
                # No fused grouped matmul on this build: loop the problems.
                D = torch.zeros(sum(Ms), N, device="cuda", dtype=dtype)
                start = 0
                for i, m in enumerate(Ms):
                    D[start : start + m] = torch.matmul(A[start : start + m], Bt[i])
                    start += m
                return D

        return {"cutlass": lambda: oasr.group_gemm(A, B, offset), "torch": torch_fn}

    M, N, K = cfg["M"], cfg["N"], cfg["K"]
    if subroutine == "gemm_log_softmax":
        # Scaled down: a log-softmax over a 32k-wide vocab of unit-variance
        # logits saturates, and the comparison stops being about the kernel.
        A = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
        W = torch.randn(N, K, device="cuda", dtype=dtype) * 0.1
        bias = torch.randn(N, device="cuda", dtype=dtype) * 0.1
        return {
            "cutlass": lambda: oasr.gemm_log_softmax(A, W, bias),
            "torch": lambda: F.log_softmax(F.linear(A, W, bias), dim=-1),
        }

    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(N, K, device="cuda", dtype=dtype)
    if subroutine == "gemm_activation":
        C = torch.randn(N, device="cuda", dtype=dtype)
        return {
            "cutlass": lambda: oasr.gemm_activation(
                A, B, C, activation_type=oasr.ACTIVATION_GELU_ERF
            ),
            "torch": lambda: F.gelu(F.linear(A, B, C)),
        }
    return {"cutlass": lambda: oasr.gemm(A, B), "torch": lambda: F.linear(A, B)}


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    if subroutine == "bmm_strided":
        M, N, K = _strided_problem(cfg["product"], cfg["T"], cfg["heads"], cfg["hidden"])
        batch = _strided_batch(cfg)
        return Work(
            shape=f"{cfg['product']}({cfg['heads'] if batch > cfg['batch'] else 1}"
            f"x{cfg['batch']}, {M}, {N}, {K})",
            params=params_of(cfg),
            flops=bmm_flops(batch, M, N, K),
        )
    if subroutine == "bmm":
        return Work(
            shape=f"({cfg['B']}, {cfg['M']}, {cfg['N']}, {cfg['K']})",
            params=params_of(cfg),
            flops=bmm_flops(cfg["B"], cfg["M"], cfg["N"], cfg["K"]),
        )
    if subroutine == "group_gemm":
        return Work(
            shape=f"{cfg['num_groups']}x({cfg['M']}, {cfg['N']}, {cfg['K']})",
            params=params_of(cfg),
            flops=group_gemm_flops(_group_problem_sizes(cfg)),
        )
    return Work(
        shape=f"({cfg['M']}, {cfg['N']}, {cfg['K']})",
        params=params_of(cfg),
        flops=gemm_flops(cfg["M"], cfg["N"], cfg["K"]),
    )
