# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused gated MLP (SwiGLU / GeGLU) against the two-GEMM path it replaces.

Three arms, all computing ``act(x @ w_gate.T) * (x @ w_up.T)``:

``cute``
    :func:`oasr.gated_mlp` -- one dual-B tensor-core GEMM, no intermediate.
``oasr``
    ``oasr.gemm_activation`` for the gate (activation folded into that
    epilogue), ``oasr.gemm`` for the up, then one elementwise multiply.  This is
    the code the fusion literally removes -- but only above the GEMM row floor.
``torch``
    ``F.linear`` twice plus the activation plus a multiply.  **Inside the fused
    kernel's band this is the honest baseline**, and quoting only ``oasr`` would
    overstate the win by ~50%: ``oasr.layers._backend.GEMM_MIN_ROWS`` sends a
    projection with fewer than 128 rows to ``F.linear`` on its own measured
    policy, so a decode-shaped ``GatedMLP`` never ran ``oasr.gemm`` at all.
    Both columns are reported; which is the baseline depends on whether ``M``
    clears the row floor.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

import oasr
from benchmarks.core.driver import Work, params_of
from benchmarks.core.metrics import dtype_size

SUBROUTINES = ["gated_mlp"]
INTERLEAVE = True

#: ``(rows, hidden, intermediate)``.  The first block is the shipped
#: Qwen2-Audio-7B LM (4096 -> 11008, 32 layers) walking a decoder's batch from
#: one stream to a full pool and then past the band into prefill.  The other two
#: are different LLM widths, and they are here for a reason: the fused kernel's
#: tile is chosen by N, not only by M, and 11008 and 18944 want different rings
#: on the same part.  A sweep at one width ships a table that loses 10% at another.
_SHAPES = [
    (1, 4096, 11008),
    (8, 4096, 11008),
    (32, 4096, 11008),
    (64, 4096, 11008),
    (128, 4096, 11008),
    (1, 3584, 18944),
    (32, 3584, 18944),
    (64, 3584, 18944),
    (1, 1536, 8960),
    (32, 1536, 8960),
    (128, 1536, 8960),
]

DEFAULT_CONFIGS: Dict[str, list] = {
    "gated_mlp": [{"rows": r, "hidden": h, "intermediate": n} for r, h, n in _SHAPES]
}


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rows", type=int, default=None, help="M -- tokens in the batch")
    parser.add_argument("--hidden", type=int, default=None, help="K -- model width")
    parser.add_argument("--intermediate", type=int, default=None, help="N -- FFN width")
    parser.add_argument(
        "--activation", default="silu", help="Gate activation (silu / relu / gelu / gelu_tanh)"
    )


def resolve_configs(args: argparse.Namespace, subroutine: str) -> list:
    dims = (args.rows, args.hidden, args.intermediate)
    if all(v is not None for v in dims):
        return [{"rows": dims[0], "hidden": dims[1], "intermediate": dims[2]}]
    return DEFAULT_CONFIGS[subroutine]


def build_fns(
    subroutine: str, cfg: dict, dtype: torch.dtype, args: argparse.Namespace
) -> Dict[str, Callable[[], Any]]:
    rows, k, n = cfg["rows"], cfg["hidden"], cfg["intermediate"]
    activation = getattr(args, "activation", "silu")
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(rows, k, device="cuda", dtype=dtype, generator=g) * 0.3
    w_gate = torch.randn(n, k, device="cuda", dtype=dtype, generator=g) * k**-0.5
    w_up = torch.randn(n, k, device="cuda", dtype=dtype, generator=g) * k**-0.5
    out = torch.empty(rows, n, device="cuda", dtype=dtype)
    gate_buf = torch.empty(rows, n, device="cuda", dtype=dtype)
    up_buf = torch.empty(rows, n, device="cuda", dtype=dtype)
    act_id = oasr.get_activation_type_id(activation)
    torch_act = {"silu": F.silu, "swish": F.silu, "relu": F.relu, "gelu": F.gelu}.get(
        activation, lambda t: F.gelu(t, approximate="tanh")
    )

    @torch.no_grad()
    def torch_fn():
        return torch_act(F.linear(x, w_gate)) * F.linear(x, w_up)

    @torch.no_grad()
    def oasr_fn():
        return torch.mul(
            oasr.gemm_activation(x, w_gate, None, act_id, out=gate_buf),
            oasr.gemm(x, w_up, None, out=up_buf),
            out=out,
        )

    fns: Dict[str, Callable] = {"torch": torch_fn, "oasr": oasr_fn}
    # `auto` is what the layer asks, so the benchmark asks it too and simply
    # reports no `cute` row where the routing declines.  OASR_GATED_MLP_CUTE=1
    # forces it above the band.
    if oasr.gated_mlp_available(x, w_gate, activation=activation):
        fns["cute"] = lambda: oasr.gated_mlp(x, w_gate, w_up, activation=activation, out=out)
    return fns


def describe(subroutine: str, cfg: dict, dtype: torch.dtype) -> Work:
    rows, k, n = cfg["rows"], cfg["hidden"], cfg["intermediate"]
    elem = dtype_size(dtype)
    weights = 2 * n * k * elem
    activations = rows * k * elem
    # The weights are read once either way.  What separates the arms is the
    # intermediate: fused writes M*N and nothing else; unfused writes gate and
    # up, reads both back, and writes the product -- five passes to one.
    fused = weights + activations + rows * n * elem
    unfused = weights + activations + 5 * rows * n * elem
    return Work(
        shape=f"M={rows},K={k},N={n}",
        params=params_of(cfg),
        flops=2 * 2 * rows * n * k,
        bytes=unfused,
        bytes_by_backend={"cute": fused},
    )
