# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused gated MLP (SwiGLU / GeGLU) against the two-GEMM path it replaces.

Three arms, all computing ``act(x @ w_gateᵀ) * (x @ w_upᵀ)``:

``cute``
    :func:`oasr.gated_mlp` -- one dual-B tensor-core GEMM, no intermediate.
``oasr``
    ``oasr.gemm_activation`` for the gate (activation folded into *that*
    epilogue), ``oasr.gemm`` for the up, then one elementwise multiply.  This is
    the code the fusion literally removes -- but only above the GEMM row floor.
``torch``
    ``F.linear`` twice plus the activation plus a multiply.  **Inside the fused
    kernel's band this is the honest baseline**, and quoting only ``oasr`` would
    overstate the win by ~50%: ``oasr.layers._backend.GEMM_MIN_ROWS`` sends a
    projection with fewer than 128 rows to ``F.linear`` on its own measured
    policy, so a decode-shaped ``GatedMLP`` never ran ``oasr.gemm`` in the first
    place.  Both columns are reported; which one is the baseline depends on
    whether ``M`` is over the row floor.
"""

from __future__ import annotations

import argparse
import statistics
from typing import Any, Callable

import torch
import torch.nn.functional as F

import oasr
from benchmarks.routines.bench_utils import (
    BenchResult,
    OutputWriter,
    bench_fn,
    compute_bandwidth_tb_s,
    dtype_size,
    parse_dtype,
)

SUBROUTINES = ["gated_mlp"]

#: ``(rows, hidden, intermediate)``.  The first block is the shipped
#: Qwen2-Audio-7B LM (``4096 -> 11008``, 32 layers) walking a decoder's batch
#: from one stream to a full pool and then past the band into prefill.  The
#: other two are different LLM widths, and they are here for a specific reason:
#: the fused kernel's tile is chosen by ``N``, not only by ``M``, and 11008 and
#: 18944 want *different* rings on the same part.  A sweep at one width would
#: have shipped a table that loses 10% at another.
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

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "gated_mlp": [
        {"rows": rows, "hidden": hidden, "intermediate": intermediate}
        for rows, hidden, intermediate in _SHAPES
    ]
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rows", type=int, default=None, help="M — tokens in the batch")
    parser.add_argument("--hidden", type=int, default=None, help="K — model width")
    parser.add_argument("--intermediate", type=int, default=None, help="N — FFN width")
    parser.add_argument(
        "--activation", default="silu", help="gate activation (silu / relu / gelu / gelu_tanh)"
    )


def _resolve_configs(args: argparse.Namespace, subroutine: str) -> list[dict[str, Any]]:
    values = (
        getattr(args, "rows", None),
        getattr(args, "hidden", None),
        getattr(args, "intermediate", None),
    )
    if all(value is not None for value in values):
        return [{"rows": values[0], "hidden": values[1], "intermediate": values[2]}]
    return DEFAULT_CONFIGS[subroutine]


def _shape_str(config: dict[str, Any]) -> str:
    return f"M={config['rows']},K={config['hidden']},N={config['intermediate']}"


#: One buffer, allocated once, larger than any L2 this runs on.
_FLUSH: torch.Tensor | None = None


def _flush_l2() -> None:
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(int(160e6) // 4, dtype=torch.int32, device="cuda")
    _FLUSH.zero_()


def _setup(config: dict[str, Any], dtype: torch.dtype, activation: str):
    rows, k, n = config["rows"], config["hidden"], config["intermediate"]
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

    @torch.no_grad()
    def cute_fn():
        return oasr.gated_mlp(x, w_gate, w_up, activation=activation, out=out)

    functions: dict[str, Callable] = {"torch": torch_fn, "oasr": oasr_fn}
    # ``always`` would answer yes here even above the band; ``auto`` is what the
    # layer asks, so the benchmark asks it too and simply reports no ``cute`` row
    # where the routing declines.  Run with OASR_GATED_MLP_CUTE=1 to force it.
    if oasr.gated_mlp_available(x, w_gate, activation=activation):
        functions["cute"] = cute_fn
    return functions, torch_fn


def _bytes(config: dict[str, Any], dtype: torch.dtype, fused: bool) -> int:
    """Compulsory traffic, in bytes.

    The weights are read once either way.  What separates the arms is the
    intermediate: fused writes ``M * N`` and nothing else; unfused writes gate
    and up, reads both back, and writes the product -- five passes to one.
    """
    rows, k, n = config["rows"], config["hidden"], config["intermediate"]
    e = dtype_size(dtype)
    weights = 2 * n * k * e
    a = rows * k * e
    intermediate = (1 if fused else 5) * rows * n * e
    return weights + a + intermediate


def _flops(config: dict[str, Any]) -> int:
    rows, k, n = config["rows"], config["hidden"], config["intermediate"]
    return 2 * 2 * rows * n * k


def _bench_interleaved(
    functions: dict[str, Callable],
    dry_run_iters: int,
    num_iters: int,
    use_cuda_events: bool,
) -> dict[str, tuple[float, float]]:
    """Alternate arm order across rounds, and flush L2 before every timed call."""
    names = list(functions)
    rounds = min(5, max(1, num_iters // 4))
    iterations = max(1, num_iters // rounds)
    samples: dict[str, list[float]] = {name: [] for name in names}
    for round_index in range(rounds):
        offset = round_index % len(names)
        for name in names[offset:] + names[:offset]:
            fn = functions[name]

            def flushed(fn=fn):
                _flush_l2()
                return fn()

            median_ms, _ = bench_fn(
                flushed,
                dry_run_iters=dry_run_iters if round_index == 0 else 0,
                num_iters=iterations,
                use_cuda_events=use_cuda_events,
            )
            # The flush is inside the timed region because ``bench_fn`` times a
            # callable, so subtract it: it is the same fixed cost for every arm.
            samples[name].append(median_ms - _flush_cost(use_cuda_events))
    return {
        name: (
            statistics.median(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )
        for name, values in samples.items()
    }


_FLUSH_MS: float | None = None


def _flush_cost(use_cuda_events: bool) -> float:
    """Measured once: what one L2 flush costs, so it can come back out."""
    global _FLUSH_MS
    if _FLUSH_MS is None:
        _FLUSH_MS = bench_fn(_flush_l2, dry_run_iters=5, num_iters=20, use_cuda_events=True)[0]
    return _FLUSH_MS


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", None) or "gated_mlp"
    if subroutine not in SUBROUTINES:
        raise ValueError(f"Unknown mlp subroutine: {subroutine}")
    dtype_str = getattr(args, "dtype", "float16")
    dtype = parse_dtype(dtype_str)
    activation = getattr(args, "activation", "silu")
    backends = getattr(args, "backends", None) or ["cute", "oasr", "torch"]

    for config in _resolve_configs(args, subroutine):
        functions, reference_fn = _setup(config, dtype, activation)
        shape = _shape_str(config)
        if getattr(args, "refcheck", False):
            reference = reference_fn()
            for backend in backends:
                if backend not in functions:
                    continue
                got = functions[backend]()
                max_diff = (got.float() - reference.float()).abs().max().item()
                output.write_verbose(f"{shape},backend={backend}: max_abs_diff={max_diff:.6f}")
                try:
                    torch.testing.assert_close(got, reference, rtol=6e-3, atol=6e-3)
                except AssertionError as error:
                    print(f"  [ERROR] Output mismatch for {shape}/{backend}: {error}")
                    if not getattr(args, "allow_output_mismatch", False):
                        return

        selected = {b: functions[b] for b in backends if b in functions}
        for backend in backends:
            if backend not in functions:
                print(f"  [WARNING] backend {backend!r} not available for {shape}, skipping")
        timings = _bench_interleaved(
            selected,
            dry_run_iters=getattr(args, "dry_run_iters", 5),
            num_iters=getattr(args, "num_iters", 30),
            use_cuda_events=getattr(args, "use_cuda_events", False),
        )
        for backend, (median_ms, std_ms) in timings.items():
            output.write_result(
                BenchResult(
                    routine="mlp",
                    subroutine=subroutine,
                    backend=backend,
                    shape=shape,
                    dtype=dtype_str,
                    median_ms=median_ms,
                    std_ms=std_ms,
                    tflops=_flops(config) / (median_ms * 1e-3) / 1e12,
                    bandwidth_tb_s=compute_bandwidth_tb_s(
                        _bytes(config, dtype, fused=backend == "cute"), median_ms
                    ),
                )
            )


def run_standalone(variant: str = "gated_mlp") -> None:
    if variant not in SUBROUTINES:
        raise ValueError(f"Unknown mlp variant: {variant}")
    raise SystemExit(
        "Use: python benchmarks/oasr_benchmark.py --routine mlp "
        f"--subroutine {variant} --backends cute oasr torch --refcheck"
    )
