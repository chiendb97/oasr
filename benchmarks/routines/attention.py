"""Attention family benchmark routines (fmha).

Compares the OASR fused multi-head attention CuteDSL kernel (``oasr.fmha``,
SM80 / SM86 / SM89 / SM120) against PyTorch's ``scaled_dot_product_attention``.
Four feature modes are exposed as subroutines:

* ``fmha_offline``       -- offline attention (no bias, no length mask).
* ``fmha_bias``          -- additive ``attn_bias`` (Transformer-XL rel-pos style).
* ``fmha_seqlens``       -- per-stream length mask via ``cache_seqlens``.
* ``fmha_bias_seqlens``  -- both bias and length mask combined.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F

import oasr
from benchmarks.routines.bench_utils import (
    BenchResult,
    OutputWriter,
    bench_fn,
    check_close,
    parse_dtype,
    run_main,
)

SUBROUTINES = [
    "fmha_offline",
    "fmha_bias",
    "fmha_seqlens",
    "fmha_bias_seqlens",
    "fmha_paged",
    "fmha_paged_bias",
]


# ---------------------------------------------------------------------------
# Default configs -- streaming-chunk shapes (Conformer-style) plus larger
# offline shapes that stress the multi-tile online softmax path.
# ---------------------------------------------------------------------------

_BASE_CONFIGS: list[dict[str, Any]] = [
    # (B, H, H_kv, T_q, T_k, D)
    # Streaming-chunk shapes (Conformer)
    {"B": 1,  "H": 4,  "H_kv": 4,  "T_q":   8,  "T_k":   16,  "D": 64},
    {"B": 4,  "H": 4,  "H_kv": 4,  "T_q":   8,  "T_k":   64,  "D": 64},
    {"B": 1,  "H": 4,  "H_kv": 4,  "T_q":  16,  "T_k":   32,  "D": 64},
    {"B": 2,  "H": 8,  "H_kv": 8,  "T_q":   8,  "T_k":  128,  "D": 64},
    {"B": 2,  "H": 8,  "H_kv": 1,  "T_q":   8,  "T_k":   64,  "D": 64},  # MQA
    {"B": 2,  "H": 8,  "H_kv": 2,  "T_q":   8,  "T_k":   64,  "D": 64},  # GQA
    # Larger / offline shapes
    {"B": 1,  "H": 4,  "H_kv": 4,  "T_q":  64,  "T_k":  256,  "D": 64},
    {"B": 1,  "H": 8,  "H_kv": 8,  "T_q": 128,  "T_k":  512,  "D": 64},
    {"B": 1,  "H": 8,  "H_kv": 8,  "T_q": 256,  "T_k": 1024,  "D": 64},
    {"B": 4,  "H": 8,  "H_kv": 8,  "T_q": 256,  "T_k":  256,  "D": 64},
]

_PAGED_CONFIGS: list[dict[str, Any]] = [
    # (B, H, H_kv, T_q, T_k, D, block_size) -- T_k is padded to a multiple
    # of block_size at setup time. head_dim must be a multiple of 32 for
    # the paged kernel.
    {"B": 1,  "H": 4,  "H_kv": 4,  "T_q":   8,  "T_k":   64,  "D": 64, "block_size": 16},
    {"B": 4,  "H": 4,  "H_kv": 4,  "T_q":   8,  "T_k":   64,  "D": 64, "block_size": 16},
    {"B": 2,  "H": 8,  "H_kv": 2,  "T_q":  16,  "T_k":  256,  "D": 64, "block_size": 16},  # GQA streaming
    {"B": 1,  "H": 8,  "H_kv": 8,  "T_q":  64,  "T_k":  512,  "D": 64, "block_size": 16},
    {"B": 1,  "H": 8,  "H_kv": 8,  "T_q": 128,  "T_k": 1024,  "D": 64, "block_size": 32},
    {"B": 4,  "H": 8,  "H_kv": 2,  "T_q":  32,  "T_k":  512,  "D": 64, "block_size": 16},
]


DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    sub: (list(_PAGED_CONFIGS) if sub.startswith("fmha_paged") else list(_BASE_CONFIGS))
    for sub in SUBROUTINES
}

PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "fmha_offline": _BASE_CONFIGS[8],   # (1, 8, 8, 256, 1024, 64)
    "fmha_bias": _BASE_CONFIGS[8],
    "fmha_seqlens": _BASE_CONFIGS[8],
    "fmha_bias_seqlens": _BASE_CONFIGS[8],
    "fmha_paged": _PAGED_CONFIGS[3],            # (1, 8, 8, 64, 512, 64)
    "fmha_paged_bias": _PAGED_CONFIGS[3],
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Performance metric: forward-pass FLOPs of FMHA = 2 * QK + 2 * SV gemm
#                                                = 4 * B * H * T_q * T_k * D
# ---------------------------------------------------------------------------


def compute_fmha_tflops(B: int, H: int, T_q: int, T_k: int, D: int, time_ms: float) -> float:
    if time_ms <= 0:
        return 0.0
    return 4.0 * B * H * T_q * T_k * D / (time_ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# Setup functions: return (oasr_fn, pytorch_fn) zero-arg callables.
# ---------------------------------------------------------------------------


def _make_inputs(cfg: dict, dtype: torch.dtype, *, with_bias: bool, with_seqlens: bool, seed: int = 0):
    B, H, H_kv = cfg["B"], cfg["H"], cfg["H_kv"]
    T_q, T_k, D = cfg["T_q"], cfg["T_k"], cfg["D"]
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, T_q, D, dtype=dtype, device="cuda", generator=g)
    k = torch.randn(B, H_kv, T_k, D, dtype=dtype, device="cuda", generator=g)
    v = torch.randn(B, H_kv, T_k, D, dtype=dtype, device="cuda", generator=g)
    bias = None
    seqlens = None
    if with_bias:
        bias = torch.randn(B, H, T_q, T_k, dtype=dtype, device="cuda", generator=g) * 0.1
    if with_seqlens:
        # Half short, half full -- exercises the per-stream length mask.
        base = max(1, T_k // 4)
        seqlens = torch.tensor(
            [base if i < B // 2 else T_k for i in range(B)],
            dtype=torch.int32, device="cuda",
        )
    return q, k, v, bias, seqlens


def _build_sdpa_mask(q, k, bias, seqlens):
    masks = []
    if bias is not None:
        masks.append(bias)
    if seqlens is not None:
        T_k = k.size(2)
        arange = torch.arange(T_k, device=seqlens.device)
        keep = arange.unsqueeze(0) < seqlens.unsqueeze(1)
        pad = torch.where(keep, 0.0, float("-inf")).to(q.dtype)
        masks.append(pad.unsqueeze(1).unsqueeze(1))
    if not masks:
        return None
    full_mask = masks[0]
    for m in masks[1:]:
        full_mask = full_mask + m
    return full_mask


def _setup_common(cfg: dict, dtype: torch.dtype, *, with_bias: bool, with_seqlens: bool):
    """Build (oasr_fn, pytorch_fn) sharing the same input tensors."""
    q, k, v, bias, seqlens = _make_inputs(
        cfg, dtype, with_bias=with_bias, with_seqlens=with_seqlens,
    )
    H, H_kv = q.size(1), k.size(1)
    D = q.size(-1)
    scale = 1.0 / (D ** 0.5)
    out = torch.empty_like(q)

    def oasr_fn():
        return oasr.fmha(
            q, k, v,
            softmax_scale=scale,
            attn_bias=bias,
            cache_seqlens=seqlens,
            out=out,
        )

    if H_kv != H:
        n_repeat = H // H_kv
        k_e = k.repeat_interleave(n_repeat, dim=1)
        v_e = v.repeat_interleave(n_repeat, dim=1)
    else:
        k_e, v_e = k, v
    full_mask = _build_sdpa_mask(q, k, bias, seqlens)

    def pytorch_fn():
        return F.scaled_dot_product_attention(
            q, k_e, v_e, attn_mask=full_mask, scale=scale,
        )

    return oasr_fn, pytorch_fn


def setup_fmha_offline(cfg: dict, dtype: torch.dtype = torch.float16):
    return _setup_common(cfg, dtype, with_bias=False, with_seqlens=False)


def setup_fmha_bias(cfg: dict, dtype: torch.dtype = torch.float16):
    return _setup_common(cfg, dtype, with_bias=True, with_seqlens=False)


def setup_fmha_seqlens(cfg: dict, dtype: torch.dtype = torch.float16):
    return _setup_common(cfg, dtype, with_bias=False, with_seqlens=True)


def setup_fmha_bias_seqlens(cfg: dict, dtype: torch.dtype = torch.float16):
    return _setup_common(cfg, dtype, with_bias=True, with_seqlens=True)


# ---------------------------------------------------------------------------
# Paged subroutines: K/V live in a (num_blocks, block_size, H_kv, D) pool
# accessed via per-stream block tables. Compares the cute paged kernel
# against gather-then-SDPA.
# ---------------------------------------------------------------------------


def _make_paged_inputs(
    cfg: dict, dtype: torch.dtype, *, with_bias: bool, seed: int = 0,
):
    B, H, H_kv = cfg["B"], cfg["H"], cfg["H_kv"]
    T_q, T_k, D = cfg["T_q"], cfg["T_k"], cfg["D"]
    block_size = int(cfg.get("block_size", 16))
    if T_k % block_size != 0:
        # Round up so T_k is a multiple of block_size for clean block tables.
        T_k = ((T_k + block_size - 1) // block_size) * block_size
    max_blocks_per_seq = T_k // block_size

    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, T_q, D, dtype=dtype, device="cuda", generator=g)
    num_pool_blocks = max(B * max_blocks_per_seq + 4, 32)
    k_pool = torch.randn(
        num_pool_blocks, block_size, H_kv, D, dtype=dtype, device="cuda", generator=g,
    )
    v_pool = torch.randn(
        num_pool_blocks, block_size, H_kv, D, dtype=dtype, device="cuda", generator=g,
    )
    block_ids = torch.randperm(num_pool_blocks)[: B * max_blocks_per_seq]
    block_table = block_ids.reshape(B, max_blocks_per_seq).to(
        dtype=torch.int32, device="cuda",
    )
    # Vary cache_seqlens across streams to exercise per-stream length mask.
    seqlens = torch.tensor(
        [max(1, T_k - 8 - 2 * b) for b in range(B)],
        dtype=torch.int32, device="cuda",
    )
    bias = None
    if with_bias:
        bias = torch.randn(B, H, T_q, T_k, dtype=dtype, device="cuda", generator=g) * 0.1
    return q, k_pool, v_pool, block_table, seqlens, bias, T_k


def _setup_paged(cfg: dict, dtype: torch.dtype, *, with_bias: bool):
    q, k_pool, v_pool, block_table, seqlens, bias, T_k = _make_paged_inputs(
        cfg, dtype, with_bias=with_bias,
    )
    H, D = q.size(1), q.size(-1)
    H_kv = k_pool.size(2)
    block_size = k_pool.size(1)
    scale = 1.0 / (D ** 0.5)
    out = torch.empty_like(q)

    def oasr_fn():
        return oasr.fmha(
            q, k_pool, v_pool,
            softmax_scale=scale,
            attn_bias=bias,
            cache_seqlens=seqlens,
            block_table=block_table,
            out=out,
        )

    # Reference: gather + SDPA (mirrors the SDPA fallback path).
    block_ids_long = block_table.long()
    k_full = k_pool[block_ids_long].reshape(
        q.size(0), -1, H_kv, D
    ).permute(0, 2, 1, 3)
    v_full = v_pool[block_ids_long].reshape(
        q.size(0), -1, H_kv, D
    ).permute(0, 2, 1, 3)
    if H_kv != H:
        n_repeat = H // H_kv
        k_full = k_full.repeat_interleave(n_repeat, dim=1)
        v_full = v_full.repeat_interleave(n_repeat, dim=1)
    full_mask = _build_sdpa_mask(q, k_full, bias, seqlens)

    def pytorch_fn():
        return F.scaled_dot_product_attention(
            q, k_full, v_full, attn_mask=full_mask, scale=scale,
        )

    # Stash the resolved T_k so the bench TFLOPS calculation uses the
    # padded value (matches what the kernel actually walks).
    cfg["T_k"] = T_k
    cfg["block_size"] = block_size
    return oasr_fn, pytorch_fn


def setup_fmha_paged(cfg: dict, dtype: torch.dtype = torch.float16):
    return _setup_paged(cfg, dtype, with_bias=False)


def setup_fmha_paged_bias(cfg: dict, dtype: torch.dtype = torch.float16):
    return _setup_paged(cfg, dtype, with_bias=True)


_SETUP_FNS = {
    "fmha_offline": setup_fmha_offline,
    "fmha_bias": setup_fmha_bias,
    "fmha_seqlens": setup_fmha_seqlens,
    "fmha_bias_seqlens": setup_fmha_bias_seqlens,
    "fmha_paged": setup_fmha_paged,
    "fmha_paged_bias": setup_fmha_paged_bias,
}


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    """Add attention-specific CLI arguments."""
    parser.add_argument("--B", type=int, default=None, help="Batch size")
    parser.add_argument("--H", type=int, default=None, help="Number of query heads")
    parser.add_argument("--H_kv", type=int, default=None,
                        help="Number of K/V heads (defaults to H; H % H_kv must be 0)")
    parser.add_argument("--T_q", type=int, default=None, help="Query length")
    parser.add_argument("--T_k", type=int, default=None, help="Key/value length")
    parser.add_argument("--D", type=int, default=None, help="Head dimension")


# ---------------------------------------------------------------------------
# run_test (unified CLI entry point)
# ---------------------------------------------------------------------------


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    """Run a single FMHA benchmark test."""
    subroutine = getattr(args, "subroutine", None) or "fmha_offline"
    if subroutine not in SUBROUTINES:
        raise ValueError(
            f"Unknown attention subroutine '{subroutine}'. "
            f"Choose from: {SUBROUTINES}"
        )
    dtype_str = getattr(args, "dtype", "float16")
    dtype = parse_dtype(dtype_str) if dtype_str else torch.float16

    do_check = getattr(args, "refcheck", False)
    allow_mismatch = getattr(args, "allow_output_mismatch", False)
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)

    configs = _resolve_configs(args, subroutine)

    setup = _SETUP_FNS[subroutine]
    for cfg in configs:
        oasr_fn, pytorch_fn = setup(cfg, dtype)
        fn_map = {"cutlass": oasr_fn, "torch": pytorch_fn}
        backends = getattr(args, "backends", None) or list(fn_map.keys())
        shape_str = _shape_str(cfg)

        output.write_verbose(f"shape = {shape_str}, dtype = {dtype_str}", level=2)

        if (
            do_check
            and "torch" in backends
            and any(b in fn_map and b != "torch" for b in backends)
        ):
            oasr_out = oasr_fn()
            pytorch_out = pytorch_fn()
            passed, max_diff = check_close(oasr_out, pytorch_out)
            if not passed:
                msg = f"[ERROR] Output mismatch for {shape_str} (max_diff={max_diff:.6f})"
                print(msg)
                if not allow_mismatch:
                    continue

        for backend in backends:
            if backend not in fn_map:
                continue
            median_ms, std_ms = bench_fn(
                fn_map[backend],
                dry_run_iters=dry_run_iters,
                num_iters=num_iters,
                use_cuda_events=use_cuda_events,
            )
            tflops = compute_fmha_tflops(
                cfg["B"], cfg["H"], cfg["T_q"], cfg["T_k"], cfg["D"], median_ms,
            )
            output.write_result(BenchResult(
                routine="attention",
                subroutine=subroutine,
                backend=backend,
                shape=shape_str,
                dtype=dtype_str,
                median_ms=median_ms,
                std_ms=std_ms,
                tflops=tflops,
            ))


def _resolve_configs(args: argparse.Namespace, subroutine: str) -> list[dict]:
    """Use CLI dims if a complete shape is specified; otherwise defaults."""
    keys = ("B", "H", "T_q", "T_k", "D")
    vals = {k: getattr(args, k, None) for k in keys}
    if all(v is not None for v in vals.values()):
        H_kv = getattr(args, "H_kv", None) or vals["H"]
        return [{**vals, "H_kv": H_kv}]
    return DEFAULT_CONFIGS.get(subroutine, _BASE_CONFIGS)


def _shape_str(cfg: dict) -> str:
    return (
        f"(B={cfg['B']}, H={cfg['H']}, H_kv={cfg['H_kv']}, "
        f"T_q={cfg['T_q']}, T_k={cfg['T_k']}, D={cfg['D']})"
    )


# ---------------------------------------------------------------------------
# Standalone entry (backwards-compat with bench_fmha.py)
# ---------------------------------------------------------------------------


def run_standalone() -> None:
    """Run all four subroutines × {fp16, bf16} with the default shape grid.

    Mirrors the standalone behavior of ``benchmarks/routines/gemm.py``.
    """
    profile_configs = {sub: _shape_tuple(PROFILE_CONFIGS[sub]) for sub in SUBROUTINES}
    setup_funcs = {
        sub: (lambda sub=sub: _SETUP_FNS[sub](PROFILE_CONFIGS[sub])) for sub in SUBROUTINES
    }

    def benchmark():
        output = OutputWriter()
        output.write_header("FMHA Benchmark (CuteDSL vs PyTorch SDPA)")
        for dtype, dtype_str in [(torch.float16, "float16"), (torch.bfloat16, "bfloat16")]:
            for sub in SUBROUTINES:
                output.write_header(f"--- {sub} / {dtype_str} ---")
                for cfg in DEFAULT_CONFIGS[sub]:
                    oasr_fn, pytorch_fn = _SETUP_FNS[sub](cfg, dtype)
                    shape = _shape_str(cfg)
                    for backend, fn in [("cutlass", oasr_fn), ("torch", pytorch_fn)]:
                        median_ms, std_ms = bench_fn(fn)
                        tflops = compute_fmha_tflops(
                            cfg["B"], cfg["H"], cfg["T_q"], cfg["T_k"], cfg["D"], median_ms,
                        )
                        output.write_result(BenchResult(
                            routine="attention", subroutine=sub, backend=backend,
                            shape=shape, dtype=dtype_str,
                            median_ms=median_ms, std_ms=std_ms, tflops=tflops,
                        ))
        output.finalize()

    run_main("FMHA Kernel", profile_configs, setup_funcs, benchmark)


def _shape_tuple(cfg: dict) -> tuple:
    """Compact tuple form for profile-config logging (cosmetic only)."""
    return (cfg["B"], cfg["H"], cfg["H_kv"], cfg["T_q"], cfg["T_k"], cfg["D"])
