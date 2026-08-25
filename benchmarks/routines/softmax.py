"""Softmax family benchmark routines."""

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
    compute_bandwidth_tb_s,
    dtype_size,
    make_bench_parser,
    run_main,
    run_profile,
)

SUBROUTINES = ["softmax", "masked_softmax"]

# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "softmax": [
        {"batch": 32, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 256},
        {"batch": 64, "seq": 250, "channels": 512},
        {"batch": 64, "seq": 500, "channels": 256},
        {"batch": 64, "seq": 500, "channels": 512},
        {"batch": 64, "seq": 250, "channels": 1024},
    ],
    # KG6: Zipformer's shared attention weights.  The score tensor is
    # (head, batch, time, time), so the row length *is* the sequence length and
    # the row count scales with it -- which is why this is a T^2 problem and why
    # the unfused arm pays six passes over it.  Sequence lengths are what the
    # stack's downsampling ladder produces for ~5-10 s of audio.
    "masked_softmax": [
        {"heads": 8, "batch": 1, "seq": 500},
        {"heads": 8, "batch": 8, "seq": 500},
        {"heads": 8, "batch": 32, "seq": 500},
        {"heads": 8, "batch": 8, "seq": 250},
        {"heads": 8, "batch": 32, "seq": 125},
        {"heads": 4, "batch": 32, "seq": 63},
    ],
}

PROFILE_CONFIGS: dict[str, tuple] = {
    "softmax": (64, 250, 512),
    "masked_softmax": (8, 8, 500),
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------


def setup_softmax(batch_size, seq_len, channels, dtype=torch.float16):
    x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.softmax(x)

    def pytorch_fn():
        return F.softmax(x, dim=-1)

    return oasr_fn, pytorch_fn


def setup_masked_softmax(heads, batch, seq, dtype=torch.float16):
    """Zipformer's ``attn_scores + rel_pos_bias -> mask -> softmax``, three ways.

    Returns ``(fused, unfused_oasr, unfused_torch)``.  The middle arm is the
    A/B that matters: it is the sequence the model ran *before* this kernel,
    already on OASR's softmax, so the gap between it and the fused arm is the
    fusion alone rather than a kernel-quality difference.
    """
    scores = torch.randn(heads, batch, seq, seq, device="cuda", dtype=dtype)
    # The relative-position product, consumed as the same shifted window the
    # encoder builds: (head, batch, time, 2T-1) read as (head, batch, T, T).
    pos = torch.randn(heads, batch, seq, 2 * seq - 1, device="cuda", dtype=dtype)
    bias = pos.as_strided(
        (heads, batch, seq, seq),
        (pos.stride(0), pos.stride(1), pos.stride(2) - pos.stride(3), pos.stride(3)),
        storage_offset=pos.stride(3) * (seq - 1),
    )
    key_padding = torch.rand(batch, seq, device="cuda") < 0.15
    key_padding[:, 0] = False
    key_padding_bcast = key_padding.unsqueeze(1)

    def oasr_fn():
        return oasr.masked_softmax(scores, bias=bias, mask2=key_padding_bcast, mask_value=-1000.0)

    def oasr_unfused_fn():
        biased = scores + bias
        biased = biased.masked_fill(key_padding_bcast, -1000)
        return oasr.softmax(biased.contiguous())

    def pytorch_fn():
        biased = scores + bias
        biased = biased.masked_fill(key_padding_bcast, -1000)
        return F.softmax(biased.contiguous(), dim=-1)

    return oasr_fn, oasr_unfused_fn, pytorch_fn


def _setup_for_config(subroutine, cfg, dtype):
    """``(oasr_fn, torch_fn)`` for the profiling path, which unpacks two."""
    if subroutine == "masked_softmax":
        fused, _unfused, torch_fn = setup_masked_softmax(**cfg, dtype=dtype)
        return fused, torch_fn
    return setup_softmax(cfg["batch"], cfg["seq"], cfg["channels"], dtype)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length")
    parser.add_argument(
        "--channels", type=int, default=None, help="Channel dimension (softmax dim)"
    )
    parser.add_argument(
        "--heads", type=int, default=None, help="Attention heads (masked_softmax only)"
    )


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


def _softmax_bytes(batch, seq, channels, dtype):
    """Bytes accessed: read input + write output."""
    elem = dtype_size(dtype)
    return 2 * batch * seq * channels * elem


def _masked_softmax_bytes(heads, batch, seq, dtype):
    """The *fused* traffic: read scores, read the bias window, write output.

    Reported for every backend, so the column is effective bandwidth for the
    same logical work rather than a measure of how many passes an arm took.
    """
    return 3 * heads * batch * seq * seq * dtype_size(dtype)


def _config_bytes(subroutine, cfg, dtype):
    if subroutine == "masked_softmax":
        return _masked_softmax_bytes(cfg["heads"], cfg["batch"], cfg["seq"], dtype)
    return _softmax_bytes(cfg["batch"], cfg["seq"], cfg["channels"], dtype)


def _config_shape(subroutine, cfg):
    if subroutine == "masked_softmax":
        return f"[{cfg['heads']}, {cfg['batch']}, {cfg['seq']}, {cfg['seq']}]"
    return f"[{cfg['batch']}, {cfg['seq']}, {cfg['channels']}]"


def _fns_for_config(subroutine, cfg, dtype):
    if subroutine == "masked_softmax":
        return setup_masked_softmax(**cfg, dtype=dtype)
    return setup_softmax(cfg["batch"], cfg["seq"], cfg["channels"], dtype)


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", "softmax")
    dtype_str = getattr(args, "dtype", "float16")
    from benchmarks.routines.bench_utils import parse_dtype

    dtype = parse_dtype(dtype_str)
    do_check = getattr(args, "refcheck", False)
    allow_mismatch = getattr(args, "allow_output_mismatch", False)
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)

    configs = _resolve_configs(args, subroutine)

    for cfg in configs:
        fns = _fns_for_config(subroutine, cfg, dtype)
        fn_map = get_fn_map(subroutine, *fns)
        backends = getattr(args, "backends", None) or list(fn_map.keys())

        bytes_accessed = _config_bytes(subroutine, cfg, dtype)
        shape_str = _config_shape(subroutine, cfg)

        if (
            do_check
            and "torch" in backends
            and any(bk in fn_map and bk != "torch" for bk in backends)
        ):
            oasr_out = fn_map["cuda"]()
            pytorch_out = fn_map["torch"]()
            passed, max_diff = check_close(oasr_out, pytorch_out.to(dtype))
            if not passed:
                print(f"  [ERROR] Output mismatch for {shape_str} (max_diff={max_diff:.6f})")
                if not allow_mismatch:
                    continue

        for backend in backends:
            if backend not in fn_map:
                print(f"  [WARNING] Unknown backend '{backend}', skipping")
                continue
            median_ms, std_ms = bench_fn(
                fn_map[backend],
                dry_run_iters=dry_run_iters,
                num_iters=num_iters,
                use_cuda_events=use_cuda_events,
            )
            bw = compute_bandwidth_tb_s(bytes_accessed, median_ms)
            output.write_result(
                BenchResult(
                    routine="softmax",
                    subroutine=subroutine,
                    backend=backend,
                    shape=shape_str,
                    dtype=dtype_str,
                    median_ms=median_ms,
                    std_ms=std_ms,
                    bandwidth_tb_s=bw,
                )
            )


def _resolve_configs(args, subroutine):
    batch = getattr(args, "batch", None)
    seq = getattr(args, "seq", None)

    if subroutine == "masked_softmax":
        heads = getattr(args, "heads", None)
        if all(v is not None for v in (heads, batch, seq)):
            return [{"heads": heads, "batch": batch, "seq": seq}]
        return DEFAULT_CONFIGS["masked_softmax"]

    channels = getattr(args, "channels", None)
    if all(v is not None for v in (batch, seq, channels)):
        return [{"batch": batch, "seq": seq, "channels": channels}]
    return DEFAULT_CONFIGS.get(subroutine, DEFAULT_CONFIGS["softmax"])


def get_fn_map(subroutine, *fns):
    """Backend name -> callable.

    ``masked_softmax`` carries a third arm, ``cuda_unfused``: the same OASR
    softmax reached through the separate add / masked_fill / contiguous ops.
    Without it a fused-vs-torch number cannot say how much of the win is the
    fusion and how much is the kernel underneath it.
    """
    if subroutine == "masked_softmax" and len(fns) == 3:
        fused, unfused_oasr, torch_fn = fns
        return {"cuda": fused, "cuda_unfused": unfused_oasr, "torch": torch_fn}
    cuda_fn, torch_fn = fns[0], fns[1]
    return {"cuda": cuda_fn, "torch": torch_fn}


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------


def run_standalone(variant: str = "softmax") -> None:
    """Standalone entry point for the ``bench_*.py`` wrappers."""
    subs = [variant]
    pcfg = {k: PROFILE_CONFIGS[k] for k in subs if k in PROFILE_CONFIGS}

    setup_funcs = {}
    for sub in subs:
        if sub in PROFILE_CONFIGS:
            setup_funcs[sub] = _make_profile_setup(sub)

    def benchmark():
        output = OutputWriter()
        for sub in subs:
            configs = DEFAULT_CONFIGS.get(sub, [])
            output.write_header(f"{sub.upper()} Benchmark")
            for cfg in configs:
                fns = _fns_for_config(sub, cfg, torch.float16)
                bytes_accessed = _config_bytes(sub, cfg, torch.float16)
                shape_str = _config_shape(sub, cfg)
                for backend, fn in get_fn_map(sub, *fns).items():
                    median_ms, std_ms = bench_fn(fn)
                    bw = compute_bandwidth_tb_s(bytes_accessed, median_ms)
                    output.write_result(
                        BenchResult(
                            routine="softmax",
                            subroutine=sub,
                            backend=backend,
                            shape=shape_str,
                            dtype="float16",
                            median_ms=median_ms,
                            std_ms=std_ms,
                            bandwidth_tb_s=bw,
                        )
                    )
        output.finalize()

    run_main(f"{variant.upper()} Softmax Kernel", pcfg, setup_funcs, benchmark)


def _make_profile_setup(subroutine):
    cfg_tuple = PROFILE_CONFIGS[subroutine]

    def _setup():
        if subroutine == "masked_softmax":
            fused, _unfused, torch_fn = setup_masked_softmax(*cfg_tuple)
            return fused, torch_fn
        return setup_softmax(*cfg_tuple)

    return _setup
