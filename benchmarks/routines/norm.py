"""Normalization family benchmark routines.

Covers: layer_norm, add_layer_norm, layer_norm_activation, rms_norm,
        batch_norm, batch_norm_swish, batch_norm_activation, group_norm.
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
    compute_bandwidth_tb_s,
    dtype_size,
    make_bench_parser,
    profile_kernel,
    run_main,
    run_profile,
)

SUBROUTINES = [
    "layer_norm",
    "add_layer_norm",
    "layer_norm_activation",
    "rms_norm",
    "batch_norm",
    "batch_norm_swish",
    "batch_norm_activation",
    "group_norm",
    "cmvn",
]

# ---------------------------------------------------------------------------
# Default configs
# ---------------------------------------------------------------------------

DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "layer_norm": [
        {"batch": 32, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 256},
        {"batch": 64, "seq": 500, "hidden": 512},
        {"batch": 32, "seq": 500, "hidden": 512},
    ],
    "add_layer_norm": [
        {"batch": 32, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
    "layer_norm_activation": [
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
    "rms_norm": [
        {"batch": 32, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
    "batch_norm": [
        {"batch": 32, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
    "batch_norm_swish": [
        {"batch": 32, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
    "batch_norm_activation": [
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 250, "hidden": 512},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
    "group_norm": [
        {"batch": 32, "seq": 250, "hidden": 256, "num_groups": 32},
        {"batch": 64, "seq": 250, "hidden": 256, "num_groups": 32},
        {"batch": 64, "seq": 250, "hidden": 512, "num_groups": 64},
        {"batch": 64, "seq": 500, "hidden": 512, "num_groups": 64},
    ],
    "cmvn": [
        {"batch": 32, "seq": 250, "hidden": 80},
        {"batch": 64, "seq": 250, "hidden": 80},
        {"batch": 64, "seq": 500, "hidden": 80},
        {"batch": 64, "seq": 250, "hidden": 256},
        {"batch": 64, "seq": 500, "hidden": 512},
    ],
}

PROFILE_CONFIGS: dict[str, tuple] = {
    "layer_norm": (64, 250, 256),
    "add_layer_norm": (64, 250, 512),
    "layer_norm_activation": (64, 250, 512),
    "rms_norm": (64, 250, 256),
    "batch_norm": (64, 250, 512),
    "batch_norm_swish": (64, 250, 512),
    "batch_norm_activation": (64, 250, 512),
    "group_norm": (64, 250, 512, 64),
    "cmvn": (64, 250, 80),
}


def get_default_configs() -> dict[str, list[dict[str, Any]]]:
    return DEFAULT_CONFIGS


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------


def setup_layer_norm(batch_size, seq_len, hidden_size, dtype=torch.float16):
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    gamma = torch.randn(hidden_size, device="cuda", dtype=dtype)
    beta = torch.randn(hidden_size, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.layer_norm(x, gamma, beta, eps)

    ln = torch.nn.LayerNorm(hidden_size, eps=eps, device="cuda", dtype=dtype)
    ln.weight.data = gamma.clone()
    ln.bias.data = beta.clone()

    def pytorch_fn():
        return ln(x)

    return oasr_fn, pytorch_fn


def setup_add_layer_norm(batch_size, seq_len, hidden_size, dtype=torch.float16):
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    gamma = torch.randn(hidden_size, device="cuda", dtype=dtype)
    beta = torch.randn(hidden_size, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.add_layer_norm(x, residual, gamma, beta, eps)

    ln = torch.nn.LayerNorm(hidden_size, eps=eps, device="cuda", dtype=dtype)
    ln.weight.data = gamma.clone()
    ln.bias.data = beta.clone()

    def pytorch_fn():
        return ln(x + residual)

    return oasr_fn, pytorch_fn


def setup_layer_norm_activation(
    batch_size, seq_len, hidden_size, activation_name="swish", dtype=torch.float16
):
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    gamma = torch.randn(hidden_size, device="cuda", dtype=dtype)
    beta = torch.randn(hidden_size, device="cuda", dtype=dtype)

    from oasr.utils.mappings import get_activation

    act_type_id = oasr.get_activation_type_id(activation_name)
    torch_act = get_activation(activation_name).cuda()

    def oasr_fn():
        return oasr.layer_norm_activation(x, gamma, beta, eps, act_type_id)

    ln = torch.nn.LayerNorm(hidden_size, eps=eps, device="cuda", dtype=dtype)
    ln.weight.data = gamma.clone()
    ln.bias.data = beta.clone()

    def pytorch_fn():
        return torch_act(ln(x))

    return oasr_fn, pytorch_fn


def setup_rms_norm(batch_size, seq_len, hidden_size, dtype=torch.float16):
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    gamma = torch.randn(hidden_size, device="cuda", dtype=dtype)

    def oasr_fn():
        return oasr.rms_norm(x, gamma, None, eps)

    def pytorch_fn():
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x / rms * gamma

    return oasr_fn, pytorch_fn


def setup_batch_norm(batch_size, seq_len, channels, dtype=torch.float32):
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)
    gamma = torch.randn(channels, device="cuda", dtype=dtype)
    beta = torch.randn(channels, device="cuda", dtype=dtype)
    running_mean = torch.randn(channels, device="cuda", dtype=dtype)
    running_var = torch.abs(torch.randn(channels, device="cuda", dtype=dtype)) + 0.1

    def oasr_fn():
        return oasr.batch_norm_1d(x, gamma, beta, running_mean, running_var, eps)

    def pytorch_fn():
        return (x - running_mean) / torch.sqrt(running_var + eps) * gamma + beta

    return oasr_fn, pytorch_fn


def setup_batch_norm_swish(batch_size, seq_len, channels, dtype=torch.float16):
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)
    gamma = torch.randn(channels, device="cuda", dtype=dtype)
    beta = torch.randn(channels, device="cuda", dtype=dtype)
    running_mean = torch.randn(channels, device="cuda", dtype=dtype)
    running_var = torch.abs(torch.randn(channels, device="cuda", dtype=dtype)) + 0.1

    def oasr_fn():
        return oasr.batch_norm_swish(x, gamma, beta, running_mean, running_var, eps)

    def pytorch_fn():
        bn_out = (x - running_mean) / torch.sqrt(running_var + eps) * gamma + beta
        return F.silu(bn_out)

    return oasr_fn, pytorch_fn


def setup_batch_norm_activation(
    batch_size, seq_len, channels, activation_name="swish", dtype=torch.float16
):
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)
    gamma = torch.randn(channels, device="cuda", dtype=dtype)
    beta = torch.randn(channels, device="cuda", dtype=dtype)
    running_mean = torch.randn(channels, device="cuda", dtype=dtype)
    running_var = torch.abs(torch.randn(channels, device="cuda", dtype=dtype)) + 0.1

    from oasr.utils.mappings import get_activation

    act_type_id = oasr.get_activation_type_id(activation_name)
    torch_act = get_activation(activation_name).cuda()

    def oasr_fn():
        return oasr.batch_norm_activation(x, gamma, beta, running_mean, running_var, eps, act_type_id)

    def pytorch_fn():
        bn_out = (x - running_mean) / torch.sqrt(running_var + eps) * gamma + beta
        return torch_act(bn_out)

    return oasr_fn, pytorch_fn


def setup_group_norm(batch_size, seq_len, channels, num_groups, dtype=torch.float32):
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, channels, device="cuda", dtype=dtype)
    gamma = torch.randn(channels, device="cuda", dtype=dtype)
    beta = torch.randn(channels, device="cuda", dtype=dtype)
    channels_per_group = channels // num_groups

    def oasr_fn():
        return oasr.group_norm(x, gamma, beta, num_groups, eps)

    def pytorch_fn():
        x_r = x.view(batch_size, seq_len, num_groups, channels_per_group)
        mean = x_r.mean(dim=-1, keepdim=True)
        var = x_r.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_r - mean) / torch.sqrt(var + eps)
        x_norm = x_norm.view(batch_size, seq_len, channels)
        return x_norm * gamma + beta

    return oasr_fn, pytorch_fn


def setup_cmvn(batch_size, seq_len, num_cols, dtype=torch.float16):
    x = torch.randn(batch_size, seq_len, num_cols, device="cuda", dtype=dtype)
    mean = torch.randn(num_cols, device="cuda", dtype=dtype)
    istd = torch.randn(num_cols, device="cuda", dtype=dtype).abs() + 0.1

    def oasr_fn():
        return oasr.cmvn(x, mean, istd)

    def pytorch_fn():
        return (x - mean) * istd

    return oasr_fn, pytorch_fn


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def parse_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--seq", type=int, default=None, help="Sequence length")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden / channel dimension")
    parser.add_argument("--num-groups", type=int, default=32, help="Number of groups (group_norm)")
    parser.add_argument(
        "--activation", type=str, default="swish",
        help="Activation for fused variants (relu, gelu, swish)",
    )


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------


def _norm_bytes(batch, seq, hidden, dtype, extra_inputs=1):
    """Estimate bytes accessed for norm: read input(s) + weight + write output."""
    elem = dtype_size(dtype)
    input_bytes = batch * seq * hidden * elem * (1 + extra_inputs)  # input(s)
    weight_bytes = hidden * elem * 2  # gamma + beta
    output_bytes = batch * seq * hidden * elem
    return input_bytes + weight_bytes + output_bytes


def run_test(args: argparse.Namespace, output: OutputWriter) -> None:
    subroutine = getattr(args, "subroutine", "layer_norm")
    dtype_str = getattr(args, "dtype", "float16")
    from benchmarks.routines.bench_utils import parse_dtype

    dtype = parse_dtype(dtype_str)
    do_check = getattr(args, "refcheck", False)
    allow_mismatch = getattr(args, "allow_output_mismatch", False)
    dry_run_iters = getattr(args, "dry_run_iters", 5)
    num_iters = getattr(args, "num_iters", 30)
    use_cuda_events = getattr(args, "use_cuda_events", False)
    activation_name = getattr(args, "activation", "swish")

    configs = _resolve_configs(args, subroutine)

    for cfg in configs:
        oasr_fn, pytorch_fn = _setup_for_config(subroutine, cfg, dtype, activation_name)
        fn_map = get_fn_map(subroutine, oasr_fn, pytorch_fn)
        backends = getattr(args, "backends", None) or list(fn_map.keys())

        b, s, h = cfg["batch"], cfg["seq"], cfg["hidden"]
        extra_inputs = 1 if subroutine == "add_layer_norm" else 0
        bytes_accessed = _norm_bytes(b, s, h, dtype, extra_inputs)
        shape_str = _shape_str(subroutine, cfg)

        if do_check and "torch" in backends and any(b in fn_map and b != "torch" for b in backends):
            oasr_out = oasr_fn()
            pytorch_out = pytorch_fn()
            passed, max_diff = check_close(oasr_out, pytorch_out)
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
            output.write_result(BenchResult(
                routine="norm",
                subroutine=subroutine,
                backend=backend,
                shape=shape_str,
                dtype=dtype_str,
                median_ms=median_ms,
                std_ms=std_ms,
                bandwidth_tb_s=bw,
            ))


def _resolve_configs(args, subroutine):
    batch = getattr(args, "batch", None)
    seq = getattr(args, "seq", None)
    hidden = getattr(args, "hidden", None)
    num_groups = getattr(args, "num_groups", 32)

    if all(v is not None for v in [batch, seq, hidden]):
        cfg = {"batch": batch, "seq": seq, "hidden": hidden}
        if subroutine == "group_norm":
            cfg["num_groups"] = num_groups
        return [cfg]

    return DEFAULT_CONFIGS.get(subroutine, DEFAULT_CONFIGS["layer_norm"])


def _setup_for_config(subroutine, cfg, dtype, activation_name="swish"):
    b, s, h = cfg["batch"], cfg["seq"], cfg["hidden"]
    if subroutine == "layer_norm":
        return setup_layer_norm(b, s, h, dtype)
    elif subroutine == "add_layer_norm":
        return setup_add_layer_norm(b, s, h, dtype)
    elif subroutine == "layer_norm_activation":
        return setup_layer_norm_activation(b, s, h, activation_name, dtype)
    elif subroutine == "rms_norm":
        return setup_rms_norm(b, s, h, dtype)
    elif subroutine == "batch_norm":
        return setup_batch_norm(b, s, h, dtype)
    elif subroutine == "batch_norm_swish":
        return setup_batch_norm_swish(b, s, h, dtype)
    elif subroutine == "batch_norm_activation":
        return setup_batch_norm_activation(b, s, h, activation_name, dtype)
    elif subroutine == "group_norm":
        num_groups = cfg.get("num_groups", 32)
        return setup_group_norm(b, s, h, num_groups, dtype)
    elif subroutine == "cmvn":
        return setup_cmvn(b, s, h, dtype)
    else:
        raise ValueError(f"Unknown norm subroutine: {subroutine}")


def _shape_str(subroutine, cfg):
    b, s, h = cfg["batch"], cfg["seq"], cfg["hidden"]
    if subroutine == "group_norm":
        return f"[{b}, {s}, {h}] g={cfg.get('num_groups', 32)}"
    return f"[{b}, {s}, {h}]"


def get_fn_map(subroutine, cuda_fn, torch_fn):
    """Return {backend_name: fn} for norm subroutines."""
    return {"cuda": cuda_fn, "torch": torch_fn}


# ---------------------------------------------------------------------------
# Standalone entry points
# ---------------------------------------------------------------------------

_STANDALONE_MAP = {
    "layer_norm": {
        "title": "LayerNorm Kernels",
        "subroutines": ["layer_norm", "add_layer_norm", "layer_norm_activation"],
    },
    "rms_norm": {
        "title": "RMSNorm Kernel",
        "subroutines": ["rms_norm"],
    },
    "batch_norm": {
        "title": "BatchNorm Kernels",
        "subroutines": ["batch_norm", "batch_norm_swish", "batch_norm_activation"],
    },
    "group_norm": {
        "title": "GroupNorm Kernel",
        "subroutines": ["group_norm"],
    },
    "cmvn": {
        "title": "CMVN Kernel",
        "subroutines": ["cmvn"],
    },
}


def run_standalone(variant: str = "layer_norm") -> None:
    """Standalone entry point for backwards-compat bench_*.py wrappers."""
    info = _STANDALONE_MAP.get(variant, _STANDALONE_MAP["layer_norm"])
    title = info["title"]
    subs = info["subroutines"]

    # Collect profile configs for relevant subroutines
    pcfg = {k: PROFILE_CONFIGS[k] for k in subs if k in PROFILE_CONFIGS}

    # Build setup_funcs for profiling
    setup_funcs = {}
    for sub in subs:
        if sub in PROFILE_CONFIGS:
            setup_funcs[sub] = _make_profile_setup(sub)

    def benchmark():
        output = OutputWriter()
        for sub in subs:
            configs = DEFAULT_CONFIGS.get(sub, [])
            output.write_header(f"{sub} Benchmark")
            for cfg in configs:
                oasr_fn, pytorch_fn = _setup_for_config(sub, cfg, torch.float16)
                b, s, h = cfg["batch"], cfg["seq"], cfg["hidden"]
                bytes_accessed = _norm_bytes(b, s, h, torch.float16)
                shape_str = _shape_str(sub, cfg)
                for backend, fn in [("cuda", oasr_fn), ("torch", pytorch_fn)]:
                    median_ms, std_ms = bench_fn(fn)
                    bw = compute_bandwidth_tb_s(bytes_accessed, median_ms)
                    output.write_result(BenchResult(
                        routine="norm", subroutine=sub, backend=backend,
                        shape=shape_str, dtype="float16",
                        median_ms=median_ms, std_ms=std_ms,
                        bandwidth_tb_s=bw,
                    ))
        output.finalize()

    run_main(title, pcfg, setup_funcs, benchmark)


def _make_profile_setup(subroutine):
    cfg_tuple = PROFILE_CONFIGS[subroutine]

    def _setup():
        if subroutine == "group_norm":
            return setup_group_norm(*cfg_tuple)
        elif subroutine == "batch_norm":
            return setup_batch_norm(*cfg_tuple)
        elif subroutine == "batch_norm_swish":
            return setup_batch_norm_swish(*cfg_tuple)
        elif subroutine == "batch_norm_activation":
            return setup_batch_norm_activation(*cfg_tuple)
        elif subroutine == "add_layer_norm":
            return setup_add_layer_norm(*cfg_tuple)
        elif subroutine == "layer_norm_activation":
            return setup_layer_norm_activation(*cfg_tuple)
        elif subroutine == "rms_norm":
            return setup_rms_norm(*cfg_tuple)
        elif subroutine == "cmvn":
            return setup_cmvn(*cfg_tuple)
        else:
            return setup_layer_norm(*cfg_tuple)

    return _setup
