#!/usr/bin/env python3
"""Performance benchmarks for BatchNorm kernels (standard, fused swish, and fused activation)."""

import torch
import torch.nn.functional as F

import oasr
try:
    from .utils import run_main
except ImportError:
    from utils import run_main


def setup_batch_norm(batch_size, seq_len, channels, dtype=torch.float32):
    """Setup tensors for BatchNorm1D."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
    gamma = torch.randn(channels, device='cuda', dtype=dtype)
    beta = torch.randn(channels, device='cuda', dtype=dtype)
    running_mean = torch.randn(channels, device='cuda', dtype=dtype)
    running_var = torch.abs(torch.randn(
        channels, device='cuda', dtype=dtype)) + 0.1

    def oasr_fn():
        return oasr.batch_norm_1d(
            x, gamma, beta, running_mean, running_var, eps
        )

    def pytorch_fn():
        return (x - running_mean) / torch.sqrt(running_var + eps) * gamma + beta

    return oasr_fn, pytorch_fn


def setup_batch_norm_swish(batch_size, seq_len, channels, dtype=torch.float16):
    """Setup tensors for fused BatchNorm + Swish."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
    gamma = torch.randn(channels, device='cuda', dtype=dtype)
    beta = torch.randn(channels, device='cuda', dtype=dtype)
    running_mean = torch.randn(channels, device='cuda', dtype=dtype)
    running_var = torch.abs(torch.randn(
        channels, device='cuda', dtype=dtype)) + 0.1

    def oasr_fn():
        return oasr.batch_norm_swish(
            x, gamma, beta, running_mean, running_var, eps
        )

    def pytorch_fn():
        bn_out = (x - running_mean) / \
            torch.sqrt(running_var + eps) * gamma + beta
        return F.silu(bn_out)

    return oasr_fn, pytorch_fn


def setup_batch_norm_activation(batch_size, seq_len, channels, activation_name="swish", dtype=torch.float16):
    """Setup tensors for fused BatchNorm + Activation."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
    gamma = torch.randn(channels, device='cuda', dtype=dtype)
    beta = torch.randn(channels, device='cuda', dtype=dtype)
    running_mean = torch.randn(channels, device='cuda', dtype=dtype)
    running_var = torch.abs(torch.randn(
        channels, device='cuda', dtype=dtype)) + 0.1

    from oasr.utils.mappings import get_activation
    act_type_id = oasr.get_activation_type_id(activation_name)
    torch_act = get_activation(activation_name).cuda()

    def oasr_fn():
        return oasr.batch_norm_activation(
            x, gamma, beta, running_mean, running_var, eps, act_type_id
        )

    def pytorch_fn():
        bn_out = (x - running_mean) / \
            torch.sqrt(running_var + eps) * gamma + beta
        return torch_act(bn_out)

    return oasr_fn, pytorch_fn


def benchmark_batch_norm():
    """Benchmark BatchNorm1D (inference). Conformer (B, T_enc, channels)."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("BatchNorm1D (Inference) Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)

    for batch_size, seq_len, channels in configs:
        oasr_fn, pytorch_fn = setup_batch_norm(batch_size, seq_len, channels)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_batch_norm_swish():
    """Benchmark fused BatchNorm + Swish."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("BatchNorm + Swish (Fused) Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)

    for batch_size, seq_len, channels in configs:
        oasr_fn, pytorch_fn = setup_batch_norm_swish(
            batch_size, seq_len, channels)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_batch_norm_activation():
    """Benchmark fused BatchNorm + Activation: OASR fused vs PyTorch separate."""
    import triton

    configs = [
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 512),
    ]
    activations = ["relu", "gelu", "swish"]

    print("\n" + "=" * 80)
    print("BatchNorm + Activation (Fused) Benchmark")
    print("=" * 80)
    print(
        f"\n{'Shape':<25} {'Act':<8} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 80)

    for batch_size, seq_len, channels in configs:
        for act_name in activations:
            oasr_fn, pytorch_fn = setup_batch_norm_activation(
                batch_size, seq_len, channels, act_name)

            oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
            pytorch_ms = triton.testing.do_bench(
                pytorch_fn, warmup=100, rep=500)

            speedup = pytorch_ms / oasr_ms
            shape_str = f"[{batch_size}, {seq_len}, {channels}]"
            print(
                f"{shape_str:<25} {act_name:<8} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_all():
    benchmark_batch_norm()
    benchmark_batch_norm_swish()
    benchmark_batch_norm_activation()


PROFILE_CONFIGS = {
    'batch_norm': (64, 250, 512),
    'batch_norm_swish': (64, 250, 512),
    'batch_norm_activation': (64, 250, 512),
}


def main():
    setup_funcs = {
        'batch_norm': lambda: setup_batch_norm(*PROFILE_CONFIGS['batch_norm']),
        'batch_norm_swish': lambda: setup_batch_norm_swish(*PROFILE_CONFIGS['batch_norm_swish']),
        'batch_norm_activation': lambda: setup_batch_norm_activation(*PROFILE_CONFIGS['batch_norm_activation']),
    }
    run_main("BatchNorm Kernels", PROFILE_CONFIGS, setup_funcs, benchmark_all)


if __name__ == "__main__":
    main()
