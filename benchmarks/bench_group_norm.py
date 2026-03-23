#!/usr/bin/env python3
"""Performance benchmarks for GroupNorm kernel."""

import torch

import oasr
try:
    from .utils import run_main
except ImportError:
    from utils import run_main


def setup_group_norm(batch_size, seq_len, channels, num_groups, dtype=torch.float32):
    """Setup tensors for GroupNorm."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
    gamma = torch.randn(channels, device='cuda', dtype=dtype)
    beta = torch.randn(channels, device='cuda', dtype=dtype)

    channels_per_group = channels // num_groups

    def oasr_fn():
        return oasr.group_norm(
            x, gamma, beta, num_groups, eps
        )

    def pytorch_fn():
        x_r = x.view(batch_size, seq_len, num_groups, channels_per_group)
        mean = x_r.mean(dim=-1, keepdim=True)
        var = x_r.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_r - mean) / torch.sqrt(var + eps)
        x_norm = x_norm.view(batch_size, seq_len, channels)
        return x_norm * gamma + beta

    return oasr_fn, pytorch_fn


def benchmark_group_norm():
    """Benchmark GroupNorm. Conformer (B, T_enc, channels, num_groups)."""
    import triton

    configs = [
        (32, 250, 256, 32),
        (64, 250, 256, 32),
        (64, 250, 512, 64),
        (64, 500, 512, 64),
    ]

    print("\n" + "=" * 70)
    print("GroupNorm Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<35} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 75)

    for batch_size, seq_len, channels, num_groups in configs:
        oasr_fn, pytorch_fn = setup_group_norm(
            batch_size, seq_len, channels, num_groups)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}] g={num_groups}"
        print(f"{shape_str:<35} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


PROFILE_CONFIGS = {
    'group_norm': (64, 250, 512, 64),
}


def main():
    setup_funcs = {
        'group_norm': lambda: setup_group_norm(*PROFILE_CONFIGS['group_norm']),
    }
    run_main("GroupNorm Kernel", PROFILE_CONFIGS, setup_funcs, benchmark_group_norm)


if __name__ == "__main__":
    main()
