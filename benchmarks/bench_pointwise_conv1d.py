#!/usr/bin/env python3
"""Performance benchmarks for Pointwise Conv1D kernels (with and without activation)."""

import torch
import torch.nn.functional as F

import oasr
try:
    from .utils import run_main
except ImportError:
    from utils import run_main


def setup_pointwise_conv1d(batch_size, seq_len, in_ch, out_ch, dtype=torch.float16):
    """Setup tensors for pointwise conv1d."""
    x = torch.randn(batch_size, seq_len, in_ch, device='cuda', dtype=dtype)
    weight = torch.randn(out_ch, in_ch, device='cuda', dtype=dtype)
    bias = torch.randn(out_ch, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.pointwise_conv1d(
            x, weight, bias,
        )

    def pytorch_fn():
        return F.linear(x, weight, bias)

    return oasr_fn, pytorch_fn


def setup_pointwise_conv1d_activation(batch_size, seq_len, in_ch, out_ch, activation, dtype=torch.float16):
    """Setup tensors for pointwise conv1d with activation."""
    x = torch.randn(batch_size, seq_len, in_ch, device='cuda', dtype=dtype)
    weight = torch.randn(out_ch, in_ch, device='cuda', dtype=dtype)
    bias = torch.randn(out_ch, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.pointwise_conv1d_activation(
            x, weight, bias,
            activation
        )

    def pytorch_fn():
        if activation == oasr.ACTIVATION_SWISH:
            return F.silu(F.linear(x, weight, bias))
        elif activation == oasr.ACTIVATION_RELU:
            return F.relu(F.linear(x, weight, bias))
        elif activation == oasr.ACTIVATION_GELU:
            return F.gelu(F.linear(x, weight, bias))
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

    return oasr_fn, pytorch_fn


def benchmark_pointwise_conv1d():
    """Benchmark PointwiseConv1D."""
    import triton

    configs = [
        (32, 250, 256, 512),
        (64, 250, 256, 512),
        (64, 250, 512, 1024),
        (64, 250, 256, 2048),
        (64, 250, 512, 2048),
        (64, 500, 256, 512),
        (64, 500, 512, 1024),
    ]

    print("\n" + "=" * 70)
    print("PointwiseConv1D Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<40} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 80)

    for batch_size, seq_len, in_ch, out_ch in configs:
        oasr_fn, pytorch_fn = setup_pointwise_conv1d(
            batch_size, seq_len, in_ch, out_ch)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {in_ch}] -> {out_ch}"
        print(f"{shape_str:<40} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_pointwise_conv1d_activation():
    """Benchmark PointwiseConv1D with fused activation."""
    import triton

    configs = [
        (32, 250, 256, 512),
        (64, 250, 256, 512),
        (64, 250, 512, 1024),
        (64, 250, 256, 2048),
        (64, 250, 512, 2048),
        (64, 500, 256, 512),
        (64, 500, 512, 1024),
    ]

    print("\n" + "=" * 70)
    print("PointwiseConv1D + Activation Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<40} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 80)

    for batch_size, seq_len, in_ch, out_ch in configs:
        oasr_fn, pytorch_fn = setup_pointwise_conv1d_activation(
            batch_size, seq_len, in_ch, out_ch, oasr.ACTIVATION_SWISH)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {in_ch}] -> {out_ch}"
        print(f"{shape_str:<40} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_all():
    benchmark_pointwise_conv1d()
    benchmark_pointwise_conv1d_activation()


PROFILE_CONFIGS = {
    'pointwise_conv1d': (64, 250, 512, 1024),
}


def main():
    setup_funcs = {
        'pointwise_conv1d': lambda: setup_pointwise_conv1d(*PROFILE_CONFIGS['pointwise_conv1d']),
    }
    run_main("PointwiseConv1D Kernels", PROFILE_CONFIGS, setup_funcs, benchmark_all)


if __name__ == "__main__":
    main()
