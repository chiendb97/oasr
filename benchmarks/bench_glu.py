#!/usr/bin/env python3
"""Performance benchmarks for GLU activation kernel."""

import torch
import torch.nn.functional as F

import oasr
try:
    from .utils import run_main
except ImportError:
    from utils import run_main


def setup_glu(batch_size, seq_len, channels, dtype=torch.float16):
    """Setup tensors for GLU."""
    x = torch.randn(batch_size, seq_len, 2 * channels,
                    device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.glu(x)

    def pytorch_fn():
        return F.glu(x, dim=-1)

    return oasr_fn, pytorch_fn


def benchmark_glu():
    """Benchmark GLU activation."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 256),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("GLU Activation Benchmark")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)

    for batch_size, seq_len, channels in configs:
        oasr_fn, pytorch_fn = setup_glu(batch_size, seq_len, channels)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {2*channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


PROFILE_CONFIGS = {
    'glu': (64, 250, 512),
}


def main():
    setup_funcs = {
        'glu': lambda: setup_glu(*PROFILE_CONFIGS['glu']),
    }
    run_main("GLU Activation Kernel", PROFILE_CONFIGS, setup_funcs, benchmark_glu)


if __name__ == "__main__":
    main()
