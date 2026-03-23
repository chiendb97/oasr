#!/usr/bin/env python3
"""Performance benchmarks for RMSNorm kernel."""

import torch

import oasr
try:
    from .utils import run_main
except ImportError:
    from utils import run_main


def setup_rms_norm(batch_size, seq_len, hidden_size, dtype=torch.float16):
    """Setup tensors for RMSNorm."""
    eps = 1e-5
    x = torch.randn(batch_size, seq_len, hidden_size,
                    device='cuda', dtype=dtype)
    gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)

    def oasr_fn():
        return oasr.rms_norm(
            x, gamma, None, eps
        )

    def pytorch_fn():
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x / rms * gamma

    return oasr_fn, pytorch_fn


def benchmark_rms_norm():
    """Benchmark RMSNorm: OASR vs manual PyTorch. Conformer (B, T_enc, d_model)."""
    import triton

    configs = [
        (32, 250, 256),
        (64, 250, 256),
        (64, 250, 512),
        (64, 500, 512),
    ]

    print("\n" + "=" * 70)
    print("RMSNorm Benchmark (Conformer workload)")
    print("=" * 70)
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)

    for batch_size, seq_len, hidden_size in configs:
        oasr_fn, pytorch_fn = setup_rms_norm(batch_size, seq_len, hidden_size)

        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)

        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {hidden_size}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


PROFILE_CONFIGS = {
    'rms_norm': (64, 250, 256),
}


def main():
    setup_funcs = {
        'rms_norm': lambda: setup_rms_norm(*PROFILE_CONFIGS['rms_norm']),
    }
    run_main("RMSNorm Kernel", PROFILE_CONFIGS, setup_funcs, benchmark_rms_norm)


if __name__ == "__main__":
    main()
