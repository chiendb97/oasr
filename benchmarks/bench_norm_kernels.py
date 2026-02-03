#!/usr/bin/env python3
"""
Performance benchmarks for normalization kernels.
Uses triton.testing.do_bench for accurate GPU timing.
"""

import sys
sys.path.insert(0, 'python')

import torch
import triton

import oasr


def benchmark_layer_norm():
    """Benchmark LayerNorm: OASR vs PyTorch."""
    print("\n" + "=" * 70)
    print("LayerNorm Benchmark")
    print("=" * 70)
    
    configs = [
        # (batch_size, seq_len, hidden_size)
        (8, 256, 256),
        (16, 512, 512),
        (32, 512, 512),
        (32, 512, 768),
        (32, 512, 1024),
        (64, 1024, 512),
    ]
    
    eps = 1e-5
    dtype = torch.float16
    
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, hidden_size in configs:
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
        gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
        beta = torch.randn(hidden_size, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        # PyTorch LayerNorm
        layer_norm = torch.nn.LayerNorm(hidden_size, eps=eps, device='cuda', dtype=dtype)
        layer_norm.weight.data = gamma.clone()
        layer_norm.bias.data = beta.clone()
        
        # OASR kernel
        def oasr_fn():
            oasr.kernels.normalization.layer_norm(
                x.data_ptr(), output.data_ptr(),
                gamma.data_ptr(), beta.data_ptr(),
                batch_size, seq_len, hidden_size,
                eps, oasr.DataType.FP16
            )
        
        # PyTorch function
        def pytorch_fn():
            return layer_norm(x)
        
        # Benchmark
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {hidden_size}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_rms_norm():
    """Benchmark RMSNorm: OASR vs manual PyTorch."""
    print("\n" + "=" * 70)
    print("RMSNorm Benchmark")
    print("=" * 70)
    
    configs = [
        (16, 512, 512),
        (32, 512, 768),
        (32, 512, 1024),
        (64, 1024, 512),
    ]
    
    eps = 1e-5
    dtype = torch.float16
    
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, hidden_size in configs:
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
        gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        # OASR kernel
        def oasr_fn():
            oasr.kernels.normalization.rms_norm(
                x.data_ptr(), output.data_ptr(),
                gamma.data_ptr(),
                batch_size, seq_len, hidden_size,
                eps, oasr.DataType.FP16
            )
        
        # PyTorch reference (manual)
        def pytorch_fn():
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
            return x / rms * gamma
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {hidden_size}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_add_layer_norm():
    """Benchmark fused Add + LayerNorm."""
    print("\n" + "=" * 70)
    print("Add + LayerNorm (Fused) Benchmark")
    print("=" * 70)
    
    configs = [
        (16, 512, 512),
        (32, 512, 768),
        (32, 512, 1024),
    ]
    
    eps = 1e-5
    dtype = torch.float16
    
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, hidden_size in configs:
        x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
        residual = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=dtype)
        gamma = torch.randn(hidden_size, device='cuda', dtype=dtype)
        beta = torch.randn(hidden_size, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        layer_norm = torch.nn.LayerNorm(hidden_size, eps=eps, device='cuda', dtype=dtype)
        layer_norm.weight.data = gamma.clone()
        layer_norm.bias.data = beta.clone()
        
        # OASR fused kernel
        def oasr_fn():
            oasr.kernels.normalization.add_layer_norm(
                x.data_ptr(), residual.data_ptr(), output.data_ptr(),
                gamma.data_ptr(), beta.data_ptr(),
                batch_size, seq_len, hidden_size,
                eps, oasr.DataType.FP16
            )
        
        # PyTorch unfused
        def pytorch_fn():
            return layer_norm(x + residual)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {hidden_size}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_group_norm():
    """Benchmark GroupNorm."""
    print("\n" + "=" * 70)
    print("GroupNorm Benchmark")
    print("=" * 70)
    
    configs = [
        # (batch_size, seq_len, channels, num_groups)
        (8, 256, 256, 32),
        (16, 512, 512, 32),
        (32, 256, 512, 64),
    ]
    
    eps = 1e-5
    dtype = torch.float32  # GroupNorm often done in FP32
    
    print(f"\n{'Shape':<35} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 75)
    
    for batch_size, seq_len, channels, num_groups in configs:
        x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        gamma = torch.randn(channels, device='cuda', dtype=dtype)
        beta = torch.randn(channels, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        # OASR kernel
        def oasr_fn():
            oasr.kernels.normalization.group_norm(
                x.data_ptr(), output.data_ptr(),
                gamma.data_ptr(), beta.data_ptr(),
                batch_size, seq_len, channels, num_groups,
                eps, oasr.DataType.FP32
            )
        
        # PyTorch reference (manual for [N, T, C] format)
        channels_per_group = channels // num_groups
        def pytorch_fn():
            x_r = x.view(batch_size, seq_len, num_groups, channels_per_group)
            mean = x_r.mean(dim=-1, keepdim=True)
            var = x_r.var(dim=-1, keepdim=True, unbiased=False)
            x_norm = (x_r - mean) / torch.sqrt(var + eps)
            x_norm = x_norm.view(batch_size, seq_len, channels)
            return x_norm * gamma + beta
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}] g={num_groups}"
        print(f"{shape_str:<35} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_batch_norm():
    """Benchmark BatchNorm1D (inference)."""
    print("\n" + "=" * 70)
    print("BatchNorm1D (Inference) Benchmark")
    print("=" * 70)
    
    configs = [
        (16, 512, 256),
        (32, 512, 512),
        (64, 1024, 256),
    ]
    
    eps = 1e-5
    dtype = torch.float32
    
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, channels in configs:
        x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        gamma = torch.randn(channels, device='cuda', dtype=dtype)
        beta = torch.randn(channels, device='cuda', dtype=dtype)
        running_mean = torch.randn(channels, device='cuda', dtype=dtype)
        running_var = torch.abs(torch.randn(channels, device='cuda', dtype=dtype)) + 0.1
        output = torch.empty_like(x)
        
        # OASR kernel
        def oasr_fn():
            oasr.kernels.normalization.batch_norm_1d(
                x.data_ptr(), output.data_ptr(),
                gamma.data_ptr(), beta.data_ptr(),
                running_mean.data_ptr(), running_var.data_ptr(),
                batch_size, seq_len, channels,
                eps, oasr.DataType.FP32
            )
        
        # PyTorch reference
        def pytorch_fn():
            return (x - running_mean) / torch.sqrt(running_var + eps) * gamma + beta
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


if __name__ == "__main__":
    print("=" * 70)
    print("OASR Normalization Kernels - Performance Benchmarks")
    print("=" * 70)
    
    benchmark_layer_norm()
    benchmark_rms_norm()
    benchmark_add_layer_norm()
    benchmark_group_norm()
    benchmark_batch_norm()
    
    print("\n" + "=" * 70)
    print("Benchmarks complete!")
    print("=" * 70)
