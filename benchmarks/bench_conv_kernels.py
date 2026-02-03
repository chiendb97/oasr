#!/usr/bin/env python3
"""
Performance benchmarks for convolution kernels.
Uses triton.testing.do_bench for accurate GPU timing.
"""

import sys
sys.path.insert(0, 'python')

import torch
import torch.nn.functional as F
import triton

import oasr

# Import types
try:
    from oasr import ConvType, ActivationType
except ImportError:
    from oasr._C import ConvType, ActivationType
    oasr.ConvType = ConvType
    oasr.ActivationType = ActivationType


def benchmark_depthwise_conv1d():
    """Benchmark DepthwiseConv1D: OASR vs PyTorch."""
    print("\n" + "=" * 70)
    print("DepthwiseConv1D Benchmark")
    print("=" * 70)
    
    configs = [
        # (batch_size, seq_len, channels, kernel_size)
        (16, 256, 256, 7),
        (16, 256, 256, 31),
        (32, 512, 512, 31),
        (32, 512, 512, 63),
        (64, 512, 256, 31),
        (64, 1024, 512, 31),
    ]
    
    dtype = torch.float16
    
    print(f"\n{'Shape':<35} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 75)
    
    for batch_size, seq_len, channels, kernel_size in configs:
        padding = kernel_size // 2
        
        x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        weight = torch.randn(channels, kernel_size, device='cuda', dtype=dtype)
        bias = torch.randn(channels, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        # PyTorch setup
        x_nchw = x.permute(0, 2, 1).contiguous()
        weight_pt = weight.view(channels, 1, kernel_size)
        
        # OASR kernel
        def oasr_fn():
            oasr.kernels.convolution.depthwise_conv1d(
                x.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
                batch_size, seq_len, channels, kernel_size, padding,
                False, oasr.DataType.FP16
            )
        
        # PyTorch function
        def pytorch_fn():
            return F.conv1d(x_nchw, weight_pt, bias, padding=padding, groups=channels)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}] k={kernel_size}"
        print(f"{shape_str:<35} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_depthwise_conv1d_causal():
    """Benchmark causal DepthwiseConv1D."""
    print("\n" + "=" * 70)
    print("DepthwiseConv1D (Causal) Benchmark")
    print("=" * 70)
    
    configs = [
        (16, 256, 256, 31),
        (32, 512, 512, 31),
        (64, 1024, 512, 31),
    ]
    
    dtype = torch.float16
    
    print(f"\n{'Shape':<35} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 75)
    
    for batch_size, seq_len, channels, kernel_size in configs:
        x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        weight = torch.randn(channels, kernel_size, device='cuda', dtype=dtype)
        bias = torch.randn(channels, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        # PyTorch setup
        x_nchw = x.permute(0, 2, 1).contiguous()
        weight_pt = weight.view(channels, 1, kernel_size)
        
        # OASR causal kernel
        def oasr_fn():
            oasr.kernels.convolution.depthwise_conv1d(
                x.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
                batch_size, seq_len, channels, kernel_size, 0,
                True, oasr.DataType.FP16  # is_causal=True
            )
        
        # PyTorch causal (manual padding)
        def pytorch_fn():
            x_padded = F.pad(x_nchw, (kernel_size - 1, 0))
            return F.conv1d(x_padded, weight_pt, bias, groups=channels)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}] k={kernel_size}"
        print(f"{shape_str:<35} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_pointwise_conv1d():
    """Benchmark PointwiseConv1D (linear projection)."""
    print("\n" + "=" * 70)
    print("PointwiseConv1D (Linear) Benchmark")
    print("=" * 70)
    
    configs = [
        # (batch_size, seq_len, in_channels, out_channels)
        (16, 256, 256, 512),
        (32, 512, 512, 1024),
        (32, 512, 512, 2048),
        (64, 512, 768, 768),
    ]
    
    dtype = torch.float16
    
    print(f"\n{'Shape':<40} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 80)
    
    for batch_size, seq_len, in_ch, out_ch in configs:
        x = torch.randn(batch_size, seq_len, in_ch, device='cuda', dtype=dtype)
        weight = torch.randn(out_ch, in_ch, device='cuda', dtype=dtype)
        bias = torch.randn(out_ch, device='cuda', dtype=dtype)
        output = torch.empty(batch_size, seq_len, out_ch, device='cuda', dtype=dtype)
        
        # OASR kernel
        def oasr_fn():
            oasr.kernels.convolution.pointwise_conv1d(
                x.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
                batch_size, seq_len, in_ch, out_ch,
                oasr.ActivationType.SWISH, False,
                oasr.DataType.FP16
            )
        
        # PyTorch F.linear
        def pytorch_fn():
            return F.linear(x, weight, bias)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {in_ch}] -> {out_ch}"
        print(f"{shape_str:<40} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_glu():
    """Benchmark GLU activation."""
    print("\n" + "=" * 70)
    print("GLU Activation Benchmark")
    print("=" * 70)
    
    configs = [
        (16, 256, 256),
        (32, 512, 512),
        (64, 512, 1024),
        (64, 1024, 512),
    ]
    
    dtype = torch.float16
    
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, channels in configs:
        x = torch.randn(batch_size, seq_len, 2 * channels, device='cuda', dtype=dtype)
        output = torch.empty(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        
        # OASR kernel
        def oasr_fn():
            oasr.kernels.convolution.glu(
                x.data_ptr(), output.data_ptr(),
                batch_size, seq_len, channels,
                oasr.DataType.FP16
            )
        
        # PyTorch F.glu
        def pytorch_fn():
            return F.glu(x, dim=-1)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {2*channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_swish():
    """Benchmark Swish (SiLU) activation."""
    print("\n" + "=" * 70)
    print("Swish Activation Benchmark")
    print("=" * 70)
    
    configs = [
        (32, 512, 512),
        (64, 512, 1024),
        (64, 1024, 512),
    ]
    
    dtype = torch.float16
    
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, channels in configs:
        x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        output = torch.empty_like(x)
        
        # OASR kernel
        def oasr_fn():
            oasr.kernels.convolution.swish(
                x.data_ptr(), output.data_ptr(),
                batch_size, seq_len, channels,
                oasr.DataType.FP16
            )
        
        # PyTorch F.silu
        def pytorch_fn():
            return F.silu(x)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_batch_norm_swish():
    """Benchmark fused BatchNorm + Swish."""
    print("\n" + "=" * 70)
    print("BatchNorm + Swish (Fused) Benchmark")
    print("=" * 70)
    
    configs = [
        (16, 512, 256),
        (32, 512, 512),
        (64, 512, 512),
    ]
    
    eps = 1e-5
    dtype = torch.float16
    
    print(f"\n{'Shape':<30} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    
    for batch_size, seq_len, channels in configs:
        x = torch.randn(batch_size, seq_len, channels, device='cuda', dtype=dtype)
        gamma = torch.randn(channels, device='cuda', dtype=dtype)
        beta = torch.randn(channels, device='cuda', dtype=dtype)
        running_mean = torch.randn(channels, device='cuda', dtype=dtype)
        running_var = torch.abs(torch.randn(channels, device='cuda', dtype=dtype)) + 0.1
        output = torch.empty_like(x)
        
        # OASR fused kernel
        def oasr_fn():
            oasr.kernels.convolution.batch_norm_swish(
                x.data_ptr(), output.data_ptr(),
                gamma.data_ptr(), beta.data_ptr(),
                running_mean.data_ptr(), running_var.data_ptr(),
                batch_size, seq_len, channels,
                eps, oasr.DataType.FP16
            )
        
        # PyTorch unfused
        def pytorch_fn():
            bn_out = (x - running_mean) / torch.sqrt(running_var + eps) * gamma + beta
            return F.silu(bn_out)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        shape_str = f"[{batch_size}, {seq_len}, {channels}]"
        print(f"{shape_str:<30} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


def benchmark_conformer_conv_block():
    """Benchmark full Conformer conv block."""
    print("\n" + "=" * 70)
    print("Conformer Conv Block (End-to-End) Benchmark")
    print("=" * 70)
    
    configs = [
        # (batch_size, seq_len, d_model, kernel_size)
        (16, 256, 256, 31),
        (32, 512, 512, 31),
        (32, 512, 256, 31),
    ]
    
    dtype = torch.float16
    
    print(f"\n{'Config':<35} {'OASR (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 75)
    
    for batch_size, seq_len, d_model, kernel_size in configs:
        x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=dtype)
        
        # Weights
        pw1_weight = torch.randn(2 * d_model, d_model, device='cuda', dtype=dtype)
        pw1_bias = torch.randn(2 * d_model, device='cuda', dtype=dtype)
        dw_weight = torch.randn(d_model, kernel_size, device='cuda', dtype=dtype)
        dw_bias = torch.randn(d_model, device='cuda', dtype=dtype)
        pw2_weight = torch.randn(d_model, d_model, device='cuda', dtype=dtype)
        pw2_bias = torch.randn(d_model, device='cuda', dtype=dtype)
        
        # Output buffers for OASR
        pw1_out = torch.empty(batch_size, seq_len, 2 * d_model, device='cuda', dtype=dtype)
        glu_out = torch.empty(batch_size, seq_len, d_model, device='cuda', dtype=dtype)
        dw_out = torch.empty(batch_size, seq_len, d_model, device='cuda', dtype=dtype)
        swish_out = torch.empty_like(dw_out)
        output = torch.empty(batch_size, seq_len, d_model, device='cuda', dtype=dtype)
        
        # PyTorch setup
        dw_weight_pt = dw_weight.view(d_model, 1, kernel_size)
        
        # OASR pipeline
        def oasr_fn():
            oasr.kernels.convolution.pointwise_conv1d(
                x.data_ptr(), pw1_weight.data_ptr(), pw1_bias.data_ptr(), pw1_out.data_ptr(),
                batch_size, seq_len, d_model, 2 * d_model,
                oasr.ActivationType.SWISH, False, oasr.DataType.FP16
            )
            oasr.kernels.convolution.glu(
                pw1_out.data_ptr(), glu_out.data_ptr(),
                batch_size, seq_len, d_model, oasr.DataType.FP16
            )
            oasr.kernels.convolution.depthwise_conv1d(
                glu_out.data_ptr(), dw_weight.data_ptr(), dw_bias.data_ptr(), dw_out.data_ptr(),
                batch_size, seq_len, d_model, kernel_size, kernel_size // 2,
                False, oasr.DataType.FP16
            )
            oasr.kernels.convolution.swish(
                dw_out.data_ptr(), swish_out.data_ptr(),
                batch_size, seq_len, d_model, oasr.DataType.FP16
            )
            oasr.kernels.convolution.pointwise_conv1d(
                swish_out.data_ptr(), pw2_weight.data_ptr(), pw2_bias.data_ptr(), output.data_ptr(),
                batch_size, seq_len, d_model, d_model,
                oasr.ActivationType.SWISH, False, oasr.DataType.FP16
            )
        
        # PyTorch pipeline
        def pytorch_fn():
            pw1 = F.linear(x, pw1_weight, pw1_bias)
            glu = F.glu(pw1, dim=-1)
            glu_nchw = glu.permute(0, 2, 1)
            dw_nchw = F.conv1d(glu_nchw, dw_weight_pt, dw_bias, padding=kernel_size // 2, groups=d_model)
            dw = dw_nchw.permute(0, 2, 1)
            swish = F.silu(dw)
            return F.linear(swish, pw2_weight, pw2_bias)
        
        oasr_ms = triton.testing.do_bench(oasr_fn, warmup=100, rep=500)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=100, rep=500)
        
        speedup = pytorch_ms / oasr_ms
        config_str = f"[{batch_size}, {seq_len}, {d_model}] k={kernel_size}"
        print(f"{config_str:<35} {oasr_ms:<12.4f} {pytorch_ms:<14.4f} {speedup:<10.2f}x")


if __name__ == "__main__":
    print("=" * 70)
    print("OASR Convolution Kernels - Performance Benchmarks")
    print("=" * 70)
    
    benchmark_depthwise_conv1d()
    benchmark_depthwise_conv1d_causal()
    benchmark_pointwise_conv1d()
    benchmark_glu()
    benchmark_swish()
    benchmark_batch_norm_swish()
    benchmark_conformer_conv_block()
    
    print("\n" + "=" * 70)
    print("Benchmarks complete!")
    print("=" * 70)
