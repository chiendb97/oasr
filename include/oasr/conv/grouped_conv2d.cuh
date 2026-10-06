// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Direct NHWC grouped/depthwise Conv2D kernels.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <oasr/common/math.h>

namespace oasr {
namespace conv {

namespace detail {

template <typename Activation>
__device__ __forceinline__ float groupedConv2dActivate(float value) {
    return Activation{}(value);
}

// True depthwise convolution (K == IC == groups) is the hot path in both
// Zipformer and Nemotron.  Threads span channels, which makes every NHWC
// activation access coalesced.  A block computes QTile adjacent output columns
// so every filter tap loaded by a thread is reused QTile times.
template <typename T, typename Activation, int KernelH, int KernelW, int QTile>
__global__ void depthwiseConv2dKernel(const T* __restrict__ input, const T* __restrict__ filter,
                                      const T* __restrict__ bias, T* __restrict__ output, int N,
                                      int H, int W, int channels, int R, int S, int P, int Q,
                                      int pad_h, int pad_w, int stride_h, int stride_w,
                                      int dilation_h, int dilation_w) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels)
        return;

    int q0 = blockIdx.y * QTile;
    int np = blockIdx.z;
    int n = np / P;
    int p = np - n * P;
    if (n >= N || p >= P || q0 >= Q)
        return;

    float bias_value = bias == nullptr ? 0.0f : static_cast<float>(bias[channel]);
    float acc[QTile];
#pragma unroll
    for (int q_it = 0; q_it < QTile; ++q_it) {
        acc[q_it] = bias_value;
    }
    int input_h0 = p * stride_h - pad_h;

    int r_end = KernelH == 0 ? R : KernelH;
    int s_end = KernelW == 0 ? S : KernelW;
#pragma unroll
    for (int r = 0; r < r_end; ++r) {
        int h = input_h0 + r * dilation_h;
        if (h < 0 || h >= H)
            continue;
#pragma unroll
        for (int s = 0; s < s_end; ++s) {
            int tap = r * S + s;
            float weight = static_cast<float>(filter[channel * R * S + tap]);
#pragma unroll
            for (int q_it = 0; q_it < QTile; ++q_it) {
                int q = q0 + q_it;
                int w = q * stride_w - pad_w + s * dilation_w;
                if (q < Q && w >= 0 && w < W) {
                    int input_offset = ((n * H + h) * W + w) * channels + channel;
                    acc[q_it] = fmaf(static_cast<float>(input[input_offset]), weight, acc[q_it]);
                }
            }
        }
    }

#pragma unroll
    for (int q_it = 0; q_it < QTile; ++q_it) {
        int q = q0 + q_it;
        if (q < Q) {
            int output_offset = ((n * P + p) * Q + q) * channels + channel;
            output[output_offset] = static_cast<T>(groupedConv2dActivate<Activation>(acc[q_it]));
        }
    }
}

// Shared-memory tiled depthwise convolution for the stride-1-in-width,
// undilated case (Zipformer's ConvNeXt 7x7).  The register-blocked kernel
// above reads every tap of every output from global memory: a 7x7 window is 49
// loads per output with only its QTile outputs of reuse, so on Zipformer's
// (64, 500, 19, 128) activation it measured 1.46 ms -- ~7x its memory bound --
// moving the window through L2 again for every output row.
//
// Here a block stages one input tile -- the (TP-1)*stride_h + R rows and
// 8*NQS + S - 1 columns its outputs read, for 32 channels -- in shared memory
// once.  A thread owns one (channel, output row) and a strip of 8 output
// columns; per filter row it reads the strip's 8 + S - 1 inputs into registers
// and applies all S taps from there, so each staged value is read S times from
// a register rather than S times from memory.
//
// Bit-identical to depthwiseConv2dKernel: every output accumulates its taps in
// the same (r, s) row-major fmaf order starting from the bias.  Taps that fall
// in the padding read a staged zero instead of being skipped, and
// fmaf(0, w, acc) == acc exactly.
template <int R, int S>
struct DepthwiseTile {
    static constexpr int kChannels = 32;  // one warp's lanes span channels
    static constexpr int kStrip = 8;      // output columns per thread
    static constexpr int kRows = 4;       // output rows per block (blockDim.y)
};

template <typename T, typename Activation, int R, int S>
__global__ void depthwiseConv2dTiledKernel(const T* __restrict__ input, const T* __restrict__ filter,
                                           const T* __restrict__ bias, T* __restrict__ output,
                                           int H, int W, int channels, int P, int Q, int pad_h,
                                           int pad_w, int stride_h, int q_tiles, int tile_rows,
                                           int tile_cols, bool vector_staging) {
    using Tile = DepthwiseTile<R, S>;
    constexpr int kC = Tile::kChannels;
    constexpr int kStrip = Tile::kStrip;
    extern __shared__ __align__(16) unsigned char depthwise_tile_smem[];
    T* tile = reinterpret_cast<T*>(depthwise_tile_smem);  // [tile_rows][tile_cols][kC]

    const int strips = blockDim.z;
    const int c_group = blockIdx.x / q_tiles;
    const int q0 = (blockIdx.x - c_group * q_tiles) * strips * kStrip;
    const int c0 = c_group * kC;
    const int p0 = blockIdx.y * blockDim.y;
    const int n = blockIdx.z;
    const int h0 = p0 * stride_h - pad_h;
    const int w0 = q0 - pad_w;

    // Stage the tile: positions x channels, zero outside the image.  Sixteen
    // bytes at a time when every channel run is whole and aligned -- the launcher
    // checks the base pointer and the channel count -- element by element
    // otherwise (and for a channel group that runs past the last channel).
    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int nthreads = blockDim.x * blockDim.y * blockDim.z;
    const int64_t image = static_cast<int64_t>(n) * H * W;
    constexpr int kVec = 16 / sizeof(T);
    if (vector_staging && c0 + kC <= channels) {
        constexpr int kVecs = kC / kVec;
        for (int i = tid; i < tile_rows * tile_cols * kVecs; i += nthreads) {
            const int v = i % kVecs;
            const int pos = i / kVecs;
            const int col = pos % tile_cols;
            const int row = pos / tile_cols;
            const int h = h0 + row;
            const int w = w0 + col;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (h >= 0 && h < H && w >= 0 && w < W) {
                val = *reinterpret_cast<const uint4*>(
                    input + (image + static_cast<int64_t>(h) * W + w) * channels + c0 + v * kVec);
            }
            reinterpret_cast<uint4*>(tile)[i] = val;
        }
    } else {
        for (int i = tid; i < tile_rows * tile_cols * kC; i += nthreads) {
            const int c = i % kC;
            const int pos = i / kC;
            const int col = pos % tile_cols;
            const int row = pos / tile_cols;
            const int h = h0 + row;
            const int w = w0 + col;
            T v = static_cast<T>(0.0f);
            if (h >= 0 && h < H && w >= 0 && w < W && c0 + c < channels) {
                v = input[(image + static_cast<int64_t>(h) * W + w) * channels + c0 + c];
            }
            tile[i] = v;
        }
    }
    __syncthreads();

    const int c = c0 + threadIdx.x;
    const int p = p0 + threadIdx.y;
    if (c >= channels || p >= P) return;
    const int qs = q0 + threadIdx.z * kStrip;
    if (qs >= Q) return;

    float acc[kStrip];
    const float b = bias == nullptr ? 0.0f : static_cast<float>(bias[c]);
#pragma unroll
    for (int j = 0; j < kStrip; ++j) acc[j] = b;

    const T* wc = filter + static_cast<int64_t>(c) * R * S;
    const int col0 = threadIdx.z * kStrip;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const T* row = tile + ((threadIdx.y * stride_h + r) * tile_cols + col0) * kC + threadIdx.x;
        float x[kStrip + S - 1];
#pragma unroll
        for (int k = 0; k < kStrip + S - 1; ++k) x[k] = static_cast<float>(row[k * kC]);
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const float wt = static_cast<float>(wc[r * S + s]);
#pragma unroll
            for (int j = 0; j < kStrip; ++j) acc[j] = fmaf(x[j + s], wt, acc[j]);
        }
    }

    T* out = output + ((static_cast<int64_t>(n) * P + p) * Q + qs) * channels + c;
#pragma unroll
    for (int j = 0; j < kStrip; ++j) {
        if (qs + j < Q) {
            out[static_cast<int64_t>(j) * channels] =
                static_cast<T>(groupedConv2dActivate<Activation>(acc[j]));
        }
    }
}

// Launch the tiled kernel; ``false`` when the shape is outside its envelope
// (the caller then takes the register-blocked kernel).
template <typename T, typename Activation, int R, int S>
bool launchDepthwiseConv2dTiled(const T* input, const T* filter, const T* bias, T* output, int N,
                                int H, int W, int channels, int P, int Q, int pad_h, int pad_w,
                                int stride_h, int stride_w, int dilation_h, int dilation_w,
                                cudaStream_t stream) {
    using Tile = DepthwiseTile<R, S>;
    if (stride_w != 1 || dilation_h != 1 || dilation_w != 1 || N > 65535) return false;
    const int strips = Q <= 8 ? 1 : (Q <= 16 ? 2 : (Q <= 24 ? 3 : 4));
    const int q_tiles = (Q + strips * Tile::kStrip - 1) / (strips * Tile::kStrip);
    const int tile_rows = (Tile::kRows - 1) * stride_h + R;
    const int tile_cols = strips * Tile::kStrip + S - 1;
    const size_t smem = static_cast<size_t>(tile_rows) * tile_cols * Tile::kChannels * sizeof(T);
    if (smem > 48 * 1024) return false;
    const int c_groups = (channels + Tile::kChannels - 1) / Tile::kChannels;
    const int p_tiles = (P + Tile::kRows - 1) / Tile::kRows;
    if (static_cast<int64_t>(c_groups) * q_tiles > 0x7fffffff || p_tiles > 65535) return false;
    dim3 block(Tile::kChannels, Tile::kRows, strips);
    dim3 grid(c_groups * q_tiles, p_tiles, N);
    const bool vector_staging = channels % (16 / static_cast<int>(sizeof(T))) == 0 &&
                                reinterpret_cast<uintptr_t>(input) % 16 == 0;
    depthwiseConv2dTiledKernel<T, Activation, R, S><<<grid, block, smem, stream>>>(
        input, filter, bias, output, H, W, channels, P, Q, pad_h, pad_w, stride_h, q_tiles,
        tile_rows, tile_cols, vector_staging);
    return true;
}

// General grouped convolution.  Each output element belongs to one output
// channel, whose group selects the contiguous input-channel slice it reduces.
// This is primarily the correctness/general-coverage path; true depthwise
// traffic takes the packed specialization above.
template <typename T, typename Activation, int KernelH, int KernelW>
__global__ void groupedConv2dKernel(const T* __restrict__ input, const T* __restrict__ filter,
                                    const T* __restrict__ bias, T* __restrict__ output, int N,
                                    int H, int W, int IC, int K, int R, int S, int P, int Q,
                                    int pad_h, int pad_w, int stride_h, int stride_w,
                                    int dilation_h, int dilation_w, int groups,
                                    int64_t num_outputs) {
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= num_outputs)
        return;

    int out_channel = linear % K;
    int64_t position = linear / K;
    int q = position % Q;
    position /= Q;
    int p = position % P;
    int n = position / P;

    int in_channels_per_group = IC / groups;
    int out_channels_per_group = K / groups;
    int group = out_channel / out_channels_per_group;
    int input_channel0 = group * in_channels_per_group;
    int input_h0 = p * stride_h - pad_h;
    int input_w0 = q * stride_w - pad_w;
    float acc = bias == nullptr ? 0.0f : static_cast<float>(bias[out_channel]);

    int r_end = KernelH == 0 ? R : KernelH;
    int s_end = KernelW == 0 ? S : KernelW;
#pragma unroll
    for (int r = 0; r < r_end; ++r) {
        int h = input_h0 + r * dilation_h;
        if (h < 0 || h >= H)
            continue;
#pragma unroll
        for (int s = 0; s < s_end; ++s) {
            int w = input_w0 + s * dilation_w;
            if (w < 0 || w >= W)
                continue;
            int input_base = ((n * H + h) * W + w) * IC + input_channel0;
            int filter_base = ((out_channel * R + r) * S + s) * in_channels_per_group;
            for (int c = 0; c < in_channels_per_group; ++c) {
                acc = fmaf(static_cast<float>(input[input_base + c]),
                           static_cast<float>(filter[filter_base + c]), acc);
            }
        }
    }
    output[linear] = static_cast<T>(groupedConv2dActivate<Activation>(acc));
}

template <typename T, typename Activation, int KernelH, int KernelW>
cudaError_t launchGroupedConv2d(const T* input, const T* filter, const T* bias, T* output, int N,
                                int H, int W, int IC, int K, int R, int S, int pad_h, int pad_w,
                                int stride_h, int stride_w, int dilation_h, int dilation_w,
                                int groups, cudaStream_t stream) {
    int P = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Q = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    if constexpr (KernelH == 7 && KernelW == 7) {
        if (groups == IC && K == IC &&
            launchDepthwiseConv2dTiled<T, Activation, 7, 7>(input, filter, bias, output, N, H, W,
                                                            IC, P, Q, pad_h, pad_w, stride_h,
                                                            stride_w, dilation_h, dilation_w,
                                                            stream)) {
            return cudaGetLastError();
        }
    }
    if (groups == IC && K == IC && N * P <= 65535) {
        int threads = IC <= 32 ? 32 : (IC <= 64 ? 64 : (IC <= 128 ? 128 : 256));
        constexpr int kQTile = KernelH == 3 && KernelW == 3 ? 4 : 2;
        dim3 grid((IC + threads - 1) / threads, (Q + kQTile - 1) / kQTile, N * P);
        depthwiseConv2dKernel<T, Activation, KernelH, KernelW, kQTile>
            <<<grid, threads, 0, stream>>>(input, filter, bias, output, N, H, W, IC, R, S, P, Q,
                                           pad_h, pad_w, stride_h, stride_w, dilation_h,
                                           dilation_w);
    } else {
        constexpr int kThreads = 256;
        int64_t num_outputs = static_cast<int64_t>(N) * P * Q * K;
        int blocks = static_cast<int>((num_outputs + kThreads - 1) / kThreads);
        groupedConv2dKernel<T, Activation, KernelH, KernelW><<<blocks, kThreads, 0, stream>>>(
            input, filter, bias, output, N, H, W, IC, K, R, S, P, Q, pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, groups, num_outputs);
    }
    return cudaGetLastError();
}

}  // namespace detail

template <typename T, typename Activation = IdentityActivation>
cudaError_t GroupedConv2D(const T* input, const T* filter, const T* bias, T* output, int N, int H,
                          int W, int IC, int K, int R, int S, int pad_h, int pad_w, int stride_h,
                          int stride_w, int dilation_h, int dilation_w, int groups,
                          cudaStream_t stream) {
    if (R == 3 && S == 3) {
        return detail::launchGroupedConv2d<T, Activation, 3, 3>(
            input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, groups, stream);
    }
    if (R == 7 && S == 7) {
        return detail::launchGroupedConv2d<T, Activation, 7, 7>(
            input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, groups, stream);
    }
    return detail::launchGroupedConv2d<T, Activation, 0, 0>(
        input, filter, bias, output, N, H, W, IC, K, R, S, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, groups, stream);
}

}  // namespace conv
}  // namespace oasr
