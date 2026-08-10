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
