// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA pooling kernels -- no framework dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <oasr/common/vec_dtypes.h>

namespace oasr {
namespace pooling {

// BTC is the native layout of the encoder residual stream.  Assigning one CTA
// to each (batch, output-time) row lets adjacent threads walk channels in
// contiguous 128-bit vectors; no BCT transpose is needed around the pool.
template <typename T, int VecSize>
__global__ void AvgPool1dK2S2Kernel(const T* __restrict__ input, T* __restrict__ output,
                                    int input_length, int output_length, int channels) {
    const int batch = static_cast<int>(blockIdx.z);
    const int output_t = static_cast<int>(blockIdx.y);
    const int vector_channels = channels / VecSize;

    for (int vector_c = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         vector_c < vector_channels; vector_c += gridDim.x * blockDim.x) {
        const int channel = vector_c * VecSize;
        const int input_t = output_t * 2;
        const int64_t input_offset =
            (static_cast<int64_t>(batch) * input_length + input_t) * channels + channel;
        const int64_t output_offset =
            (static_cast<int64_t>(batch) * output_length + output_t) * channels + channel;

        Vec<T, VecSize> first;
        Vec<T, VecSize> second;
        first.load(input + input_offset);
        second.load(input + input_offset + channels);

        Vec<T, VecSize> result;
#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            const float sum = static_cast<float>(first[i]) + static_cast<float>(second[i]);
            result[i] = static_cast<T>(sum * 0.5f);
        }
        result.store(output + output_offset);
    }
}

template <typename T, int VecSize>
__global__ void AvgPool1dGenericKernel(const T* __restrict__ input, T* __restrict__ output,
                                       int input_length, int output_length, int channels,
                                       int kernel_size, int stride, int padding,
                                       bool count_include_pad) {
    const int batch = static_cast<int>(blockIdx.z);
    const int output_t = static_cast<int>(blockIdx.y);
    const int vector_channels = channels / VecSize;

    const int window_start = output_t * stride - padding;
    const int window_end = window_start + kernel_size;
    const int valid_start = window_start > 0 ? window_start : 0;
    const int valid_end = window_end < input_length ? window_end : input_length;
    const int padded_end =
        window_end < input_length + padding ? window_end : input_length + padding;
    const int divisor = count_include_pad ? padded_end - window_start : valid_end - valid_start;
    const float scale = 1.0f / static_cast<float>(divisor);

    for (int vector_c = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         vector_c < vector_channels; vector_c += gridDim.x * blockDim.x) {
        const int channel = vector_c * VecSize;
        float accum[VecSize] = {};

        for (int input_t = valid_start; input_t < valid_end; ++input_t) {
            const int64_t input_offset =
                (static_cast<int64_t>(batch) * input_length + input_t) * channels + channel;
            Vec<T, VecSize> values;
            values.load(input + input_offset);
#pragma unroll
            for (int i = 0; i < VecSize; ++i) {
                accum[i] += static_cast<float>(values[i]);
            }
        }

        Vec<T, VecSize> result;
#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            result[i] = static_cast<T>(accum[i] * scale);
        }
        const int64_t output_offset =
            (static_cast<int64_t>(batch) * output_length + output_t) * channels + channel;
        result.store(output + output_offset);
    }
}

// The CTA-per-(batch, output-time) mapping above assigns *channels* to threads,
// so it fills a warp only when the tensor is wide.  A per-frame trace is one
// channel: `vector_channels` is 1, 31 of every 32 threads idle, and the launch
// degrades to a thread per CTA -- measured at 0.50x torch on (32, 1500, 1)
// while the same kernel is 2.15x on (4, 500, 512).  These flatten
// (batch, output-time, channel) across threads instead, which fills the
// machine at any width but cannot vectorize the channel axis.  The launcher
// picks between them on `vector_channels < 32`: whether the wide mapping can
// fill a single warp is exactly the question.
template <typename T>
__global__ void AvgPool1dNarrowKernel(const T* __restrict__ input, T* __restrict__ output,
                                      int input_length, int output_length, int channels,
                                      int kernel_size, int stride, int padding,
                                      bool count_include_pad, int64_t total) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % channels);
        const int64_t row = index / channels;
        const int output_t = static_cast<int>(row % output_length);
        const int batch = static_cast<int>(row / output_length);

        const int window_start = output_t * stride - padding;
        const int window_end = window_start + kernel_size;
        const int valid_start = window_start > 0 ? window_start : 0;
        const int valid_end = window_end < input_length ? window_end : input_length;
        const int padded_end =
            window_end < input_length + padding ? window_end : input_length + padding;
        const int divisor = count_include_pad ? padded_end - window_start : valid_end - valid_start;

        const T* base = input + static_cast<int64_t>(batch) * input_length * channels + channel;
        // Reciprocal-multiply, not divide: the wide kernel above scales by
        // 1/divisor, and the two paths must agree bit-for-bit or the threshold
        // that chooses between them becomes observable in the output.
        const float scale = 1.0f / static_cast<float>(divisor);
        float accum = 0.0f;
        for (int input_t = valid_start; input_t < valid_end; ++input_t) {
            accum += static_cast<float>(base[static_cast<int64_t>(input_t) * channels]);
        }
        output[index] = static_cast<T>(accum * scale);
    }
}

template <typename T>
__global__ void MaxPool1dNarrowKernel(const T* __restrict__ input, T* __restrict__ output,
                                      int input_length, int output_length, int channels,
                                      int kernel_size, int stride, int padding, int64_t total) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index % channels);
        const int64_t row = index / channels;
        const int output_t = static_cast<int>(row % output_length);
        const int batch = static_cast<int>(row / output_length);

        const int window_start = output_t * stride - padding;
        const int window_end = window_start + kernel_size;
        const int valid_start = window_start > 0 ? window_start : 0;
        const int valid_end = window_end < input_length ? window_end : input_length;

        const T* base = input + static_cast<int64_t>(batch) * input_length * channels + channel;
        float best = static_cast<float>(base[static_cast<int64_t>(valid_start) * channels]);
        for (int input_t = valid_start + 1; input_t < valid_end; ++input_t) {
            const float value = static_cast<float>(base[static_cast<int64_t>(input_t) * channels]);
            best = best > value ? best : value;
        }
        output[index] = static_cast<T>(best);
    }
}

// Threads per CTA and CTA count for the narrow path, capped at the grid limit
// so a long trace still launches (the wide path is bounded by 65535 rows).
struct NarrowLaunch {
    int block;
    int grid;
};

inline NarrowLaunch NarrowLaunchFor(int64_t total) {
    constexpr int kBlock = 256;
    const int64_t blocks = (total + kBlock - 1) / kBlock;
    return NarrowLaunch{kBlock, static_cast<int>(blocks < 65535 ? blocks : 65535)};
}

//: The wide mapping fills a warp only above this many vector channels.
constexpr int kNarrowChannelThreshold = 32;

template <typename T>
cudaError_t AvgPool1d(const T* input, T* output, int batch_size, int input_length,
                      int output_length, int channels, int kernel_size, int stride, int padding,
                      bool count_include_pad, cudaStream_t stream) {
    if (batch_size == 0) {
        return cudaSuccess;
    }

    constexpr int kVectorSize = VecTypeTrait<T>::VecSize;
    const bool vectorized = channels % kVectorSize == 0;
    const int vector_size = vectorized ? kVectorSize : 1;
    const int vector_channels = channels / vector_size;

    if (vector_channels < kNarrowChannelThreshold) {
        const int64_t total = static_cast<int64_t>(batch_size) * output_length * channels;
        const NarrowLaunch launch = NarrowLaunchFor(total);
        AvgPool1dNarrowKernel<T><<<launch.grid, launch.block, 0, stream>>>(
            input, output, input_length, output_length, channels, kernel_size, stride, padding,
            count_include_pad, total);
        return cudaGetLastError();
    }

    const int block_size = std::min(256, std::max(32, ((vector_channels + 31) / 32) * 32));
    const int channel_blocks = (vector_channels + block_size - 1) / block_size;
    const dim3 grid(channel_blocks, output_length, batch_size);

    // The only in-tree call is exactly this case.  Removing window bounds and
    // divisor branches is material for this bandwidth-bound, sub-0.1 ms op.
    const bool use_k2_s2 = kernel_size == 2 && stride == 2 && padding == 0 &&
                           input_length % 2 == 0 && output_length == input_length / 2;
    if (vectorized) {
        if (use_k2_s2) {
            AvgPool1dK2S2Kernel<T, kVectorSize><<<grid, block_size, 0, stream>>>(
                input, output, input_length, output_length, channels);
        } else {
            AvgPool1dGenericKernel<T, kVectorSize><<<grid, block_size, 0, stream>>>(
                input, output, input_length, output_length, channels, kernel_size, stride, padding,
                count_include_pad);
        }
    } else if (use_k2_s2) {
        AvgPool1dK2S2Kernel<T, 1>
            <<<grid, block_size, 0, stream>>>(input, output, input_length, output_length, channels);
    } else {
        AvgPool1dGenericKernel<T, 1>
            <<<grid, block_size, 0, stream>>>(input, output, input_length, output_length, channels,
                                              kernel_size, stride, padding, count_include_pad);
    }
    return cudaGetLastError();
}

// Max pooling shares AvgPool1d's CTA mapping: one CTA per (batch, output-time)
// row, adjacent threads walking channels in contiguous 128-bit vectors.  What
// differs is the identity and the padding rule.  PyTorch pads max pooling with
// -inf rather than zero, so a padded position must contribute nothing; the
// launcher's `padding <= kernel_size / 2` bound guarantees every window still
// overlaps at least one real element, so the reduction is never empty and no
// -inf can survive into the output.
template <typename T, int VecSize>
__global__ void MaxPool1dGenericKernel(const T* __restrict__ input, T* __restrict__ output,
                                       int input_length, int output_length, int channels,
                                       int kernel_size, int stride, int padding) {
    const int batch = static_cast<int>(blockIdx.z);
    const int output_t = static_cast<int>(blockIdx.y);
    const int vector_channels = channels / VecSize;

    const int window_start = output_t * stride - padding;
    const int window_end = window_start + kernel_size;
    const int valid_start = window_start > 0 ? window_start : 0;
    const int valid_end = window_end < input_length ? window_end : input_length;

    for (int vector_c = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         vector_c < vector_channels; vector_c += gridDim.x * blockDim.x) {
        const int channel = vector_c * VecSize;
        // Seed from the first valid position instead of -inf: the window is
        // never empty, and seeding from real data keeps the accumulator in the
        // input's own range for every dtype.
        const int64_t seed_offset =
            (static_cast<int64_t>(batch) * input_length + valid_start) * channels + channel;
        Vec<T, VecSize> seed;
        seed.load(input + seed_offset);
        float accum[VecSize];
#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            accum[i] = static_cast<float>(seed[i]);
        }

        for (int input_t = valid_start + 1; input_t < valid_end; ++input_t) {
            const int64_t input_offset =
                (static_cast<int64_t>(batch) * input_length + input_t) * channels + channel;
            Vec<T, VecSize> values;
            values.load(input + input_offset);
#pragma unroll
            for (int i = 0; i < VecSize; ++i) {
                const float value = static_cast<float>(values[i]);
                accum[i] = accum[i] > value ? accum[i] : value;
            }
        }

        Vec<T, VecSize> result;
#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            result[i] = static_cast<T>(accum[i]);
        }
        const int64_t output_offset =
            (static_cast<int64_t>(batch) * output_length + output_t) * channels + channel;
        result.store(output + output_offset);
    }
}

template <typename T>
cudaError_t MaxPool1d(const T* input, T* output, int batch_size, int input_length,
                      int output_length, int channels, int kernel_size, int stride, int padding,
                      cudaStream_t stream) {
    if (batch_size == 0) {
        return cudaSuccess;
    }

    constexpr int kVectorSize = VecTypeTrait<T>::VecSize;
    const bool vectorized = channels % kVectorSize == 0;
    const int vector_size = vectorized ? kVectorSize : 1;
    const int vector_channels = channels / vector_size;

    if (vector_channels < kNarrowChannelThreshold) {
        const int64_t total = static_cast<int64_t>(batch_size) * output_length * channels;
        const NarrowLaunch launch = NarrowLaunchFor(total);
        MaxPool1dNarrowKernel<T><<<launch.grid, launch.block, 0, stream>>>(
            input, output, input_length, output_length, channels, kernel_size, stride, padding,
            total);
        return cudaGetLastError();
    }

    const int block_size = std::min(256, std::max(32, ((vector_channels + 31) / 32) * 32));
    const int channel_blocks = (vector_channels + block_size - 1) / block_size;
    const dim3 grid(channel_blocks, output_length, batch_size);

    if (vectorized) {
        MaxPool1dGenericKernel<T, kVectorSize><<<grid, block_size, 0, stream>>>(
            input, output, input_length, output_length, channels, kernel_size, stride, padding);
    } else {
        MaxPool1dGenericKernel<T, 1><<<grid, block_size, 0, stream>>>(
            input, output, input_length, output_length, channels, kernel_size, stride, padding);
    }
    return cudaGetLastError();
}

}  // namespace pooling
}  // namespace oasr
