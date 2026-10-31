// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA activation kernels — no framework dependencies.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <oasr/common/math.h>
#include <oasr/common/utils.h>
#include <oasr/common/vec_dtypes.h>

namespace oasr {
namespace activation {

// =============================================================================
// GLU (Gated Linear Unit) Kernel
// =============================================================================

template <typename T, int VecSize>
__global__ void gluKernel(const T* __restrict__ input,  // [rows, 2 * channels]
                          T* __restrict__ output,       // [rows, channels]
                          int64_t rows, int64_t channels) {
    const int64_t vec_channels = channels / VecSize;
    const int64_t total_vec_elements = rows * vec_channels;

    for (int64_t vid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         vid < total_vec_elements; vid += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t pos = vid / vec_channels;
        const int64_t c_offset = (vid - pos * vec_channels) * VecSize;

        const int64_t input_idx1 = pos * (2 * channels) + c_offset;
        Vec<T, VecSize> val_vec;
        val_vec.load(input + input_idx1);

        const int64_t input_idx2 = input_idx1 + channels;
        Vec<T, VecSize> gate_vec;
        gate_vec.load(input + input_idx2);

        Vec<T, VecSize> out_vec;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            float x = static_cast<float>(val_vec[v]);
            float gate = static_cast<float>(gate_vec[v]);
            float sigmoid_gate = oasr::sigmoid(gate);
            out_vec[v] = static_cast<T>(x * sigmoid_gate);
        }

        out_vec.store(output + pos * channels + c_offset);
    }
}

// =============================================================================
// Unary Activation Kernel (vectorized, elementwise over a flat buffer)
// =============================================================================

template <typename T, int VecSize, typename Activation>
__global__ void elementwiseKernel(const T* __restrict__ input, T* __restrict__ output, int64_t n,
                                  Activation activation) {
    const int64_t total_vec_elements = n / VecSize;
    for (int64_t vid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         vid < total_vec_elements; vid += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        Vec<T, VecSize> v_in;
        v_in.load(input + vid * VecSize);

        Vec<T, VecSize> v_out;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            v_out[v] = static_cast<T>(activation(static_cast<float>(v_in[v])));
        }
        v_out.store(output + vid * VecSize);
    }
}

template <typename T, int VecSize, typename Activation>
__global__ void elementwiseStridedRowsKernel(const T* __restrict__ input, T* __restrict__ output,
                                             int64_t rows, int64_t columns,
                                             int64_t input_row_stride, Activation activation) {
    const int64_t vec_columns = columns / VecSize;
    const int64_t total_vec_elements = rows * vec_columns;
    for (int64_t vid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         vid < total_vec_elements; vid += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t row = vid / vec_columns;
        const int64_t column = (vid - row * vec_columns) * VecSize;

        Vec<T, VecSize> v_in;
        v_in.load(input + row * input_row_stride + column);

        Vec<T, VecSize> v_out;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
            v_out[v] = static_cast<T>(activation(static_cast<float>(v_in[v])));
        }
        v_out.store(output + row * columns + column);
    }
}

// =============================================================================
// Typed Launchers — raw pointer interface, returns cudaError_t
// =============================================================================

// GLU over the last axis of a contiguous [rows, 2 * channels] input.  The
// vector path also needs both pointers on a vector boundary: the halves sit
// `channels` apart, so a whole number of vectors keeps the gate half aligned
// with the value half.
template <typename T>
cudaError_t GLU(const T* input, T* output, int64_t rows, int64_t channels, cudaStream_t stream) {
    const int64_t total_elements = rows * channels;
    if (total_elements == 0) {
        return cudaSuccess;
    }

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;
    constexpr int kBlockSize = 256;

    if (channels % kVecSize == 0 && oasr::isAligned<T, kVecSize>(input) &&
        oasr::isAligned<T, kVecSize>(output)) {
        int64_t grid_size = (total_elements / kVecSize + kBlockSize - 1) / kBlockSize;
        grid_size = std::min<int64_t>(grid_size, 65535);
        gluKernel<T, kVecSize>
            <<<static_cast<int>(grid_size), kBlockSize, 0, stream>>>(input, output, rows, channels);
    } else {
        int64_t grid_size = (total_elements + kBlockSize - 1) / kBlockSize;
        grid_size = std::min<int64_t>(grid_size, 65535);
        gluKernel<T, 1>
            <<<static_cast<int>(grid_size), kBlockSize, 0, stream>>>(input, output, rows, channels);
    }

    return cudaGetLastError();
}

template <typename T, typename Activation>
cudaError_t Elementwise(const T* input, T* output, int64_t n, Activation activation,
                        cudaStream_t stream) {
    if (n == 0) {
        return cudaSuccess;
    }

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;
    constexpr int kBlockSize = 256;

    if (n % kVecSize == 0 && oasr::isAligned<T, kVecSize>(input) &&
        oasr::isAligned<T, kVecSize>(output)) {
        const int64_t total_vec_elements = n / kVecSize;
        int64_t grid_size = (total_vec_elements + kBlockSize - 1) / kBlockSize;
        grid_size = std::min<int64_t>(grid_size, 65535);
        elementwiseKernel<T, kVecSize>
            <<<static_cast<int>(grid_size), kBlockSize, 0, stream>>>(input, output, n, activation);
    } else {
        int64_t grid_size = (n + kBlockSize - 1) / kBlockSize;
        grid_size = std::min<int64_t>(grid_size, 65535);
        elementwiseKernel<T, 1>
            <<<static_cast<int>(grid_size), kBlockSize, 0, stream>>>(input, output, n, activation);
    }

    return cudaGetLastError();
}

template <typename T, typename Activation>
cudaError_t ElementwiseStridedRows(const T* input, T* output, int64_t rows, int64_t columns,
                                   int64_t input_row_stride, Activation activation,
                                   cudaStream_t stream) {
    if (rows == 0 || columns == 0) {
        return cudaSuccess;
    }

    constexpr int kVecSize = VecTypeTrait<T>::VecSize;
    constexpr int kBlockSize = 256;
    const bool aligned = columns % kVecSize == 0 && input_row_stride % kVecSize == 0 &&
                         oasr::isAligned<T, kVecSize>(input) &&
                         oasr::isAligned<T, kVecSize>(output);
    const int64_t elements_per_row = aligned ? columns / kVecSize : columns;
    int64_t grid_size = (rows * elements_per_row + kBlockSize - 1) / kBlockSize;
    grid_size = std::min<int64_t>(grid_size, 65535);

    if (aligned) {
        elementwiseStridedRowsKernel<T, kVecSize>
            <<<static_cast<int>(grid_size), kBlockSize, 0, stream>>>(input, output, rows, columns,
                                                                     input_row_stride, activation);
    } else {
        elementwiseStridedRowsKernel<T, 1><<<static_cast<int>(grid_size), kBlockSize, 0, stream>>>(
            input, output, rows, columns, input_row_stride, activation);
    }
    return cudaGetLastError();
}

template <typename T>
cudaError_t GeluErf(const T* input, T* output, int64_t n, cudaStream_t stream) {
    return Elementwise(input, output, n, GeluErfActivation{}, stream);
}

template <typename T>
cudaError_t Sigmoid(const T* input, T* output, int64_t n, cudaStream_t stream) {
    return Elementwise(input, output, n, SigmoidActivation{}, stream);
}

template <typename T>
cudaError_t Tanh(const T* input, T* output, int64_t n, cudaStream_t stream) {
    return Elementwise(input, output, n, TanhActivation{}, stream);
}

template <typename T>
cudaError_t Relu(const T* input, T* output, int64_t n, cudaStream_t stream) {
    return Elementwise(input, output, n, ReluActivation{}, stream);
}

// Swish and the Zipformer Swooshes are plain elementwise maps.  They had
// kernels of their own that counted elements in `int` and took the vector path
// on `n % VecSize` alone, so an unaligned pointer (any `x[1:]`) faulted with
// `misaligned address`; the shared kernel checks both and counts in 64 bits.
template <typename T>
cudaError_t Swish(const T* input, T* output, int64_t n, cudaStream_t stream) {
    return Elementwise(input, output, n, SwishActivation{}, stream);
}

template <typename T>
cudaError_t SwooshL(const T* input, T* output, int64_t n, cudaStream_t stream) {
    return Elementwise(input, output, n, SwooshLActivation{}, stream);
}

template <typename T>
cudaError_t SwooshR(const T* input, T* output, int64_t n, cudaStream_t stream) {
    return Elementwise(input, output, n, SwooshRActivation{}, stream);
}

}  // namespace activation
}  // namespace oasr
