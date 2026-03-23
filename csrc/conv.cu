// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for conv1d kernels.

#include <oasr/conv/conv1d.cuh>
#include <oasr/gemm/gemm.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

// =============================================================================
// Depthwise Conv1D launcher
// =============================================================================

void depthwise_conv1d(TensorView output, TensorView input, TensorView weight, Optional bias_opt,
                      int64_t padding) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_DIM(3, input);

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int kernel_size = weight.size(0);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = conv::DepthwiseConv1D<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), batch_size, seq_len, channels, kernel_size,
            static_cast<int>(padding), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "DepthwiseConv1D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Depthwise Conv1D + SiLU launcher
// =============================================================================

void depthwise_conv1d_silu(TensorView output, TensorView input, TensorView weight,
                           Optional bias_opt, int64_t padding) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_DIM(3, input);

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int channels = input.size(2);
    int kernel_size = weight.size(0);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = conv::DepthwiseConv1DSilu<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), batch_size, seq_len, channels, kernel_size,
            static_cast<int>(padding), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "DepthwiseConv1DSilu kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Pointwise Conv1D launcher (delegates to GEMM)
// =============================================================================
// Note: PointwiseConv1D is essentially a GEMM operation.
// The TVM-FFI binding calls into the gemm launcher layer directly.
// This launcher reshapes the problem to match the GEMM interface.

void pointwise_conv1d(TensorView output, TensorView input, TensorView weight, Optional bias_opt) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_DIM(3, input);

    // input: [batch, seq_len, in_channels]
    // weight: [out_channels, in_channels]
    // output: [batch, seq_len, out_channels]
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int in_channels = input.size(2);
    int out_channels = weight.size(0);

    int M = batch_size * seq_len;
    int N = out_channels;
    int K = in_channels;

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        using CutlassType =
            typename std::conditional<std::is_same<c_type, half>::value, cutlass::half_t,
                                      typename std::conditional<std::is_same<c_type, __nv_bfloat16>::value,
                                                                cutlass::bfloat16_t, c_type>::type>::type;

        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }

        // Pointwise conv1d = GEMM: [M, K] @ [N, K]^T -> [M, N]
        cudaError_t status = gemm::Gemm<CutlassType, CutlassType, CutlassType>(
            reinterpret_cast<const CutlassType*>(input.data_ptr()),
            reinterpret_cast<const CutlassType*>(weight.data_ptr()),
            reinterpret_cast<const CutlassType*>(bias_ptr),
            reinterpret_cast<CutlassType*>(output.data_ptr()), M, N, K, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "PointwiseConv1D (GEMM) failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Pointwise Conv1D + Activation launcher (delegates to GEMM)
// =============================================================================

void pointwise_conv1d_activation(TensorView output, TensorView input, TensorView weight,
                                 Optional bias_opt, int64_t activation_type) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(weight);
    CHECK_DIM(3, input);

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int in_channels = input.size(2);
    int out_channels = weight.size(0);

    int M = batch_size * seq_len;
    int N = out_channels;
    int K = in_channels;
    auto activation = static_cast<ActivationType>(activation_type);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        using CutlassType =
            typename std::conditional<std::is_same<c_type, half>::value, cutlass::half_t,
                                      typename std::conditional<std::is_same<c_type, __nv_bfloat16>::value,
                                                                cutlass::bfloat16_t, c_type>::type>::type;

        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }

        cudaError_t status = gemm::GemmActivation<CutlassType, CutlassType, CutlassType>(
            reinterpret_cast<const CutlassType*>(input.data_ptr()),
            reinterpret_cast<const CutlassType*>(weight.data_ptr()),
            reinterpret_cast<const CutlassType*>(bias_ptr),
            reinterpret_cast<CutlassType*>(output.data_ptr()), M, N, K, activation, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "PointwiseConv1DActivation (GEMM) failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Causal Conv1D launcher
// =============================================================================

void causal_conv1d(TensorView output, TensorView input, TensorView state, TensorView weight,
                   Optional bias_opt) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(state);
    CHECK_INPUT(weight);
    CHECK_DIM(3, input);

    int batch_size = input.size(0);
    int chunk_len = input.size(1);
    int channels = input.size(2);
    int kernel_size = weight.size(2);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_HALF_DTYPE(input.dtype(), c_type, [&] {
        const c_type* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = static_cast<const c_type*>(bias_opt.value().data_ptr());
        }
        cudaError_t status = conv::CausalConv1D<c_type>(
            static_cast<const c_type*>(input.data_ptr()),
            static_cast<c_type*>(state.data_ptr()),
            static_cast<const c_type*>(weight.data_ptr()), bias_ptr,
            static_cast<c_type*>(output.data_ptr()), batch_size, chunk_len, channels, kernel_size,
            stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "CausalConv1D kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
