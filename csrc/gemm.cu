// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for GEMM kernels.

#include <oasr/gemm/gemm.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

// =============================================================================
// Helper: map CUDA types to CUTLASS types
// =============================================================================

namespace {

template <typename T>
struct ToCutlassType {
    using type = T;
};

template <>
struct ToCutlassType<half> {
    using type = cutlass::half_t;
};

template <>
struct ToCutlassType<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;
};

}  // namespace

// =============================================================================
// GEMM launcher: D = A @ B [+ C]
// =============================================================================
// A: [M, K] row-major, B: [N, K] column-major, C/D: [M, N] row-major

void gemm(TensorView output, TensorView A, TensorView B, Optional C_opt,
          int64_t split_k_slices) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(output);
    CHECK_DIM(2, A);
    CHECK_DIM(2, B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    cudaStream_t stream = get_stream(A.device());

    DISPATCH_DLPACK_HALF_DTYPE(A.dtype(), c_type, [&] {
        using CutlassType = typename ToCutlassType<c_type>::type;

        const CutlassType* C_ptr = nullptr;
        if (C_opt.has_value()) {
            C_ptr = reinterpret_cast<const CutlassType*>(C_opt.value().data_ptr());
        }

        cudaError_t status = gemm::Gemm<CutlassType, CutlassType, CutlassType>(
            reinterpret_cast<const CutlassType*>(A.data_ptr()),
            reinterpret_cast<const CutlassType*>(B.data_ptr()), C_ptr,
            reinterpret_cast<CutlassType*>(output.data_ptr()), M, N, K, stream,
            static_cast<int>(split_k_slices));
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "GEMM kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// GEMM + Activation launcher: D = activation(A @ B [+ C])
// =============================================================================

void gemm_activation(TensorView output, TensorView A, TensorView B, Optional C_opt,
                     int64_t activation_type, int64_t split_k_slices) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(output);
    CHECK_DIM(2, A);
    CHECK_DIM(2, B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    auto activation = static_cast<ActivationType>(activation_type);

    cudaStream_t stream = get_stream(A.device());

    DISPATCH_DLPACK_HALF_DTYPE(A.dtype(), c_type, [&] {
        using CutlassType = typename ToCutlassType<c_type>::type;

        const CutlassType* C_ptr = nullptr;
        if (C_opt.has_value()) {
            C_ptr = reinterpret_cast<const CutlassType*>(C_opt.value().data_ptr());
        }

        cudaError_t status = gemm::GemmActivation<CutlassType, CutlassType, CutlassType>(
            reinterpret_cast<const CutlassType*>(A.data_ptr()),
            reinterpret_cast<const CutlassType*>(B.data_ptr()), C_ptr,
            reinterpret_cast<CutlassType*>(output.data_ptr()), M, N, K, activation, stream,
            static_cast<int>(split_k_slices));
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "GemmActivation kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
