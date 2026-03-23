// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for batched matrix multiplication (BMM).

#include <oasr/gemm/bmm.cuh>

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
// BMM launcher: D[b] = A[b] @ B[b]
// =============================================================================
// A: [batch, M, K] row-major, B: [batch, N, K] column-major, D: [batch, M, N] row-major

void bmm(TensorView output, TensorView A, TensorView B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(output);
    CHECK_DIM(3, A);
    CHECK_DIM(3, B);

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(1);

    cudaStream_t stream = get_stream(A.device());

    DISPATCH_DLPACK_HALF_DTYPE(A.dtype(), c_type, [&] {
        using CutlassType = typename ToCutlassType<c_type>::type;

        cudaError_t status = gemm::Bmm<CutlassType, CutlassType, CutlassType>(
            reinterpret_cast<const CutlassType*>(A.data_ptr()),
            reinterpret_cast<const CutlassType*>(B.data_ptr()),
            reinterpret_cast<CutlassType*>(output.data_ptr()), batch_size, M, N, K, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "BMM kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
