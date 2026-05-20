// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher for fused GEMM + log_softmax.

#include <oasr/gemm/gemm_log_softmax.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

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
// gemm_log_softmax launcher
//   output: [M, N] row-major; in-place valid (callers usually pre-allocate)
//   A:      [M, K] row-major (activations)
//   B:      [N, K] row-major weight tensor (CUTLASS treats it as column-major
//                  [K, N] for the matmul — same convention as oasr.gemm)
//   bias_opt: optional [N] vector, broadcast over rows via stride-0 C
// =============================================================================

void gemm_log_softmax(TensorView output, TensorView A, TensorView B, Optional bias_opt,
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

        const CutlassType* bias_ptr = nullptr;
        if (bias_opt.has_value()) {
            bias_ptr = reinterpret_cast<const CutlassType*>(bias_opt.value().data_ptr());
        }

        cudaError_t status = gemm::GemmLogSoftmax<CutlassType, CutlassType, CutlassType>(
            reinterpret_cast<const CutlassType*>(A.data_ptr()),
            reinterpret_cast<const CutlassType*>(B.data_ptr()), bias_ptr,
            reinterpret_cast<CutlassType*>(output.data_ptr()), M, N, K, stream,
            static_cast<int>(split_k_slices));
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "GemmLogSoftmax kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
