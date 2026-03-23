// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for grouped GEMM.

#include <oasr/gemm/group_gemm.cuh>

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
// GroupGemm launcher
// =============================================================================
// A:       Concatenated [L, K] buffer on device (L = sum of M_i)
// B_ptrs:  1D tensor of problem_count device pointers, each to [N, K] col-major
// D:       Concatenated [L, N] buffer on device
// offsets: 1D int32 tensor of cumulative row offsets (host-accessible)
//          offsets[i] = M_0 + M_1 + ... + M_i

void group_gemm(TensorView output, TensorView A, TensorView B, TensorView offsets) {
    CHECK_INPUT(A);
    CHECK_INPUT(output);

    int64_t problem_count = offsets.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    cudaStream_t stream = get_stream(A.device());

    // Copy offsets from device to host — GroupedGemmProblemDesc iterates on CPU.
    std::vector<int> offsets_host(problem_count);
    cudaMemcpy(offsets_host.data(), offsets.data_ptr(),
               problem_count * sizeof(int), cudaMemcpyDeviceToHost);

    DISPATCH_DLPACK_HALF_DTYPE(A.dtype(), c_type, [&] {
        using CutlassType = typename ToCutlassType<c_type>::type;

        cudaError_t status = gemm::GroupGemm<CutlassType, CutlassType, CutlassType>(
            reinterpret_cast<const CutlassType*>(A.data_ptr()),
            reinterpret_cast<const CutlassType*>(B.data_ptr()),
            reinterpret_cast<CutlassType*>(output.data_ptr()),
            static_cast<int>(problem_count), static_cast<int>(K), static_cast<int>(N),
            offsets_host.data(), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "GroupGemm kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
