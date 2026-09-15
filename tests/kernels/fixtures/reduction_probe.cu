// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Test-only probe for include/oasr/common/reduction.h.
//
// The reduction helpers are a shared header with no kernel of their own, so
// nothing in the suite exercised them directly -- which is how a butterfly that
// shuffled the wrong operand sat in `warpReduceMax` unnoticed.  This TU exports
// one entry point per helper so a test can state what each must return.
//
// Built by tests/kernels/test_reduction.py through the ordinary JIT spec, so it
// compiles with the same flags and include paths as the shipped kernels.

#include <oasr/common/reduction.h>

#include "tvm_ffi_utils.h"

using namespace oasr;
using namespace oasr::reduction;

namespace {

// One block, `n` lanes' worth of data.  Every thread writes the value its own
// lane ended up with, so a reduction that leaves lanes disagreeing is visible
// rather than averaged away by reading lane 0 alone.
__global__ void WarpMaxKernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    const int tid = threadIdx.x;
    const float v = tid < n ? in[tid] : -3.0e38f;
    const float m = warpReduceMax(v);
    if (tid < n) out[tid] = m;
}

__global__ void BlockMaxKernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    const int tid = threadIdx.x;
    const float v = tid < n ? in[tid] : -3.0e38f;
    const float m = blockReduceMax(v);
    if (tid == 0) out[0] = m;
}

}  // namespace

void warp_reduce_max(TensorView out, TensorView in) {
    CHECK_INPUT(in);
    CHECK_INPUT(out);
    const int n = static_cast<int>(in.size(0));
    cudaStream_t stream = get_stream(in.device());
    WarpMaxKernel<<<1, 32, 0, stream>>>(static_cast<const float*>(in.data_ptr()),
                                        static_cast<float*>(out.data_ptr()), n);
}

void block_reduce_max(TensorView out, TensorView in, int64_t threads) {
    CHECK_INPUT(in);
    CHECK_INPUT(out);
    const int n = static_cast<int>(in.size(0));
    cudaStream_t stream = get_stream(in.device());
    BlockMaxKernel<<<1, static_cast<int>(threads), 0, stream>>>(
        static_cast<const float*>(in.data_ptr()), static_cast<float*>(out.data_ptr()), n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(warp_reduce_max, warp_reduce_max);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(block_reduce_max, block_reduce_max);
