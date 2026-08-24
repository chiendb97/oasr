// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher for the *general* strided batched GEMM -- the lane the
// rendered alignment-8 tile variants refuse.  Everything shape-dependent is
// here; the kernel selection rule is in `include/oasr/gemm/bmm.cuh`.
//
// Contract: `D = A @ B.transpose(-1, -2)` with `A[..., M, K]`, `B[..., N, K]`
// and one or two batch dimensions that broadcast against each other.  A's K
// axis must be contiguous; B may be contiguous along *either* of its two
// trailing axes, which is what lets a caller hand over a permuted view instead
// of materializing a transpose.

#include <oasr/gemm/bmm.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

namespace {

/// Batch stride of *tensor* along output batch axis *output_axis*, in elements.
///
/// Zero means "broadcast": either the operand has fewer batch dimensions than
/// the output (leading axes are implicit) or its extent on this axis is 1.
int64_t batchStride(const TensorView& tensor, int output_batch_dims, int output_axis) {
    const int tensor_batch_dims = tensor.ndim() - 2;
    const int leading_broadcast_dims = output_batch_dims - tensor_batch_dims;
    if (output_axis < leading_broadcast_dims) {
        return 0;
    }
    const int tensor_axis = output_axis - leading_broadcast_dims;
    return tensor.size(tensor_axis) == 1 ? 0 : tensor.stride(tensor_axis);
}

/// Extent of *tensor* along output batch axis *output_axis* (1 when implicit).
int64_t logicalBatchSize(const TensorView& tensor, int output_batch_dims, int output_axis) {
    const int tensor_batch_dims = tensor.ndim() - 2;
    const int leading_broadcast_dims = output_batch_dims - tensor_batch_dims;
    if (output_axis < leading_broadcast_dims) {
        return 1;
    }
    return tensor.size(output_axis - leading_broadcast_dims);
}

void checkNonNegativeStrides(const TensorView& tensor, const char* name) {
    for (int i = 0; i < tensor.ndim(); ++i) {
        TVM_FFI_ICHECK(tensor.stride(i) >= 0)
            << name << " must not have negative strides, got stride(" << i
            << ")=" << tensor.stride(i);
    }
}

bool sameDtype(const TensorView& a, const TensorView& b) {
    return a.dtype().code == b.dtype().code && a.dtype().bits == b.dtype().bits &&
           a.dtype().lanes == b.dtype().lanes;
}

}  // namespace

// =============================================================================
// bmm: D[..., M, N] = A[..., M, K] @ B[..., N, K]^T
// =============================================================================

void bmm(TensorView output, TensorView A, TensorView B) {
    CHECK_INPUT(output);
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_DEVICE(output, A);
    CHECK_DEVICE(output, B);
    TVM_FFI_ICHECK(A.ndim() >= 3 && A.ndim() <= 4)
        << "bmm expects A to be 3D or 4D, got " << A.ndim() << "D";
    TVM_FFI_ICHECK(B.ndim() >= 3 && B.ndim() <= 4)
        << "bmm expects B to be 3D or 4D, got " << B.ndim() << "D";
    TVM_FFI_ICHECK(output.ndim() >= 3 && output.ndim() <= 4)
        << "bmm expects output to be 3D or 4D, got " << output.ndim() << "D";
    TVM_FFI_ICHECK(sameDtype(A, B)) << "bmm operands must have the same dtype";
    TVM_FFI_ICHECK(sameDtype(A, output)) << "bmm output must have the operands' dtype";
    checkNonNegativeStrides(A, "A");
    checkNonNegativeStrides(B, "B");
    checkNonNegativeStrides(output, "output");

    const int output_batch_dims = output.ndim() - 2;
    TVM_FFI_ICHECK(A.ndim() - 2 <= output_batch_dims && B.ndim() - 2 <= output_batch_dims)
        << "bmm output has fewer batch dimensions than an operand";

    const int M = A.size(A.ndim() - 2);
    const int K = A.size(A.ndim() - 1);
    const int b_n_axis = B.ndim() - 2;
    const int b_k_axis = B.ndim() - 1;
    const int N = B.size(b_n_axis);
    TVM_FFI_ICHECK(B.size(b_k_axis) == K)
        << "bmm contraction mismatch: A has K=" << K << " but B has K=" << B.size(b_k_axis);
    TVM_FFI_ICHECK(output.size(output.ndim() - 2) == M && output.size(output.ndim() - 1) == N)
        << "bmm output matrix shape must be (..., " << M << ", " << N << ")";

    // A is consumed as CUTLASS RowMajor and D is written as RowMajor, so their
    // trailing axes have to be the contiguous ones.  B may be contiguous along
    // either trailing axis; which one it is picks CUTLASS's LayoutB.
    TVM_FFI_ICHECK(A.stride(A.ndim() - 1) == 1)
        << "bmm requires A's K axis to be contiguous (stride 1), got stride "
        << A.stride(A.ndim() - 1);
    TVM_FFI_ICHECK(output.stride(output.ndim() - 1) == 1)
        << "bmm requires the output's N axis to be contiguous (stride 1), got stride "
        << output.stride(output.ndim() - 1);
    const bool b_contiguous_k = B.stride(b_k_axis) == 1;
    TVM_FFI_ICHECK(b_contiguous_k || B.stride(b_n_axis) == 1)
        << "bmm requires B to be contiguous along one of its two trailing axes, got strides ("
        << B.stride(b_n_axis) << ", " << B.stride(b_k_axis) << ") for shape (" << N << ", " << K
        << "). Call .contiguous() on the operand, or pass the untransposed view.";

    for (int axis = 0; axis < output_batch_dims; ++axis) {
        const int64_t a_size = logicalBatchSize(A, output_batch_dims, axis);
        const int64_t b_size = logicalBatchSize(B, output_batch_dims, axis);
        const int64_t out_size = output.size(axis);
        TVM_FFI_ICHECK((a_size == out_size || a_size == 1) && (b_size == out_size || b_size == 1))
            << "bmm batch dimensions are not broadcastable at axis " << axis << ": A has " << a_size
            << ", B has " << b_size << ", output has " << out_size;
    }

    const int batch0 = output_batch_dims == 2 ? static_cast<int>(output.size(0)) : 1;
    const int batch1 = static_cast<int>(output.size(output_batch_dims - 1));
    const int64_t out_stride0 = output_batch_dims == 2 ? output.stride(0) : 0;
    const int64_t out_stride1 = output.stride(output_batch_dims - 1);
    const int64_t a_stride0 = output_batch_dims == 2 ? batchStride(A, output_batch_dims, 0) : 0;
    const int64_t a_stride1 = batchStride(A, output_batch_dims, output_batch_dims - 1);
    const int64_t b_stride0 = output_batch_dims == 2 ? batchStride(B, output_batch_dims, 0) : 0;
    const int64_t b_stride1 = batchStride(B, output_batch_dims, output_batch_dims - 1);

    // CUTLASS GemmBatched advances every operand by one constant stride per
    // matrix, so two batch axes become one batched launch only when all three
    // tensors are affine in the flattened batch index.  There are two ways to
    // flatten and they are *not* equivalent: with axis 0 outer the test is
    // `stride0 == batch1 * stride1`, with axis 1 outer it is
    // `stride1 == batch0 * stride0`.  A contiguous output satisfies the first;
    // a head-major view of a `(time, batch, head, dim)` activation satisfies
    // the second.  Try both, because getting one launch instead of
    // `min(batch0, batch1)` is worth more here than any tile choice.
    auto affine = [](int64_t outer_stride, int inner_extent, int64_t inner_stride) {
        return outer_stride == static_cast<int64_t>(inner_extent) * inner_stride;
    };
    const bool degenerate = batch0 == 1 || batch1 == 1;
    const bool collapse_axis0_outer = degenerate || (affine(a_stride0, batch1, a_stride1) &&
                                                     affine(b_stride0, batch1, b_stride1) &&
                                                     affine(out_stride0, batch1, out_stride1));
    const bool collapse_axis1_outer =
        !collapse_axis0_outer &&
        (affine(a_stride1, batch0, a_stride0) && affine(b_stride1, batch0, b_stride0) &&
         affine(out_stride1, batch0, out_stride0));

    int outer_count = 1;
    int inner_count = batch0 * batch1;
    int64_t a_outer_stride = 0, b_outer_stride = 0, out_outer_stride = 0;
    int64_t a_inner_stride = 0, b_inner_stride = 0, out_inner_stride = 0;
    if (collapse_axis0_outer) {
        // The degenerate cases fold to whichever axis is not 1.
        const bool use_axis1 = batch1 != 1;
        a_inner_stride = use_axis1 ? a_stride1 : a_stride0;
        b_inner_stride = use_axis1 ? b_stride1 : b_stride0;
        out_inner_stride = use_axis1 ? out_stride1 : out_stride0;
    } else if (collapse_axis1_outer) {
        a_inner_stride = a_stride0;
        b_inner_stride = b_stride0;
        out_inner_stride = out_stride0;
    } else {
        // Neither flattening is affine -- a broadcast *inner* axis is the usual
        // reason, and Zipformer's relative-position operand (shared over the
        // request batch) is exactly that.  Loop over the shorter axis rather
        // than forcing an expand + copy, which costs a launch of its own.
        const bool loop_axis0 = batch0 <= batch1;
        outer_count = loop_axis0 ? batch0 : batch1;
        inner_count = loop_axis0 ? batch1 : batch0;
        a_outer_stride = loop_axis0 ? a_stride0 : a_stride1;
        b_outer_stride = loop_axis0 ? b_stride0 : b_stride1;
        out_outer_stride = loop_axis0 ? out_stride0 : out_stride1;
        a_inner_stride = loop_axis0 ? a_stride1 : a_stride0;
        b_inner_stride = loop_axis0 ? b_stride1 : b_stride0;
        out_inner_stride = loop_axis0 ? out_stride1 : out_stride0;
    }

    gemm::GeneralBmmParams params;
    params.batch = inner_count;
    params.M = M;
    params.N = N;
    params.K = K;
    params.lda = A.stride(A.ndim() - 2);
    params.ldb = b_contiguous_k ? B.stride(b_n_axis) : B.stride(b_k_axis);
    params.ldd = output.stride(output.ndim() - 2);
    params.batch_stride_a = a_inner_stride;
    params.batch_stride_b = b_inner_stride;
    params.batch_stride_d = out_inner_stride;
    params.b_contiguous_k = b_contiguous_k;
    params.stream = get_stream(A.device());

    // The shared macro is what refuses fp32 (and says so), so the enum is
    // derived through it rather than from the DLPack fields directly.
    gemm::BmmElement element = gemm::BmmElement::kFloat16;
    DISPATCH_DLPACK_HALF_DTYPE(A.dtype(), c_type, [&] {
        element = std::is_same_v<c_type, __nv_bfloat16> ? gemm::BmmElement::kBFloat16
                                                        : gemm::BmmElement::kFloat16;
        return true;
    });

    // Byte arithmetic, not element arithmetic: the outer stride is an element
    // count but the pointers are typed only inside the instantiation.  Strides
    // are validated non-negative above, so this cannot wrap.
    const int64_t element_bytes = A.dtype().bits / 8;
    const auto* a_bytes = static_cast<const char*>(A.data_ptr());
    const auto* b_bytes = static_cast<const char*>(B.data_ptr());
    auto* d_bytes = static_cast<char*>(output.data_ptr());

    for (int outer = 0; outer < outer_count; ++outer) {
        params.A = a_bytes + static_cast<int64_t>(outer) * a_outer_stride * element_bytes;
        params.B = b_bytes + static_cast<int64_t>(outer) * b_outer_stride * element_bytes;
        params.D = d_bytes + static_cast<int64_t>(outer) * out_outer_stride * element_bytes;
        const gemm::GemmStatus status = gemm::generalBmm(params, element);
        TVM_FFI_ICHECK(status == gemm::GemmStatus::SUCCESS)
            << "general BMM failed (" << gemm::getGemmStatusString(status) << ") for M=" << M
            << " N=" << N << " K=" << K << " batch=" << inner_count
            << " b_contiguous_k=" << b_contiguous_k;
    }
}
