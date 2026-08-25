// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for softmax kernel.

#include <oasr/softmax.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

void softmax(TensorView output, TensorView input) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);

    unsigned int num_rows = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        num_rows *= input.size(i);
    }
    unsigned int num_cols = input.size(input.ndim() - 1);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = softmax::Softmax<c_type>(static_cast<const c_type*>(input.data_ptr()),
                                                      static_cast<c_type*>(output.data_ptr()),
                                                      num_rows, num_cols, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "Softmax kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// Row-wise log_softmax over the last dimension.  In-place valid
// (output == input): the online kernel reads each element once per pass and
// writes each element exactly once in the final pass.
void log_softmax(TensorView output, TensorView input) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);

    unsigned int num_rows = 1;
    for (int i = 0; i < input.ndim() - 1; ++i) {
        num_rows *= input.size(i);
    }
    unsigned int num_cols = input.size(input.ndim() - 1);

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = softmax::LogSoftmax<c_type>(
            static_cast<const c_type*>(input.data_ptr()), static_cast<c_type*>(output.data_ptr()),
            num_rows, num_cols, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "LogSoftmax kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}

// =============================================================================
// Fused masked / biased softmax
// =============================================================================
//
// Replaces `softmax((scores + bias).masked_fill(m1, v).masked_fill(m2, v))`.
// The bias and both masks are consumed as *strided broadcast views*, so a
// caller hands over the shifted / step-sliced / unsqueezed view it already has
// instead of materializing a copy of a T^2 tensor per operand.

namespace {

using oasr::softmax::BroadcastView;

//! Leading axes the broadcast operands can be addressed on -- one per grid axis.
constexpr int kMaxLeadDims = 3;

bool same_dtype(const TensorView& a, const TensorView& b) {
    return a.dtype().code == b.dtype().code && a.dtype().bits == b.dtype().bits &&
           a.dtype().lanes == b.dtype().lanes;
}

/// Layout of an operand broadcast against the score tensor, stated per *grid*
/// axis (innermost leading axis on x) and dtype-erased so it can be built once
/// outside the dtype dispatch.
struct BroadcastLayout {
    const void* ptr = nullptr;
    int64_t col_stride = 0;
    int64_t grid_stride[kMaxLeadDims] = {0, 0, 0};
};

/// The score tensor's leading extents as a grid, innermost on x.  Blocks then
/// read their leading indices from `blockIdx` instead of dividing a flat row
/// index -- and, more to the point, the broadcast views need no stride *array*,
/// which is what would push the kernel's parameters into local memory.
dim3 makeGrid(const TensorView& scores) {
    const int num_lead_dims = scores.ndim() - 1;
    TVM_FFI_ICHECK(num_lead_dims <= kMaxLeadDims)
        << "masked_softmax cannot map " << num_lead_dims << " leading axes onto " << kMaxLeadDims
        << " grid axes";  // broadcastAgainst rejects this first; belt and braces
    unsigned int extent[kMaxLeadDims] = {1, 1, 1};
    for (int axis = 0; axis < num_lead_dims && axis < kMaxLeadDims; ++axis) {
        extent[axis] = static_cast<unsigned int>(scores.size(num_lead_dims - 1 - axis));
    }
    return dim3(extent[0], extent[1], extent[2]);
}

/// NumPy broadcast of `operand` against `scores`: trailing axes align, an
/// extent of one (or a missing leading axis) becomes a zero stride.
BroadcastLayout broadcastAgainst(const TensorView& operand, const TensorView& scores,
                                 const char* name) {
    CHECK_INPUT(operand);
    CHECK_DEVICE(operand, scores);

    const int scores_ndim = scores.ndim();
    const int operand_ndim = operand.ndim();
    TVM_FFI_ICHECK(scores_ndim <= kMaxLeadDims + 1)
        << "masked_softmax addresses a broadcast operand on at most " << kMaxLeadDims
        << " leading axes, so a score tensor carrying one must be at most " << (kMaxLeadDims + 1)
        << "-D; got " << scores_ndim << "-D. Reshape the leading axes together.";
    TVM_FFI_ICHECK(operand_ndim >= 1 && operand_ndim <= scores_ndim)
        << "masked_softmax " << name << " is " << operand_ndim
        << "-D; it must broadcast against the " << scores_ndim << "-D score tensor";

    BroadcastLayout layout;
    layout.ptr = operand.data_ptr();

    const int num_lead_dims = scores_ndim - 1;
    const int64_t operand_cols = operand.size(operand_ndim - 1);
    TVM_FFI_ICHECK(operand_cols == scores.size(num_lead_dims) || operand_cols == 1)
        << "masked_softmax " << name << " has extent " << operand_cols
        << " on the softmax axis, which does not broadcast against " << scores.size(num_lead_dims);
    layout.col_stride = operand_cols == 1 ? 0 : operand.stride(operand_ndim - 1);

    // Grid axis `g` is score axis `num_lead_dims - 1 - g`; the operand's axes
    // align to the score's from the right, so anything it is missing on the
    // left broadcasts.
    const int implicit_dims = scores_ndim - operand_ndim;
    for (int g = 0; g < num_lead_dims; ++g) {
        const int scores_axis = num_lead_dims - 1 - g;
        if (scores_axis < implicit_dims) {
            continue;  // operand has no such axis: broadcast
        }
        const int operand_axis = scores_axis - implicit_dims;
        const int64_t extent = operand.size(operand_axis);
        TVM_FFI_ICHECK(extent == scores.size(scores_axis) || extent == 1)
            << "masked_softmax " << name << " has extent " << extent << " on axis " << operand_axis
            << ", which does not broadcast against " << scores.size(scores_axis);
        layout.grid_stride[g] = extent == 1 ? 0 : operand.stride(operand_axis);
    }
    return layout;
}

/// Booleans arrive as one byte per element, which is what torch's `bool` is.
void checkMaskDtype(const TensorView& mask, const char* name) {
    const bool byte_sized =
        (mask.dtype().code == kDLBool || mask.dtype().code == kDLUInt) && mask.dtype().bits == 8;
    TVM_FFI_ICHECK(byte_sized) << "masked_softmax " << name << " must be bool or uint8";
}

template <typename T>
BroadcastView<T> toView(const BroadcastLayout& layout) {
    BroadcastView<T> view;
    view.ptr = static_cast<const T*>(layout.ptr);
    view.col_stride = layout.col_stride;
    view.stride_x = layout.grid_stride[0];
    view.stride_y = layout.grid_stride[1];
    view.stride_z = layout.grid_stride[2];
    return view;
}

}  // namespace

// Row-wise `softmax(cast(input + bias) where(mask | mask2) -> mask_value)`.
// `bias`, `mask` and `mask2` broadcast against `input` and may be arbitrarily
// strided views; `input` and `output` must be contiguous, since a block's
// leading indices come from its position in the grid.
void masked_softmax(TensorView output, TensorView input, Optional bias_opt, Optional mask_opt,
                    Optional mask2_opt, double mask_value) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_CONTIGUOUS_INPUT(input);
    CHECK_CONTIGUOUS_INPUT(output);
    CHECK_SAME_LAYOUT(input, output);
    if (input.numel() == 0) {
        return;  // before FLATTENED_ROWS, which divides by the trailing extent
    }

    const unsigned int num_cols = static_cast<unsigned int>(input.size(input.ndim() - 1));

    BroadcastLayout bias_layout;
    if (bias_opt.has_value()) {
        TVM_FFI_ICHECK(same_dtype(input, bias_opt.value()))
            << "masked_softmax bias must have the score tensor's dtype";
        bias_layout = broadcastAgainst(bias_opt.value(), input, "bias");
    }
    BroadcastLayout mask_layout;
    if (mask_opt.has_value()) {
        checkMaskDtype(mask_opt.value(), "mask");
        mask_layout = broadcastAgainst(mask_opt.value(), input, "mask");
    }
    BroadcastLayout mask2_layout;
    if (mask2_opt.has_value()) {
        checkMaskDtype(mask2_opt.value(), "mask2");
        mask2_layout = broadcastAgainst(mask2_opt.value(), input, "mask2");
    }

    // With no broadcast operand the leading axes carry no meaning beyond the row
    // count, so a flat grid serves any rank; `broadcastAgainst` is what caps the
    // rank when one is present.
    const bool has_operand = bias_opt.has_value() || mask_opt.has_value() || mask2_opt.has_value();
    const dim3 grid = has_operand ? makeGrid(input)
                                  : dim3(static_cast<unsigned int>(FLATTENED_ROWS(input)), 1, 1);
    TVM_FFI_ICHECK(grid.y <= 65535 && grid.z <= 65535)
        << "masked_softmax maps the score tensor's leading extents onto the CUDA grid, whose "
           "y and z axes cap at 65535; got y="
        << grid.y << " z=" << grid.z;

    cudaStream_t stream = get_stream(input.device());

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
        cudaError_t status = softmax::MaskedSoftmax<c_type>(
            static_cast<const c_type*>(input.data_ptr()), static_cast<c_type*>(output.data_ptr()),
            toView<c_type>(bias_layout), toView<uint8_t>(mask_layout),
            toView<uint8_t>(mask2_layout), grid, static_cast<float>(mask_value), num_cols, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "MaskedSoftmax kernel failed: " << cudaGetErrorString(status);
        return true;
    });
}
