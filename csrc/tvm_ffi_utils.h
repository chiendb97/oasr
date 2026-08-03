// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Shared TVM-FFI utilities for OASR launchers.
// Mirrors FlashInfer's tvm_ffi_utils.h pattern.

#pragma once

#include <cuda_runtime.h>

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

using TensorView = tvm::ffi::TensorView;
using Optional = tvm::ffi::Optional<TensorView>;

namespace oasr {

// =============================================================================
// DLPack dtype constants
// =============================================================================

static constexpr DLDataType dl_float16 = {kDLFloat, 16, 1};
static constexpr DLDataType dl_bfloat16 = {kDLBfloat, 16, 1};
static constexpr DLDataType dl_float32 = {kDLFloat, 32, 1};
static constexpr DLDataType dl_int8 = {kDLInt, 8, 1};
static constexpr DLDataType dl_int32 = {kDLInt, 32, 1};

// =============================================================================
// Validation macros
// =============================================================================

#define CHECK_INPUT(x)                                                                    \
    TVM_FFI_ICHECK((x).device().device_type == kDLCUDA) << "Input must be a CUDA tensor"

#define CHECK_DIM(expected, x)                                                            \
    TVM_FFI_ICHECK((x).ndim() == (expected))                                             \
        << "Expected " << (expected) << "D tensor, got " << (x).ndim() << "D"

#define CHECK_DEVICE(x, y)                                                                \
    TVM_FFI_ICHECK((x).device().device_id == (y).device().device_id)                     \
        << "Tensors must be on the same device"

#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x)                                                \
    TVM_FFI_ICHECK((x).stride((x).ndim() - 1) == 1)                                     \
        << "Tensor must be contiguous along the last dimension"

// Full row-major contiguity.  Stronger than CHECK_LAST_DIM_CONTIGUOUS_INPUT and
// what a kernel indexing rows as `base + row * row_len` actually needs: a
// tensor can have stride(-1) == 1 and still have a padded row stride (any
// `x[:, -1]`-style slice of a wider buffer), in which case the weaker check
// passes and the kernel reads the wrong memory.
#define CHECK_CONTIGUOUS_INPUT(x)                                                         \
    TVM_FFI_ICHECK((x).IsContiguous())                                                    \
        << "Tensor must be contiguous (row-major, no padded strides)"

// True when the trailing dimension is contiguous *and* the resulting rows tile
// the tensor's memory exactly -- no gaps, no overlap, in some order.
//
// This is the real precondition of a row-wise kernel that walks
// `base + row * row_len`, and it is both weaker and stronger than
// `IsContiguous()` in useful ways:
//
//   * Stronger than `stride(-1) == 1`, which was the check these kernels used.
//     `x[..., :32]` of a wider buffer, or `x[:, -1]` of a `(B, T, D)` tensor,
//     satisfies that and still has a padded row stride -- the kernel then reads
//     the wrong memory and returns a plausible wrong answer, silently.
//
//   * Weaker than `IsContiguous()`, which needlessly refuses a *permuted* dense
//     view.  Zipformer works in `(T, B, C)`, a transpose of a contiguous
//     `(B, T, C)`: its rows still tile memory exactly, just visited in a
//     different order.  Since normalization is per-row and independent, and the
//     output carries the same strides (`torch.empty_like` preserves them),
//     processing rows in *memory* order computes exactly the same result.
//     Forcing contiguity in the model instead would copy the whole activation
//     (~18 MiB at T=1500 B=16 d=384) to save a few microseconds of norm.
//
// Zero/negative strides (`expand`, reversed views) fail the density test, which
// is what we want: those alias or run backwards and are not row-tilings.
inline bool IsRowDense(const TensorView& x) {
    int n = x.ndim();
    if (n < 1 || x.stride(n - 1) != 1) return false;
    int64_t span = 1;
    for (int i = 0; i < n; ++i) {
        int64_t extent = x.size(i);
        int64_t stride = x.stride(i);
        if (extent > 1 && stride <= 0) return false;
        span += (extent - 1) * stride;
    }
    return span == x.numel();
}

#define CHECK_ROW_DENSE_INPUT(x)                                                          \
    TVM_FFI_ICHECK(oasr::IsRowDense(x))                                                   \
        << "Tensor rows must tile memory exactly (trailing dim contiguous, no "           \
           "padded row stride); a row-wise kernel walks `base + row * row_len`"

// Two tensors must agree on layout, not just shape.  A row-wise kernel visits
// rows in memory order, so input row `j` and output row `j` are the same logical
// row only if the strides match.
#define CHECK_SAME_LAYOUT(a, b)                                                           \
    TVM_FFI_ICHECK((a).ndim() == (b).ndim() && [&] {                                       \
        for (int _i = 0; _i < (a).ndim(); ++_i) {                                          \
            if ((a).size(_i) != (b).size(_i) || (a).stride(_i) != (b).stride(_i))          \
                return false;                                                              \
        }                                                                                  \
        return true;                                                                       \
    }()) << "Tensors must have identical shape and strides"

// Rows of a matrix whose trailing dimension is the reduction axis: every
// leading dimension is flattened.  Lets a launcher take the caller's N-D
// activation directly instead of making Python `reshape(-1, K)` first, which
// cost ~1.3 us per call on shapes where the kernel itself costs ~10.
#define FLATTENED_ROWS(x) ((x).numel() / (x).size((x).ndim() - 1))

// CUTLASS 2.x alignment-8 iterators: both GEMM free dimensions must divide by 8.
//
// Checked here, in every GEMM-family launcher, for one reason: the *same*
// question used to get two different answers.  `gemm` let CUTLASS fail and
// surfaced "GEMM kernel failed", which says nothing about what to do;
// `gemm_log_softmax` silently rerouted to cuBLAS, which says nothing at all.
// Neither is acceptable for a precondition the caller can trivially satisfy —
// every unaligned case in this repo is an output projection, and padding it is
// the established fix (`oasr.models.base.align_out_features` +
// `pad_output_projection`, which the WeNet CTC head has used since day one).
#define CHECK_GEMM_ALIGNMENT(N, K)                                                        \
    TVM_FFI_ICHECK((N) % 8 == 0 && (K) % 8 == 0)                                          \
        << "GEMM needs both free dimensions 8-aligned (CUTLASS alignment-8 "              \
           "iterators), got N=" << (N) << " K=" << (K)                                    \
        << ". Pad the projection at the model layer -- see "                               \
           "oasr.models.base.align_out_features / pad_output_projection."

// =============================================================================
// Dtype dispatch macros
// =============================================================================

// Dispatch for FP16/BF16/FP32 dtypes
#define DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(dl_dtype, c_type, ...)                        \
    [&]() -> bool {                                                                       \
        if ((dl_dtype).code == kDLFloat && (dl_dtype).bits == 32) {                       \
            using c_type = float;                                                         \
            return __VA_ARGS__();                                                         \
        } else if ((dl_dtype).code == kDLFloat && (dl_dtype).bits == 16) {                \
            using c_type = half;                                                          \
            return __VA_ARGS__();                                                         \
        } else if ((dl_dtype).code == kDLBfloat && (dl_dtype).bits == 16) {               \
            using c_type = __nv_bfloat16;                                                 \
            return __VA_ARGS__();                                                         \
        } else {                                                                          \
            TVM_FFI_ICHECK(false) << "Unsupported dtype: code=" << (dl_dtype).code        \
                                  << " bits=" << (dl_dtype).bits;                         \
            return false;                                                                 \
        }                                                                                 \
    }()

// Alias: same as DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16 (dispatches FP32/FP16/BF16)
#define DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16 DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16

// Dispatch for FP16/BF16 only (half-precision input types for GEMM/conv)
#define DISPATCH_DLPACK_HALF_DTYPE(dl_dtype, c_type, ...)                                 \
    [&]() -> bool {                                                                       \
        if ((dl_dtype).code == kDLFloat && (dl_dtype).bits == 16) {                       \
            using c_type = half;                                                          \
            return __VA_ARGS__();                                                         \
        } else if ((dl_dtype).code == kDLBfloat && (dl_dtype).bits == 16) {               \
            using c_type = __nv_bfloat16;                                                 \
            return __VA_ARGS__();                                                         \
        } else {                                                                          \
            TVM_FFI_ICHECK(false) << "Unsupported input dtype for GEMM/conv: "            \
                                  << "code=" << (dl_dtype).code                           \
                                  << " bits=" << (dl_dtype).bits                          \
                                  << ". Only FP16 and BF16 are supported.";               \
            return false;                                                                 \
        }                                                                                 \
    }()

// =============================================================================
// CUDA helpers
// =============================================================================

inline cudaStream_t get_stream(DLDevice device) {
    // Use the caller's current CUDA stream (set on the FFI env by the
    // framework, e.g. ``tvm_ffi.use_torch_stream`` or torch's autograd
    // dispatcher). Returning ``nullptr`` (the null/default stream) here
    // breaks CUDA Graph capture: ``torch.cuda.graph`` records kernels
    // launched on the *capture* stream and silently skips any launched on
    // the null stream, so every JIT-compiled OASR op (conv2d, gemm, glu,
    // norm, ...) is left out of the captured graph, the resulting graph is
    // empty (PyTorch warns ``CUDA Graph is empty``), and replays produce
    // wrong / NaN outputs because none of the encoder ops actually run.
    TVMFFIStreamHandle s = TVMFFIEnvGetStream(device.device_type, device.device_id);
    return static_cast<cudaStream_t>(s);
}

inline size_t get_element_size(DLDataType dtype) {
    return (dtype.bits * dtype.lanes + 7) / 8;
}

}  // namespace oasr
