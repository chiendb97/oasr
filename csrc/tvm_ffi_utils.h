// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Shared TVM-FFI utilities for OASR launchers.
// Mirrors FlashInfer's tvm_ffi_utils.h pattern.

#pragma once

#include <cuda_runtime.h>
#include <oasr/common/arch_dispatch.h>
#include <oasr/common/utils.h>

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>
#include <utility>

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

// `stride(-1) == 1`.  Not optional wherever a kernel's layout carries a
// compile-time unit stride on the last axis (every CuTe family's `_1`): a
// tensor whose last stride is not 1 is then read at wrong addresses silently,
// with no fault.  That is the price of accepting an arbitrary stride on every
// other axis.
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x)                                                \
    TVM_FFI_ICHECK((x).stride((x).ndim() - 1) == 1)                                     \
        << #x << " must have a contiguous last dimension; got stride "                    \
        << (x).stride((x).ndim() - 1)

// The 128-bit vector contract: a 16-byte-aligned base *and* a row stride that
// is a whole number of `elems`-element vectors.  Either one alone is not
// enough -- a misaligned stride makes every row after the first misaligned,
// which surfaces as `misaligned address` on a good day and as a wrong answer
// on a bad one.
#define CHECK_VECTOR_ALIGNED(x, elems)                                                    \
    do {                                                                                  \
        TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>((x).data_ptr()) % 16, 0u)            \
            << #x << " must be 16-byte aligned (the 128-bit load/store width)";           \
        TVM_FFI_ICHECK_EQ((x).stride(0) % (elems), 0)                                      \
            << #x << " needs a row stride that is a multiple of " << (elems)              \
            << " elements; got " << (x).stride(0);                                        \
    } while (0)

// The same contract for a tensor of any rank that a kernel walks with 128-bit
// accesses along its last axis: a 16-byte-aligned base, and the stride of every
// other axis longer than one a whole number of vectors (an axis of extent 1 is
// never stepped along, so its stride is free).  The vector width comes from the
// dtype.  An odd element anywhere in `base + sum(i_d * stride_d)` misaligns the
// whole access, and that is a sticky `misaligned address`, not an exception.
inline void CheckVectorAlignedEveryAxis(TensorView const& x, char const* name) {
    int64_t const elems = 128 / (int64_t(x.dtype().bits) * x.dtype().lanes);
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16, 0u)
        << name << " must be 16-byte aligned (the 128-bit load/store width)";
    for (int d = 0; d + 1 < x.ndim(); ++d) {
        TVM_FFI_ICHECK(x.size(d) <= 1 || x.stride(d) % elems == 0)
            << name << ": the stride of axis " << d << " must be a multiple of " << elems
            << " elements (one 128-bit vector); got " << x.stride(d);
    }
}

// A vector a kernel reads as `ptr[i]`: a CUDA tensor on `like`'s device, rank 1,
// a signed integer of exactly `bits` bits, unit stride, and at least `min_len`
// entries.  The Python wrappers convert lengths and offsets to the right width
// before every call, so it is a direct caller who skips that -- and an int64
// vector read as int32 is a wrong answer, not a fault.
inline void CheckIntVector(TensorView const& x, int bits, int64_t min_len, char const* name,
                           TensorView const& like) {
    TVM_FFI_ICHECK(x.device().device_type == kDLCUDA &&
                   x.device().device_id == like.device().device_id)
        << name << " must be a CUDA tensor on the inputs' device";
    TVM_FFI_ICHECK_EQ(x.ndim(), 1) << name << " must be 1-D";
    TVM_FFI_ICHECK(x.dtype().code == kDLInt && x.dtype().bits == bits && x.dtype().lanes == 1)
        << name << " must be int" << bits;
    TVM_FFI_ICHECK(x.size(0) <= 1 || x.stride(0) == 1) << name << " must be contiguous";
    TVM_FFI_ICHECK_GE(x.size(0), min_len)
        << name << " needs at least " << min_len << " entries; got " << x.size(0);
}

// Full row-major contiguity.  Stronger than CHECK_LAST_DIM_CONTIGUOUS_INPUT and
// what a kernel indexing rows as `base + row * row_len` actually needs: a
// tensor can have stride(-1) == 1 and still have a padded row stride (any
// `x[:, -1]`-style slice of a wider buffer), in which case the weaker check
// passes and the kernel reads the wrong memory.
#define CHECK_CONTIGUOUS_INPUT(x)                                                         \
    TVM_FFI_ICHECK((x).IsContiguous())                                                    \
        << "Tensor must be contiguous (row-major, no padded strides)"

// True when contiguous trailing rows tile storage without gaps or overlap.
// Permuted dense views are valid because row-wise operations are independent;
// padded, expanded, and reversed views are not.
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

// The data pointer of an optional tensor, or null when it is absent.
template <class T>
inline T* OptionalDataPtr(Optional const& t) {
    return t.has_value() ? static_cast<T*>(t.value().data_ptr()) : nullptr;
}

// Alignment-8 iterators require both free dimensions to divide by eight. Check
// uniformly at the launcher boundary instead of failing or rerouting later.
#define CHECK_GEMM_ALIGNMENT(N, K)                                                        \
    TVM_FFI_ICHECK((N) % 8 == 0 && (K) % 8 == 0)                                          \
        << "GEMM needs both free dimensions 8-aligned (CUTLASS alignment-8 "              \
           "iterators), got N=" << (N) << " K=" << (K)                                    \
        << ". Pad the projection at the model layer -- see "                               \
           "oasr.models.base.align_out_features / pad_output_projection."

// CUTLASS's implicit-GEMM convolutions address each tensor with 32-bit byte
// offsets.  Past 2 GiB they refuse the problem with a bare failure status, which
// once surfaced as an engine-wide empty transcript; name the limit instead.
#define CHECK_CONV_LAUNCH_BYTES(x)                                                  \
    TVM_FFI_ICHECK(static_cast<int64_t>((x).numel()) *                              \
                       static_cast<int64_t>(oasr::get_element_size((x).dtype())) <= \
                   int64_t{0x7fffffff})                                             \
        << "convolution tensor " #x " is "                                          \
        << static_cast<int64_t>((x).numel()) *                                      \
               static_cast<int64_t>(oasr::get_element_size((x).dtype()))            \
        << " bytes, more than one launch can address (2 GiB); split the batch -- "  \
           "the oasr.conv* entry points do"

// Raise for a CUTLASS launcher's GemmStatus, keeping the two kinds of failure
// apart.  NOT_SUPPORTED -- the variant's can_implement refused the problem, or it
// has no path for these arguments -- raises OasrVariantUnsupported
// (oasr.jit.errors.VariantUnsupportedError in Python), which the dispatch layer
// answers by running the default variant.  Anything else is a launch that did not
// happen (a failed shared-memory opt-in, initialize(), the launch itself) and
// raises RuntimeError: a fallback there would hide a CUDA error until a later call
// trips over it, and a variant that can never launch until someone profiles.
#define CHECK_GEMM_STATUS(status, what)                                                   \
    do {                                                                                  \
        const ::oasr::gemm::GemmStatus oasr_status_ = (status);                           \
        if (oasr_status_ == ::oasr::gemm::GemmStatus::NOT_SUPPORTED) {                    \
            TVM_FFI_THROW(OasrVariantUnsupported)                                         \
                << (what) << " cannot implement this problem (NOT_SUPPORTED)";            \
        }                                                                                 \
        if (oasr_status_ != ::oasr::gemm::GemmStatus::SUCCESS) {                          \
            const cudaError_t oasr_cuda_ = cudaPeekAtLastError();                         \
            TVM_FFI_THROW(RuntimeError)                                                   \
                << (what) << " failed: "                                                  \
                << ::oasr::gemm::getGemmStatusString(oasr_status_)                        \
                << (oasr_cuda_ == cudaSuccess ? "" : " -- ")                              \
                << (oasr_cuda_ == cudaSuccess ? "" : cudaGetErrorString(oasr_cuda_));     \
        }                                                                                 \
    } while (0)

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
    //
    // The architecture check lives here, and it looks out of place until you ask
    // where else it could go.  A module is compiled for one SM family; the
    // tensor's device is the only thing that says which card the work is
    // actually for; and every launcher in the package calls this exactly once
    // with that device.  A ``CHECK_*`` macro would have to be remembered at some
    // forty call sites and at every new one -- and the failure it guards against
    // is silent, so forgetting it would not show up.  After the first call per
    // ordinal it is an array read and an integer compare.
    oasr::checkDeviceMatchesBuild(device.device_id);
    TVMFFIStreamHandle s = TVMFFIEnvGetStream(device.device_type, device.device_id);
    return static_cast<cudaStream_t>(s);
}

inline size_t get_element_size(DLDataType dtype) {
    return (dtype.bits * dtype.lanes + 7) / 8;
}

// A tensor's storage as `rows` byte intervals of `width` bytes, `pitch` apart:
// one interval for a contiguous tensor, one per row for a 2-D view whose rows
// do not overlap each other, and otherwise the whole span (conservative).
struct StorageRows {
    int64_t base;
    int64_t rows;
    int64_t pitch;
    int64_t width;
};

inline StorageRows AsStorageRows(const TensorView& t) {
    const int64_t esize = static_cast<int64_t>(get_element_size(t.dtype()));
    const auto base = static_cast<int64_t>(reinterpret_cast<uintptr_t>(t.data_ptr()));
    if (t.numel() == 0)
        return {base, 0, 0, 0};
    if (t.IsContiguous())
        return {base, 1, 0, t.numel() * esize};
    if (t.ndim() == 2 && t.stride(1) == 1 && t.stride(0) >= t.size(1)) {
        return {base, t.size(0), t.stride(0) * esize, t.size(1) * esize};
    }
    int64_t lowest = 0, highest = 0;  // element offsets, whatever the strides' signs
    for (int i = 0; i < t.ndim(); ++i) {
        const int64_t reach = (t.size(i) - 1) * t.stride(i);
        (reach < 0 ? lowest : highest) += reach;
    }
    return {base + lowest * esize, 1, 0, (highest - lowest + 1) * esize};
}

// Whether two tensors share any byte of storage.  Exact for contiguous tensors
// and row-strided 2-D views -- two column halves of one buffer interleave
// without overlapping -- and conservative for any other layout.  The row walk
// runs only when the two spans intersect, which separate allocations never do.
inline bool StorageOverlaps(const TensorView& a, const TensorView& b) {
    StorageRows x = AsStorageRows(a), y = AsStorageRows(b);
    if (x.rows == 0 || y.rows == 0)
        return false;
    const auto end = [](const StorageRows& s) { return s.base + (s.rows - 1) * s.pitch + s.width; };
    if (end(x) <= y.base || end(y) <= x.base)
        return false;
    if (x.rows > y.rows)
        std::swap(x, y);
    const auto floor_div = [](int64_t n, int64_t d) { return n / d - (n % d != 0 && n < 0); };
    for (int64_t i = 0; i < x.rows; ++i) {
        // Row i of x, relative to y's base: [lo, hi].  Row k of y is
        // [k * pitch, k * pitch + width - 1]; they meet iff kmin <= k <= kmax.
        const int64_t lo = x.base + i * x.pitch - y.base;
        const int64_t hi = lo + x.width - 1;
        if (y.rows == 1) {
            if (hi >= 0 && lo <= y.width - 1)
                return true;
            continue;
        }
        const int64_t kmax = std::min(floor_div(hi, y.pitch), y.rows - 1);
        const int64_t kmin = std::max(-floor_div(-(lo - y.width + 1), y.pitch), int64_t{0});
        if (kmin <= kmax)
            return true;
    }
    return false;
}

}  // namespace oasr
