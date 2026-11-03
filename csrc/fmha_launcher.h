// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The TVM-FFI side of the C++ attention launchers, shared by every rendered
// FMHA translation unit (`fmha_template.cu.jinja` for dense, paged and packed,
// `fmha_split_template.cu.jinja` for split-KV, `fmha_kv8_template.cu.jinja` for
// an fp8 KV cache): one statement of what the kernel assumes of its operands.
//
// `oasr.fmha` / `oasr.fmha_varlen` establish all of it before a call -- they
// copy a tensor the kernel cannot address, convert lengths and offsets to the
// width it reads, and refuse a head_dim it cannot load -- so these checks are
// for a caller of the module itself, who skips all of that.  What they stand in
// front of is not an exception: an operand the kernel addresses wrongly is a
// sticky `misaligned address` / `illegal memory access`, which ends the CUDA
// context, or a plausible wrong answer.

#pragma once

#include <initializer_list>

#include "tvm_ffi_utils.h"

namespace oasr {
namespace attention {
namespace launcher {

/*! \brief A q, k, v or output operand of the fp16/bf16 mainloop and epilogue.
 *
 * Both move it in 128-bit vectors along the head dimension and step every
 * other axis by its stride, so it must be the variant's element type, on the
 * device, with a contiguous last axis, a 16-byte base and whole-vector strides.
 */
inline void checkVectorOperand(TensorView const& x, DLDataType element, char const* dtype_name,
                               char const* name, TensorView const& like) {
    TVM_FFI_ICHECK(x.device().device_type == kDLCUDA &&
                   x.device().device_id == like.device().device_id)
        << name << " must be a CUDA tensor on q's device";
    TVM_FFI_ICHECK(x.dtype() == element)
        << name << " must be " << dtype_name << ", this variant's element type";
    TVM_FFI_ICHECK_EQ(x.stride(x.ndim() - 1), 1)
        << name << " must have a contiguous last dimension; got stride " << x.stride(x.ndim() - 1);
    CheckVectorAlignedEveryAxis(x, name);
}

/*! \brief The head dimension the 128-bit loads can cover.
 *
 * A load covers `elems` columns at once and is predicated per vector, so a
 * head_dim that is not a whole number of vectors reads and writes the columns
 * past it -- the next row's, or past the allocation.
 */
inline void checkHeadDim(int D, int padded, int elems) {
    TVM_FFI_ICHECK_LE(D, padded) << "head_dim " << D << " exceeds this variant's " << padded;
    TVM_FFI_ICHECK_EQ(D % elems, 0)
        << "head_dim must be a multiple of " << elems << " (one 128-bit load); got " << D;
}

/*! \brief Two tensors of one shape (k and v; q and the output). */
inline void checkSameShape(TensorView const& a, char const* a_name, TensorView const& b,
                           char const* b_name) {
    bool same = a.ndim() == b.ndim();
    for (int d = 0; same && d < a.ndim(); ++d) {
        same = a.size(d) == b.size(d);
    }
    TVM_FFI_ICHECK(same) << a_name << " and " << b_name << " must have the same shape";
}

/*! \brief `cache_seqlens` / `cache_seqstarts`: one int32 per batch row, read as `ptr[b]`. */
inline void checkLengths(Optional const& cache_seqlens, Optional const& cache_seqstarts, int B,
                         TensorView const& like) {
    if (cache_seqlens.has_value()) {
        CheckIntVector(cache_seqlens.value(), 32, B, "cache_seqlens", like);
    }
    if (cache_seqstarts.has_value()) {
        CheckIntVector(cache_seqstarts.value(), 32, B, "cache_seqstarts", like);
    }
}

/*! \brief A paged pool's `(B, max_pages)` int32 table, read as `ptr[b * stride + page]`. */
inline void checkBlockTable(Optional const& block_table, int B, TensorView const& like) {
    TVM_FFI_ICHECK(block_table.has_value()) << "paged variant requires a block_table";
    TensorView const t = block_table.value();
    TVM_FFI_ICHECK(t.device().device_type == kDLCUDA &&
                   t.device().device_id == like.device().device_id)
        << "block_table must be a CUDA tensor on q's device";
    TVM_FFI_ICHECK(t.ndim() == 2 && t.dtype().code == kDLInt && t.dtype().bits == 32)
        << "block_table must be a 2-D int32 tensor";
    TVM_FFI_ICHECK(t.size(1) <= 1 || t.stride(1) == 1)
        << "block_table must have a contiguous last dimension";
    TVM_FFI_ICHECK_GE(t.size(0), B) << "block_table needs a row per batch row";
}

/*! \brief Split-KV's fp32 workspaces, indexed as contiguous
 *  `(num_splits, B, H, T_q, D)` and `(num_splits, B, H, T_q)`.
 */
inline void checkSplitWorkspace(TensorView const& o_partial, TensorView const& lse_partial,
                                int64_t num_splits, int B, int H, int T_q, int D,
                                TensorView const& like) {
    for (TensorView const* t : {&o_partial, &lse_partial}) {
        TVM_FFI_ICHECK(t->device().device_type == kDLCUDA &&
                       t->device().device_id == like.device().device_id)
            << "the split workspaces must be CUDA tensors on q's device";
        TVM_FFI_ICHECK(t->dtype() == dl_float32) << "the split workspaces must be fp32";
        TVM_FFI_ICHECK(t->IsContiguous()) << "the split workspaces must be contiguous";
    }
    TVM_FFI_ICHECK_GE(num_splits, 1) << "num_splits must be at least 1";
    TVM_FFI_ICHECK_GE(o_partial.numel(), int64_t(num_splits) * B * H * T_q * D)
        << "o_partial workspace is too small for " << num_splits << " splits";
    TVM_FFI_ICHECK_GE(lse_partial.numel(), int64_t(num_splits) * B * H * T_q)
        << "lse_partial workspace is too small for " << num_splits << " splits";
}

/*! \brief A dense `(B, H, rows, cols)` bias: the element type, and a plane per
 *  (batch row, query head).
 *
 * Its row and column extents may be short of `(T_q, T_k)` -- the tile loader
 * predicates against them, and a paged caller's bias is its own logical extent
 * -- but its batch and head axes are stepped by stride with no predicate, so a
 * short one is read past.
 */
inline void checkDenseBias(TensorView const& b, DLDataType element, char const* dtype_name, int B,
                           int H, TensorView const& like) {
    TVM_FFI_ICHECK(b.device().device_type == kDLCUDA &&
                   b.device().device_id == like.device().device_id)
        << "bias must be a CUDA tensor on q's device";
    TVM_FFI_ICHECK_EQ(b.ndim(), 4) << "bias must be (B, H, T_q, T_k)";
    TVM_FFI_ICHECK(b.dtype() == element) << "bias must be " << dtype_name << ", like q";
    TVM_FFI_ICHECK_EQ(b.stride(3), 1) << "bias must have a contiguous last dimension";
    TVM_FFI_ICHECK(b.size(0) == B && b.size(1) == H)
        << "bias must be (B, H, T_q, T_k) with q's B = " << B << " and H = " << H << "; got ("
        << b.size(0) << ", " << b.size(1) << ", ...)";
}

/*! \brief A packed block-diagonal bias and its offsets.
 *
 * The kernel reads segment `s`'s block as `(H, T_q_s, T_k_s)` row-major at
 * element `bias_offsets[s]` of a flat buffer and ignores the tensor's own
 * strides, so the buffer must be contiguous.  The offsets are int64: the
 * buffer is `sum_s H * T_s^2` elements, which an int32 index runs out of at 256
 * segments of 1024 frames and 8 heads.  Its base may be at any element; a base
 * that is not 4-byte aligned takes the scalar read (`FmhaParams::
 * bias_vectorizable`).
 */
inline void checkPackedBias(TensorView const& b, TensorView const& bias_offsets, DLDataType element,
                            char const* dtype_name, int S, TensorView const& like) {
    TVM_FFI_ICHECK(b.device().device_type == kDLCUDA &&
                   b.device().device_id == like.device().device_id)
        << "bias must be a CUDA tensor on q's device";
    TVM_FFI_ICHECK(b.dtype() == element) << "bias must be " << dtype_name << ", like q";
    TVM_FFI_ICHECK(b.IsContiguous())
        << "a packed bias is read as one flat buffer and must be contiguous";
    CheckIntVector(bias_offsets, 64, int64_t(S) + 1, "bias_offsets", like);
}

}  // namespace launcher
}  // namespace attention
}  // namespace oasr
