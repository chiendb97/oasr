// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Host-side arguments and device-side params for the FMHA kernel.
//
// Framework-agnostic by construction: raw pointers, runtime shapes, runtime
// strides.  Nothing here knows about DLPack, torch or TVM-FFI -- that lives in
// `csrc/fmha.cu`, one layer up.
//
// ---------------------------------------------------------------------------
// Why one stride type covers every layout OASR passes
// ---------------------------------------------------------------------------
//
// `StrideQKV` is `Stride<int64_t, _1, int64_t, int64_t>` over a
// `(seqlen, head_dim, head, batch)` shape.  Only the head-dim stride is static.
// That one type expresses all five layouts the engine hands over:
//
//   dense q  (B, H, T, D)             {q.stride(2), _1, q.stride(1), q.stride(0)}
//   packed q (total, H, D)            {q.stride(0), _1, q.stride(1), 0}
//   paged KV (nblk, page, H_kv, D)    {k.stride(1), _1, k.stride(2), k.stride(0)}
//   dense bias (B, H, T_q, T_k)       {b.stride(2), b.stride(3), b.stride(1), b.stride(0)}
//   packed bias (flat, block-diag)    {T_k_seg, 1, T_q_seg*T_k_seg, 0}
//
// A batch stride of 0 is what makes a packed tensor a dense one with one batch.
// This is the whole reason this backend needs neither `_ensure_canonical`'s
// copy (measured 1.24-2.15x of the call at real call-site strides,
// `.artifacts/fmha_cpp_validation.md` § L1) nor a second varlen kernel.
//
// \warning The `_1` is a **compile-time** 1.  A tensor whose last stride is not
//   1 will be read at wrong addresses, silently and without a fault.
//   `csrc/fmha.cu` must therefore check `stride(-1) == 1` on q/k/v/o (and on
//   the bias) -- that check is not optional, and it is the price of the
//   flexibility above.  A capacity *slice* such as `k_buf[:, :, :t]` has a
//   stride gap rather than a wrong last stride; it is rejected by
//   `oasr::IsRowDense`, not by this one.

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>

#include <cmath>
#include <cstdint>

namespace oasr {
namespace attention {

using namespace cute;

//! `(seqlen, head_dim, head, batch)`.
using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
using StrideQKV = cute::Stride<int64_t, cute::_1, int64_t, int64_t>;

//! `(q_row, k_col, head, batch)` -- the additive bias, in the served dtype.
using ShapeBias = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
using StrideBias = cute::Stride<int64_t, cute::_1, int64_t, int64_t>;

//! `(batch, max_pages_per_seq)`.
using ShapePageTable = cute::Shape<int32_t, int32_t>;
using StridePageTable = cute::Stride<int64_t, cute::_1>;

/*! \brief What the caller supplies. */
template <class Element>
struct FmhaArguments {
    Element const* ptr_q;
    ShapeQKV shape_q;  //!< (T_q, D, H, B)
    StrideQKV stride_q;

    Element const* ptr_k;
    ShapeQKV shape_k;  //!< (T_k, D, H_kv, B); paged: (page_size, D, H_kv, num_pages)
    StrideQKV stride_k;

    Element const* ptr_v;
    StrideQKV stride_v;

    Element* ptr_o;
    StrideQKV stride_o;

    //! Additive bias, post-scale (SDPA semantics).  Null when absent.
    Element const* ptr_bias = nullptr;
    ShapeBias shape_bias = {};
    StrideBias stride_bias = {};
    int32_t const* bias_offsets = nullptr;  //!< packed block-diagonal bias

    //! Paged KV.  Null when the K/V above are dense.
    int32_t const* ptr_pagetable = nullptr;
    ShapePageTable shape_pagetable = {};
    StridePageTable stride_pagetable = {};
    int32_t page_size = 0;

    //! Per-stream key window `[seqstart_k, seqlen_k)`.  Null means the full extent.
    int32_t const* seqused_k = nullptr;
    int32_t const* seqstarts_k = nullptr;

    //! Packed (varlen) row offsets.  Null means dense.
    int32_t const* cu_seqlens_q = nullptr;
    int32_t const* cu_seqlens_k = nullptr;

    //! Split-KV (flash-decoding).  `num_splits <= 1` is the unsplit kernel and
    //! the two buffers are unused.  `O_partial` is
    //! `(num_splits, B, H, T_q, D)` fp32 and `LSE_partial` is
    //! `(num_splits, B, H, T_q)` fp32, both contiguous.
    //!
    //! Allocated by the **caller**, in Python, from torch's caching allocator.
    //! Deliberately not `oasr::getCachedWorkspace`: that branches on
    //! `cudaStreamIsCapturing` and hands back null during capture, which would
    //! force the dispatch to differ under a graph -- exactly what rule 11
    //! forbids, and for exactly the reason split-KV makes acute (splitting
    //! changes the fp32 summation order).
    float* ptr_o_partial = nullptr;
    float* ptr_lse_partial = nullptr;
    int num_splits = 1;

    float softmax_scale = 1.0f;
    int window_size_left = -1;   //!< -1 == unbounded
    int window_size_right = -1;  //!< -1 == unbounded
};

/*! \brief What the kernel reads. */
template <class Element>
struct FmhaParams {
    Element const* ptr_q;
    ShapeQKV shape_q;
    StrideQKV stride_q;

    Element const* ptr_k;
    ShapeQKV shape_k;
    StrideQKV stride_k;

    Element const* ptr_v;
    StrideQKV stride_v;

    Element* ptr_o;
    StrideQKV stride_o;

    Element const* ptr_bias;
    ShapeBias shape_bias;
    StrideBias stride_bias;
    int32_t const* bias_offsets;
    //! True when a bias row can be read as 32-bit pairs: the row stride is an
    //! even number of elements and the column stride is 1.  A CTA-uniform
    //! runtime bool rather than a template parameter, because the vectorised
    //! and predicated paths both have to exist anyway (a boundary tile takes
    //! the predicated one whatever the alignment), so specialising on it would
    //! only delete the fast path from half the binaries.
    bool bias_vectorizable;

    int32_t const* ptr_pagetable;
    ShapePageTable shape_pagetable;
    StridePageTable stride_pagetable;
    cutlass::FastDivmod page_size_divmod;

    int32_t const* seqused_k;
    int32_t const* seqstarts_k;
    int32_t const* cu_seqlens_q;
    int32_t const* cu_seqlens_k;

    cutlass::FastDivmod qhead_per_khead_divmod;

    //! `softmax_scale * log2(e)`, folded on the host so the kernel uses `exp2`.
    float softmax_scale_log2;
    //! `1 / softmax_scale`, folded on the host.  The bias is a *post*-scale
    //! logit under SDPA semantics, and it is added to the *pre*-scale
    //! accumulator, so it is pre-divided here.  Host-side and exact, rather
    //! than an in-kernel `rcp.approx` whose result would depend on a compiler
    //! flag -- this backend is compared against the CuTeDSL one at tight
    //! tolerance, so a divide that moves with `--use_fast_math` is a hazard.
    float inv_softmax_scale;

    float* ptr_o_partial;
    float* ptr_lse_partial;
    int num_splits;

    int window_size_left;
    int window_size_right;
};

/*! \brief Fold the host-side constants and build the divmods. */
template <class Element>
FmhaParams<Element> to_underlying_arguments(FmhaArguments<Element> const& args) {
    int const qhead_per_khead =
        cute::ceil_div(cute::get<2>(args.shape_q), cute::get<2>(args.shape_k));
    // A bias row is vectorisable when consecutive columns are adjacent *and*
    // every row start is 4-byte aligned for a 2-element fp16 pair.  "Every row
    // start" means the head and batch planes too: the MMA-C partition hands a
    // thread an *even* column offset within its row, so the pair address is
    // `ptr + b*sB + h*sH + m*sM + even`, and a single odd term in that sum
    // misaligns the whole plane.  Checking only the row stride -- which is what
    // the obvious reading of "row start" gets you -- leaves a `bias[:, 1:]`
    // style view passing the test and issuing a misaligned 32-bit load.
    bool const bias_vec = args.ptr_bias != nullptr &&
                          (cute::get<1>(args.stride_bias) == 1) &&
                          ((cute::get<0>(args.stride_bias) % 2) == 0) &&
                          ((cute::get<2>(args.stride_bias) % 2) == 0) &&
                          ((cute::get<3>(args.stride_bias) % 2) == 0) &&
                          ((reinterpret_cast<uintptr_t>(args.ptr_bias) % 4) == 0);
    return FmhaParams<Element>{
        args.ptr_q,
        args.shape_q,
        args.stride_q,
        args.ptr_k,
        args.shape_k,
        args.stride_k,
        args.ptr_v,
        args.stride_v,
        args.ptr_o,
        args.stride_o,
        args.ptr_bias,
        args.shape_bias,
        args.stride_bias,
        args.bias_offsets,
        bias_vec,
        args.ptr_pagetable,
        args.shape_pagetable,
        args.stride_pagetable,
        cutlass::FastDivmod(args.page_size > 0 ? args.page_size : 1),
        args.seqused_k,
        args.seqstarts_k,
        args.cu_seqlens_q,
        args.cu_seqlens_k,
        cutlass::FastDivmod(qhead_per_khead > 0 ? qhead_per_khead : 1),
        float(args.softmax_scale * float(M_LOG2E)),
        args.softmax_scale != 0.0f ? 1.0f / args.softmax_scale : 0.0f,
        args.ptr_o_partial,
        args.ptr_lse_partial,
        args.num_splits > 0 ? args.num_splits : 1,
        args.window_size_left,
        args.window_size_right,
    };
}

}  // namespace attention
}  // namespace oasr
