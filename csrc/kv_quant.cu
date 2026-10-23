// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher for FP8 KV-cache quantization (`include/oasr/kv_quant/`).

#include <oasr/kv_quant/kv_quant.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

namespace {

bool isFp8E4M3(DLDataType t) {
    return t.code == kDLFloat8_e4m3fn && t.bits == 8 && t.lanes == 1;
}

bool isFloat32(DLDataType t) {
    return t.code == kDLFloat && t.bits == 32 && t.lanes == 1;
}

/*! \brief Strides of a `(B, H, T, D)` input, which must be 16-byte vectorisable. */
template <typename Element>
void bindInput(kv_quant::KvQuantParams<Element>& p, int i, TensorView x, int B, int H, int T,
               int D) {
    CHECK_INPUT(x);
    CHECK_DIM(4, x);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(x);
    TVM_FFI_ICHECK(x.size(0) == B && x.size(1) == H && x.size(2) == T && x.size(3) == D)
        << "kv_quant_store: k and v must have the same (B, H, T, D) shape";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16, 0u)
        << "kv_quant_store: the input must be 16-byte aligned (128-bit loads)";
    for (int d = 0; d < 3; ++d) {
        TVM_FFI_ICHECK_EQ(x.stride(d) % 8, 0)
            << "kv_quant_store: every input stride must be a whole number of 8-element "
               "vectors; got stride "
            << x.stride(d) << " on axis " << d;
    }
    p.x[i] = static_cast<Element const*>(x.data_ptr());
    p.x_stride_b[i] = x.stride(0);
    p.x_stride_h[i] = x.stride(1);
    p.x_stride_t[i] = x.stride(2);
}

/*! \brief An output viewed `(B or 1, H, slots, D)` and its scale `(B or 1, H, slots)`.
 *
 * A leading extent of 1 with `B > 1` means the destination is shared by every
 * row and addressed by `pos` alone -- a paged pool.  Its batch stride is then 0.
 */
template <typename P>
void bindOutput(P& p, int i, TensorView out, TensorView scale, int B, int H, int D) {
    CHECK_INPUT(out);
    CHECK_INPUT(scale);
    CHECK_DIM(4, out);
    CHECK_DIM(3, scale);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(out);
    TVM_FFI_ICHECK(isFp8E4M3(out.dtype())) << "kv_quant_store: the cache must be float8_e4m3fn";
    TVM_FFI_ICHECK(isFloat32(scale.dtype())) << "kv_quant_store: scales must be float32";
    TVM_FFI_ICHECK(out.size(1) == H && out.size(3) == D && scale.size(1) == H)
        << "kv_quant_store: the cache's head axes do not match the input";
    TVM_FFI_ICHECK(out.size(0) == B || out.size(0) == 1)
        << "kv_quant_store: the cache's leading axis must be the batch or 1 (a shared pool)";
    TVM_FFI_ICHECK(out.size(0) == scale.size(0) && out.size(2) == scale.size(2))
        << "kv_quant_store: the cache and its scales must address the same slots";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(out.data_ptr()) % 8, 0u)
        << "kv_quant_store: the cache must be 8-byte aligned";
    for (int d = 0; d < 3; ++d) {
        TVM_FFI_ICHECK_EQ(out.stride(d) % 8, 0)
            << "kv_quant_store: every cache stride must be a multiple of 8 bytes";
    }
    bool const shared = out.size(0) == 1 && B > 1;
    p.out[i] = static_cast<uint8_t*>(out.data_ptr());
    p.o_stride_b[i] = shared ? 0 : out.stride(0);
    p.o_stride_h[i] = out.stride(1);
    p.o_stride_t[i] = out.stride(2);
    p.scale[i] = static_cast<float*>(scale.data_ptr());
    p.s_stride_b[i] = shared ? 0 : scale.stride(0);
    p.s_stride_h[i] = scale.stride(1);
    p.s_stride_t[i] = scale.stride(2);
}

}  // namespace

/*! \brief Quantize K (and V) vectors into an fp8 cache, one pow2 scale per vector.
 *
 * Outputs first (`AGENTS.md` rule 4).
 *
 * \param k_out, v_out       fp8 caches viewed `(B or 1, H, slots, D)`
 * \param k_scale, v_scale   fp32 `(B or 1, H, slots)`
 * \param k, v               fp16/bf16 `(B, H, T, D)`; `v_out`/`v_scale`/`v` optional
 * \param pos                optional int64 `(B * T,)` slot per new token; absent = `t`
 */
void kv_quant_store(TensorView k_out, TensorView k_scale, Optional v_out, Optional v_scale,
                    TensorView k, Optional v, Optional pos) {
    int const B = int(k.size(0));
    int const H = int(k.size(1));
    int const T = int(k.size(2));
    int const D = int(k.size(3));
    TVM_FFI_ICHECK(D % 8 == 0 && D <= kv_quant::kMaxHeadDim)
        << "kv_quant_store: head_dim must be a multiple of 8 and at most " << kv_quant::kMaxHeadDim
        << ", got " << D;
    bool const two = v.has_value();
    TVM_FFI_ICHECK(two == v_out.has_value() && two == v_scale.has_value())
        << "kv_quant_store: v, v_out and v_scale come together";

    int64_t const* pos_ptr = nullptr;
    if (pos.has_value()) {
        TensorView const pv = pos.value();
        CHECK_INPUT(pv);
        CHECK_CONTIGUOUS_INPUT(pv);
        TVM_FFI_ICHECK(pv.dtype().code == kDLInt && pv.dtype().bits == 64)
            << "kv_quant_store: pos must be int64";
        TVM_FFI_ICHECK_EQ(pv.numel(), int64_t(B) * T) << "kv_quant_store: pos needs B*T entries";
        pos_ptr = static_cast<int64_t const*>(pv.data_ptr());
    } else {
        TVM_FFI_ICHECK(k_out.size(0) == B)
            << "kv_quant_store: a shared pool needs explicit write positions";
    }

    cudaStream_t stream = get_stream(k.device());
    DISPATCH_DLPACK_HALF_DTYPE(k.dtype(), c_type, [&] {
        kv_quant::KvQuantParams<c_type> p{};
        bindInput(p, 0, k, B, H, T, D);
        bindOutput(p, 0, k_out, k_scale, B, H, D);
        if (two) {
            TVM_FFI_ICHECK(v.value().dtype().code == k.dtype().code &&
                           v.value().dtype().bits == k.dtype().bits)
                << "kv_quant_store: k and v must share a dtype";
            bindInput(p, 1, v.value(), B, H, T, D);
            bindOutput(p, 1, v_out.value(), v_scale.value(), B, H, D);
        }
        p.pos = pos_ptr;
        p.B = B;
        p.H = H;
        p.T = T;
        p.D = D;
        cudaError_t status = kv_quant::KvQuantStore<c_type>(p, two ? 2 : 1, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "kv_quant_store failed: " << cudaGetErrorString(status);
        return true;
    });
}
