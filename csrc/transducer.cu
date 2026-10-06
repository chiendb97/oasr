// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher for the fused stateless-transducer greedy decode.

#include <oasr/transducer/greedy_decode.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

namespace {

bool is_dtype(const TensorView& t, DLDataType d) {
    return t.dtype().code == d.code && t.dtype().bits == d.bits && t.dtype().lanes == d.lanes;
}

constexpr DLDataType dl_int64 = {kDLInt, 64, 1};

void check_vec16(const TensorView& t, const char* name) {
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16, 0u)
        << name << " must be 16-byte aligned (the kernel reads 8-element vectors)";
}

}  // namespace

// Greedy-decode `enc_proj` (encoder output already in joiner space) from the
// predictor state `(window_in, dec_proj_in)`, writing the emitted tokens, their
// frames (and posteriors when `probs` is given), the per-row emission counts,
// and the final predictor state.  `window_out` / `dec_proj_out` may alias the
// inputs: each CTA reads its rows' state before it writes any of it.
void stateless_greedy_decode(TensorView tokens, TensorView frames, Optional probs,
                             TensorView counts, TensorView window_out, TensorView dec_proj_out,
                             TensorView enc_proj, TensorView lengths, TensorView window_in,
                             TensorView dec_proj_in, TensorView w_out_t, Optional b_out,
                             TensorView emb, Optional conv_w, TensorView w_dp_t, Optional b_dp,
                             int64_t vocab, int64_t group, int64_t max_sym, int64_t blank,
                             int64_t activation, int64_t rows_per_cta) {
    CHECK_INPUT(enc_proj);
    CHECK_DIM(3, enc_proj);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(enc_proj);
    const int64_t B = enc_proj.size(0);
    const int64_t J = enc_proj.size(2);
    TVM_FFI_ICHECK(J % transducer::kStageK == 0)
        << "joiner dim must be a multiple of " << transducer::kStageK << "; got " << J;
    TVM_FFI_ICHECK(enc_proj.stride(0) % 8 == 0 && enc_proj.stride(1) % 8 == 0)
        << "enc_proj row strides must be multiples of 8 elements";
    check_vec16(enc_proj, "enc_proj");

    CHECK_INPUT(lengths);
    CHECK_CONTIGUOUS_INPUT(lengths);
    TVM_FFI_ICHECK(is_dtype(lengths, dl_int64) && lengths.ndim() == 1 && lengths.size(0) == B)
        << "lengths must be int64 of shape (B,)";

    CHECK_INPUT(window_in);
    CHECK_CONTIGUOUS_INPUT(window_in);
    CHECK_DIM(2, window_in);
    TVM_FFI_ICHECK(is_dtype(window_in, dl_int64) && window_in.size(0) == B)
        << "window_in must be int64 of shape (B, ctx)";
    const int64_t ctx = window_in.size(1);
    TVM_FFI_ICHECK(ctx >= 1 && ctx <= transducer::kMaxContext)
        << "predictor context must be in [1, " << transducer::kMaxContext << "], got " << ctx;
    CHECK_INPUT(window_out);
    CHECK_CONTIGUOUS_INPUT(window_out);
    TVM_FFI_ICHECK(is_dtype(window_out, dl_int64) && window_out.ndim() == 2 &&
                   window_out.size(0) == B && window_out.size(1) == ctx)
        << "window_out must be int64 of shape (B, ctx)";

    CHECK_INPUT(w_out_t);
    CHECK_CONTIGUOUS_INPUT(w_out_t);
    CHECK_DIM(2, w_out_t);
    const int64_t ld_out = w_out_t.size(1);
    TVM_FFI_ICHECK(w_out_t.size(0) == J && ld_out % transducer::kGemvTile == 0 && vocab >= 1 &&
                   vocab <= ld_out)
        << "w_out_t must be the K-major (J, V_pad) output projection, V_pad a multiple of "
        << transducer::kGemvTile << " covering the vocabulary";
    check_vec16(w_out_t, "w_out_t");

    CHECK_INPUT(emb);
    CHECK_CONTIGUOUS_INPUT(emb);
    CHECK_DIM(2, emb);
    const int64_t D = emb.size(1);
    TVM_FFI_ICHECK(D % transducer::kStageK == 0)
        << "decoder dim must be a multiple of " << transducer::kStageK << "; got " << D;
    TVM_FFI_ICHECK(emb.size(0) >= vocab) << "embedding rows must cover the vocabulary";

    CHECK_INPUT(w_dp_t);
    CHECK_CONTIGUOUS_INPUT(w_dp_t);
    TVM_FFI_ICHECK(w_dp_t.ndim() == 2 && w_dp_t.size(0) == D &&
                   w_dp_t.size(1) % transducer::kGemvTile == 0 && w_dp_t.size(1) >= J)
        << "w_dp_t must be the K-major (D, J_pad) decoder projection, J_pad a multiple of "
        << transducer::kGemvTile << " covering the joiner dim";
    check_vec16(w_dp_t, "w_dp_t");

    TVM_FFI_ICHECK(group >= 1 && D % group == 0) << "conv group size must divide D";
    if (ctx > 1) {
        TVM_FFI_ICHECK(conv_w.has_value()) << "a context > 1 predictor needs its conv weight";
        const TensorView cw = conv_w.value();
        CHECK_INPUT(cw);
        CHECK_CONTIGUOUS_INPUT(cw);
        TVM_FFI_ICHECK(cw.numel() == D * ctx * group)
            << "conv weight must hold D * ctx * group elements";
    }

    CHECK_INPUT(dec_proj_in);
    CHECK_CONTIGUOUS_INPUT(dec_proj_in);
    TVM_FFI_ICHECK(dec_proj_in.ndim() == 2 && dec_proj_in.size(0) == B && dec_proj_in.size(1) == J)
        << "dec_proj_in must be (B, J)";
    CHECK_INPUT(dec_proj_out);
    CHECK_CONTIGUOUS_INPUT(dec_proj_out);
    TVM_FFI_ICHECK(dec_proj_out.ndim() == 2 && dec_proj_out.size(0) == B &&
                   dec_proj_out.size(1) == J)
        << "dec_proj_out must be (B, J)";

    CHECK_INPUT(tokens);
    CHECK_CONTIGUOUS_INPUT(tokens);
    TVM_FFI_ICHECK(is_dtype(tokens, dl_int32) && tokens.ndim() == 2 && tokens.size(0) == B)
        << "tokens must be int32 of shape (B, cap)";
    const int64_t cap = tokens.size(1);
    CHECK_INPUT(frames);
    CHECK_CONTIGUOUS_INPUT(frames);
    TVM_FFI_ICHECK(is_dtype(frames, dl_int32) && frames.ndim() == 2 && frames.size(0) == B &&
                   frames.size(1) == cap)
        << "frames must be int32 of shape (B, cap)";
    CHECK_INPUT(counts);
    CHECK_CONTIGUOUS_INPUT(counts);
    TVM_FFI_ICHECK(is_dtype(counts, dl_int32) && counts.ndim() == 1 && counts.size(0) == B)
        << "counts must be int32 of shape (B,)";
    if (probs.has_value()) {
        const TensorView pr = probs.value();
        CHECK_INPUT(pr);
        CHECK_CONTIGUOUS_INPUT(pr);
        TVM_FFI_ICHECK(is_dtype(pr, dl_float32) && pr.ndim() == 2 && pr.size(0) == B &&
                       pr.size(1) == cap)
            << "probs must be float32 of shape (B, cap)";
    }
    TVM_FFI_ICHECK(activation == transducer::kJoinerTanh || activation == transducer::kJoinerRelu)
        << "unknown joiner activation " << activation;

    cudaStream_t stream = get_stream(enc_proj.device());

    // Half precision only: the GEMVs multiply packed half-precision pairs.
    DISPATCH_DLPACK_HALF_DTYPE(enc_proj.dtype(), c_type, [&] {
        for (const TensorView* t : {&dec_proj_in, &dec_proj_out, &w_out_t, &emb, &w_dp_t}) {
            TVM_FFI_ICHECK(t->dtype().code == enc_proj.dtype().code &&
                           t->dtype().bits == enc_proj.dtype().bits)
                << "every floating-point operand must share enc_proj's dtype";
        }
        transducer::StatelessGreedyParams<c_type> p;
        p.enc_proj = static_cast<const c_type*>(enc_proj.data_ptr());
        p.enc_stride_b = enc_proj.stride(0);
        p.enc_stride_t = enc_proj.stride(1);
        p.lengths = static_cast<const int64_t*>(lengths.data_ptr());
        p.window_in = static_cast<const int64_t*>(window_in.data_ptr());
        p.dec_proj_in = static_cast<const c_type*>(dec_proj_in.data_ptr());
        p.w_out_t = static_cast<const c_type*>(w_out_t.data_ptr());
        p.b_out = OptionalDataPtr<const c_type>(b_out);
        p.emb = static_cast<const c_type*>(emb.data_ptr());
        p.conv_w = OptionalDataPtr<const c_type>(conv_w);
        p.w_dp_t = static_cast<const c_type*>(w_dp_t.data_ptr());
        p.b_dp = OptionalDataPtr<const c_type>(b_dp);
        p.tokens = static_cast<int32_t*>(tokens.data_ptr());
        p.frames = static_cast<int32_t*>(frames.data_ptr());
        p.probs = OptionalDataPtr<float>(probs);
        p.counts = static_cast<int32_t*>(counts.data_ptr());
        p.window_out = static_cast<int64_t*>(window_out.data_ptr());
        p.dec_proj_out = static_cast<c_type*>(dec_proj_out.data_ptr());
        p.B = static_cast<int>(B);
        p.J = static_cast<int>(J);
        p.D = static_cast<int>(D);
        p.V = static_cast<int>(vocab);
        p.ld_out = static_cast<int>(ld_out);
        p.ld_dp = static_cast<int>(w_dp_t.size(1));
        p.ctx = static_cast<int>(ctx);
        p.group = static_cast<int>(group);
        p.cap = static_cast<int>(cap);
        p.max_sym = static_cast<int>(max_sym);
        p.blank = static_cast<int>(blank);
        p.activation = static_cast<int>(activation);
        cudaError_t status =
            transducer::StatelessGreedyDecode<c_type>(p, static_cast<int>(rows_per_cta), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "stateless greedy decode failed: " << cudaGetErrorString(status);
        return true;
    });
}
