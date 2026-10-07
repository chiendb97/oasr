// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher for the fused stateless-transducer greedy decode.

#include <oasr/transducer/beam_decode.cuh>
#include <oasr/transducer/beam_topk.cuh>
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

// One modified-beam-search frame after the joiner: log-softmax, score add,
// top-k over the beam's k * V candidates, the (parent, label) split, the mask
// for rows past their utterance and the label-window reorder.  `logits` is the
// joiner's (B * k, V) output, its rows possibly strided (the joiner slices a
// padded projection).  Scores are bit-identical to the torch composition; see
// include/oasr/transducer/beam_topk.cuh for the tie order.
void transducer_beam_topk(TensorView context_out, TensorView scores_out, TensorView parent_out,
                          TensorView label_out, TensorView logits, TensorView scores,
                          TensorView context, TensorView active, int64_t blank) {
    CHECK_INPUT(scores);
    CHECK_CONTIGUOUS_INPUT(scores);
    CHECK_DIM(2, scores);
    TVM_FFI_ICHECK(is_dtype(scores, dl_float32)) << "scores must be float32 of shape (B, k)";
    const int64_t B = scores.size(0);
    const int64_t k = scores.size(1);
    TVM_FFI_ICHECK(k >= 1 && k <= transducer::kBeamTopkMaxBeam)
        << "beam must be in [1, " << transducer::kBeamTopkMaxBeam << "], got " << k;

    CHECK_INPUT(logits);
    CHECK_DIM(2, logits);
    TVM_FFI_ICHECK(logits.stride(1) == 1) << "logits rows must be contiguous";
    TVM_FFI_ICHECK(logits.size(0) == B * k) << "logits must be (B * k, V)";
    const int64_t V = logits.size(1);
    TVM_FFI_ICHECK(V >= k && V <= (int64_t{1} << transducer::kBeamTopkMaxLog2Vocab))
        << "vocabulary must be in [beam, " << (1 << transducer::kBeamTopkMaxLog2Vocab) << "], got "
        << V;
    TVM_FFI_ICHECK(logits.stride(0) >= V) << "logits row stride must cover the vocabulary";

    CHECK_INPUT(context);
    CHECK_CONTIGUOUS_INPUT(context);
    TVM_FFI_ICHECK(is_dtype(context, dl_int64) && context.ndim() == 3 && context.size(0) == B &&
                   context.size(1) == k)
        << "context must be int64 of shape (B, k, ctx)";
    const int64_t ctx = context.size(2);
    TVM_FFI_ICHECK(ctx >= 1) << "context must hold at least one label";

    CHECK_INPUT(active);
    CHECK_CONTIGUOUS_INPUT(active);
    TVM_FFI_ICHECK(active.dtype().code == kDLBool && active.dtype().bits == 8 &&
                   active.ndim() == 1 && active.size(0) == B)
        << "active must be bool of shape (B,)";

    CHECK_INPUT(scores_out);
    CHECK_CONTIGUOUS_INPUT(scores_out);
    TVM_FFI_ICHECK(is_dtype(scores_out, dl_float32) && scores_out.ndim() == 2 &&
                   scores_out.size(0) == B && scores_out.size(1) == k)
        << "scores_out must be float32 of shape (B, k)";
    for (const TensorView* t : {&parent_out, &label_out}) {
        CHECK_INPUT(*t);
        CHECK_CONTIGUOUS_INPUT(*t);
        TVM_FFI_ICHECK(is_dtype(*t, dl_int64) && t->ndim() == 2 && t->size(0) == B &&
                       t->size(1) == k)
            << "parent_out / label_out must be int64 of shape (B, k)";
    }
    CHECK_INPUT(context_out);
    CHECK_CONTIGUOUS_INPUT(context_out);
    TVM_FFI_ICHECK(is_dtype(context_out, dl_int64) && context_out.ndim() == 3 &&
                   context_out.size(0) == B && context_out.size(1) == k &&
                   context_out.size(2) == ctx)
        << "context_out must be int64 of shape (B, k, ctx)";

    cudaStream_t stream = get_stream(logits.device());

    DISPATCH_DLPACK_HALF_DTYPE(logits.dtype(), c_type, [&] {
        transducer::BeamTopkParams<c_type> p;
        p.logits = static_cast<const c_type*>(logits.data_ptr());
        p.scores = static_cast<const float*>(scores.data_ptr());
        p.context = static_cast<const int64_t*>(context.data_ptr());
        p.active = static_cast<const bool*>(active.data_ptr());
        p.scores_out = static_cast<float*>(scores_out.data_ptr());
        p.parent_out = static_cast<int64_t*>(parent_out.data_ptr());
        p.label_out = static_cast<int64_t*>(label_out.data_ptr());
        p.context_out = static_cast<int64_t*>(context_out.data_ptr());
        p.B = static_cast<int>(B);
        p.k = static_cast<int>(k);
        p.V = static_cast<int>(V);
        p.ld = logits.stride(0);
        p.ctx = static_cast<int>(ctx);
        p.blank = blank;
        cudaError_t status = transducer::BeamTopk<c_type>(p, stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "transducer beam top-k failed: " << cudaGetErrorString(status);
        return true;
    });
}

// Modified beam search over every frame of `enc_proj` (encoder output already
// in joiner space) from the beam `(context_in, scores_in)`, in one launch: the
// beam after the chunk into `(context_out, scores_out)` -- which may alias the
// inputs, each CTA reads its utterance's beam before it writes any of it -- and
// each frame's back-pointers and labels into the frame-major `(frames, B, k)`
// `parents` / `labels`.  With `walk`, also every final slot walked back to the
// chunk's start: int32 roots (B * k), counts (B * k), then each hypothesis's
// tokens (B * k * frames, the first `count` valid).  `cluster` is the CTAs per
// utterance (0 picks; the result does not depend on it).  The weights are the
// greedy decode's.  See include/oasr/transducer/beam_decode.cuh.
void stateless_beam_decode(TensorView context_out, TensorView scores_out, TensorView parents,
                           TensorView labels, Optional walk, TensorView enc_proj,
                           TensorView lengths, TensorView context_in, TensorView scores_in,
                           TensorView w_out_t, Optional b_out, TensorView emb, Optional conv_w,
                           TensorView w_dp_t, Optional b_dp, int64_t vocab, int64_t group,
                           int64_t blank, int64_t activation, int64_t cluster) {
    CHECK_INPUT(enc_proj);
    CHECK_DIM(3, enc_proj);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(enc_proj);
    const int64_t B = enc_proj.size(0);
    const int64_t frames = enc_proj.size(1);
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

    CHECK_INPUT(scores_in);
    CHECK_CONTIGUOUS_INPUT(scores_in);
    TVM_FFI_ICHECK(is_dtype(scores_in, dl_float32) && scores_in.ndim() == 2 &&
                   scores_in.size(0) == B)
        << "scores_in must be float32 of shape (B, k)";
    const int64_t k = scores_in.size(1);
    TVM_FFI_ICHECK(k >= 1 && k <= transducer::kBeamDecodeMaxBeam)
        << "beam must be in [1, " << transducer::kBeamDecodeMaxBeam << "], got " << k;
    CHECK_INPUT(context_in);
    CHECK_CONTIGUOUS_INPUT(context_in);
    TVM_FFI_ICHECK(is_dtype(context_in, dl_int64) && context_in.ndim() == 3 &&
                   context_in.size(0) == B && context_in.size(1) == k)
        << "context_in must be int64 of shape (B, k, ctx)";
    const int64_t ctx = context_in.size(2);
    TVM_FFI_ICHECK(ctx >= 1 && ctx <= transducer::kMaxContext)
        << "predictor context must be in [1, " << transducer::kMaxContext << "], got " << ctx;

    CHECK_INPUT(w_out_t);
    CHECK_CONTIGUOUS_INPUT(w_out_t);
    CHECK_DIM(2, w_out_t);
    const int64_t ld_out = w_out_t.size(1);
    TVM_FFI_ICHECK(w_out_t.size(0) == J && ld_out % transducer::kGemvTile == 0 && vocab >= k &&
                   vocab <= transducer::kBeamDecodeMaxVocab && vocab <= ld_out)
        << "w_out_t must be the K-major (J, V_pad) output projection, V_pad a multiple of "
        << transducer::kGemvTile << " covering a vocabulary in [beam, "
        << transducer::kBeamDecodeMaxVocab << "]";
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

    CHECK_INPUT(context_out);
    CHECK_CONTIGUOUS_INPUT(context_out);
    TVM_FFI_ICHECK(is_dtype(context_out, dl_int64) && context_out.ndim() == 3 &&
                   context_out.size(0) == B && context_out.size(1) == k &&
                   context_out.size(2) == ctx)
        << "context_out must be int64 of shape (B, k, ctx)";
    CHECK_INPUT(scores_out);
    CHECK_CONTIGUOUS_INPUT(scores_out);
    TVM_FFI_ICHECK(is_dtype(scores_out, dl_float32) && scores_out.ndim() == 2 &&
                   scores_out.size(0) == B && scores_out.size(1) == k)
        << "scores_out must be float32 of shape (B, k)";
    for (const TensorView* t : {&parents, &labels}) {
        CHECK_INPUT(*t);
        CHECK_CONTIGUOUS_INPUT(*t);
        TVM_FFI_ICHECK(is_dtype(*t, dl_int64) && t->ndim() == 3 && t->size(0) == frames &&
                       t->size(1) == B && t->size(2) == k)
            << "parents / labels must be int64 of shape (frames, B, k)";
    }
    TVM_FFI_ICHECK(activation == transducer::kJoinerTanh || activation == transducer::kJoinerRelu)
        << "unknown joiner activation " << activation;
    if (walk.has_value()) {
        const TensorView wk = walk.value();
        CHECK_INPUT(wk);
        CHECK_CONTIGUOUS_INPUT(wk);
        TVM_FFI_ICHECK(is_dtype(wk, dl_int32) && wk.numel() == B * k * (2 + frames))
            << "walk must be int32 holding B * k * (2 + frames) elements";
    }

    cudaStream_t stream = get_stream(enc_proj.device());

    // Half precision only: the GEMVs multiply packed half-precision pairs.
    DISPATCH_DLPACK_HALF_DTYPE(enc_proj.dtype(), c_type, [&] {
        for (const TensorView* t : {&w_out_t, &emb, &w_dp_t}) {
            TVM_FFI_ICHECK(t->dtype().code == enc_proj.dtype().code &&
                           t->dtype().bits == enc_proj.dtype().bits)
                << "every floating-point operand must share enc_proj's dtype";
        }
        transducer::StatelessBeamParams<c_type> p;
        p.enc_proj = static_cast<const c_type*>(enc_proj.data_ptr());
        p.enc_stride_b = enc_proj.stride(0);
        p.enc_stride_t = enc_proj.stride(1);
        p.lengths = static_cast<const int64_t*>(lengths.data_ptr());
        p.context_in = static_cast<const int64_t*>(context_in.data_ptr());
        p.scores_in = static_cast<const float*>(scores_in.data_ptr());
        p.w_out_t = static_cast<const c_type*>(w_out_t.data_ptr());
        p.b_out = OptionalDataPtr<const c_type>(b_out);
        p.emb = static_cast<const c_type*>(emb.data_ptr());
        p.conv_w = OptionalDataPtr<const c_type>(conv_w);
        p.w_dp_t = static_cast<const c_type*>(w_dp_t.data_ptr());
        p.b_dp = OptionalDataPtr<const c_type>(b_dp);
        p.context_out = static_cast<int64_t*>(context_out.data_ptr());
        p.scores_out = static_cast<float*>(scores_out.data_ptr());
        p.parents = static_cast<int64_t*>(parents.data_ptr());
        p.labels = static_cast<int64_t*>(labels.data_ptr());
        p.walk = OptionalDataPtr<int32_t>(walk);
        p.B = static_cast<int>(B);
        p.frames = static_cast<int>(frames);
        p.k = static_cast<int>(k);
        p.J = static_cast<int>(J);
        p.D = static_cast<int>(D);
        p.V = static_cast<int>(vocab);
        p.ld_out = static_cast<int>(ld_out);
        p.ld_dp = static_cast<int>(w_dp_t.size(1));
        p.ctx = static_cast<int>(ctx);
        p.group = static_cast<int>(group);
        p.blank = static_cast<int>(blank);
        p.activation = static_cast<int>(activation);
        cudaError_t status =
            transducer::StatelessBeamDecode<c_type>(p, static_cast<int>(cluster), stream);
        TVM_FFI_ICHECK(status == cudaSuccess)
            << "stateless beam decode failed: " << cudaGetErrorString(status);
        return true;
    });
}

// Whether the fused beam search can run a beam of `beam` over these dims on the
// current device: its working set, static and dynamic shared memory, against
// the opt-in limit.  Shared memory is the same for fp16 and bf16.
bool stateless_beam_fits(int64_t beam, int64_t J, int64_t D, int64_t vocab) {
    bool fits = false;
    cudaError_t status = transducer::StatelessBeamFits<__nv_bfloat16>(
        static_cast<int>(beam), static_cast<int>(J), static_cast<int>(D), static_cast<int>(vocab),
        &fits);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "stateless beam fit query failed: " << cudaGetErrorString(status);
    return fits;
}
