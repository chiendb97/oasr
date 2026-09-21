// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Everything a CTA needs to know about its sequence lengths, read once.
//
// Structurally modelled on FlashAttention's `hopper/seqlen.h`.  The one
// substantive difference is `seqstart_k`, and it is worth stating why it is a
// *predicate* here rather than a pointer offset the way FA expresses left
// padding (`leftpad_k`).
//
// FA shifts the K/V base pointer so padded keys are never addressed at all,
// which is strictly better -- where it applies.  It applies because FA aligns
// the causal diagonal to the **bottom right**: shifting K shifts the diagonal
// with it and the two stay consistent.  OASR is **top-left** aligned, to match
// `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`, which is
// what every parity test in the suite compares against.  A K-only offset would
// slide the diagonal out from under the mask.  So the start stays a predicate,
// and the waste FA avoids by construction is instead avoided by bounding the
// block loop (`fmha_block.h`).
//
// The varlen fold is the other half of this struct's job.  A packed
// `(total_q, H, D)` tensor and a dense `(B, H, T, D)` one differ only in the
// per-batch row offset and the per-batch length; with batch stride 0 and these
// two numbers, one mainloop serves both.  That is what lets this backend drop
// the CuTeDSL side's separate ~600-line varlen kernel.

#pragma once

#include <cutlass/cutlass.h>

namespace oasr {
namespace attention {

/*! \brief Per-CTA sequence geometry, resolved from runtime pointers.
 *
 * Every field is read in the constructor so the mainloop never re-touches
 * global memory for a length.  A null pointer means "not supplied" and folds to
 * the dense default, so `has_seqlen` / `has_seqstart` / `varlen` need not be
 * template parameters -- which keeps four axes out of the JIT variant space for
 * the price of a few registers.
 */
struct SeqlenInfoQK {
    int const offset_q;    //!< first row of this batch in a packed Q, else 0
    int const offset_k;    //!< first row of this batch in a packed K/V, else 0
    int const seqlen_q;
    int const seqlen_k;    //!< exclusive upper bound on valid keys
    int const seqstart_k;  //!< inclusive lower bound on valid keys (left padding)
    int const bias_offset; //!< element offset of this batch's block in a packed bias

    CUTLASS_DEVICE
    SeqlenInfoQK(int const bidb, int const seqlen_q_static, int const seqlen_k_static,
                 int const* const cu_seqlens_q, int const* const cu_seqlens_k,
                 int const* const seqused_k, int const* const seqstarts_k,
                 int const* const bias_offsets)
        : offset_q(cu_seqlens_q == nullptr ? 0 : cu_seqlens_q[bidb]),
          offset_k(cu_seqlens_k == nullptr ? 0 : cu_seqlens_k[bidb]),
          seqlen_q(cu_seqlens_q == nullptr ? seqlen_q_static
                                           : cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb]),
          // `seqused_k` (OASR's `cache_seqlens`) wins over the packed extent:
          // the paged and streaming callers hand over a whole capacity buffer
          // plus a length rather than a slice, which is worth 1.23-1.88x and is
          // the only form a strided capacity buffer can arrive in at all.
          seqlen_k(seqused_k != nullptr
                       ? seqused_k[bidb]
                       : (cu_seqlens_k == nullptr
                              ? seqlen_k_static
                              : cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb])),
          seqstart_k(seqstarts_k == nullptr ? 0 : seqstarts_k[bidb]),
          bias_offset(bias_offsets == nullptr ? 0 : bias_offsets[bidb]) {}
};

}  // namespace attention
}  // namespace oasr
