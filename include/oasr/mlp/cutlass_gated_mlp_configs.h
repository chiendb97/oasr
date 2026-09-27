// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-architecture traits for the fused gated MLP.  This is the
// `cutlass_*_configs.h` third of the three-header CUTLASS pattern (config /
// template / dispatch) that GEMM, BMM, Conv2D, attention and the recurrent step
// all follow -- see `include/oasr/attention/cutlass_fmha_configs.h`.
//
// One thing lives here: `GatedMlpArch<SM>` -- which architectures this family
// serves, and its own ceilings.  What an architecture *is* -- its
// shared-memory budget, its warp slots, its MMA and copy atoms and its
// capability bools -- is `oasr::cute_sm80::ArchAmpere`
// (`include/oasr/common/cute_sm80.h`), shared with attention and the recurrent
// step.
//
// It is `constexpr` over an `sm` **argument**, never over `__CUDA_ARCH__` and
// never over a live device query, for the reasons `oasr/common/arch_facts.h`
// sets out.  The CTA tiles themselves and the arithmetic that picks one live
// next door in `gated_mlp_tiles.h`, which needs no CuTe.

#pragma once

// `cute_sm80.h` carries the order-sensitive CuTe include block (with its own
// `// clang-format off` guard); nothing here needs an individual CuTe header.
#include <oasr/common/cute_sm80.h>

#include "gated_mlp_tiles.h"

namespace oasr {
namespace mlp {

// ---------------------------------------------------------------------------
// Per-architecture traits
// ---------------------------------------------------------------------------

/*! \brief What one architecture is, for this kernel family.
 *
 * The instruction selection -- the MMA atom, the ZFILL gmem copy, `ldmatrix`
 * -- and the budgets are `oasr::cute_sm80::ArchAmpere`'s, shared with the
 * attention and recurrent-step families.  What a specialization here adds is
 * the *decision to serve* that architecture, plus this family's own ceilings.
 *
 * X and both weights are K-major -- `nn.Linear`'s own `(out, in)` layout for
 * the weights -- so the A and B operands of the TN `mma.sync` read through
 * the same non-transposed `ldmatrix` (`SmemCopyAtom`).  The ZFILL
 * `GmemCopyAtom` is what lets the K residue be predicated instead of refusing
 * every `K` that is not a whole number of K tiles -- the contract the CuTeDSL
 * lane has to impose.
 *
 * \warning Never branch on `Tag::kMinComputeCapability >= 90`; see
 *   `cute_sm80::ArchAmpere`.
 */
template <int SmVersion>
struct GatedMlpArch;

namespace detail {

template <int SmVersion>
struct GatedMlpArchAmpere : cute_sm80::ArchAmpere<SmVersion> {
    static constexpr int kMaxThreadsPerBlock = 1024;
};

}  // namespace detail

template <>
struct GatedMlpArch<80> : detail::GatedMlpArchAmpere<80> {};
template <>
struct GatedMlpArch<86> : detail::GatedMlpArchAmpere<86> {};
template <>
struct GatedMlpArch<89> : detail::GatedMlpArchAmpere<89> {};
template <>
struct GatedMlpArch<120> : detail::GatedMlpArchAmpere<120> {};

// sm_90 and sm_100 deliberately have no specialization yet.  The Ampere-class
// mainloop would *run* there (mma.sync and cp.async are both available), but
// routing to it would be a performance claim with no measurement behind it,
// and the right answer on those parts is a wgmma/TMA mainloop of their own.
// `oasr/jit/gated_mlp.py` declares the omission and counts it rather than
// hiding it.  Adding one later is: a `GatedMlpArch<90>` here, a collective,
// and one `conditional_t` arm in `gated_mlp_launch_template.h`.  The kernel
// shell, the epilogue, the arguments, the FFI signature and the whole Python
// side are arch-free and do not move.

}  // namespace mlp
}  // namespace oasr
