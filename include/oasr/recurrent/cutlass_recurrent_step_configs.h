// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-architecture traits for the fused recurrent step.  This is the
// `cutlass_*_configs.h` third of the three-header CUTLASS pattern (config /
// template / dispatch) that GEMM, BMM, Conv2D, attention and the gated MLP all
// follow -- see `include/oasr/mlp/cutlass_gated_mlp_configs.h`.
//
// One thing lives here: `RecurrentStepArch<SM>` -- which architectures this
// family serves, and its own ceilings.  What an architecture *is* -- its
// shared-memory budget, its warp slots, its MMA and copy atoms and its
// capability bools -- is `oasr::cute_sm80::ArchAmpere`
// (`include/oasr/common/cute_sm80.h`), shared with attention and the gated MLP.
//
// It is `constexpr` over an `sm` **argument**, never over `__CUDA_ARCH__` and
// never over a live device query, for the reasons `oasr/common/arch_facts.h`
// sets out.  The CTA tiles themselves and the ladder that picks one live next
// door in `recurrent_step_tiles.h`, which needs no CuTe.

#pragma once

// `cute_sm80.h` carries the order-sensitive CuTe include block (with its own
// `// clang-format off` guard); nothing here needs an individual CuTe header.
#include <oasr/common/cute_sm80.h>

#include "recurrent_step_tiles.h"

namespace oasr {
namespace recurrent {

// ---------------------------------------------------------------------------
// Per-architecture traits
// ---------------------------------------------------------------------------

/*! \brief What one architecture is, for this kernel family.
 *
 * The instruction selection -- the MMA atom, the ZFILL gmem copy, `ldmatrix`
 * -- and the budgets are `oasr::cute_sm80::ArchAmpere`'s, shared with the
 * attention and gated-MLP families.  What a specialization here adds is the
 * *decision to serve* that architecture, plus this family's own ceilings.
 *
 * `previous_h` is `(M, K)` row-major and `weight_hh` is `(N, K)` -- the
 * gate-interleaved `(out, in)` layout the packer produces -- so both operands
 * of the TN `mma.sync` read through the same non-transposed `ldmatrix`
 * (`SmemCopyAtom`).  The ZFILL `GmemCopyAtom` is what lets a hidden width that
 * is not a whole number of K tiles be predicated rather than refused: a
 * predicated-off copy writes zeros, the identity for the dot product.
 *
 * \warning Never branch on `Tag::kMinComputeCapability >= 90`; see
 *   `cute_sm80::ArchAmpere`.
 */
template <int SmVersion>
struct RecurrentStepArch;

namespace detail {

template <int SmVersion>
struct RecurrentStepArchAmpere : cute_sm80::ArchAmpere<SmVersion> {
    static constexpr int kMaxThreadsPerBlock = 1024;
};

}  // namespace detail

template <>
struct RecurrentStepArch<80> : detail::RecurrentStepArchAmpere<80> {};
template <>
struct RecurrentStepArch<86> : detail::RecurrentStepArchAmpere<86> {};
template <>
struct RecurrentStepArch<89> : detail::RecurrentStepArchAmpere<89> {};
template <>
struct RecurrentStepArch<120> : detail::RecurrentStepArchAmpere<120> {};

// sm_90 and sm_100 deliberately have no specialization yet.  The Ampere-class
// mainloop would *run* there (mma.sync and cp.async are both available), but
// routing to it would be a performance claim with no measurement behind it,
// and the right answer on those parts is a wgmma / tcgen05 mainloop of their
// own -- each of which needs its own epilogue as well, because neither
// partitions the accumulator the way `partition_fragment_C` does here.
// `oasr/jit/recurrent_step.py` declares the omission and counts it rather than
// hiding it (`AGENTS.md` rule 3).
//
// Adding one later is exactly three things, and none of them is in this file's
// consumers:
//
//   1. a `RecurrentStepArch<90>` here, with `kIsWarpSpecialized = true` and
//      `kHasTma = true`;
//   2. a `CollectiveRecurrentStepMainloopSm90` beside the Sm80 one, plus its
//      epilogue if the accumulator partition differs;
//   3. one more arm in the `conditional_t` that
//      `recurrent_step_launch_template.h` describes and `static_assert`s
//      against today.
//
// The kernel shell, the arguments, the tile table, the FFI signature and the
// whole Python side are arch-free and do not move.

}  // namespace recurrent
}  // namespace oasr
