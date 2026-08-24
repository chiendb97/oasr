// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Strided batched GEMM (BMM) -- the *general* lane: arbitrary batch strides,
// either memory layout for the B operand, and small / unaligned N and K.
//
// The rendered tile variants in ``csrc/templates/bmm_cutlass_template*.jinja``
// remain the fast lane for contiguous 3-D alignment-8 problems and are selected
// by the shape heuristic in ``oasr/jit/gemm.py``.  This header is what the shapes
// that lane refuses land on: Zipformer's decomposed attention, whose head dims
// (query 32, pos 4, value 12) and relative-position extent (always odd, 2T-1)
// violate the 8-element iterator contract three different ways.
//
// Four things are chosen at run time and turned into a compile-time
// instantiation:
//
//   * **B's layout.**  The public contract is ``A[..., M, K] @ B[..., N, K]^T``.
//     A logical ``[N, K]`` operand can be *contiguous along K* (CUTLASS
//     ColumnMajor -- the historical OASR contract) or *contiguous along N*
//     (CUTLASS RowMajor).  Both are accepted, so a caller holding
//     ``value.permute(...)`` never has to materialize a transpose.  This is
//     worth more than it looks: the alternative is one extra copy *and one
//     extra launch* per attention product, 57 of them per Zipformer forward.
//   * **Operand alignment along K** (A always, B when it is ColumnMajor):
//     8 / 4 / 2 elements, derived from K, the leading dimension, the batch
//     stride *and* the base pointer.  Alignment 1 is not a tensor-op case at
//     all -- ``cp.async`` cannot issue a two-byte copy -- so it routes to SIMT.
//   * **Alignment along N** (the epilogue always, B when it is RowMajor):
//     8 / 4 / 2 / 1 elements, derived from N, ``ldd``, the output batch stride
//     and the output pointer.
//   * **The threadblock tile**, and this is the one that decides whether the
//     lane is worth having.  These problems are 2-70 MFLOP, so they are
//     latency-bound and all that matters is how much of the device the grid
//     covers; see ``WideTile`` below for the measurement that establishes it.
//
// Anything the tensor-op grid cannot express falls to CUTLASS SIMT
// ``GemmBatched``, which has no alignment constraint.  Nothing falls back to
// PyTorch: a shape either has a kernel here or the call fails.

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wstrict-aliasing"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#include <oasr/common/arch_dispatch.h>
#include <oasr/common/utils.h>

namespace oasr {
namespace gemm {

//==============================================================================
// Problem description
//==============================================================================

/// One strided batched GEMM: ``D[b] = A[b] @ B[b]^T``, all extents in elements.
///
/// ``b_contiguous_k`` says how the logical ``[N, K]`` operand sits in memory,
/// which selects CUTLASS's LayoutB (ColumnMajor when K is the contiguous axis,
/// RowMajor when N is).  ``ldb`` is the stride of the *other* axis either way.
struct GeneralBmmParams {
    const void* A = nullptr;
    const void* B = nullptr;
    void* D = nullptr;

    int batch = 0;
    int M = 0;
    int N = 0;
    int K = 0;

    int64_t lda = 0;  ///< A stride between rows (M axis); K axis is contiguous.
    int64_t ldb = 0;  ///< B stride along the non-contiguous axis.
    int64_t ldd = 0;  ///< D stride between rows (M axis); N axis is contiguous.

    int64_t batch_stride_a = 0;
    int64_t batch_stride_b = 0;
    int64_t batch_stride_d = 0;

    bool b_contiguous_k = true;

    cudaStream_t stream = nullptr;
};

namespace detail {

/// The general lane is a CUTLASS 2.x tensor-op kernel for every target.  SM90+
/// keeps its TMA warp-specialised builders for the aligned lane; those schedules
/// require 128-bit operands, which is exactly the contract this file exists to
/// break, so the Sm80 ``mma.sync`` specialisations (forward-compatible on every
/// later architecture) are used instead.
template <int kSmVersion>
struct GeneralBmmArch {
    using Type = std::conditional_t<(kSmVersion >= 80), cutlass::arch::Sm80, cutlass::arch::Sm75>;
    using InstructionShape =
        std::conditional_t<(kSmVersion >= 80), cutlass::gemm::GemmShape<16, 8, 16>,
                           cutlass::gemm::GemmShape<16, 8, 8>>;
};

/// Threadblock/warp tiles for the general lane.
///
/// The ladder is not a tuning detail, it is the whole difference between this
/// lane and a regression.  These problems are *small* -- a Zipformer attention
/// product is 2-70 MFLOP -- so they are latency-bound, and the only thing that
/// matters is how much of the device the grid covers.  Measured on an RTX 5090
/// (170 SMs), a single 128x128 tile put 8-64 CTAs on the device at 0.02-0.19
/// waves/SM, ran at 3.75% of compute and 2.24% of memory peak, and burned 246
/// registers/thread for 8.3% achieved occupancy: 1.74x *slower* than cuBLAS
/// geomean.  Narrowing the tile is worth 1.5x on its own.
struct WideTile {
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
};

struct MidTile {
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
};

/// One warp per threadblock.  CUTLASS's epilogue thread map divides the tile's
/// rows by the warp count in M before it computes its iteration count, so a
/// 32-row tile with two warps in M asserts ("Iteration Count Row must be > 0")
/// -- 32x32 is a single-warp shape or nothing.
struct SmallTile {
    using ThreadblockShape = cutlass::gemm::GemmShape<32, 32, 32>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
};

/// For an N that a square tile would mostly predicate away.  Deep kK because
/// the callers that land here contract over time (K = T, N = 12).
struct ThinNTile {
    using ThreadblockShape = cutlass::gemm::GemmShape<64, 16, 64>;
    using WarpShape = cutlass::gemm::GemmShape<32, 16, 64>;
};

/// N at or below this uses ``ThinNTile``.  It is the tile's own kN, so the
/// threshold and the tile cannot drift apart.
constexpr int kThinNThreshold = ThinNTile::ThreadblockShape::kN;

/// Three stages, not two, and this is a measured choice rather than a default.
/// Stages=2 selects CUTLASS's ``MmaPipelined`` mainloop, which has a cheaper
/// prologue and is 3-5% faster on the shallow-K products (K = 4 and K = 32 run
/// a single K iteration behind the pipeline fill) -- but 1.20x and 1.64x slower
/// on the deep-K ones, where the ``cp.async`` multistage overlap is the whole
/// point.  Keying stages on K as well would double the instantiation grid for
/// ~2% geomean, so K wins and the shallow products keep the deeper pipeline.
constexpr int kGeneralStages = 3;

constexpr int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
}

/// CTAs a tile shape produces for this problem.
inline int64_t tileGridSize(int M, int N, int batch, int tile_m, int tile_n) {
    return static_cast<int64_t>(ceilDiv(M, tile_m)) * ceilDiv(N, tile_n) * batch;
}

/// Largest power of two ≤ *cap* that divides every quantity.
inline int maxAlignment(int cap, int64_t q0, int64_t q1, int64_t q2, int64_t q3) {
    int a = cap;
    while (a > 1 && (q0 % a != 0 || q1 % a != 0 || q2 % a != 0 || q3 % a != 0)) {
        a /= 2;
    }
    return a;
}

/// Pointer alignment in *elements*.  A DLPack tensor is always element-aligned,
/// so a non-zero remainder here means the caller handed us a byte offset that
/// cannot be a tensor of this dtype at all.
template <typename Element>
inline int64_t pointerElementOffset(const void* ptr) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    if (address % sizeof(Element) != 0) {
        return 1;  // forces alignment 1 -> SIMT
    }
    return static_cast<int64_t>(address / sizeof(Element));
}

//==============================================================================
// One tensor-op instantiation
//==============================================================================

template <typename Element, typename LayoutB, int kAlignK, int kAlignN, typename Tile,
          int kSmVersion>
struct GeneralBmmKernel {
    static constexpr bool kBIsColumnMajor = std::is_same_v<LayoutB, cutlass::layout::ColumnMajor>;
    /// B's alignment is along whichever axis is contiguous in memory.
    static constexpr int kAlignB = kBIsColumnMajor ? kAlignK : kAlignN;

    using Gemm = cutlass::gemm::device::GemmBatched<
        Element, cutlass::layout::RowMajor, Element, LayoutB, Element, cutlass::layout::RowMajor,
        float, cutlass::arch::OpClassTensorOp, typename GeneralBmmArch<kSmVersion>::Type,
        typename Tile::ThreadblockShape, typename Tile::WarpShape,
        typename GeneralBmmArch<kSmVersion>::InstructionShape,
        cutlass::epilogue::thread::LinearCombination<Element, kAlignN, float, float>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, kGeneralStages, kAlignK,
        kAlignB>;

    static GemmStatus run(const GeneralBmmParams& p) {
        const auto* A = static_cast<const Element*>(p.A);
        const auto* B = static_cast<const Element*>(p.B);
        auto* D = static_cast<Element*>(p.D);

        typename Gemm::Arguments args({p.M, p.N, p.K}, {A, p.lda}, p.batch_stride_a, {B, p.ldb},
                                      p.batch_stride_b, {D, p.ldd}, p.batch_stride_d, {D, p.ldd},
                                      p.batch_stride_d, {1.0f, 0.0f}, p.batch);

        Gemm gemm_op;
        if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {
            return GemmStatus::NOT_SUPPORTED;
        }
        // GemmBatched needs no workspace at all -- no split-K partials, no
        // semaphores -- so there is nothing to allocate here and the whole lane
        // is CUDA-graph safe by construction rather than by a graph check.
        if (gemm_op.initialize(args, nullptr, p.stream) != cutlass::Status::kSuccess) {
            return GemmStatus::INTERNAL_ERROR;
        }
        return gemm_op(p.stream) == cutlass::Status::kSuccess ? GemmStatus::SUCCESS
                                                              : GemmStatus::CUTLASS_ERROR;
    }
};

/// CUTLASS SIMT ``GemmBatched`` -- the alignment-1 lane.  Every template
/// argument past the layouts is CUTLASS's default configuration for
/// ``OpClassSimt``, which imposes no alignment on any operand.
template <typename Element, typename LayoutB>
struct SimtBmmKernel {
    using Gemm =
        cutlass::gemm::device::GemmBatched<Element, cutlass::layout::RowMajor, Element, LayoutB,
                                           Element, cutlass::layout::RowMajor, float>;

    static GemmStatus run(const GeneralBmmParams& p) {
        const auto* A = static_cast<const Element*>(p.A);
        const auto* B = static_cast<const Element*>(p.B);
        auto* D = static_cast<Element*>(p.D);

        typename Gemm::Arguments args({p.M, p.N, p.K}, {A, p.lda}, p.batch_stride_a, {B, p.ldb},
                                      p.batch_stride_b, {D, p.ldd}, p.batch_stride_d, {D, p.ldd},
                                      p.batch_stride_d, {1.0f, 0.0f}, p.batch);

        Gemm gemm_op;
        if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) {
            return GemmStatus::NOT_SUPPORTED;
        }
        if (gemm_op.initialize(args, nullptr, p.stream) != cutlass::Status::kSuccess) {
            return GemmStatus::INTERNAL_ERROR;
        }
        return gemm_op(p.stream) == cutlass::Status::kSuccess ? GemmStatus::SUCCESS
                                                              : GemmStatus::CUTLASS_ERROR;
    }
};

}  // namespace detail

//==============================================================================
// Alignment selection
//==============================================================================

/// ``(align_k, align_n)`` for *p*: the largest power-of-two operand access the
/// pointers, extents and strides all permit.  Computed here rather than in the
/// launcher so the C++ and the test that pins the routing read the same rule.
template <typename Element>
inline void generalBmmAlignments(const GeneralBmmParams& p, int* align_k, int* align_n) {
    const int64_t ptr_a = detail::pointerElementOffset<Element>(p.A);
    const int64_t ptr_b = detail::pointerElementOffset<Element>(p.B);
    const int64_t ptr_d = detail::pointerElementOffset<Element>(p.D);

    // A is RowMajor: the contiguous axis is K, so K, lda, its batch stride and
    // its base offset all have to divide the access width.
    int ak = detail::maxAlignment(8, p.K, p.lda, p.batch_stride_a, ptr_a);
    int an = detail::maxAlignment(8, p.N, p.ldd, p.batch_stride_d, ptr_d);

    if (p.b_contiguous_k) {
        ak = std::min(ak, detail::maxAlignment(8, p.K, p.ldb, p.batch_stride_b, ptr_b));
    } else {
        an = std::min(an, detail::maxAlignment(8, p.N, p.ldb, p.batch_stride_b, ptr_b));
    }

    *align_k = ak;
    *align_n = an;
}

//==============================================================================
// Dispatch: (align_k, align_n) -> one instantiation
//==============================================================================

namespace detail {

/// The general lane is one architecture family, because ``GeneralBmmArch``
/// already collapses every SM >= 80 onto the Sm80 tensor-op specialisations --
/// instantiating 80 and 120 separately would produce byte-identical kernels.
#if defined(OASR_TARGET_SM)
constexpr int kGeneralBmmSm = OASR_TARGET_SM >= 80 ? 80 : 75;
#else
constexpr int kGeneralBmmSm = 80;
#endif

/// Walk ``kAlignN`` down from 8 and run the first instantiation the output
/// permits.  ``an`` is a power of two, so the first ``an >= kAlignN`` is the
/// exact match; the remaining, narrower cases are the fallback chain for a
/// ``can_implement`` refusal.
template <typename Element, typename LayoutB, typename Tile, int kAlignK, int kAlignN,
          int kMinAlignN>
GemmStatus dispatchAlignN(const GeneralBmmParams& p, int an) {
    if (an >= kAlignN) {
        const GemmStatus status =
            GeneralBmmKernel<Element, LayoutB, kAlignK, kAlignN, Tile, kGeneralBmmSm>::run(p);
        if (status != GemmStatus::NOT_SUPPORTED) {
            return status;
        }
    }
    if constexpr (kAlignN > kMinAlignN) {
        return dispatchAlignN<Element, LayoutB, Tile, kAlignK, kAlignN / 2, kMinAlignN>(p, an);
    } else {
        return GemmStatus::NOT_SUPPORTED;
    }
}

template <typename Element, typename LayoutB, typename Tile, int kAlignK, int kMinAlignN>
GemmStatus dispatchAlignK(const GeneralBmmParams& p, int ak, int an) {
    if (ak >= kAlignK) {
        const GemmStatus status =
            dispatchAlignN<Element, LayoutB, Tile, kAlignK, 8, kMinAlignN>(p, an);
        if (status != GemmStatus::NOT_SUPPORTED) {
            return status;
        }
    }
    if constexpr (kAlignK > 2) {
        return dispatchAlignK<Element, LayoutB, Tile, kAlignK / 2, kMinAlignN>(p, ak, an);
    } else {
        return GemmStatus::NOT_SUPPORTED;
    }
}

/// Which tile a problem gets.
enum class BmmTile { kThinN, kSmall, kMid, kWide };

/// Pick the *largest* tile whose grid still covers the device.
///
/// Large is better per CTA -- more MMA work amortizing the same prologue -- so
/// the rule walks down only as far as it must.  ``kWaveTarget`` is in full
/// waves of the device's SMs; one wave is the point at which adding CTAs stops
/// buying parallelism and starts buying tail.
inline BmmTile selectBmmTile(int M, int N, int batch, int sm_count, bool thin_available) {
    if (thin_available && N <= kThinNThreshold) {
        return BmmTile::kThinN;
    }
    constexpr int kWaveTarget = 1;
    const int64_t target = static_cast<int64_t>(sm_count) * kWaveTarget;
    if (tileGridSize(M, N, batch, WideTile::ThreadblockShape::kM, WideTile::ThreadblockShape::kN) >=
        target) {
        return BmmTile::kWide;
    }
    if (tileGridSize(M, N, batch, MidTile::ThreadblockShape::kM, MidTile::ThreadblockShape::kN) >=
        target) {
        return BmmTile::kMid;
    }
    return BmmTile::kSmall;
}

/// The instantiation grid, stated once.
///
/// ColumnMajor B (the historical ``[batch, N, K]`` contract) needs an epilogue
/// down to a single element -- Zipformer's relative-position product has an
/// always-odd N -- but its own operand alignment rides on K.  RowMajor B ties B
/// *and* the epilogue to N, so an odd N there has no tensor-op instantiation and
/// lands on SIMT; in exchange the thin-N tile is only worth instantiating there,
/// because that is the layout ``N = 12`` arrives in.
template <typename Element, typename LayoutB>
GemmStatus dispatchGeneralBmmLayout(const GeneralBmmParams& p) {
    int align_k = 0;
    int align_n = 0;
    generalBmmAlignments<Element>(p, &align_k, &align_n);

    constexpr bool kBIsColumnMajor = std::is_same_v<LayoutB, cutlass::layout::ColumnMajor>;
    constexpr int kMinAlignN = kBIsColumnMajor ? 1 : 4;

    // The thin-N tile is instantiated for RowMajor B only -- that is the layout
    // ``N = 12`` arrives in, and a ColumnMajor thin-N problem belongs to the
    // tuned lane.  Saying so here keeps an unaligned ColumnMajor thin-N call on
    // the square ladder instead of landing it on a 128-column tile.
    const BmmTile tile = selectBmmTile(p.M, p.N, p.batch, oasr::getDeviceMultiProcessorCount(),
                                       /*thin_available=*/!kBIsColumnMajor);

    GemmStatus status = GemmStatus::NOT_SUPPORTED;
    switch (tile) {
        case BmmTile::kThinN:
            if constexpr (!kBIsColumnMajor) {
                status =
                    dispatchAlignK<Element, LayoutB, ThinNTile, 8, kMinAlignN>(p, align_k, align_n);
            }
            break;
        case BmmTile::kSmall:
            status =
                dispatchAlignK<Element, LayoutB, SmallTile, 8, kMinAlignN>(p, align_k, align_n);
            break;
        case BmmTile::kMid:
            status = dispatchAlignK<Element, LayoutB, MidTile, 8, kMinAlignN>(p, align_k, align_n);
            break;
        case BmmTile::kWide:
            status = dispatchAlignK<Element, LayoutB, WideTile, 8, kMinAlignN>(p, align_k, align_n);
            break;
    }
    // A tile that has no instantiation for these alignments still has to run:
    // walk out to the widest square tile, then to SIMT, which constrains
    // nothing.
    if (status == GemmStatus::NOT_SUPPORTED && tile != BmmTile::kWide) {
        status = dispatchAlignK<Element, LayoutB, WideTile, 8, kMinAlignN>(p, align_k, align_n);
    }
    if (status == GemmStatus::NOT_SUPPORTED) {
        status = SimtBmmKernel<Element, LayoutB>::run(p);
    }
    return status;
}

}  // namespace detail

/// Which element type a call carries -- how the FFI layer names a dtype without
/// dragging DLPack into the kernel header.
enum class BmmElement { kFloat16, kBFloat16 };

/// One entry point per (B layout, dtype) cell of the instantiation grid.
///
/// Each is defined in its own rendered translation unit
/// (``csrc/templates/bmm_general_template.cu.jinja`` →
/// ``oasr/jit/gemm.py::_render_bmm_general_variants``), for one reason only:
/// nvcc parallelizes across translation units and not within one, and this
/// module's cold build time is whatever its largest TU takes.  Measured on a
/// 64-core box, the ``bmm`` module builds in 112 s with the grid in a single TU,
/// 70 s split by layout, and 42 s split this way.
GemmStatus generalBmm_column_major_f16(const GeneralBmmParams& p);
GemmStatus generalBmm_column_major_bf16(const GeneralBmmParams& p);
GemmStatus generalBmm_row_major_f16(const GeneralBmmParams& p);
GemmStatus generalBmm_row_major_bf16(const GeneralBmmParams& p);

/// Run one general strided batched GEMM, selecting layout and dtype cell.
inline GemmStatus generalBmm(const GeneralBmmParams& p, BmmElement dtype) {
    if (p.A == nullptr || p.B == nullptr || p.D == nullptr) {
        return GemmStatus::INVALID_ARGUMENT;
    }
    if (p.batch <= 0 || p.M <= 0 || p.N <= 0 || p.K <= 0) {
        return GemmStatus::INVALID_ARGUMENT;
    }
    const bool bf16 = dtype == BmmElement::kBFloat16;
    if (p.b_contiguous_k) {
        return bf16 ? generalBmm_column_major_bf16(p) : generalBmm_column_major_f16(p);
    }
    return bf16 ? generalBmm_row_major_bf16(p) : generalBmm_row_major_f16(p);
}

}  // namespace gemm
}  // namespace oasr
