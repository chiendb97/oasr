// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Tile configurations for the native (CUTLASS 3.x) scaled GEMM on Hopper (sm90)
// and Blackwell data-center (sm100): TMA loads into a multistage shared-memory
// ring, a warp-specialized persistent kernel, and the architecture's own
// tensor-core instruction -- wgmma on sm90, tcgen05 (UMMA, accumulating in
// tensor memory) on sm100.  ``cutlass_scaled_mm_configs.h`` is the CUTLASS 2.x
// ``mma.sync`` counterpart these replace on those architectures.
//
// Overlapping the mainloop with the epilogue:
//
// * sm90 ``kPingpong`` -- two consumer warpgroups own alternate output tiles:
//   while one runs its tile's wgmma mainloop, the other runs the previous
//   tile's dequant / activation / store epilogue, and an ordered barrier hands
//   the tensor cores from one to the other.  The epilogue is hidden behind
//   tensor-core work, which matters here: at K = 256 (every Conformer width) a
//   tile is two K iterations, so its epilogue is a large share of its life.
// * sm90 ``kCooperative`` -- both consumer warpgroups split one larger tile
//   (BM >= 128); the producer warp keeps prefetching the next tile while they
//   run the epilogue, but the tensor cores idle through it.  For large M, where
//   the bigger tile's operand reuse is worth more.
// * sm100 ``k1Sm`` / ``k2Sm`` -- tcgen05 accumulates in tensor memory, not
//   registers, and the kernel double-buffers the accumulator there: the MMA warp
//   starts tile i+1 while the epilogue warps drain tile i.  Blackwell's form of
//   ping-pong, with no register cost.  ``k2Sm`` pairs two SMs on one MMA
//   (BM is the pair's extent: 128 or 256 rows), halving each SM's B traffic.
//
// Numerics (shared with the 2.x lane, ``scaled_mm_epilogue.h``):
//
// * int8 accumulates exactly in int32 (wgmma s8 / tcgen05 kind::i8), so every
//   tile and both kernel generations compute the same accumulator.
// * fp8 e4m3 accumulates in fp32.  On sm90 the mainloop is CUTLASS's
//   *promoting* fp8 mainloop: wgmma accumulates ``mma_promotion_interval`` (4)
//   MMAs -- 128 K elements -- in the tensor core's accumulator, then the kernel
//   adds that partial into an fp32 register accumulator.  Hopper's fp8 wgmma
//   accumulator keeps fewer bits than fp32 (DeepSeek-V3 reports about 14 on
//   H800), so the "fast accumulation" alternative trades accuracy over a deep K
//   for speed; it is not used.  The promotion boundary is a count of MMAs, so
//   with every tile's BK dividing 128 each output is summed in the same order
//   by every tile: the lane stays batch-invariant (``BK`` is pinned to 128
//   below).  On sm100, tcgen05 accumulates kind::f8f6f4 into fp32 tensor
//   memory in K order.  Neither splits K.

#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/float8.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>

#include <cute/tensor.hpp>

namespace oasr {
namespace scaled_mm {

// The kernel schedule.  sm90: ping-pong or cooperative warpgroups; sm100: the
// 1-SM or the 2-SM (CTA pair) tcgen05 MMA.
enum class NativeSchedule { kPingpong = 0, kCooperative = 1, k1Sm = 2, k2Sm = 3 };

template <bool kFp8>
struct NativeOperands {
    using ElementAB = int8_t;
    using ElementAccumulator = int32_t;
};

template <>
struct NativeOperands<true> {
    using ElementAB = cutlass::float_e4m3_t;
    using ElementAccumulator = float;
};

template <int kSm, NativeSchedule kSchedule>
struct NativeScheduleTypes;

template <>
struct NativeScheduleTypes<90, NativeSchedule::kPingpong> {
    using ArchTag = cutlass::arch::Sm90;
    // The ping-pong kernel pairs with the plain warp-specialized epilogue, as in
    // CUTLASS's own examples; there is no dedicated ping-pong epilogue type.
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
};

template <>
struct NativeScheduleTypes<90, NativeSchedule::kCooperative> {
    using ArchTag = cutlass::arch::Sm90;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
};

template <>
struct NativeScheduleTypes<100, NativeSchedule::k1Sm> {
    using ArchTag = cutlass::arch::Sm100;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
};

template <>
struct NativeScheduleTypes<100, NativeSchedule::k2Sm> {
    using ArchTag = cutlass::arch::Sm100;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
    using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
};

// One native tile configuration.
//
//   kFp8     -- e4m3 operands with an fp32 accumulator, else int8 with int32
//   BM/BN/BK -- the MMA tile (for ``k2Sm`` the CTA pair's combined extent)
//   CM/CN    -- the cluster shape (CK is 1); TMA multicasts B across CM and A
//               across CN
//   kSm      -- 90 or 100
//
// The constraints are CUTLASS's own (the builders assert them) restated here so
// a bad config fails with a readable message, and the Python config space
// (``oasr/jit/scaled_mm.py::_native_buildable``) mirrors them so no unbuildable
// variant is ever rendered -- one would fail the whole JIT module.
template <bool kFp8, int BM, int BN, int BK, int CM, int CN, NativeSchedule kSchedule, int kSm>
struct CutlassScaledMmConfigNative {
    using Operands = NativeOperands<kFp8>;
    using ElementAB = typename Operands::ElementAB;
    using ElementAccumulator = typename Operands::ElementAccumulator;
    using Schedules = NativeScheduleTypes<kSm, kSchedule>;
    using ArchTag = typename Schedules::ArchTag;
    using EpilogueSchedule = typename Schedules::EpilogueSchedule;
    using MainloopSchedule = typename Schedules::MainloopSchedule;

    static constexpr bool kIsFp8 = kFp8;
    static constexpr bool kIs2Sm = kSchedule == NativeSchedule::k2Sm;

    using TileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<BK>>;
    using ClusterShape = cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::_1>;
    // The tile one CTA stores -- what the epilogue's broadcast nodes are sized by.
    using CtaTileShape = cute::Shape<cute::Int<(kIs2Sm ? BM / 2 : BM)>, cute::Int<BN>, cute::Int<BK>>;

    // One K tile is one 128-byte swizzle row of 8-bit operands, and (fp8 on
    // sm90) divides the 128-element promotion interval: the same summation
    // order for every tile.
    static_assert(BK == 128, "8-bit native tiles take BK = 128");
    static_assert(BN % 16 == 0 && BN <= 256, "BN must be a multiple of 16, at most 256");
    static_assert(kSm != 90 || kSchedule == NativeSchedule::kPingpong ||
                      kSchedule == NativeSchedule::kCooperative,
                  "sm90 schedules are ping-pong or cooperative");
    static_assert(kSm != 90 || kSchedule != NativeSchedule::kPingpong || BM == 64 || BM == 128,
                  "a ping-pong warpgroup owns a 64- or 128-row tile");
    static_assert(kSm != 90 || kSchedule != NativeSchedule::kCooperative || BM % 128 == 0,
                  "a cooperative tile splits at least 128 rows across two warpgroups");
    static_assert(kSm != 100 || kSchedule == NativeSchedule::k1Sm ||
                      kSchedule == NativeSchedule::k2Sm,
                  "sm100 schedules are 1-SM or 2-SM");
    static_assert(kSm != 100 || kSchedule != NativeSchedule::k1Sm || BM == 64 || BM == 128,
                  "a 1-SM tcgen05 MMA is 64 or 128 rows");
    static_assert(kSm != 100 || kSchedule != NativeSchedule::k2Sm ||
                      ((BM == 128 || BM == 256) && CM % 2 == 0),
                  "a 2-SM tcgen05 MMA is 128 or 256 rows over an even cluster M");
};

}  // namespace scaled_mm
}  // namespace oasr
