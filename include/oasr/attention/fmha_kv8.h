// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// FP8 (e4m3) K/V for the Ampere-class mainloop: the gather into an fp8 ring and
// the in-tile dequantization to the served dtype.
//
// ---------------------------------------------------------------------------
// The design, and why it is shaped this way
// ---------------------------------------------------------------------------
//
// The cache stores e4m3 bytes plus one fp32 scale per `(token, kv-head)`
// (`include/oasr/kv_quant/kv_quant.cuh`, power-of-two scales).  `mma.sync` has
// no fp16 x fp8 shape, and quantizing Q to fp8 to use the e4m3 MMA would throw
// away the precision the cache was designed to keep -- so the K and V tiles are
// dequantized to the served dtype **in shared memory**, one tile at a time:
//
//   gmem (fp8) --cp.async--> fp8 ring stage --LDS.128, x scale, cvt--> served
//   tile --ldmatrix--> mma
//
// * The cp.async ring is unchanged in depth and in fence accounting; it simply
//   carries half the bytes (and the scales ride in the same commit groups as
//   their rows, so one `cp_async_wait` covers both).
// * Each thread converts **exactly the 16-byte chunks it issued**, so the fp8
//   bytes it reads are its own completed copies; only the scales, written by the
//   one thread per row that loads them, need the barrier the ring already takes.
// * One served-dtype tile serves K and then V: the QK gemm is done with it
//   before the barrier the PV half of the step already takes, and the PV gemm is
//   done with it before the next step's first barrier.  So the extra shared
//   memory is one tile, not a second ring, and the extra cost is one
//   `__syncthreads` per conversion.
// * The scale is applied in fp32 and rounded once to the served dtype.  With
//   the cache's power-of-two scales that rounding is exact, so this kernel is
//   **bit-identical** to the served-dtype kernel run on the dequantized cache --
//   the oracle `tests/kernels/test_fmha_kv8.py` holds it to.
//
// Everything after the conversion -- masks, online softmax, P @ V, split-KV,
// the epilogue -- is the served-dtype mainloop's own code.

#pragma once

#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cutlass_fmha_configs.h"
#include "fmha_params.h"
#include "fmha_seqlen.h"
#include "fmha_utils.h"

namespace oasr {
namespace attention {

using namespace cute;

//! The fp8 storage type of a quantized K/V cache.
using ElementKV8 = cutlass::float_e4m3_t;

/*! \brief Geometry of one fp8 `(kBlockN, kHeadDim)` K or V tile.
 *
 * Plain row-major bytes in shared memory, not swizzled: the only accesses are
 * 16-byte cp.async writes and 16-byte reads of the same chunk by the same
 * thread, laid out so that consecutive threads touch consecutive chunks of a
 * row -- conflict-free without a swizzle.
 */
template <int kBlockN_, int kHeadDim_, int kStages_, int kNumThreads_>
struct Kv8Tile {
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kStages = kStages_;
    static constexpr int kNumThreads = kNumThreads_;
    static constexpr int kVec = kFmhaKv8ElemsPerLoad;
    static_assert(fmhaKv8Covers(kBlockN, kHeadDim, kNumThreads),
                  "the fp8 gather does not tile this (block_n, head_dim) exactly; "
                  "fmhaResolveTileKv8 must refuse it before it gets here");
    static constexpr int kThreadsPerRow = kHeadDim / kVec;
    static constexpr int kRowsPerPass = kNumThreads / kThreadsPerRow;
    static constexpr int kPasses = kBlockN / kRowsPerPass;
    //! Bytes of one ring stage.
    static constexpr int kStageBytes = kBlockN * kHeadDim;
};

//! 16-byte cp.async that zero-fills instead of reading when `pred` is false.
CUTLASS_DEVICE void kv8_cp_async_16(void* smem, void const* gmem, bool pred) {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    uint32_t const dst = cute::cast_smem_ptr_to_uint(smem);
    int const src_size = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem),
                 "r"(src_size));
#else
    CUTE_INVALID_CONTROL_PATH("the fp8 KV gather needs cp.async (sm_80+)");
#endif
}

//! 4-byte cp.async (one scale) that zero-fills instead of reading when `pred` is false.
CUTLASS_DEVICE void kv8_cp_async_4(void* smem, void const* gmem, bool pred) {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    uint32_t const dst = cute::cast_smem_ptr_to_uint(smem);
    int const src_size = pred ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(gmem),
                 "r"(src_size));
#else
    CUTE_INVALID_CONTROL_PATH("the fp8 KV gather needs cp.async (sm_80+)");
#endif
}

/*! \brief Gather one fp8 K or V tile and its row scales into a ring stage.
 *
 * Dense or paged by `PagedKV`; rows bounded by `seqlen_k` on the first tile only
 * (`Seqlenk_mask`), as in the served-dtype loader.  **Everything skipped arrives
 * as zeros** -- rows past the length or past the block table, and the head-dim
 * residue -- data and scale alike, so a skipped row dequantizes to exact zeros
 * (the V half of that is required: a stale NaN reaches the output through
 * `P @ V`).  The page id is read only for a real row, exactly as
 * `paged_gather_tile` does; a predicated-off copy's source is the tensor base,
 * which is never dereferenced.
 *
 * Issues cp.asyncs only; the caller fences them into the stage's commit group.
 */
template <class Tile, bool PagedKV, bool Seqlenk_mask, class Params>
CUTLASS_DEVICE void kv8_load_tile(Params const& params, bool const is_v, ElementKV8* smem_stage,
                                  float* smem_scale_stage, int const thread_idx, int const bidb,
                                  int const bidh_kv, int const n_block, int const seqlen_k) {
    ElementKV8 const* const base = is_v ? params.ptr_v_fp8 : params.ptr_k_fp8;
    float const* const scale_base = is_v ? params.ptr_v_scale : params.ptr_k_scale;
    auto const& stride = is_v ? params.stride_v : params.stride_k;
    auto const& sstride = is_v ? params.stride_v_scale : params.stride_k_scale;
    int const head_dim = int(get<1>(params.shape_k));
    int const max_pages = PagedKV ? int(get<1>(params.shape_pagetable)) : 0;

    int const row0 = thread_idx / Tile::kThreadsPerRow;
    int const chunk = thread_idx % Tile::kThreadsPerRow;
    int const col = chunk * Tile::kVec;
    bool const col_ok = col < head_dim;

    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < Tile::kPasses; ++p) {
        int const row = row0 + p * Tile::kRowsPerPass;
        int const abs_row = n_block * Tile::kBlockN + row;
        bool row_ok = !Seqlenk_mask || abs_row < seqlen_k;
        int64_t outer = bidb;  // the batch row, or the physical page
        int inner = abs_row;   // the token, or its offset within that page
        if constexpr (PagedKV) {
            int page_idx = 0;
            params.page_size_divmod(page_idx, inner, abs_row);
            row_ok = row_ok && page_idx < max_pages;
            outer = row_ok ? params.ptr_pagetable[int64_t(bidb) * get<0>(params.stride_pagetable) +
                                                  page_idx]
                           : 0;
        }
        int64_t const off = outer * get<3>(stride) + int64_t(inner) * get<0>(stride) +
                            int64_t(bidh_kv) * get<2>(stride) + col;
        bool const ok = row_ok && col_ok;
        kv8_cp_async_16(smem_stage + row * Tile::kHeadDim + col, ok ? base + off : base, ok);
        if (chunk == 0) {
            int64_t const soff = outer * get<2>(sstride) + int64_t(inner) * get<0>(sstride) +
                                 int64_t(bidh_kv) * get<1>(sstride);
            kv8_cp_async_4(smem_scale_stage + row, row_ok ? scale_base + soff : scale_base,
                           row_ok);
        }
    }
}

/*! \brief Dequantize one fp8 ring stage into the served-dtype tile.
 *
 * Each thread converts the chunks it gathered itself (the same `(row, chunk)`
 * walk as `kv8_load_tile`), so its own `cp_async_wait` makes the bytes visible;
 * the scales come from other threads and need the barrier the caller has
 * already taken.  The caller takes another before anything reads `sTile`.
 *
 * `q * scale` is formed in fp32 and rounded once to `Element`; it is exact for
 * the cache's power-of-two scales, which is what makes this kernel equal to the
 * served-dtype one on the dequantized cache.
 */
template <class Tile, class Element, class TensorTile>
CUTLASS_DEVICE void kv8_dequant_tile(ElementKV8 const* smem_stage, float const* smem_scale_stage,
                                     TensorTile&& sTile, int const thread_idx) {
    using Vec8 = cutlass::Array<ElementKV8, Tile::kVec>;
    using VecF = cutlass::Array<float, Tile::kVec>;
    using VecE = cutlass::Array<Element, Tile::kVec>;
    static_assert(sizeof(VecE) == 2 * sizeof(cute::uint128_t), "two 128-bit stores per chunk");
    cutlass::NumericArrayConverter<float, ElementKV8, Tile::kVec> to_float;
    cutlass::NumericArrayConverter<Element, float, Tile::kVec> to_elem;

    int const row0 = thread_idx / Tile::kThreadsPerRow;
    int const col = (thread_idx % Tile::kThreadsPerRow) * Tile::kVec;
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < Tile::kPasses; ++p) {
        int const row = row0 + p * Tile::kRowsPerPass;
        cute::uint128_t const raw =
            *reinterpret_cast<cute::uint128_t const*>(smem_stage + row * Tile::kHeadDim + col);
        float const scale = smem_scale_stage[row];
        VecF f = to_float(*reinterpret_cast<Vec8 const*>(&raw));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Tile::kVec; ++i) {
            f[i] = __fmul_rn(f[i], scale);
        }
        VecE const e = to_elem(f);
        // The swizzle permutes 16-byte chunks, never the 8 elements inside one,
        // so each half of the chunk is one aligned 128-bit store.
        cute::uint128_t const* src = reinterpret_cast<cute::uint128_t const*>(&e);
        *reinterpret_cast<cute::uint128_t*>(raw_pointer_cast(&sTile(row, col))) = src[0];
        *reinterpret_cast<cute::uint128_t*>(raw_pointer_cast(&sTile(row, col + 8))) = src[1];
    }
}

}  // namespace attention
}  // namespace oasr
