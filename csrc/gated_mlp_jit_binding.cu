// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The gated-MLP module's one *static* translation unit.
//
// Everything else in this module is rendered per variant from
// `csrc/templates/gated_mlp_template.cu.jinja`, and a symbol exported from a
// rendered file would be defined once per variant -- a duplicate-symbol link
// error.  `csrc/fmha_jit_binding.cu` and `csrc/gemm_ws_cache.cu` exist for the
// same reason and say so.
//
// What lives here is the **capability oracle**: for any architecture, any SM
// count and any problem shape it answers which tile the kernel would pick and
// what that tile costs -- without launching anything and without a GPU of that
// architecture being present.
//
// That last part is the point.  `oasr/jit/gated_mlp.py` carries a Python
// mirror of the same table and the same arithmetic, because the layer waist
// has to answer "can you serve this shape, and at what occupancy?" before
// anything is built.  A mirror that is never checked against the original is a
// mirror that drifts.  Exporting the C++ answer for `sm` and `num_sms` as
// *arguments* lets one built module hold the line for every supported
// architecture and every machine width from one box -- which is exactly the
// property that caught sm_86/sm_89 being budgeted with A100's shared memory in
// the attention family (`.artifacts/arch_portability_audit.md` § A3), and
// exactly what a live device query would have destroyed.
//
// `gated_mlp_tiles.h` deliberately includes no CuTe, so this TU builds in a
// couple of seconds and the mirror test can afford to compile it.

#include <oasr/mlp/gated_mlp_tiles.h>

#include "tvm_ffi_utils.h"

/*! \brief How many tiles this lane compiles per cell. */
int64_t gated_mlp_tile_count() {
    return int64_t(oasr::mlp::kGatedMlpTileCount);
}

/*! \brief The 128-bit vector contract, in elements. */
int64_t gated_mlp_alignment() {
    return int64_t(oasr::mlp::kGatedMlpAlignment);
}

/*! \brief Shared memory this architecture budgets a block, in bytes. */
int64_t gated_mlp_smem_budget(int64_t sm) {
    return int64_t(oasr::smemBudgetForSm(int(sm)));
}

/*! \brief Threads this architecture can hold resident per SM. */
int64_t gated_mlp_max_threads_per_sm(int64_t sm) {
    return int64_t(oasr::maxThreadsPerSmForSm(int(sm)));
}

/*! \brief Describe one tile, as this architecture would run it.
 *
 * \param out int32 `(9,)` on the host:
 *   `{block_m, block_n, block_k, stages, threads, warps_n, valid, smem_bytes,
 *     ctas_per_sm}`
 *
 * `valid`, `smem_bytes` and `ctas_per_sm` are functions of \p sm and
 * \p elem_bits; the first six fields are not.
 */
void gated_mlp_tile_info(TensorView out, int64_t index, int64_t sm, int64_t elem_bits) {
    CHECK_DIM(1, out);
    TVM_FFI_ICHECK_GE(out.size(0), 9) << "gated_mlp_tile_info writes 9 int32 fields";
    TVM_FFI_ICHECK(out.dtype() == oasr::dl_int32) << "gated_mlp_tile_info wants int32";
    TVM_FFI_ICHECK(out.device().device_type == kDLCPU)
        << "gated_mlp_tile_info answers on the host; it launches nothing";
    TVM_FFI_ICHECK(index >= 0 && index < oasr::mlp::kGatedMlpTileCount)
        << "tile index " << index << " is outside [0, " << oasr::mlp::kGatedMlpTileCount
        << ")";

    int const elem_size = int(elem_bits) / 8;
    oasr::mlp::GatedMlpTile const t = oasr::mlp::kGatedMlpTiles[index];

    int32_t* p = static_cast<int32_t*>(out.data_ptr());
    p[0] = t.block_m;
    p[1] = t.block_n;
    p[2] = t.block_k;
    p[3] = t.stages;
    p[4] = t.threads;
    p[5] = t.warps_n;
    p[6] = oasr::mlp::gatedMlpTileValid(t, int(sm), elem_size) ? 1 : 0;
    p[7] = oasr::mlp::gatedMlpSmemBytes(t, elem_size);
    p[8] = oasr::mlp::gatedMlpCtasPerSm(t, int(sm), elem_size);
}

/*! \brief `gatedMlpSelectTile` for an architecture and a machine given as arguments.
 *
 * **This must never depend on CUDA-graph capture state** (`AGENTS.md` rule
 * 11).  Two tiles sum the K loop in different orders, so a capture-dependent
 * answer would make a replayed graph produce different numbers than eager, and
 * a one-ulp difference has changed a decoded token in this repo before.
 * Everything it reads is an argument, `num_sms` included, and the result is
 * asserted equal to the Python mirror by `tests/kernels/test_gated_mlp_cpp.py`.
 *
 * \return an index into `kGatedMlpTiles`, or -1 if nothing fits.
 */
int64_t gated_mlp_select_tile(int64_t sm, int64_t num_sms, int64_t rows, int64_t n,
                              int64_t elem_bits) {
    return int64_t(oasr::mlp::gatedMlpSelectTile(int(sm), int(num_sms), int(rows), int(n),
                                                 int(elem_bits) / 8));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gated_mlp_tile_count, gated_mlp_tile_count);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gated_mlp_alignment, gated_mlp_alignment);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gated_mlp_smem_budget, gated_mlp_smem_budget);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gated_mlp_max_threads_per_sm, gated_mlp_max_threads_per_sm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gated_mlp_tile_info, gated_mlp_tile_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gated_mlp_select_tile, gated_mlp_select_tile);
