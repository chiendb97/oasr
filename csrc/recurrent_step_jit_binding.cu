// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The fused-recurrent-step module's one *static* translation unit.
//
// Everything else in this module is rendered per variant from
// `csrc/templates/recurrent_step_template.cu.jinja`, and a symbol exported
// from a rendered file would be defined once per variant -- a duplicate-symbol
// link error.  `csrc/fmha_jit_binding.cu`, `csrc/gated_mlp_jit_binding.cu` and
// `csrc/gemm_ws_cache.cu` exist for the same reason and say so.
//
// What lives here is the **capability oracle**: for any architecture, any
// dtype and any gate count it answers which tile the kernel would pick and
// what that tile costs -- without launching anything and without a GPU of that
// architecture being present.
//
// That last part is the point.  `oasr/jit/recurrent_step.py` carries a Python
// mirror of the same table and the same arithmetic, because the router has to
// answer "can you serve this shape, and at what occupancy?" before anything is
// built.  A mirror that is never checked against the original is a mirror that
// drifts.  Exporting the C++ answer for `sm` as an *argument* lets one built
// module hold the line for every supported architecture from one box -- which
// is exactly the property that caught sm_86/sm_89 being budgeted with A100's
// shared memory in the attention family
// (`.artifacts/arch_portability_audit.md` § A3), and exactly what a live
// device query would have destroyed.
//
// `recurrent_step_tiles.h` deliberately includes no CuTe, so this TU builds in
// a couple of seconds and the mirror test can afford to compile it.

#include <oasr/recurrent/recurrent_step_tiles.h>

#include "tvm_ffi_utils.h"

/*! \brief How many tiles this lane compiles per cell. */
int64_t recurrent_step_tile_count() {
    return int64_t(oasr::recurrent::kRecurrentStepTileCount);
}

/*! \brief How many rungs the `(hidden, batch)` ladder has. */
int64_t recurrent_step_route_count() {
    return int64_t(oasr::recurrent::kRecurrentStepRouteCount);
}

/*! \brief The 128-bit vector contract, in elements. */
int64_t recurrent_step_alignment() {
    return int64_t(oasr::recurrent::kRecurrentStepAlignment);
}

/*! \brief Floats of padding per staged accumulator row. */
int64_t recurrent_step_acc_pad() {
    return int64_t(oasr::recurrent::kRecurrentStepAccPad);
}

/*! \brief Shared memory this architecture budgets a block, in bytes. */
int64_t recurrent_step_smem_budget(int64_t sm) {
    return int64_t(oasr::smemBudgetForSm(int(sm)));
}

/*! \brief Threads this architecture can hold resident per SM. */
int64_t recurrent_step_max_threads_per_sm(int64_t sm) {
    return int64_t(oasr::maxThreadsPerSmForSm(int(sm)));
}

/*! \brief Describe one tile, as this architecture would run it.
 *
 * \param out int32 `(9,)` on the host:
 *   `{block_m, block_n, block_k, stages, threads, warps_n, valid, smem_bytes,
 *     ctas_per_sm}`
 *
 * `valid` and `ctas_per_sm` are functions of \p sm, \p elem_bits and
 * \p gates; the first six fields are not.
 */
void recurrent_step_tile_info(TensorView out, int64_t index, int64_t sm, int64_t elem_bits,
                              int64_t gates) {
    CHECK_DIM(1, out);
    TVM_FFI_ICHECK_GE(out.size(0), 9) << "recurrent_step_tile_info writes 9 int32 fields";
    TVM_FFI_ICHECK(out.dtype() == oasr::dl_int32) << "recurrent_step_tile_info wants int32";
    TVM_FFI_ICHECK(out.device().device_type == kDLCPU)
        << "recurrent_step_tile_info answers on the host; it launches nothing";
    TVM_FFI_ICHECK(index >= 0 && index < oasr::recurrent::kRecurrentStepTileCount)
        << "tile index " << index << " is outside [0, " << oasr::recurrent::kRecurrentStepTileCount
        << ")";

    int const elem_size = int(elem_bits) / 8;
    oasr::recurrent::RecurrentStepTile const t = oasr::recurrent::kRecurrentStepTiles[index];

    int32_t* p = static_cast<int32_t*>(out.data_ptr());
    p[0] = t.block_m;
    p[1] = t.block_n;
    p[2] = t.block_k;
    p[3] = t.stages;
    p[4] = t.threads;
    p[5] = t.warps_n;
    p[6] = oasr::recurrent::recurrentStepTileValid(t, int(sm), elem_size, int(gates)) ? 1 : 0;
    p[7] = oasr::recurrent::recurrentStepSmemBytes(t, elem_size);
    p[8] = oasr::recurrent::recurrentStepCtasPerSm(t, int(sm), elem_size);
}

/*! \brief Describe one rung of the `(hidden, batch)` ladder.
 *
 * \param out int32 `(3,)` on the host: `{hidden_max, batch_max, tile}`
 *
 * Exported so the Python mirror can be checked rung for rung.  A permutation
 * of the ladder is a silent routing change, and the ladder is also what the
 * CuTeDSL lane uses -- the two lanes running *different* tiles would make an
 * A/B between them measure tile choices rather than kernels.
 */
void recurrent_step_route_info(TensorView out, int64_t index) {
    CHECK_DIM(1, out);
    TVM_FFI_ICHECK_GE(out.size(0), 3) << "recurrent_step_route_info writes 3 int32 fields";
    TVM_FFI_ICHECK(out.dtype() == oasr::dl_int32) << "recurrent_step_route_info wants int32";
    TVM_FFI_ICHECK(out.device().device_type == kDLCPU)
        << "recurrent_step_route_info answers on the host; it launches nothing";
    TVM_FFI_ICHECK(index >= 0 && index < oasr::recurrent::kRecurrentStepRouteCount)
        << "route index " << index << " is outside [0, "
        << oasr::recurrent::kRecurrentStepRouteCount << ")";

    oasr::recurrent::RecurrentStepRoute const r = oasr::recurrent::kRecurrentStepRoutes[index];
    int32_t* p = static_cast<int32_t*>(out.data_ptr());
    p[0] = r.hidden_max;
    p[1] = r.batch_max;
    p[2] = r.tile;
}

/*! \brief `recurrentStepSelectTile` for an architecture given as an argument.
 *
 * **This must never depend on CUDA-graph capture state** (`AGENTS.md` rule
 * 11).  Two tiles sum the K loop in different orders, so a capture-dependent
 * answer would make a replayed graph produce different numbers than eager, and
 * a one-ulp difference has changed a decoded token in this repo before.
 * Everything it reads is an argument, and the result is asserted equal to the
 * Python mirror by `tests/kernels/test_recurrent_cpp.py`.
 *
 * \return an index into `kRecurrentStepTiles`, or -1 if nothing fits.
 */
int64_t recurrent_step_select_tile(int64_t sm, int64_t hidden, int64_t batch, int64_t elem_bits,
                                   int64_t gates) {
    return int64_t(oasr::recurrent::recurrentStepSelectTile(int(sm), int(hidden), int(batch),
                                                            int(elem_bits) / 8, int(gates)));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(recurrent_step_tile_count, recurrent_step_tile_count);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(recurrent_step_route_count, recurrent_step_route_count);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(recurrent_step_alignment, recurrent_step_alignment);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(recurrent_step_acc_pad, recurrent_step_acc_pad);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(recurrent_step_smem_budget, recurrent_step_smem_budget);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(recurrent_step_max_threads_per_sm, recurrent_step_max_threads_per_sm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(recurrent_step_tile_info, recurrent_step_tile_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(recurrent_step_route_info, recurrent_step_route_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(recurrent_step_select_tile, recurrent_step_select_tile);
