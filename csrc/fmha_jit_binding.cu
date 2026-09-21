// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The FMHA module's one *static* translation unit.
//
// Everything else in this module is rendered per variant from
// `csrc/templates/fmha_template.cu.jinja`, and a symbol exported from a
// rendered file would be defined once per variant -- a duplicate-symbol link
// error.  `csrc/gemm_ws_cache.cu` exists for the same reason and says so.
//
// What lives here is the **capability oracle**: `fmha_resolved_config` answers,
// for any architecture and any head dim, which tile the kernel would pick and
// how much shared memory it would need -- without launching anything and
// without a GPU of that architecture being present.
//
// That last part is the point.  `oasr/jit/fmha.py` carries a Python mirror of
// the same resolver so the layer waist can ask "can you serve this shape?"
// before building anything, and a mirror that is never checked against the
// original is a mirror that drifts.  Exporting the C++ answer for an `sm`
// *argument* lets one built module hold the line for every supported
// architecture from one box -- which is exactly the property that caught
// sm_86/sm_89 being budgeted with A100's shared memory
// (`.artifacts/arch_portability_audit.md` § A3), and exactly what a live device
// query would have destroyed.

#include <oasr/attention/cutlass_fmha_configs.h>

#include "tvm_ffi_utils.h"

namespace {

using oasr::attention::FmhaTile;

/*! \brief `fmhaResolveTile` for an `sm` known only at run time.
 *
 * The resolver is `constexpr` over `sm`, so this is the one place the module is
 * not specialised to `OASR_TARGET_SM`.  It is host code and launches nothing.
 */
FmhaTile resolve_for(int sm, int head_dim_padded, int head_dim_v, int elem_size) {
    switch (sm) {
        case 80:
            return oasr::attention::fmhaResolveTile(80, head_dim_padded, head_dim_v, elem_size);
        case 86:
            return oasr::attention::fmhaResolveTile(86, head_dim_padded, head_dim_v, elem_size);
        case 89:
            return oasr::attention::fmhaResolveTile(89, head_dim_padded, head_dim_v, elem_size);
        case 120:
            return oasr::attention::fmhaResolveTile(120, head_dim_padded, head_dim_v,
                                                    elem_size);
        default:
            return FmhaTile{false, 0, 0, 0, 0, false, 0};
    }
}

}  // namespace

/*! \brief Write the resolved tile for `(sm, elem_bits, head_dim)` into a CPU tensor.
 *
 * \param out int32 `(7,)` on the host:
 *   `{valid, block_m, block_n, num_warps, num_stages, q_in_regs, smem_bytes}`
 *
 * `head_dim` is the **raw** value; it is padded here exactly as the kernel
 * pads it, so a caller cannot get the padding wrong either.
 */
void fmha_resolved_config(TensorView out, int64_t sm, int64_t elem_bits, int64_t head_dim) {
    CHECK_DIM(1, out);
    TVM_FFI_ICHECK_GE(out.size(0), 7) << "fmha_resolved_config writes 7 int32 fields";
    TVM_FFI_ICHECK(out.dtype() == oasr::dl_int32) << "fmha_resolved_config wants int32";
    TVM_FFI_ICHECK(out.device().device_type == kDLCPU)
        << "fmha_resolved_config answers on the host; it launches nothing";

    int const elem_size = int(elem_bits) / 8;
    int const d = oasr::attention::fmhaPaddedHeadDim(int(head_dim));
    FmhaTile const t = resolve_for(int(sm), d, d, elem_size);

    int32_t* p = static_cast<int32_t*>(out.data_ptr());
    p[0] = t.valid ? 1 : 0;
    p[1] = t.block_m;
    p[2] = t.block_n;
    p[3] = t.num_warps;
    p[4] = t.num_stages;
    p[5] = t.q_in_regs ? 1 : 0;
    p[6] = t.smem_bytes;
}

/*! \brief Shared memory this architecture budgets a block, in bytes. */
int64_t fmha_smem_budget(int64_t sm) {
    return int64_t(oasr::attention::fmhaSmemBudget(int(sm)));
}

/*! \brief `fmhaNumSplits` for a `num_sms` given as an argument.
 *
 * Exported for the same reason `fmha_resolved_config` is: the Python side
 * carries a mirror so it can size the split workspace before calling, and a
 * mirror nobody checks is a mirror that drifts.  Taking `num_sms` as an
 * argument rather than querying the device is what lets one box hold the line
 * for an A100's 108, an H100's 132 and this part's 170 -- and it is also the
 * property that makes the answer provably independent of anything but its
 * arguments, which is what `AGENTS.md` rule 11 requires of it.
 */
int64_t fmha_num_splits(int64_t cta_count, int64_t n_blocks, int64_t num_sms) {
    return int64_t(
        oasr::attention::fmhaNumSplits(int(cta_count), int(n_blocks), int(num_sms)));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fmha_resolved_config, fmha_resolved_config);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fmha_smem_budget, fmha_smem_budget);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fmha_num_splits, fmha_num_splits);
