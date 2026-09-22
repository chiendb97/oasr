// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Hardware facts about a compute capability, as `constexpr` functions of an
// `sm` **argument**.
//
// These are properties of the *part*, not of any kernel family, so they live
// here rather than being restated per family.  Two families already need them
// -- attention (`include/oasr/attention/cutlass_fmha_configs.h`) and the gated
// MLP (`include/oasr/mlp/cutlass_gated_mlp_configs.h`) -- and a third copy is
// how a table like this drifts.
//
// Taking `sm` as an argument rather than reading `__CUDA_ARCH__` or querying
// the device is deliberate and load-bearing: it is what lets one built module
// answer the capability question for every architecture, so a single box can
// hold the line for all of them in CI.  A live
// `cudaDevAttrMaxSharedMemoryPerBlockOptin` read would collapse every row to
// whatever card happens to be present, which is precisely the coverage that
// caught sm_86/sm_89 being budgeted with A100's 163 KB
// (`.artifacts/arch_portability_audit.md` § A3).
//
// The device query is not absent, it is *demoted*: a launcher asserts its
// resolved shared-memory size against `oasr::getDeviceMaxSharedMemoryOptin()`
// before launching, so a part that grants less than this table claims fails
// loudly with both numbers instead of as a driver error from inside
// `cudaFuncSetAttribute`.

#pragma once

namespace oasr {

/*! \brief Shared memory the driver keeps for itself, in bytes.
 *
 * Budgeting against the architectural maximum is what let the CuTeDSL
 * recurrent step clear `can_implement` at `num_stages=5` and then die at
 * *launch* with an empty error.  Same constant the CuTeDSL backends use
 * (`oasr/kernels/cute/mlp/gated.py::_DRIVER_SMEM_RESERVE`), so every backend
 * budgets alike.
 */
inline constexpr int kDriverSmemReserve = 1024;

/*! \brief Opt-in shared memory an architecture offers a single block, in bytes.
 *
 * The architectural maxima.  They match what CuTeDSL's
 * `get_smem_capacity_in_bytes("sm_NN")` returns, which is what the CuTeDSL
 * backends budget against, so the C++ and CuTeDSL lanes approve the same set
 * of shapes.
 *
 * Returns 0 for an architecture no table here knows, which every caller turns
 * into "nothing fits" rather than into a wrong answer.
 */
constexpr int smemCapacityForSm(int sm) {
    return sm == 80    ? 166912   // A100, A30
           : sm == 86  ? 101376   // A10G, A40, RTX 3090
           : sm == 89  ? 101376   // L4, L40S, RTX 4090
           : sm == 90  ? 232448   // H100, H200
           : sm == 100 ? 232448   // B200
           : sm == 120 ? 101376   // RTX 5090, consumer Blackwell
                       : 0;
}

/*! \brief Shared memory a launch on \p sm will actually be granted, in bytes. */
constexpr int smemBudgetForSm(int sm) {
    return smemCapacityForSm(sm) == 0 ? 0 : smemCapacityForSm(sm) - kDriverSmemReserve;
}

/*! \brief Threads one SM can hold resident, across all its blocks.
 *
 * The other half of an occupancy estimate: shared memory says how many blocks
 * *fit*, this says how many the warp slots allow.  Returns 0 for an unknown
 * architecture, which a caller must treat the same way it treats a zero
 * shared-memory budget.
 */
constexpr int maxThreadsPerSmForSm(int sm) {
    return sm == 80    ? 2048   // A100
           : sm == 86  ? 1536   // consumer Ampere
           : sm == 89  ? 1536   // Ada
           : sm == 90  ? 2048   // Hopper
           : sm == 100 ? 2048   // Blackwell datacenter
           : sm == 120 ? 1536   // consumer Blackwell
                       : 0;
}

}  // namespace oasr
