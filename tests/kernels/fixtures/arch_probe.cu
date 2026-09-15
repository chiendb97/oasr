// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Test-only probe for include/oasr/common/arch_dispatch.h.
//
// The heterogeneous-node guard cannot be provoked on a homogeneous box, so what
// a test can hold is the *predicate* it decides on, evaluated against the
// `OASR_TARGET_SM` this module was actually compiled with.

#include <oasr/common/arch_dispatch.h>

#include "tvm_ffi_utils.h"

// The SM family this TU was built for, as baked in by the JIT flags.
int64_t target_sm() {
#ifdef OASR_TARGET_SM
    return OASR_TARGET_SM;
#else
    return -1;
#endif
}

// The running device's raw compute capability, per ordinal.
int64_t device_sm(int64_t device_id) {
    return oasr::getDeviceSmVersionFor(static_cast<int>(device_id));
}

// Would `checkDeviceMatchesBuild` accept a device of this raw capability?
// Exposed rather than the void guard so a test can state both answers.
bool arch_matches_build(int64_t sm) {
#ifdef OASR_TARGET_SM
    return oasr::resolveSmVersion(static_cast<int>(sm)) == OASR_TARGET_SM;
#else
    (void)sm;
    return true;
#endif
}

// Does the guard actually refuse?  Returns true when it threw.
bool guard_refuses(int64_t device_id) {
    try {
        oasr::checkDeviceMatchesBuild(static_cast<int>(device_id));
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

// The device's opt-in dynamic shared-memory budget, per ordinal.
int64_t max_smem_optin(int64_t device_id) {
    return oasr::getDeviceMaxSharedMemoryOptin(static_cast<int>(device_id));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(max_smem_optin, max_smem_optin);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(target_sm, target_sm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(device_sm, device_sm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(arch_matches_build, arch_matches_build);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(guard_refuses, guard_refuses);
