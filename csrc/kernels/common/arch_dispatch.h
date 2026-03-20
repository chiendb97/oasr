// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Runtime architecture dispatch utilities

#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace oasr {
namespace kernels {

/**
 * @brief Query the compute capability of the current CUDA device.
 *
 * Returns SM version as (major * 10 + minor), e.g. 80 for SM8.0.
 * Result is cached per device ordinal.
 */
inline int getDeviceSmVersion() {
    int device = 0;
    cudaGetDevice(&device);

    // Thread-safe: cudaDeviceGetAttribute is reentrant and the result is
    // idempotent for a given device, so racing writes produce the same value.
    static int cached[16] = {};
    if (cached[device] != 0) {
        return cached[device];
    }

    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    cached[device] = major * 10 + minor;
    return cached[device];
}

/**
 * @brief Map a runtime SM version to the nearest compiled architecture family.
 *
 * Each GPU maps to the highest ArchTraits specialization that does not exceed it:
 *   SM120+       → 120 (next-gen)
 *   SM103-SM119  → 103 (Blackwell consumer, e.g. RTX 5090)
 *   SM100-SM102  → 100 (Blackwell datacenter, e.g. B200)
 *   SM90-SM99    → 90  (Hopper, e.g. H100)
 *   SM89         → 89  (Ada Lovelace, e.g. RTX 4090)
 *   SM86-SM88    → 86  (Ampere GA102, e.g. RTX 3090, A40)
 *   SM80-SM85    → 80  (Ampere GA100, e.g. A100)
 *   SM75-SM79    → 75  (Turing)
 *   SM70-SM74    → 70  (Volta)
 */
inline int resolveSmVersion(int sm) {
    if (sm >= 120) return 120;
    if (sm >= 103) return 103;
    if (sm >= 100) return 100;
    if (sm >= 90) return 90;
    if (sm >= 89) return 89;
    if (sm >= 86) return 86;
    if (sm >= 80) return 80;
    if (sm >= 75) return 75;
    if (sm >= 70) return 70;
    throw std::runtime_error("Unsupported GPU architecture: SM" + std::to_string(sm));
}

}  // namespace kernels
}  // namespace oasr

/**
 * @brief Runtime dispatch macro — converts runtime SM version to a compile-time constant.
 *
 * Usage:
 *   int sm = oasr::kernels::getDeviceSmVersion();
 *   OASR_DISPATCH_ARCH(sm, SM_VERSION, {
 *       using Traits = oasr::kernels::ArchTraits<SM_VERSION>;
 *       // ... use Traits ...
 *   });
 */
#define OASR_DISPATCH_ARCH(sm_version, ARCH_VAR, ...)                                    \
    do {                                                                                  \
        const int _resolved = oasr::kernels::resolveSmVersion(sm_version);               \
        switch (_resolved) {                                                              \
            case 120: {                                                                   \
                constexpr int ARCH_VAR = 120;                                             \
                __VA_ARGS__                                                               \
            } break;                                                                      \
            case 103: {                                                                   \
                constexpr int ARCH_VAR = 103;                                             \
                __VA_ARGS__                                                               \
            } break;                                                                      \
            case 100: {                                                                   \
                constexpr int ARCH_VAR = 100;                                             \
                __VA_ARGS__                                                               \
            } break;                                                                      \
            case 90: {                                                                    \
                constexpr int ARCH_VAR = 90;                                              \
                __VA_ARGS__                                                               \
            } break;                                                                      \
            case 89: {                                                                    \
                constexpr int ARCH_VAR = 89;                                              \
                __VA_ARGS__                                                               \
            } break;                                                                      \
            case 86: {                                                                    \
                constexpr int ARCH_VAR = 86;                                              \
                __VA_ARGS__                                                               \
            } break;                                                                      \
            case 80: {                                                                    \
                constexpr int ARCH_VAR = 80;                                              \
                __VA_ARGS__                                                               \
            } break;                                                                      \
            case 75: {                                                                    \
                constexpr int ARCH_VAR = 75;                                              \
                __VA_ARGS__                                                               \
            } break;                                                                      \
            case 70: {                                                                    \
                constexpr int ARCH_VAR = 70;                                              \
                __VA_ARGS__                                                               \
            } break;                                                                      \
            default:                                                                      \
                throw std::runtime_error("Unsupported SM version: " +                     \
                                         std::to_string(_resolved));                      \
        }                                                                                 \
    } while (0)
