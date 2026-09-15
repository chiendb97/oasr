// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Runtime architecture dispatch utilities

#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace oasr {

/**
 * @brief Query the compute capability of the current CUDA device.
 *
 * Returns SM version as (major * 10 + minor), e.g. 80 for SM8.0.
 * Result is cached per device ordinal.
 */
inline int getDeviceSmVersionFor(int device) {
    // Thread-safe: cudaDeviceGetAttribute is reentrant and the result is
    // idempotent for a given device, so racing writes produce the same value.
    static int cached[16] = {};
    if (device < 0 || device >= 16) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        return major * 10 + minor;
    }
    if (cached[device] != 0) {
        return cached[device];
    }

    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    cached[device] = major * 10 + minor;
    return cached[device];
}

inline int getDeviceSmVersion() {
    int device = 0;
    cudaGetDevice(&device);
    return getDeviceSmVersionFor(device);
}

/**
 * @brief Number of SMs on the current CUDA device.
 *
 * Cached per device ordinal, like getDeviceSmVersion().  A kernel that has to
 * choose a tile shape needs this: the CTA count a tile produces is only
 * meaningful against the device it will run on.
 */
inline int getDeviceMultiProcessorCount() {
    int device = 0;
    cudaGetDevice(&device);

    static int cached[16] = {};
    if (cached[device] != 0) {
        return cached[device];
    }

    int count = 0;
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    cached[device] = count > 0 ? count : 1;
    return cached[device];
}

/**
 * @brief Maximum *dynamic* shared memory a block may opt in to, in bytes.
 *
 * Cached per device ordinal, like the queries above.  This is the number a
 * kernel that sizes its own shared memory should be asking for; the familiar
 * 48 KiB is the limit a block gets *without* asking, and it is the same on every
 * architecture, which is exactly why hardcoding it looks portable and is not:
 * it leaves 52 KiB unused on sm_86/89/120 and 115 KiB on sm_80.
 *
 * Opting in is two steps, and the second is per kernel: ask for the budget here,
 * then ``cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
 * bytes)`` before the launch.  ``oasr::optInSharedMemory`` does the second.
 *
 * Note this does **not** apply to ``__shared__`` arrays declared inside a
 * kernel.  Static shared memory is capped at 48 KiB per block on every
 * architecture and no attribute raises it -- so a gate on static usage (CUB's
 * ``BlockRadixSort::TempStorage`` in ``topk.cuh``, for one) is correct at
 * ``48 * 1024`` and must stay there.
 */
inline int getDeviceMaxSharedMemoryOptin(int device) {
    static int cached[16] = {};
    if (device >= 0 && device < 16 && cached[device] != 0) {
        return cached[device];
    }
    int bytes = 0;
    cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // A driver that cannot answer leaves the caller on the guaranteed floor
    // rather than on zero.
    if (bytes < 48 * 1024) {
        bytes = 48 * 1024;
    }
    if (device >= 0 && device < 16) {
        cached[device] = bytes;
    }
    return bytes;
}

inline int getDeviceMaxSharedMemoryOptin() {
    int device = 0;
    cudaGetDevice(&device);
    return getDeviceMaxSharedMemoryOptin(device);
}

/**
 * @brief Raise *kernel*'s dynamic shared-memory ceiling to @p bytes if needed.
 *
 * A no-op at or below 48 KiB, which every block gets for free.  Returns the
 * driver's verdict so a launcher can decline rather than launch a kernel the
 * driver has already refused to configure.
 */
template <typename KernelFn>
inline cudaError_t optInSharedMemory(KernelFn kernel, size_t bytes) {
    if (bytes <= 48 * 1024) {
        return cudaSuccess;
    }
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(bytes));
}

/** Map a runtime SM version to the highest compiled family not exceeding it. */
inline int resolveSmVersion(int sm) {
    if (sm >= 120) return 120;
    if (sm >= 100) return 100;
    if (sm >= 90) return 90;
    if (sm >= 89) return 89;
    if (sm >= 86) return 86;
    if (sm >= 80) return 80;
    if (sm >= 75) return 75;
    throw std::runtime_error("Unsupported GPU architecture: SM" + std::to_string(sm) +
                             " (OASR requires SM75 / Turing or newer)");
}

/**
 * @brief Refuse a device this module was not compiled for.
 *
 * A JIT module is built for exactly one SM family -- ``OASR_TARGET_SM`` is baked
 * in at compile time and the gencode with it -- and the family is resolved once,
 * from whichever device happened to be current the first time a kernel was
 * called.  On a **heterogeneous node** that is a silent wrong answer: a process
 * holding an A100 and an H100 loads one library for both, and the tensors that
 * arrive on the other card either fail to launch with a cryptic driver error or,
 * where the binary happens to be loadable, run kernels tuned and specialised for
 * the wrong architecture.
 *
 * ``getDeviceSmVersionFor`` already caches per ordinal, so after the first call
 * per device this is an array read and an integer compare -- cheap enough to sit
 * on the launch path, which is the only place that sees the tensor's device.
 *
 * Deliberately a *refusal* rather than a recompile.  Recompiling per device is
 * the right long-term answer (the cache key already covers the arch flags), but
 * it has to happen above the launcher, in the Python module cache, and silently
 * doing the wrong thing is what this is here to stop.
 */
inline void checkDeviceMatchesBuild(int device_id) {
#ifdef OASR_TARGET_SM
    const int sm = getDeviceSmVersionFor(device_id);
    if (resolveSmVersion(sm) != OASR_TARGET_SM) {
        throw std::runtime_error(
            "this OASR kernel module was compiled for SM" + std::to_string(OASR_TARGET_SM) +
            " but the tensor is on CUDA device " + std::to_string(device_id) + ", which is SM" +
            std::to_string(sm) +
            ". One module serves one architecture, so a process spanning two different GPUs "
            "needs one process per architecture (set CUDA_VISIBLE_DEVICES), or the kernels "
            "rebuilt for this device.");
    }
#else
    (void)device_id;
#endif
}

}  // namespace oasr

/** Convert a runtime SM to a compile-time constant; JIT builds instantiate only
 * OASR_TARGET_SM when defined. */
#ifdef OASR_TARGET_SM

#define OASR_DISPATCH_SM(sm_version, ARCH_VAR, ...)                                        \
    do {                                                                                    \
        constexpr int ARCH_VAR = OASR_TARGET_SM;                                           \
        __VA_ARGS__                                                                         \
    } while (0)

#else  // Full runtime dispatch (AOT builds)

#define OASR_DISPATCH_SM(sm_version, ARCH_VAR, ...)                                        \
    do {                                                                                    \
        const int _resolved = oasr::resolveSmVersion(sm_version);                          \
        switch (_resolved) {                                                                \
            case 120: {                                                                     \
                constexpr int ARCH_VAR = 120;                                               \
                __VA_ARGS__                                                                 \
            } break;                                                                        \
            case 100: {                                                                     \
                constexpr int ARCH_VAR = 100;                                               \
                __VA_ARGS__                                                                 \
            } break;                                                                        \
            case 90: {                                                                      \
                constexpr int ARCH_VAR = 90;                                                \
                __VA_ARGS__                                                                 \
            } break;                                                                        \
            case 89: {                                                                      \
                constexpr int ARCH_VAR = 89;                                                \
                __VA_ARGS__                                                                 \
            } break;                                                                        \
            case 86: {                                                                      \
                constexpr int ARCH_VAR = 86;                                                \
                __VA_ARGS__                                                                 \
            } break;                                                                        \
            case 80: {                                                                      \
                constexpr int ARCH_VAR = 80;                                                \
                __VA_ARGS__                                                                 \
            } break;                                                                        \
            case 75: {                                                                      \
                constexpr int ARCH_VAR = 75;                                                \
                __VA_ARGS__                                                                 \
            } break;                                                                        \
            default:                                                                        \
                throw std::runtime_error("Unsupported SM version: " +                       \
                                         std::to_string(_resolved));                        \
        }                                                                                   \
    } while (0)

#endif  // OASR_TARGET_SM

// Backward compatibility alias
#define OASR_DISPATCH_ARCH OASR_DISPATCH_SM
