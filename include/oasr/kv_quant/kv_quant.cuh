// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// FP8 (e4m3) KV-cache quantization -- no framework dependencies.
//
// One scale per **(token, kv-head)** vector, chosen from that vector alone, when
// the token is written to the cache: no calibration, and row-local, so a token
// quantizes identically whatever else shares the batch (rule: batch invariance,
// the property every capture bucket and merge relies on).
//
// The scale is a **power of two**, the smallest one that brings the vector's
// amax inside e4m3's +-448:
//
//     amax  = max_j |f32(x_j)|                         (0 for an all-zero vector)
//     amax  = m * 2^e,  m in [0.5, 1)                  (frexp)
//     k     = (m <= 0.875 ? e - 9 : e - 8), clamped to >= -126
//     scale = 2^k,  inv = 2^-k                         (both exact)
//     q_j   = e4m3_rne_satfinite(f32(x_j) * inv)       (the product is exact)
//
// (448 = 0.875 * 2^9, so `k` is the least integer with 448 * 2^k >= amax.)
//
// Why a power of two rather than `amax / 448`: dequantizing `q * scale` is then
// **exact** in fp32 and in the served fp16/bf16, so the attention kernel's
// in-tile dequantization reproduces, bit for bit, what a torch reference that
// dequantizes the same bytes would hand the fp16 kernel -- which is what makes
// the kernel testable against an exact oracle.  The cost is that the amax maps
// into (224, 448] instead of onto 448: e4m3's relative precision is the same
// across its whole normal range, so the only loss is a factor <= 2 more of the
// smallest values falling into the subnormal band.  On the shipped Whisper and
// Qwen2-Audio decoders both scale rules measure the same ~31 dB SQNR per layer.
//
// `torch` mirrors this exactly in `oasr.functionals.kv_quant.reference_quantize_kv`.
//
// Addressing.  Input `x[b, h, t, :]` with arbitrary strides (unit last stride);
// output vector `(b, h, t)` lands at `out + b*o_b + p*o_t + h*o_h` with its scale
// at `scale + b*s_b + p*s_t + h*s_h`, where `p = pos[b * T + t]` (or `t` when no
// position vector is given).  That one rule covers both caches the decoders use:
// a paged pool viewed as `(slots, H, D)` (`o_b = 0`, `p` = flat slot) and a dense
// capacity buffer `(B, H, cap, D)` (`p` = the row's write offset).

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <oasr/quant/quantize.cuh>

namespace oasr {
namespace kv_quant {

//! The largest finite e4m3 (`float8_e4m3fn` has no infinity; all-ones is NaN).
constexpr float kFp8Max = quant::QuantTraits<__nv_fp8_e4m3>::kMax;

//! Most head dims one call handles: one 8-element vector per lane of a warp.
constexpr int kMaxHeadDim = 256;

/*! \brief Up to two tensors (K and V) quantized in one launch. */
template <typename Element>
struct KvQuantParams {
    Element const* x[2];
    int64_t x_stride_b[2];
    int64_t x_stride_h[2];
    int64_t x_stride_t[2];

    uint8_t* out[2];
    int64_t o_stride_b[2];
    int64_t o_stride_h[2];
    int64_t o_stride_t[2];

    float* scale[2];
    int64_t s_stride_b[2];
    int64_t s_stride_h[2];
    int64_t s_stride_t[2];

    int64_t const* pos;  //!< `(B * T,)` write positions, or null for `p = t`
    int B;
    int H;
    int T;
    int D;
};

/*! \brief The power-of-two scale exponent for a vector whose amax is `amax`. */
__device__ __forceinline__ int scaleExponent(float amax) {
    int e = 0;
    float const m = frexpf(amax, &e);
    int const k = (m <= 0.875f) ? e - 9 : e - 8;
    return k < -126 ? -126 : k;
}

/*! \brief `2^k` for `k` in `[-126, 127]`, built from its bits (exact). */
__device__ __forceinline__ float exp2Int(int k) { return __int_as_float((k + 127) << 23); }

template <typename T>
__device__ __forceinline__ void loadVec8(T const* p, float (&f)[8]) {
    uint4 const raw = *reinterpret_cast<uint4 const*>(p);
    T const* v = reinterpret_cast<T const*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        f[j] = static_cast<float>(v[j]);
    }
}

/*! \brief One `(b, h, t)` vector per `kLanes` lanes; `32 / kLanes` vectors per warp.
 *
 * `kLanes` is the power of two covering `D / 8` (each lane owns one 8-element,
 * 16-byte vector), so the amax is a butterfly over an aligned lane group --
 * every lane of the warp takes part in the shuffle, the ones past the end with
 * a zero, because a `__shfl_xor_sync` with a lane missing from its mask is
 * undefined.
 */
template <typename T, int kLanes>
__global__ void kvQuantStoreKernel(KvQuantParams<T> const p) {
    static_assert(kLanes >= 1 && kLanes <= 32 && (32 % kLanes) == 0);
    constexpr int kRowsPerWarp = 32 / kLanes;
    int const which = blockIdx.y;
    int const lane = threadIdx.x % 32;
    int const warp = threadIdx.x / 32;
    int64_t const rows = int64_t(p.B) * p.T * p.H;
    int64_t const row = (int64_t(blockIdx.x) * (blockDim.x / 32) + warp) * kRowsPerWarp +
                        lane / kLanes;
    int const vec = lane % kLanes;
    bool const row_ok = row < rows;
    bool const vec_ok = row_ok && (vec * 8 < p.D);

    int b = 0, t = 0, h = 0;
    if (row_ok) {
        h = int(row % p.H);
        int64_t const bt = row / p.H;
        t = int(bt % p.T);
        b = int(bt / p.T);
    }

    float f[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        f[j] = 0.0f;
    }
    if (vec_ok) {
        T const* src = p.x[which] + int64_t(b) * p.x_stride_b[which] +
                       int64_t(h) * p.x_stride_h[which] + int64_t(t) * p.x_stride_t[which] +
                       vec * 8;
        loadVec8(src, f);
    }
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        amax = fmaxf(amax, fabsf(f[j]));
    }
#pragma unroll
    for (int off = kLanes / 2; off >= 1; off /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    if (!row_ok) {
        return;
    }
    int const k = scaleExponent(amax);
    float const inv = exp2Int(-k);
    int64_t const pos = p.pos != nullptr ? p.pos[int64_t(b) * p.T + t] : int64_t(t);
    if (vec == 0) {
        p.scale[which][int64_t(b) * p.s_stride_b[which] + pos * p.s_stride_t[which] +
                       int64_t(h) * p.s_stride_h[which]] = exp2Int(k);
    }
    if (!vec_ok) {
        return;
    }
    // The activation quantizer's e4m3 rounding (RNE, saturating), so the two
    // 8-bit paths in the tree cannot drift apart on what "round to fp8" means.
    alignas(8) __nv_fp8_e4m3 q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        q[j] = quant::QuantTraits<__nv_fp8_e4m3>::quantize(f[j], inv);
    }
    uint8_t* dst = p.out[which] + int64_t(b) * p.o_stride_b[which] +
                   pos * p.o_stride_t[which] + int64_t(h) * p.o_stride_h[which] + vec * 8;
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<uint2 const*>(q);
}

/*! \brief Quantize `num_tensors` (1 or 2) KV tensors into their caches. */
template <typename T>
cudaError_t KvQuantStore(KvQuantParams<T> const& p, int num_tensors, cudaStream_t stream) {
    int64_t const rows = int64_t(p.B) * p.T * p.H;
    if (rows == 0) {
        return cudaSuccess;
    }
    int const vecs = (p.D + 7) / 8;
    int lanes = 1;
    while (lanes < vecs) {
        lanes *= 2;
    }
    constexpr int kThreads = 128;
    int64_t const rows_per_block = int64_t(kThreads / 32) * (32 / lanes);
    dim3 const grid(unsigned((rows + rows_per_block - 1) / rows_per_block),
                    unsigned(num_tensors));
    switch (lanes) {
        case 1:
            kvQuantStoreKernel<T, 1><<<grid, kThreads, 0, stream>>>(p);
            break;
        case 2:
            kvQuantStoreKernel<T, 2><<<grid, kThreads, 0, stream>>>(p);
            break;
        case 4:
            kvQuantStoreKernel<T, 4><<<grid, kThreads, 0, stream>>>(p);
            break;
        case 8:
            kvQuantStoreKernel<T, 8><<<grid, kThreads, 0, stream>>>(p);
            break;
        case 16:
            kvQuantStoreKernel<T, 16><<<grid, kThreads, 0, stream>>>(p);
            break;
        case 32:
            kvQuantStoreKernel<T, 32><<<grid, kThreads, 0, stream>>>(p);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

}  // namespace kv_quant
}  // namespace oasr
