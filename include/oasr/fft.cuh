// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA real-to-complex FFT kernel for FBANK / MFCC pipelines.
//
// Algorithm
// ---------
// Batched real-to-complex FFT (rfft) for power-of-two sizes via the
// "half-complex" method:
//
//   1. Pack N real samples into N/2 complex samples z[k] = x[2k] + i*x[2k+1].
//   2. Run an in-place radix-2 decimation-in-time complex FFT of length N/2
//      in shared memory (iterative Cooley-Tukey, log2(N/2) stages).
//   3. Post-process to recover the N/2+1 unique bins of the real DFT:
//        X[0]   = Re(Z[0]) + Im(Z[0])           (purely real)
//        X[N/2] = Re(Z[0]) - Im(Z[0])           (purely real)
//        X[k]   = 0.5*(Z[k] + conj(Z[N/2-k]))
//                 - 0.5j * exp(-2πi*k/N) * (Z[k] - conj(Z[N/2-k]))
//                                                for k = 1..N/2-1
//
// Layout: one block per FFT, blockDim.x = N/4 (every thread is active in every
// butterfly stage), shared memory = N/2 * float2 + N/4 * float2 (cached
// twiddles for the inner FFT). Supports N in {8, 16, 32, ..., 2048}.

#pragma once

#include <cuda_runtime.h>

namespace oasr {
namespace fft {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;

// Compile-time integer log2 for power-of-two N.
constexpr int Log2Int(int n) {
    int r = 0;
    while (n > 1) {
        n >>= 1;
        ++r;
    }
    return r;
}

// LOG2_N2-bit reverse of x using __brev.
template <int LOG2_N2>
__device__ __forceinline__ unsigned int BitReverse(unsigned int x) {
    return __brev(x) >> (32 - LOG2_N2);
}

namespace detail {

// Run the bit-reversal load and the iterative N/2-point complex FFT in shared
// memory. After this returns, smem[k] holds Z[k] (k = 0..N/2-1) for the
// half-complex FFT, with the largest-stage twiddle table cached in `twiddles`.
//
// blockDim.x must equal N/4. Each thread processes two packed complex slots
// during the load and one butterfly per stage.
template <int N>
__device__ __forceinline__ void RfftCore(const float* __restrict__ in_ptr,
                                         float2* __restrict__ smem,
                                         float2* __restrict__ twiddles) {
    constexpr int N2 = N / 2;
    constexpr int N4 = N / 4;
    constexpr int LOG2_N2 = Log2Int(N2);

    const int tid = threadIdx.x;  // 0..N/4-1

    // Stage 0: precompute the largest-stage twiddle table.
    //   twiddles[k] = exp(-2πi * k / N2)   for k = 0..N/4-1
    // Twiddles for stage s use stride 2^(LOG2_N2 - s) into this table.
    {
        const float angle = -kTwoPi * static_cast<float>(tid) / static_cast<float>(N2);
        float wr, wi;
        __sincosf(angle, &wi, &wr);
        twiddles[tid] = make_float2(wr, wi);
    }

    // Stage 1: bit-reversal load. Each thread packs two pairs of real samples.
    {
        const int t0 = tid;
        const int t1 = tid + N4;
        const unsigned int rev0 = BitReverse<LOG2_N2>(static_cast<unsigned int>(t0));
        const unsigned int rev1 = BitReverse<LOG2_N2>(static_cast<unsigned int>(t1));
        smem[rev0] = make_float2(in_ptr[2 * t0], in_ptr[2 * t0 + 1]);
        smem[rev1] = make_float2(in_ptr[2 * t1], in_ptr[2 * t1 + 1]);
    }
    __syncthreads();

    // Stage 2..LOG2_N2+1: iterative N/2-point DIT FFT. Each stage has N/4
    // butterflies; with blockDim = N/4 every thread does exactly one butterfly.
#pragma unroll
    for (int s = 1; s <= LOG2_N2; ++s) {
        const int m = 1 << s;
        const int m2 = m >> 1;
        const int k = tid & (m2 - 1);
        const int g = tid >> (s - 1);
        const int idx0 = g * m + k;
        const int idx1 = idx0 + m2;

        const float2 w = twiddles[k << (LOG2_N2 - s)];

        const float2 a = smem[idx0];
        const float2 b = smem[idx1];
        const float tx = w.x * b.x - w.y * b.y;
        const float ty = w.x * b.y + w.y * b.x;

        smem[idx0] = make_float2(a.x + tx, a.y + ty);
        smem[idx1] = make_float2(a.x - tx, a.y - ty);
        __syncthreads();
    }
}

// Compute the half-complex post-processed bin X[k] for k in [1, N/2-1].
// Inputs zk = Z[k], zc = Z[N/2-k]; the function applies the conjugate to zc.
template <int N>
__device__ __forceinline__ float2 PostprocessBin(float2 zk, float2 zc, int k) {
    constexpr int N2 = N / 2;  // unused, kept for symmetry
    (void)N2;

    zc.y = -zc.y;  // conj(Z[N/2-k])

    const float sumx = zk.x + zc.x;
    const float sumy = zk.y + zc.y;
    const float diffx = zk.x - zc.x;
    const float diffy = zk.y - zc.y;

    const float angle = -kTwoPi * static_cast<float>(k) / static_cast<float>(N);
    float wr, wi;
    __sincosf(angle, &wi, &wr);

    // X[k] = 0.5 * sum - 0.5j * exp(-2πi*k/N) * diff
    const float r = wr * diffx - wi * diffy;
    const float im = wr * diffy + wi * diffx;
    return make_float2(0.5f * sumx + 0.5f * im, 0.5f * sumy - 0.5f * r);
}

}  // namespace detail

// Single-block batched real-to-complex FFT.
//
// Input  : (batch, N) float32  (each row contiguous, see input_stride).
// Output : (batch, N/2+1) float2 (complex; see output_stride).
template <int N>
__global__ void RfftKernel(const float* __restrict__ input,
                           float2* __restrict__ output,
                           int input_stride,
                           int output_stride) {
    static_assert(N >= 8 && (N & (N - 1)) == 0, "N must be power of 2 >= 8");
    constexpr int N2 = N / 2;
    constexpr int N4 = N / 4;

    __shared__ float2 smem[N2];
    __shared__ float2 twiddles[N4];

    const int batch = blockIdx.x;
    const int tid = threadIdx.x;  // 0..N/4-1

    const float* in_ptr = input + batch * input_stride;
    float2* out_ptr = output + batch * output_stride;

    detail::RfftCore<N>(in_ptr, smem, twiddles);

    // Post-process. Each thread emits two outputs at indices tid and tid+N/4.
    if (tid == 0) {
        const float2 z0 = smem[0];
        out_ptr[0] = make_float2(z0.x + z0.y, 0.0f);
        out_ptr[N2] = make_float2(z0.x - z0.y, 0.0f);
    } else {
        out_ptr[tid] = detail::PostprocessBin<N>(smem[tid], smem[N2 - tid], tid);
    }

    const int k1 = tid + N4;  // N/4..N/2-1
    out_ptr[k1] = detail::PostprocessBin<N>(smem[k1], smem[N2 - k1], k1);
}

// Same as RfftKernel, but the output is the power spectrum |X[k]|^2 (real).
// Used as a fused output for FBANK / MFCC pipelines.
template <int N>
__global__ void RfftPowerKernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int input_stride,
                                int output_stride) {
    static_assert(N >= 8 && (N & (N - 1)) == 0, "N must be power of 2 >= 8");
    constexpr int N2 = N / 2;
    constexpr int N4 = N / 4;

    __shared__ float2 smem[N2];
    __shared__ float2 twiddles[N4];

    const int batch = blockIdx.x;
    const int tid = threadIdx.x;  // 0..N/4-1

    const float* in_ptr = input + batch * input_stride;
    float* out_ptr = output + batch * output_stride;

    detail::RfftCore<N>(in_ptr, smem, twiddles);

    if (tid == 0) {
        const float2 z0 = smem[0];
        const float x0 = z0.x + z0.y;
        const float xN2 = z0.x - z0.y;
        out_ptr[0] = x0 * x0;
        out_ptr[N2] = xN2 * xN2;
    } else {
        const float2 xk = detail::PostprocessBin<N>(smem[tid], smem[N2 - tid], tid);
        out_ptr[tid] = xk.x * xk.x + xk.y * xk.y;
    }

    const int k1 = tid + N4;
    const float2 xk = detail::PostprocessBin<N>(smem[k1], smem[N2 - k1], k1);
    out_ptr[k1] = xk.x * xk.x + xk.y * xk.y;
}

// =============================================================================
// Typed launchers (raw-pointer interface, returns cudaError_t)
// =============================================================================

template <int N>
inline cudaError_t LaunchRfft(const float* input, float2* output, int batch_size,
                              int input_stride, int output_stride, cudaStream_t stream) {
    constexpr int kThreads = N / 4;
    RfftKernel<N><<<batch_size, kThreads, 0, stream>>>(input, output, input_stride,
                                                       output_stride);
    return cudaGetLastError();
}

template <int N>
inline cudaError_t LaunchRfftPower(const float* input, float* output, int batch_size,
                                   int input_stride, int output_stride,
                                   cudaStream_t stream) {
    constexpr int kThreads = N / 4;
    RfftPowerKernel<N><<<batch_size, kThreads, 0, stream>>>(input, output, input_stride,
                                                            output_stride);
    return cudaGetLastError();
}

#define OASR_FFT_DISPATCH_N(n_fft, ...)        \
    do {                                       \
        switch (n_fft) {                       \
            case 8: {                          \
                constexpr int kN = 8;          \
                __VA_ARGS__;                   \
                break;                         \
            }                                  \
            case 16: {                         \
                constexpr int kN = 16;         \
                __VA_ARGS__;                   \
                break;                         \
            }                                  \
            case 32: {                         \
                constexpr int kN = 32;         \
                __VA_ARGS__;                   \
                break;                         \
            }                                  \
            case 64: {                         \
                constexpr int kN = 64;         \
                __VA_ARGS__;                   \
                break;                         \
            }                                  \
            case 128: {                        \
                constexpr int kN = 128;        \
                __VA_ARGS__;                   \
                break;                         \
            }                                  \
            case 256: {                        \
                constexpr int kN = 256;        \
                __VA_ARGS__;                   \
                break;                         \
            }                                  \
            case 512: {                        \
                constexpr int kN = 512;        \
                __VA_ARGS__;                   \
                break;                         \
            }                                  \
            case 1024: {                       \
                constexpr int kN = 1024;       \
                __VA_ARGS__;                   \
                break;                         \
            }                                  \
            case 2048: {                       \
                constexpr int kN = 2048;       \
                __VA_ARGS__;                   \
                break;                         \
            }                                  \
            default:                           \
                return cudaErrorInvalidValue;  \
        }                                      \
    } while (0)

inline cudaError_t Rfft(const float* input, float2* output, int n_fft, int batch_size,
                        int input_stride, int output_stride, cudaStream_t stream) {
    OASR_FFT_DISPATCH_N(n_fft, return LaunchRfft<kN>(input, output, batch_size, input_stride,
                                                     output_stride, stream));
    return cudaSuccess;
}

inline cudaError_t RfftPower(const float* input, float* output, int n_fft, int batch_size,
                             int input_stride, int output_stride, cudaStream_t stream) {
    OASR_FFT_DISPATCH_N(n_fft, return LaunchRfftPower<kN>(input, output, batch_size,
                                                          input_stride, output_stride, stream));
    return cudaSuccess;
}

}  // namespace fft
}  // namespace oasr
