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
// Layout: one block per FFT, blockDim.x = N/2, shared memory = N/2 * float2.
// Supports N in {8, 16, 32, ..., 2048}.

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

// Single-block batched real-to-complex FFT.
//
// Input  : (batch, N) float32 (each row contiguous, see input_stride).
// Output : (batch, N/2+1) float2 (complex; see output_stride).
template <int N>
__global__ void RfftKernel(const float* __restrict__ input,
                           float2* __restrict__ output,
                           int input_stride,
                           int output_stride) {
    static_assert(N >= 8 && (N & (N - 1)) == 0, "N must be power of 2 >= 8");
    constexpr int N2 = N / 2;
    constexpr int LOG2_N2 = Log2Int(N2);

    __shared__ float2 smem[N2];

    const int batch = blockIdx.x;
    const int tid = threadIdx.x;  // 0..N/2-1

    const float* in_ptr = input + batch * input_stride;
    float2* out_ptr = output + batch * output_stride;

    // Step 1: load N reals as N/2 complex with bit-reversal permutation.
    const unsigned int rev = BitReverse<LOG2_N2>(static_cast<unsigned int>(tid));
    smem[rev] = make_float2(in_ptr[2 * tid], in_ptr[2 * tid + 1]);
    __syncthreads();

    // Step 2: iterative N/2-point complex DIT FFT.
    // Each stage has N/4 butterflies; threads with tid < N/4 do work.
#pragma unroll
    for (int s = 1; s <= LOG2_N2; ++s) {
        const int m = 1 << s;
        const int m2 = m >> 1;
        if (tid < N2 / 2) {
            const int k = tid & (m2 - 1);
            const int g = tid >> (s - 1);
            const int idx0 = g * m + k;
            const int idx1 = idx0 + m2;

            const float angle = -kTwoPi * static_cast<float>(k) / static_cast<float>(m);
            float wr, wi;
            __sincosf(angle, &wi, &wr);

            const float2 a = smem[idx0];
            const float2 b = smem[idx1];
            const float tx = wr * b.x - wi * b.y;
            const float ty = wr * b.y + wi * b.x;

            smem[idx0] = make_float2(a.x + tx, a.y + ty);
            smem[idx1] = make_float2(a.x - tx, a.y - ty);
        }
        __syncthreads();
    }

    // Step 3: post-process to N/2+1 real-FFT bins.
    if (tid == 0) {
        const float2 z0 = smem[0];
        out_ptr[0] = make_float2(z0.x + z0.y, 0.0f);
        out_ptr[N2] = make_float2(z0.x - z0.y, 0.0f);
    } else {
        const int k = tid;  // 1..N/2-1
        const float2 zk = smem[k];
        float2 zc = smem[N2 - k];
        zc.y = -zc.y;  // conj(Z[N/2-k])

        const float sumx = zk.x + zc.x;
        const float sumy = zk.y + zc.y;
        const float diffx = zk.x - zc.x;
        const float diffy = zk.y - zc.y;

        const float angle = -kTwoPi * static_cast<float>(k) / static_cast<float>(N);
        float wr, wi;
        __sincosf(angle, &wi, &wr);

        // X[k] = 0.5 * sum - 0.5j * (wr + wi*i) * diff
        const float r = wr * diffx - wi * diffy;
        const float i = wr * diffy + wi * diffx;
        out_ptr[k] = make_float2(0.5f * sumx + 0.5f * i,
                                 0.5f * sumy - 0.5f * r);
    }
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
    constexpr int LOG2_N2 = Log2Int(N2);

    __shared__ float2 smem[N2];

    const int batch = blockIdx.x;
    const int tid = threadIdx.x;

    const float* in_ptr = input + batch * input_stride;
    float* out_ptr = output + batch * output_stride;

    const unsigned int rev = BitReverse<LOG2_N2>(static_cast<unsigned int>(tid));
    smem[rev] = make_float2(in_ptr[2 * tid], in_ptr[2 * tid + 1]);
    __syncthreads();

#pragma unroll
    for (int s = 1; s <= LOG2_N2; ++s) {
        const int m = 1 << s;
        const int m2 = m >> 1;
        if (tid < N2 / 2) {
            const int k = tid & (m2 - 1);
            const int g = tid >> (s - 1);
            const int idx0 = g * m + k;
            const int idx1 = idx0 + m2;

            const float angle = -kTwoPi * static_cast<float>(k) / static_cast<float>(m);
            float wr, wi;
            __sincosf(angle, &wi, &wr);

            const float2 a = smem[idx0];
            const float2 b = smem[idx1];
            const float tx = wr * b.x - wi * b.y;
            const float ty = wr * b.y + wi * b.x;

            smem[idx0] = make_float2(a.x + tx, a.y + ty);
            smem[idx1] = make_float2(a.x - tx, a.y - ty);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float2 z0 = smem[0];
        const float x0 = z0.x + z0.y;
        const float xN2 = z0.x - z0.y;
        out_ptr[0] = x0 * x0;
        out_ptr[N2] = xN2 * xN2;
    } else {
        const int k = tid;
        const float2 zk = smem[k];
        float2 zc = smem[N2 - k];
        zc.y = -zc.y;

        const float sumx = zk.x + zc.x;
        const float sumy = zk.y + zc.y;
        const float diffx = zk.x - zc.x;
        const float diffy = zk.y - zc.y;

        const float angle = -kTwoPi * static_cast<float>(k) / static_cast<float>(N);
        float wr, wi;
        __sincosf(angle, &wi, &wr);

        const float r = wr * diffx - wi * diffy;
        const float im = wr * diffy + wi * diffx;
        const float xr = 0.5f * sumx + 0.5f * im;
        const float xi = 0.5f * sumy - 0.5f * r;
        out_ptr[k] = xr * xr + xi * xi;
    }
}

// =============================================================================
// Typed launchers (raw-pointer interface, returns cudaError_t)
// =============================================================================

template <int N>
inline cudaError_t LaunchRfft(const float* input, float2* output, int batch_size,
                              int input_stride, int output_stride, cudaStream_t stream) {
    constexpr int kThreads = N / 2;
    RfftKernel<N><<<batch_size, kThreads, 0, stream>>>(input, output, input_stride,
                                                       output_stride);
    return cudaGetLastError();
}

template <int N>
inline cudaError_t LaunchRfftPower(const float* input, float* output, int batch_size,
                                   int input_stride, int output_stride,
                                   cudaStream_t stream) {
    constexpr int kThreads = N / 2;
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
