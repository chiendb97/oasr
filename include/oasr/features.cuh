// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA kernels for the FBANK / MFCC feature extraction pipeline.
//
// Three building blocks run between framing (handled by torch.unfold) and the
// final feature tensor:
//
//   1. FbankPreprocess  -- DC removal + pre-emphasis + windowing + zero-pad
//                          (Total_frames, frame_length) -> (Total_frames, n_fft)
//   2. (rfft_power)     -- power spectrum  (see oasr/fft.cuh)
//   3. MelLog           -- mel filterbank + log floor
//                          (Total_frames, n_fft/2+1) -> (Total_frames, num_mel)
//   4. DctLifter        -- DCT-II + cepstral lifter (MFCC only)
//                          (Total_frames, num_mel) -> (Total_frames, num_ceps)
//
// All kernels use `Total_frames = batch * num_frames` and one block per frame.

#pragma once

#include <cuda_runtime.h>

#include <oasr/reduction.cuh>  // brings in oasr/common/utils.h (WARP_SIZE)

namespace oasr {
namespace features {

// =============================================================================
// 1. Fbank preprocess: DC removal + pre-emphasis + windowing + zero-pad.
// =============================================================================
//
// For each frame x[0..L-1]:
//   mean = sum(x) / L
//   x[i] -= mean
//   y[0] = (1 - coef) * x[0]                          (Kaldi "replicate" boundary)
//   y[i] = x[i] - coef * x[i-1]    for i = 1..L-1
//   out[i] = y[i] * window[i]      for i = 0..L-1
//   out[i] = 0                     for i = L..n_fft-1
//
// Layout:
//   gridDim.x  = total_frames
//   blockDim.x = power-of-two thread count (256 by default)
//   shared     = frame_length floats (input cache for in-place transform)
__global__ inline void FbankPreprocessKernel(const float* __restrict__ frames,
                                             const float* __restrict__ window,
                                             float* __restrict__ output, int frame_length,
                                             int n_fft, float preemph_coef,
                                             int remove_dc_offset, int apply_preemph) {
    extern __shared__ float smem[];

    const int frame_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;

    const float* in_ptr = frames + frame_idx * frame_length;
    float* out_ptr = output + frame_idx * n_fft;

    // Phase 1: load and accumulate sum for DC removal.
    float local_sum = 0.0f;
    for (int i = tid; i < frame_length; i += bs) {
        const float v = in_ptr[i];
        smem[i] = v;
        local_sum += v;
    }

    __shared__ float s_mean;
    if (remove_dc_offset) {
        __syncthreads();
        const float total = oasr::reduction::blockReduceSum<float>(local_sum);
        if (tid == 0) {
            s_mean = total / static_cast<float>(frame_length);
        }
    } else if (tid == 0) {
        s_mean = 0.0f;
    }
    __syncthreads();
    const float mean = s_mean;

    // Phase 2: emit pre-emphasized + windowed samples.
    for (int i = tid; i < frame_length; i += bs) {
        const float xi = smem[i] - mean;
        float yi;
        if (apply_preemph) {
            const float xim1 = (i > 0) ? (smem[i - 1] - mean) : xi;
            yi = xi - preemph_coef * xim1;
        } else {
            yi = xi;
        }
        out_ptr[i] = yi * window[i];
    }

    // Phase 3: zero-pad the tail [frame_length, n_fft).
    // Use float4 stores when the start offset and remaining length are aligned.
    const int pad_start = frame_length;
    const int pad_end = n_fft;
    const int pad_len = pad_end - pad_start;
    if (pad_len > 0) {
        const bool aligned = ((pad_start & 3) == 0) && ((pad_len & 3) == 0);
        if (aligned) {
            float4* out4 = reinterpret_cast<float4*>(out_ptr + pad_start);
            const float4 zero4 = make_float4(0.f, 0.f, 0.f, 0.f);
            const int n4 = pad_len >> 2;
            for (int j = tid; j < n4; j += bs) {
                out4[j] = zero4;
            }
        } else {
            for (int i = pad_start + tid; i < pad_end; i += bs) {
                out_ptr[i] = 0.0f;
            }
        }
    }
}

inline cudaError_t FbankPreprocess(const float* frames, const float* window, float* output,
                                   int total_frames, int frame_length, int n_fft,
                                   float preemph_coef, bool remove_dc_offset,
                                   bool apply_preemph, cudaStream_t stream) {
    const int threads = 256;
    const size_t smem_bytes = static_cast<size_t>(frame_length) * sizeof(float);
    FbankPreprocessKernel<<<total_frames, threads, smem_bytes, stream>>>(
        frames, window, output, frame_length, n_fft, preemph_coef,
        remove_dc_offset ? 1 : 0, apply_preemph ? 1 : 0);
    return cudaGetLastError();
}

// =============================================================================
// 2. Mel filterbank + log-floor.
// =============================================================================
//
// For each frame's power spectrum p[0..F-1] (F = n_fft/2+1), compute
// log(max(mel_mat[b] @ p, log_floor)) for b = 0..num_mel-1.
//
// Layout:
//   gridDim.x  = total_frames
//   blockDim.x = 128 (4 warps); each warp computes one mel bin per pass via a
//                warp-strided dot product, giving fully-coalesced reads of
//                `mel_mat`.
//   shared     = F floats (cached power spectrum).
__global__ inline void MelLogKernel(const float* __restrict__ power,
                                    const float* __restrict__ mel_mat,
                                    float* __restrict__ output, int num_freq, int num_mel,
                                    float log_floor) {
    extern __shared__ float spec[];

    const int frame_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wid = tid >> 5;
    const int n_warps = bs >> 5;

    const float* in_ptr = power + frame_idx * num_freq;
    for (int i = tid; i < num_freq; i += bs) {
        spec[i] = in_ptr[i];
    }
    __syncthreads();

    float* out_ptr = output + frame_idx * num_mel;
    for (int b = wid; b < num_mel; b += n_warps) {
        const float* fb = mel_mat + b * num_freq;
        float acc = 0.0f;
        for (int i = lane; i < num_freq; i += WARP_SIZE) {
            acc += fb[i] * spec[i];
        }
        acc = oasr::reduction::warpReduceSum(acc);
        if (lane == 0) {
            if (acc < log_floor) {
                acc = log_floor;
            }
            out_ptr[b] = logf(acc);
        }
    }
}

inline cudaError_t MelLog(const float* power, const float* mel_mat, float* output,
                          int total_frames, int num_freq, int num_mel, float log_floor,
                          cudaStream_t stream) {
    const int threads = 128;  // 4 warps
    const size_t smem_bytes = static_cast<size_t>(num_freq) * sizeof(float);
    MelLogKernel<<<total_frames, threads, smem_bytes, stream>>>(power, mel_mat, output,
                                                                num_freq, num_mel, log_floor);
    return cudaGetLastError();
}

// =============================================================================
// 3. DCT-II + cepstral lifter (MFCC only).
// =============================================================================
//
// For each frame's log-mel vector m[0..M-1], compute
//   c[k] = lifter[k] * sum_i dct_mat[k, i] * m[i]   for k = 0..num_ceps-1
//
// `lifter_weights` may be null (no lifter applied).  When `replace_c0_with_energy`
// is true, c[0] is overwritten with `energy[frame_idx]` (typically log-energy
// of the windowed frame) -- matches Kaldi's `use_energy=true`.
//
// Layout:
//   gridDim.x  = total_frames
//   blockDim.x = 128 (4 warps); each warp emits one cepstral coefficient via a
//                warp-strided dot product (coalesced reads of `dct_mat`).
//   shared     = num_mel floats.
__global__ inline void DctLifterKernel(const float* __restrict__ log_mel,
                                       const float* __restrict__ dct_mat,
                                       const float* __restrict__ lifter_weights,
                                       const float* __restrict__ energy,
                                       float* __restrict__ output, int num_mel, int num_ceps,
                                       int replace_c0_with_energy) {
    extern __shared__ float smel[];

    const int frame_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wid = tid >> 5;
    const int n_warps = bs >> 5;

    const float* in_ptr = log_mel + frame_idx * num_mel;
    for (int i = tid; i < num_mel; i += bs) {
        smel[i] = in_ptr[i];
    }
    __syncthreads();

    float* out_ptr = output + frame_idx * num_ceps;
    for (int k = wid; k < num_ceps; k += n_warps) {
        const float* row = dct_mat + k * num_mel;
        float acc = 0.0f;
        for (int i = lane; i < num_mel; i += WARP_SIZE) {
            acc += row[i] * smel[i];
        }
        acc = oasr::reduction::warpReduceSum(acc);
        if (lane == 0) {
            if (lifter_weights != nullptr) {
                acc *= lifter_weights[k];
            }
            if (replace_c0_with_energy && k == 0 && energy != nullptr) {
                acc = energy[frame_idx];
            }
            out_ptr[k] = acc;
        }
    }
}

inline cudaError_t DctLifter(const float* log_mel, const float* dct_mat,
                             const float* lifter_weights, const float* energy, float* output,
                             int total_frames, int num_mel, int num_ceps,
                             bool replace_c0_with_energy, cudaStream_t stream) {
    const int threads = 128;  // 4 warps
    const size_t smem_bytes = static_cast<size_t>(num_mel) * sizeof(float);
    DctLifterKernel<<<total_frames, threads, smem_bytes, stream>>>(
        log_mel, dct_mat, lifter_weights, energy, output, num_mel, num_ceps,
        replace_c0_with_energy ? 1 : 0);
    return cudaGetLastError();
}

}  // namespace features
}  // namespace oasr
