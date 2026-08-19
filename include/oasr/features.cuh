// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Pure CUDA kernels for the log-mel / FBANK / MFCC feature extraction pipelines.
//
// Building blocks, in pipeline order:
//
//   0. StftFrame        -- framing + optional per-frame DC removal + pre-emphasis
//                          + windowing + zero-pad, straight off the waveform
//                          (B, T_wav) -> (B, num_frames, n_fft)
//   1. FbankPreprocess  -- DC removal + pre-emphasis + windowing + zero-pad for
//                          input that is *already* framed
//                          (Total_frames, frame_length) -> (Total_frames, n_fft)
//   2. (rfft_power)     -- power spectrum  (see oasr/fft.cuh)
//   3. MelLog           -- mel filterbank + log floor / additive guard
//                          (Total_frames, n_fft/2+1) -> (Total_frames, num_mel)
//   4. DctLifter        -- DCT-II + cepstral lifter (MFCC only)
//                          (Total_frames, num_mel) -> (Total_frames, num_ceps)
//   5. WhisperLogMel    -- mel + log10 + per-row max floor / normalization
//   6. LfrGather        -- varlen low-frame-rate frame stacking
//
// Stage 0 and stage 1 are alternatives, not a sequence: 0 owns the framing (so
// the caller needs no `unfold` / `torch.stft`) and pre-emphasises in the *signal*
// domain (NeMo / Nemotron), while 1 takes pre-framed input and pre-emphasises
// per frame with Kaldi's replicate boundary plus per-frame DC removal.
//
// Kernels 1-4 use `Total_frames = batch * num_frames` and one block per frame.

#pragma once

#include <cuda_runtime.h>

#include <oasr/reduction.cuh>  // brings in oasr/common/utils.h (WARP_SIZE)

namespace oasr {
namespace features {

__device__ __forceinline__ int ReflectIndex(int t, int signal_length) {
    const int period = 2 * (signal_length - 1);
    t %= period;
    if (t < 0) {
        t += period;
    }
    return t < signal_length ? t : period - t;
}

__device__ __forceinline__ int ResolveWaveformIndex(int wav_stride, int valid_length,
                                                    int signal_length, int t, int reflect_pad) {
    if (reflect_pad) {
        t = ReflectIndex(t, signal_length);
    } else if (t < 0 || t >= signal_length) {
        return -1;
    }
    if (t < 0 || t >= valid_length || t >= wav_stride) {
        return -1;
    }
    return t;
}

__device__ __forceinline__ float ReadWaveformSample(const float* row, int wav_stride,
                                                    int valid_length, int signal_length, int t,
                                                    int reflect_pad) {
    const int index = ResolveWaveformIndex(wav_stride, valid_length, signal_length, t, reflect_pad);
    return index >= 0 ? row[index] : 0.0f;
}

// =============================================================================
// 0. STFT framing: waveform -> pre-emphasised, windowed, zero-padded frames.
// =============================================================================
//
// Fused framing, signal-domain pre-emphasis, windowing, and zero padding.
// `center_offset` selects centered or snip-edges framing; samples outside each
// row's valid length are zero. `preemph_replicate` selects whether x[-1] is zero
// or x[0], preserving the two supported boundary conventions.
__global__ inline void StftFrameKernel(const float* __restrict__ waveform,
                                       const int32_t* __restrict__ lengths,
                                       const float* __restrict__ window, float* __restrict__ output,
                                       int64_t total_elems, int wav_stride, int num_frames,
                                       int n_fft, int win_length, int win_offset, int hop_length,
                                       int center_offset, float preemph_coef, int preemph_replicate,
                                       int signal_length, int reflect_pad) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_elems; idx += stride) {
        const int i = static_cast<int>(idx % n_fft);
        const int64_t frame_flat = idx / n_fft;
        const int f = static_cast<int>(frame_flat % num_frames);
        const int b = static_cast<int>(frame_flat / num_frames);

        const int w = i - win_offset;
        if (w < 0 || w >= win_length) {
            output[idx] = 0.0f;
            continue;
        }

        const int len = lengths[b];
        const int t = f * hop_length - center_offset + i;
        const float* row = waveform + static_cast<int64_t>(b) * wav_stride;
        const int sample_index =
            ResolveWaveformIndex(wav_stride, len, signal_length, t, reflect_pad);
        if (sample_index < 0) {
            output[idx] = 0.0f;
            continue;
        }
        float y = row[sample_index];
        if (preemph_coef != 0.0f) {
            float prev;
            if (t == 0) {
                prev = preemph_replicate ? y : 0.0f;
            } else {
                prev = ReadWaveformSample(row, wav_stride, len, signal_length, t - 1, reflect_pad);
            }
            y -= preemph_coef * prev;
        }
        output[idx] = y * window[w];
    }
}

// DC removal is a per-frame reduction, so one block owns a complete frame.
// Pre-emphasis is frame-local in this mode, including Kaldi's leading replicate
// boundary; the signal-domain kernel above remains the NeMo/Whisper path.
__global__ inline void StftFrameDcKernel(
    const float* __restrict__ waveform, const int32_t* __restrict__ lengths,
    const float* __restrict__ window, float* __restrict__ output, int wav_stride, int num_frames,
    int n_fft, int win_length, int win_offset, int hop_length, int center_offset,
    float preemph_coef, int preemph_replicate, int signal_length, int reflect_pad) {
    extern __shared__ float frame[];

    const int frame_idx = blockIdx.x;
    const int f = frame_idx % num_frames;
    const int b = frame_idx / num_frames;
    const int tid = threadIdx.x;
    const float* row = waveform + static_cast<int64_t>(b) * wav_stride;

    float local_sum = 0.0f;
    for (int w = tid; w < win_length; w += blockDim.x) {
        const int t = f * hop_length - center_offset + win_offset + w;
        const float x =
            ReadWaveformSample(row, wav_stride, lengths[b], signal_length, t, reflect_pad);
        frame[w] = x;
        local_sum += x;
    }
    __syncthreads();
    const float total = oasr::reduction::blockReduceSum<float>(local_sum);
    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = total / static_cast<float>(win_length);
    }
    __syncthreads();

    float* out = output + static_cast<int64_t>(frame_idx) * n_fft;
    for (int i = tid; i < n_fft; i += blockDim.x) {
        out[i] = 0.0f;
    }
    __syncthreads();
    for (int w = tid; w < win_length; w += blockDim.x) {
        const float x = frame[w] - s_mean;
        float y = x;
        if (preemph_coef != 0.0f) {
            const float prev = w > 0 ? frame[w - 1] - s_mean : (preemph_replicate ? x : 0.0f);
            y -= preemph_coef * prev;
        }
        out[win_offset + w] = y * window[w];
    }
}

inline cudaError_t StftFrame(const float* waveform, const int32_t* lengths, const float* window,
                             float* output, int batch, int wav_stride, int num_frames, int n_fft,
                             int win_length, int win_offset, int hop_length, int center_offset,
                             float preemph_coef, bool preemph_replicate, bool remove_dc_offset,
                             int signal_length, bool reflect_pad, cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(batch) * static_cast<int64_t>(num_frames) *
                          static_cast<int64_t>(n_fft);
    if (total == 0) {
        return cudaSuccess;
    }
    const int threads = 256;
    if (remove_dc_offset) {
        const int total_frames = batch * num_frames;
        const size_t smem_bytes = static_cast<size_t>(win_length) * sizeof(float);
        StftFrameDcKernel<<<total_frames, threads, smem_bytes, stream>>>(
            waveform, lengths, window, output, wav_stride, num_frames, n_fft, win_length,
            win_offset, hop_length, center_offset, preemph_coef, preemph_replicate ? 1 : 0,
            signal_length, reflect_pad ? 1 : 0);
        return cudaGetLastError();
    }
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 65535) {
        blocks = 65535;  // grid-stride handles the remainder
    }
    StftFrameKernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
        waveform, lengths, window, output, total, wav_stride, num_frames, n_fft, win_length,
        win_offset, hop_length, center_offset, preemph_coef, preemph_replicate ? 1 : 0,
        signal_length, reflect_pad ? 1 : 0);
    return cudaGetLastError();
}

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
                                             int n_fft, float preemph_coef, int remove_dc_offset,
                                             int apply_preemph) {
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
                                   float preemph_coef, bool remove_dc_offset, bool apply_preemph,
                                   cudaStream_t stream) {
    const int threads = 256;
    const size_t smem_bytes = static_cast<size_t>(frame_length) * sizeof(float);
    FbankPreprocessKernel<<<total_frames, threads, smem_bytes, stream>>>(
        frames, window, output, frame_length, n_fft, preemph_coef, remove_dc_offset ? 1 : 0,
        apply_preemph ? 1 : 0);
    return cudaGetLastError();
}

// =============================================================================
// 2. Mel filterbank + log-floor.
// =============================================================================
//
// Computes log(max(mel_mat @ power, log_floor) + log_offset). Floor and additive
// guard remain separate because supported frontends use different silence
// scales. Invalid frames are zeroed after log; otherwise padded silence becomes
// a large negative feature value.
__global__ inline void MelLogKernel(const float* __restrict__ power,
                                    const float* __restrict__ mel_mat,
                                    const int32_t* __restrict__ frame_lengths,
                                    float* __restrict__ output, int num_freq, int num_mel,
                                    int frames_per_row, float log_floor, float log_offset) {
    extern __shared__ float spec[];

    const int frame_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wid = tid >> 5;
    const int n_warps = bs >> 5;

    float* out_ptr = output + static_cast<int64_t>(frame_idx) * num_mel;

    if (frame_lengths != nullptr) {
        const int row = frame_idx / frames_per_row;
        if (frame_idx - row * frames_per_row >= frame_lengths[row]) {
            for (int b = tid; b < num_mel; b += bs) {
                out_ptr[b] = 0.0f;
            }
            return;
        }
    }

    const float* in_ptr = power + static_cast<int64_t>(frame_idx) * num_freq;
    for (int i = tid; i < num_freq; i += bs) {
        spec[i] = in_ptr[i];
    }
    __syncthreads();

    for (int b = wid; b < num_mel; b += n_warps) {
        const float* fb = mel_mat + static_cast<int64_t>(b) * num_freq;
        float acc = 0.0f;
        for (int i = lane; i < num_freq; i += WARP_SIZE) {
            acc += fb[i] * spec[i];
        }
        acc = oasr::reduction::warpReduceSum(acc);
        if (lane == 0) {
            if (acc < log_floor) {
                acc = log_floor;
            }
            out_ptr[b] = logf(acc + log_offset);
        }
    }
}

inline cudaError_t MelLog(const float* power, const float* mel_mat, const int32_t* frame_lengths,
                          float* output, int total_frames, int num_freq, int num_mel,
                          int frames_per_row, float log_floor, float log_offset,
                          cudaStream_t stream) {
    const int threads = 128;  // 4 warps
    const size_t smem_bytes = static_cast<size_t>(num_freq) * sizeof(float);
    MelLogKernel<<<total_frames, threads, smem_bytes, stream>>>(
        power, mel_mat, frame_lengths, output, num_freq, num_mel, frames_per_row, log_floor,
        log_offset);
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
                                       const float* __restrict__ energy, float* __restrict__ output,
                                       int num_mel, int num_ceps, int replace_c0_with_energy) {
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

// =============================================================================
// 5. Whisper mel + log10 + per-utterance max-floor normalization.
// =============================================================================

__global__ inline void WhisperMelLogKernel(const float* __restrict__ power,
                                           const float* __restrict__ mel_mat,
                                           float* __restrict__ output, int num_freq, int num_mel,
                                           float log_floor) {
    extern __shared__ float spec[];

    const int frame_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wid = tid >> 5;
    const int n_warps = blockDim.x >> 5;
    const float* in = power + static_cast<int64_t>(frame_idx) * num_freq;
    float* out = output + static_cast<int64_t>(frame_idx) * num_mel;

    for (int i = tid; i < num_freq; i += blockDim.x) {
        spec[i] = in[i];
    }
    __syncthreads();

    for (int m = wid; m < num_mel; m += n_warps) {
        const float* filter = mel_mat + static_cast<int64_t>(m) * num_freq;
        float acc = 0.0f;
        for (int i = lane; i < num_freq; i += WARP_SIZE) {
            acc += filter[i] * spec[i];
        }
        acc = oasr::reduction::warpReduceSum(acc);
        if (lane == 0) {
            out[m] = log10f(acc < log_floor ? log_floor : acc);
        }
    }
}

__global__ inline void WhisperNormalizeKernel(float* __restrict__ output, int row_elems,
                                              float max_floor, float offset, float scale) {
    float* row = output + static_cast<int64_t>(blockIdx.x) * row_elems;
    float local_max = -3.402823466e+38F;
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }
    for (int delta = WARP_SIZE / 2; delta > 0; delta /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, delta));
    }
    __shared__ float warp_max[32];
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_max[warp] = local_max;
    }
    __syncthreads();
    if (warp == 0) {
        local_max = lane < blockDim.x / WARP_SIZE ? warp_max[lane] : -3.402823466e+38F;
        for (int delta = WARP_SIZE / 2; delta > 0; delta /= 2) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, delta));
        }
    }
    __shared__ float s_floor;
    if (threadIdx.x == 0) {
        s_floor = local_max - max_floor;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        row[i] = (fmaxf(row[i], s_floor) + offset) * scale;
    }
}

inline cudaError_t WhisperLogMel(const float* power, const float* mel_mat, float* output, int batch,
                                 int num_frames, int num_freq, int num_mel, float log_floor,
                                 float max_floor, float offset, float scale, cudaStream_t stream) {
    const int threads = 256;
    WhisperMelLogKernel<<<batch * num_frames, threads,
                          static_cast<size_t>(num_freq) * sizeof(float), stream>>>(
        power, mel_mat, output, num_freq, num_mel, log_floor);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    WhisperNormalizeKernel<<<batch, threads, 0, stream>>>(output, num_frames * num_mel, max_floor,
                                                          offset, scale);
    return cudaGetLastError();
}

// =============================================================================
// 6. Varlen low-frame-rate gather.
// =============================================================================

template <typename T>
__global__ void LfrGatherKernel(const T* __restrict__ input, const int32_t* __restrict__ lengths,
                                T* __restrict__ output, int64_t total_elems, int input_frames,
                                int output_frames, int feature_dim, int lfr_m, int lfr_n) {
    const int stacked_dim = feature_dim * lfr_m;
    const int left = (lfr_m - 1) / 2;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_elems; idx += stride) {
        const int d = static_cast<int>(idx % stacked_dim);
        const int64_t row = idx / stacked_dim;
        const int t = static_cast<int>(row % output_frames);
        const int b = static_cast<int>(row / output_frames);
        const int length = lengths[b];
        const int valid_output = length > 0 ? (length + lfr_n - 1) / lfr_n : 0;
        if (t >= valid_output) {
            output[idx] = static_cast<T>(0.0f);
            continue;
        }
        const int stack_slot = d / feature_dim;
        const int f = d - stack_slot * feature_dim;
        int source_t = t * lfr_n + stack_slot - left;
        source_t = max(0, min(source_t, min(length, input_frames) - 1));
        const int64_t source =
            (static_cast<int64_t>(b) * input_frames + source_t) * feature_dim + f;
        output[idx] = input[source];
    }
}

template <typename T>
inline cudaError_t LfrGather(const T* input, const int32_t* lengths, T* output, int batch,
                             int input_frames, int output_frames, int feature_dim, int lfr_m,
                             int lfr_n, cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(batch) * output_frames * feature_dim * lfr_m;
    if (total == 0) {
        return cudaSuccess;
    }
    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 65535) {
        blocks = 65535;
    }
    LfrGatherKernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
        input, lengths, output, total, input_frames, output_frames, feature_dim, lfr_m, lfr_n);
    return cudaGetLastError();
}

}  // namespace features
}  // namespace oasr
