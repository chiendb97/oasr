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

#include <cstdint>

#include <oasr/common/reduction.h>

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
//
// Three launches over one `(batch, num_frames, num_mel)` buffer:
//
//   InitRowMax        row_max[b] = -FLT_MAX
//   WhisperMelLog     mel projection + log10, and the per-row max as a
//                     by-product of the tile each block already holds
//   WhisperNormalize  floor at `row_max - max_floor`, then the affine
//
// The row maximum is a reduction over a *whole utterance* (num_frames * num_mel
// values), so it cannot be folded into the projection's own block and cannot be
// recomputed per output element.  Carrying it on an `atomicMax` costs one atomic
// per block and keeps the normalization a flat elementwise pass -- the previous
// form reduced and rewrote with one block per batch row, which is `batch` blocks
// of work however large the GPU is.  `max` is exact and associative, so the
// atomic changes nothing about the result, including bit-for-bit.
//
// The projection tiles `TILE_F` consecutive frames per block.  One frame per
// block makes every mel bin its own warp reduction and re-reads the filterbank
// for each frame; a tile amortizes both over `TILE_F` dot products that share
// one filter load.  The tile never straddles a batch row (gridDim.y is the
// batch), which is what lets one block own a piece of exactly one row max.

//: Frames per projection block.  The tile amortizes the filterbank load, which
//: is otherwise re-read from L2 once per frame: 16 * 201 * 4 = 12.9 KB of
//: shared memory for Whisper's 400-point transform.
constexpr int kMelTileFrames = 16;

__device__ __forceinline__ void AtomicMaxFloat(float* address, float value) {
    // Ordered on each sign domain: non-negative floats compare as signed ints,
    // negative floats compare in reverse as unsigned.  Both start from -FLT_MAX,
    // whose bit pattern is the largest unsigned of the negative domain and a
    // negative signed int, so either branch admits any first writer.
    //
    // -0.0 is the one value the split mishandles: it takes the non-negative
    // branch, where its bit pattern is INT_MIN and therefore loses to every
    // other candidate including the initial -FLT_MAX.  It is unreachable from
    // log10f, but the helper should not depend on knowing that.
    if (value == 0.0f) {
        value = 0.0f;
    }
    if (value >= 0.0f) {
        atomicMax(reinterpret_cast<int*>(address), __float_as_int(value));
    } else {
        atomicMin(reinterpret_cast<unsigned int*>(address), __float_as_uint(value));
    }
}

__global__ inline void InitRowMaxKernel(float* __restrict__ row_max, int batch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch) {
        row_max[idx] = -3.402823466e+38F;
    }
}

// `input` is either a `(batch, num_frames, num_freq)` power spectrum or, when
// `is_complex`, the interleaved real/imaginary halves of the transform itself.
// Reading the complex spectrum directly is what removes `abs().square()`: two
// full-size elementwise passes and a full-size temporary, for two multiplies
// inside a staging loop that already pays the load.
//
// `mel_mat` is the projection matrix in `(num_freq, num_mel)` order, so one
// thread owns one mel bin and reads its filter weights coalesced across the
// warp.  The alternative -- a warp per mel bin, striding the filter along
// frequency -- spends five shuffles on every output, which at Whisper's
// 201-point spectrum is half the kernel's instructions.
template <int TILE_F>
__global__ void WhisperMelLogKernel(const float* __restrict__ input,
                                    const float* __restrict__ mel_mat,
                                    float* __restrict__ output, float* __restrict__ row_max,
                                    int num_frames, int num_freq, int num_mel, float log_floor,
                                    int is_complex) {
    extern __shared__ float smem[];
    float* spec = smem;                          // TILE_F * num_freq
    float* warp_max = smem + TILE_F * num_freq;  // blockDim.x / WARP_SIZE

    const int batch_idx = blockIdx.y;
    const int frame_base = blockIdx.x * TILE_F;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int64_t row_base = static_cast<int64_t>(batch_idx) * num_frames;

    // Stage the tile.  Frames past the end stage zeros so every thread can run
    // the same unrolled dot product; their outputs are simply never stored.
    const int tile_elems = TILE_F * num_freq;
    for (int idx = tid; idx < tile_elems; idx += bs) {
        const int t = idx / num_freq;
        const int i = idx - t * num_freq;
        const int f = frame_base + t;
        float v = 0.0f;
        if (f < num_frames) {
            const int64_t offset = (row_base + f) * num_freq + i;
            if (is_complex) {
                const float2 z = reinterpret_cast<const float2*>(input)[offset];
                v = z.x * z.x + z.y * z.y;
            } else {
                v = input[offset];
            }
        }
        spec[idx] = v;
    }
    __syncthreads();

    float local_max = -3.402823466e+38F;
    for (int m = tid; m < num_mel; m += bs) {
        float acc[TILE_F];
#pragma unroll
        for (int t = 0; t < TILE_F; ++t) {
            acc[t] = 0.0f;
        }
        // `mel_mat[i][m]` is coalesced across the warp; `spec[t][i]` is one
        // address for every thread, which shared memory broadcasts.
        const float* filter = mel_mat + m;
        for (int i = 0; i < num_freq; ++i) {
            const float fv = filter[static_cast<int64_t>(i) * num_mel];
            const float* row = spec + i;
#pragma unroll
            for (int t = 0; t < TILE_F; ++t) {
                acc[t] += fv * row[t * num_freq];
            }
        }
#pragma unroll
        for (int t = 0; t < TILE_F; ++t) {
            const int f = frame_base + t;
            if (f < num_frames) {
                const float v = log10f(acc[t] < log_floor ? log_floor : acc[t]);
                output[(row_base + f) * num_mel + m] = v;
                local_max = fmaxf(local_max, v);
            }
        }
    }

    // One atomic per block: reduce across the warp, then across the block.
    for (int delta = WARP_SIZE / 2; delta > 0; delta /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, delta));
    }
    const int lane = tid & (WARP_SIZE - 1);
    const int wid = tid >> 5;
    if (lane == 0) {
        warp_max[wid] = local_max;
    }
    __syncthreads();
    if (tid == 0) {
        const int n_warps = bs >> 5;
        float block_max = warp_max[0];
        for (int w = 1; w < n_warps; ++w) {
            block_max = fmaxf(block_max, warp_max[w]);
        }
        AtomicMaxFloat(row_max + batch_idx, block_max);
    }
}

__global__ inline void WhisperNormalizeKernel(float* __restrict__ output,
                                              const float* __restrict__ row_max, int row_elems,
                                              float max_floor, float offset, float scale) {
    const int batch_idx = blockIdx.y;
    const float floor_value = row_max[batch_idx] - max_floor;
    float* row = output + static_cast<int64_t>(batch_idx) * row_elems;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < row_elems; i += stride) {
        row[i] = (fmaxf(row[i], floor_value) + offset) * scale;
    }
}

inline cudaError_t WhisperLogMel(const float* input, const float* mel_mat, float* output,
                                 float* row_max, int batch, int num_frames, int num_freq,
                                 int num_mel, float log_floor, float max_floor, float offset,
                                 float scale, bool is_complex, cudaStream_t stream) {
    constexpr int kInitThreads = 256;
    const int init_blocks = (batch + kInitThreads - 1) / kInitThreads;
    InitRowMaxKernel<<<init_blocks, kInitThreads, 0, stream>>>(row_max, batch);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    // One thread per mel bin, rounded to whole warps: 80 mels take 3 warps and
    // 128 take 4, rather than a fixed block leaving a third of its lanes idle.
    int threads = ((num_mel + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    threads = threads < WARP_SIZE ? WARP_SIZE : (threads > 256 ? 256 : threads);
    const int n_warps = threads / WARP_SIZE;
    const size_t smem =
        (static_cast<size_t>(kMelTileFrames) * num_freq + n_warps) * sizeof(float);
    const dim3 grid((num_frames + kMelTileFrames - 1) / kMelTileFrames, batch);
    WhisperMelLogKernel<kMelTileFrames><<<grid, threads, smem, stream>>>(
        input, mel_mat, output, row_max, num_frames, num_freq, num_mel, log_floor,
        is_complex ? 1 : 0);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    constexpr int kNormThreads = 256;
    const int row_elems = num_frames * num_mel;
    const int tile_blocks = (row_elems + kNormThreads - 1) / kNormThreads;
    const int norm_blocks = tile_blocks < 1024 ? tile_blocks : 1024;
    WhisperNormalizeKernel<<<dim3(norm_blocks, batch), kNormThreads, 0, stream>>>(
        output, row_max, row_elems, max_floor, offset, scale);
    return cudaGetLastError();
}

// =============================================================================
// 6. Varlen low-frame-rate gather.
// =============================================================================

//: All-zero bytes is zero for every dtype this kernel moves, but the transfer
//: type is not always the element type, so the constant is a trait.
template <typename V>
struct LfrZero {
    __device__ __forceinline__ static V value() { return static_cast<V>(0.0f); }
};
template <>
struct LfrZero<int4> {
    __device__ __forceinline__ static int4 value() { return make_int4(0, 0, 0, 0); }
};

// LFR stacking is a gather and nothing else, so every instruction that is not a
// load or a store is waste.  The grid carries the batch and the output frame,
// and a warp owns one stack slot, which leaves the inner loop with a single
// running index: no integer division anywhere on the element path.  Expressing
// the same mapping as one flat index over `(b, t, slot, f)` costs five integer
// divisions per output element -- two of them 64-bit -- and turns a
// bandwidth-bound copy into an arithmetic-bound one.
//
// `V` is the transfer type, not the element type: a 16-byte vector whenever the
// feature dimension and the pointers allow it, otherwise the element itself.
template <typename V>
__global__ void LfrGatherKernel(const V* __restrict__ input,
                                const int32_t* __restrict__ lengths, V* __restrict__ output,
                                int input_frames, int output_frames, int feature_vecs, int lfr_m,
                                int lfr_n) {
    const int b = blockIdx.y;
    const int left = (lfr_m - 1) / 2;
    const int length = lengths[b];
    // One division per block, on a value every thread in it shares.
    const int valid_output = length > 0 ? (length + lfr_n - 1) / lfr_n : 0;
    const int upper = (length < input_frames ? length : input_frames) - 1;

    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wid = tid >> 5;
    const int n_warps = bs >> 5;
    const int row_vecs = feature_vecs * lfr_m;

    for (int t = blockIdx.x; t < output_frames; t += gridDim.x) {
        V* out_row = output + (static_cast<int64_t>(b) * output_frames + t) * row_vecs;
        if (t >= valid_output) {
            // Past this row's own LFR length.  The gather would otherwise
            // replicate the last valid frame into the padding.
            for (int i = tid; i < row_vecs; i += bs) {
                out_row[i] = LfrZero<V>::value();
            }
            continue;
        }
        for (int slot = wid; slot < lfr_m; slot += n_warps) {
            // FunASR's `apply_lfr`: `left` copies of the first frame prepended,
            // and the trailing partial window completed with the last valid
            // frame -- both are this clamp.
            int source_t = t * lfr_n + slot - left;
            source_t = source_t < 0 ? 0 : (source_t > upper ? upper : source_t);
            const V* src =
                input + (static_cast<int64_t>(b) * input_frames + source_t) * feature_vecs;
            V* dst = out_row + slot * feature_vecs;
            for (int i = lane; i < feature_vecs; i += WARP_SIZE) {
                dst[i] = src[i];
            }
        }
    }
}

template <typename T>
inline cudaError_t LfrGather(const T* input, const int32_t* lengths, T* output, int batch,
                             int input_frames, int output_frames, int feature_dim, int lfr_m,
                             int lfr_n, cudaStream_t stream) {
    if (batch == 0 || output_frames == 0 || feature_dim == 0) {
        return cudaSuccess;
    }
    const int threads = 256;
    // gridDim.x is capped; the kernel strides over the remaining frames.
    const int grid_x = output_frames < 65535 ? output_frames : 65535;
    const dim3 grid(grid_x, batch);

    // Every offset the kernel forms is a whole number of feature rows, so a
    // 16-byte transfer is safe as soon as one feature row is a whole number of
    // them and the two base pointers are aligned.  A batch-sliced view is the
    // case that is not, hence the runtime check rather than a static one.
    constexpr int kVecElems = static_cast<int>(16 / sizeof(T));
    const bool aligned = (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(output) % 16 == 0);
    if (kVecElems > 0 && feature_dim % kVecElems == 0 && aligned) {
        LfrGatherKernel<int4><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const int4*>(input), lengths, reinterpret_cast<int4*>(output),
            input_frames, output_frames, feature_dim / kVecElems, lfr_m, lfr_n);
    } else {
        LfrGatherKernel<T><<<grid, threads, 0, stream>>>(
            input, lengths, output, input_frames, output_frames, feature_dim, lfr_m, lfr_n);
    }
    return cudaGetLastError();
}

}  // namespace features
}  // namespace oasr
