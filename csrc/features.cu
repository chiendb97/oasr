// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// TVM-FFI launcher layer for the FBANK / MFCC feature-extraction kernels.

#include <oasr/features.cuh>

#include "tvm_ffi_utils.h"

using namespace oasr;

namespace {

inline void check_fp32(const TensorView& t, const char* what) {
    TVM_FFI_ICHECK(t.dtype().code == kDLFloat && t.dtype().bits == 32)
        << what << " must be float32";
}

inline int product_until_last(const TensorView& t) {
    int p = 1;
    for (int i = 0; i < t.ndim() - 1; ++i) {
        p *= t.size(i);
    }
    return p;
}

}  // namespace

// ---------------------------------------------------------------------------
// fbank_preprocess(output, frames, window, preemph, remove_dc, apply_preemph)
//   frames : (..., frame_length) float32
//   window : (frame_length,)     float32
//   output : (..., n_fft)        float32
// ---------------------------------------------------------------------------
void fbank_preprocess(TensorView output, TensorView frames, TensorView window,
                      double preemph_coef, bool remove_dc_offset, bool apply_preemph) {
    CHECK_INPUT(frames);
    CHECK_INPUT(window);
    CHECK_INPUT(output);
    CHECK_DEVICE(frames, output);
    CHECK_DEVICE(frames, window);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(frames);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
    check_fp32(frames, "frames");
    check_fp32(window, "window");
    check_fp32(output, "output");

    TVM_FFI_ICHECK(window.ndim() == 1) << "window must be 1-D";
    TVM_FFI_ICHECK(frames.ndim() >= 2) << "frames must be (..., frame_length)";
    TVM_FFI_ICHECK(output.ndim() == frames.ndim())
        << "output ndim must match frames ndim";

    const int frame_length = static_cast<int>(frames.size(frames.ndim() - 1));
    const int n_fft = static_cast<int>(output.size(output.ndim() - 1));
    TVM_FFI_ICHECK(window.size(0) == frame_length)
        << "window length (" << window.size(0) << ") must equal frame_length ("
        << frame_length << ")";
    TVM_FFI_ICHECK(n_fft >= frame_length)
        << "output last dim n_fft (" << n_fft << ") must be >= frame_length ("
        << frame_length << ")";

    const int total_frames = product_until_last(frames);
    TVM_FFI_ICHECK(product_until_last(output) == total_frames)
        << "output and frames must have matching leading dims";

    cudaStream_t stream = get_stream(frames.device());
    cudaError_t status = features::FbankPreprocess(
        static_cast<const float*>(frames.data_ptr()),
        static_cast<const float*>(window.data_ptr()),
        static_cast<float*>(output.data_ptr()), total_frames, frame_length, n_fft,
        static_cast<float>(preemph_coef), remove_dc_offset, apply_preemph, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "fbank_preprocess kernel failed: " << cudaGetErrorString(status);
}

// ---------------------------------------------------------------------------
// mel_log(output, power, mel_mat, log_floor)
//   power   : (..., n_fft/2+1)               float32
//   mel_mat : (num_mel, n_fft/2+1)           float32
//   output  : (..., num_mel)                 float32
// ---------------------------------------------------------------------------
void mel_log(TensorView output, TensorView power, TensorView mel_mat, double log_floor) {
    CHECK_INPUT(power);
    CHECK_INPUT(mel_mat);
    CHECK_INPUT(output);
    CHECK_DEVICE(power, output);
    CHECK_DEVICE(power, mel_mat);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(power);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(mel_mat);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
    check_fp32(power, "power");
    check_fp32(mel_mat, "mel_mat");
    check_fp32(output, "output");

    TVM_FFI_ICHECK(mel_mat.ndim() == 2) << "mel_mat must be 2-D";
    TVM_FFI_ICHECK(power.ndim() >= 2) << "power must be (..., n_freq)";
    TVM_FFI_ICHECK(output.ndim() == power.ndim())
        << "output ndim must match power ndim";

    const int num_freq = static_cast<int>(power.size(power.ndim() - 1));
    const int num_mel = static_cast<int>(output.size(output.ndim() - 1));
    TVM_FFI_ICHECK(mel_mat.size(0) == num_mel && mel_mat.size(1) == num_freq)
        << "mel_mat shape (" << mel_mat.size(0) << ", " << mel_mat.size(1)
        << ") must equal (num_mel=" << num_mel << ", num_freq=" << num_freq << ")";

    const int total_frames = product_until_last(power);
    TVM_FFI_ICHECK(product_until_last(output) == total_frames)
        << "output and power must have matching leading dims";

    cudaStream_t stream = get_stream(power.device());
    cudaError_t status = features::MelLog(static_cast<const float*>(power.data_ptr()),
                                          static_cast<const float*>(mel_mat.data_ptr()),
                                          static_cast<float*>(output.data_ptr()),
                                          total_frames, num_freq, num_mel,
                                          static_cast<float>(log_floor), stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "mel_log kernel failed: " << cudaGetErrorString(status);
}

// ---------------------------------------------------------------------------
// dct_lifter(output, log_mel, dct_mat, lifter, energy, replace_c0_with_energy)
//   log_mel : (..., num_mel)         float32
//   dct_mat : (num_ceps, num_mel)    float32
//   lifter  : optional (num_ceps,)   float32 -- pass empty tensor (numel=0) to skip
//   energy  : optional (total_frames,) float32 -- pass empty tensor to skip
//   output  : (..., num_ceps)        float32
// ---------------------------------------------------------------------------
void dct_lifter(TensorView output, TensorView log_mel, TensorView dct_mat,
                Optional lifter_opt, Optional energy_opt,
                bool replace_c0_with_energy) {
    CHECK_INPUT(log_mel);
    CHECK_INPUT(dct_mat);
    CHECK_INPUT(output);
    CHECK_DEVICE(log_mel, output);
    CHECK_DEVICE(log_mel, dct_mat);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(log_mel);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(dct_mat);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
    check_fp32(log_mel, "log_mel");
    check_fp32(dct_mat, "dct_mat");
    check_fp32(output, "output");

    TVM_FFI_ICHECK(dct_mat.ndim() == 2) << "dct_mat must be 2-D";
    TVM_FFI_ICHECK(log_mel.ndim() >= 2) << "log_mel must be (..., num_mel)";
    TVM_FFI_ICHECK(output.ndim() == log_mel.ndim())
        << "output ndim must match log_mel ndim";

    const int num_mel = static_cast<int>(log_mel.size(log_mel.ndim() - 1));
    const int num_ceps = static_cast<int>(output.size(output.ndim() - 1));
    TVM_FFI_ICHECK(dct_mat.size(0) == num_ceps && dct_mat.size(1) == num_mel)
        << "dct_mat shape (" << dct_mat.size(0) << ", " << dct_mat.size(1)
        << ") must equal (num_ceps=" << num_ceps << ", num_mel=" << num_mel << ")";

    const int total_frames = product_until_last(log_mel);
    TVM_FFI_ICHECK(product_until_last(output) == total_frames)
        << "output and log_mel must have matching leading dims";

    const float* lifter_ptr = nullptr;
    if (lifter_opt.has_value()) {
        const TensorView lifter = lifter_opt.value();
        check_fp32(lifter, "lifter");
        TVM_FFI_ICHECK(lifter.size(0) == num_ceps)
            << "lifter length (" << lifter.size(0) << ") must equal num_ceps (" << num_ceps
            << ")";
        lifter_ptr = static_cast<const float*>(lifter.data_ptr());
    }

    const float* energy_ptr = nullptr;
    if (energy_opt.has_value()) {
        const TensorView energy = energy_opt.value();
        check_fp32(energy, "energy");
        TVM_FFI_ICHECK(energy.numel() == total_frames)
            << "energy must have total_frames=" << total_frames << " elements, got "
            << energy.numel();
        energy_ptr = static_cast<const float*>(energy.data_ptr());
    }

    cudaStream_t stream = get_stream(log_mel.device());
    cudaError_t status = features::DctLifter(static_cast<const float*>(log_mel.data_ptr()),
                                             static_cast<const float*>(dct_mat.data_ptr()),
                                             lifter_ptr, energy_ptr,
                                             static_cast<float*>(output.data_ptr()),
                                             total_frames, num_mel, num_ceps,
                                             replace_c0_with_energy, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "dct_lifter kernel failed: " << cudaGetErrorString(status);
}
