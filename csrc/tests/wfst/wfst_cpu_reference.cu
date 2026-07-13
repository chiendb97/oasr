// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Test-only TVM-FFI launcher for the exact-semantics WFST CPU reference oracle
// (see csrc/tests/wfst/cpu_reference.h).  Compiled into a SEPARATE JIT module
// (oasr.jit.wfst_decoder.gen_wfst_cpu_reference_module) so the production
// `wfst_decoder` module carries no reference-decoder code.
//
// Unlike the production launcher, this entry point is fully self-contained: it takes
// a graph-image *path* and loads the GraphImage on the host itself, so nothing here
// depends on the production launcher's opaque graph handle or internals.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <tvm/ffi/string.h>

#include "decoder/wfst/config.h"
#include "decoder/wfst/graph.h"
#include "tests/wfst/cpu_reference.h"
#include "tvm_ffi_utils.h"

namespace {

using oasr::wfst::CpuDecode;
using oasr::wfst::CpuDecodeResult;
using oasr::wfst::DecoderConfig;
using oasr::wfst::GraphImage;
using oasr::wfst::LoadGraphImage;

inline void check_cpu(TensorView t, const char* what) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCPU) << what << " must be a CPU tensor";
}

// Copy up to `cap` words into `dst`; write the TRUE (unclamped) word count into
// `*len_dst` so the caller can detect truncation.
inline void write_words(int32_t* dst, int64_t cap, int32_t* len_dst,
                        const std::vector<int32_t>& words) {
  int64_t n = std::min<int64_t>(static_cast<int64_t>(words.size()), cap);
  for (int64_t i = 0; i < n; ++i) dst[i] = words[i];
  *len_dst = static_cast<int32_t>(words.size());
}

}  // namespace

// CPU reference (exact-semantics oracle, used by tests). Loads the graph image at
// `graph_path`, runs the host reference decoder over CPU float32 log_probs [T, V], and
// marshals the best path's words/score into caller-provided CPU output tensors
// (out_meta = {ok, reached_final, 0}).
void wfst_cpu_decode(tvm::ffi::String graph_path, TensorView log_probs, double search_beam,
                     double output_beam, int64_t min_active, int64_t max_active,
                     int64_t allow_partial, int64_t online, int64_t eps_iterations,
                     TensorView out_words, TensorView out_word_len, TensorView out_score,
                     TensorView out_meta) {
  std::unique_ptr<GraphImage> graph =
      LoadGraphImage(std::string(graph_path.data(), graph_path.size()));
  TVM_FFI_ICHECK(graph != nullptr) << "failed to load graph image";
  check_cpu(log_probs, "log_probs");
  TVM_FFI_ICHECK(log_probs.ndim() == 2) << "log_probs must be [T, V]";
  const int64_t num_frames = log_probs.size(0);
  const int64_t vocab = log_probs.size(1);

  DecoderConfig cfg;
  cfg.search_beam = static_cast<float>(search_beam);
  cfg.output_beam = static_cast<float>(output_beam);
  cfg.min_active_states = static_cast<int32_t>(min_active);
  cfg.max_active_states = static_cast<int32_t>(max_active);
  cfg.allow_partial = allow_partial != 0;
  cfg.eps_iterations = static_cast<int32_t>(eps_iterations);

  CpuDecodeResult r = CpuDecode(*graph, static_cast<const float*>(log_probs.data_ptr()),
                                static_cast<int32_t>(num_frames), static_cast<int32_t>(vocab),
                                cfg, online != 0);
  const int64_t cap = out_words.size(out_words.ndim() - 1);
  write_words(static_cast<int32_t*>(out_words.data_ptr()), cap,
              static_cast<int32_t*>(out_word_len.data_ptr()), r.words);
  static_cast<double*>(out_score.data_ptr())[0] = r.score;
  int32_t* meta = static_cast<int32_t*>(out_meta.data_ptr());
  meta[0] = r.ok ? 1 : 0;
  meta[1] = r.reached_final ? 1 : 0;
  meta[2] = 0;
}
