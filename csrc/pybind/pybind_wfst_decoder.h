// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// pybind11 bindings for the in-tree GPU WFST beam-search decoder.  Migrated from
// the standalone `wfst` project's csrc/pybind/wfst_pybind.cc and re-homed into the
// oasr._C.decoder submodule via registerWfstDecoderBindings(), which pybind_decoder.h
// calls under OASR_USE_WFST_DECODER.  The exposed names (Graph / load_graph /
// cpu_decode / GpuDecoder) mirror the original _wfst module 1:1 so the in-tree API
// is a drop-in for the external decoder.

#pragma once

#include <torch/extension.h>

#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "oasr/wfst/config.h"
#include "decoder/wfst/cpu_reference.h"
#include "decoder/wfst/decoder.h"
#include "decoder/wfst/graph.h"

namespace py = pybind11;

namespace oasr {
namespace pybind {
namespace wfst_decoder {

using namespace oasr::wfst;

inline DecoderConfig ConfigFromKwargs(float search_beam, float output_beam, int32_t min_active,
                                      int32_t max_active, bool allow_partial) {
  DecoderConfig cfg;
  cfg.search_beam = search_beam;
  cfg.output_beam = output_beam;
  cfg.min_active_states = min_active;
  cfg.max_active_states = max_active;
  cfg.allow_partial = allow_partial;
  return cfg;
}

inline py::dict CpuDecodeWrapper(const GraphImage& g, torch::Tensor log_probs, float search_beam,
                                 float output_beam, int32_t min_active, int32_t max_active,
                                 bool allow_partial, bool online, int32_t eps_iterations) {
  TORCH_CHECK(log_probs.dim() == 2, "log_probs must be [T, V]");
  TORCH_CHECK(log_probs.device().is_cpu(), "cpu_decode wants a cpu tensor");
  log_probs = log_probs.contiguous().to(torch::kFloat32);
  auto cfg = ConfigFromKwargs(search_beam, output_beam, min_active, max_active, allow_partial);
  cfg.eps_iterations = eps_iterations;
  TORCH_CHECK(log_probs.size(1) >= g.vocab_size, "log_probs vocab dim (", log_probs.size(1),
              ") smaller than graph vocab (", g.vocab_size, ")");
  CpuDecodeResult res =
      CpuDecode(g, log_probs.data_ptr<float>(), static_cast<int32_t>(log_probs.size(0)),
                static_cast<int32_t>(log_probs.size(1)), cfg, online);

  py::dict out;
  out["ok"] = res.ok;
  out["reached_final"] = res.reached_final;
  out["score"] = res.score;
  out["arc_path"] = res.arc_path;
  out["words"] = res.words;
  py::list frames;
  for (const auto& frame : res.frames) {
    py::list f;
    for (const auto& [state, score] : frame) f.append(py::make_tuple(state, score));
    frames.append(f);
  }
  out["frames"] = frames;
  return out;
}

// Owns one GpuDecoder + a shared_ptr to the graph image it was built from.  The
// shared_ptr keeps the host graph alive for the decoder's whole lifetime — the
// decoder stores a borrowed pointer into it for aux (word) lookups during backtrack.
class PyGpuDecoder {
 public:
  PyGpuDecoder(std::shared_ptr<GraphImage> graph, float search_beam, float output_beam,
               int32_t min_active, int32_t max_active, bool allow_partial, int32_t max_lanes,
               int32_t max_frames, int device, bool debug_snapshots, int32_t main_q_factor,
               int32_t cand_factor, bool use_cuda_graphs, bool lattice, bool fp16_logprobs,
               bool streaming, int32_t lat_prune_interval, int32_t eps_iterations)
      : graph_(std::move(graph)) {
    GpuDecoder::Options opts;
    opts.cfg = ConfigFromKwargs(search_beam, output_beam, min_active, max_active, allow_partial);
    opts.cfg.max_lanes = max_lanes;
    opts.cfg.max_frames = max_frames;
    opts.cfg.main_q_factor = main_q_factor;
    opts.cfg.cand_factor = cand_factor;
    opts.cfg.lattice = lattice;
    opts.cfg.fp16_logprobs = fp16_logprobs;
    opts.cfg.streaming = streaming;
    opts.cfg.lat_prune_interval = lat_prune_interval;
    opts.cfg.eps_iterations = eps_iterations;
    fp16_ = fp16_logprobs;
    opts.device = device;
    opts.debug_snapshots = debug_snapshots;
    opts.use_cuda_graphs = use_cuda_graphs;
    decoder_ = std::make_unique<GpuDecoder>(*graph_, opts);
  }

  int32_t CreateStream() { return decoder_->CreateStream(); }
  void ReleaseStream(int32_t channel) { decoder_->ReleaseStream(channel); }

  py::list AdvanceChunk(std::vector<int32_t> channels, torch::Tensor log_probs,
                        torch::Tensor lengths, bool partial) {
    TORCH_CHECK(log_probs.dim() == 3, "log_probs must be [B, Tc, V]");
    TORCH_CHECK(log_probs.is_cuda(), "log_probs must be on GPU");
    const auto want = fp16_ ? torch::kFloat16 : torch::kFloat32;
    TORCH_CHECK(log_probs.scalar_type() == want, "log_probs dtype mismatch");
    log_probs = log_probs.contiguous();
    lengths = lengths.to(torch::kInt32).cpu().contiguous();
    TORCH_CHECK(lengths.numel() == static_cast<int64_t>(channels.size()));
    std::vector<int32_t> lens(lengths.data_ptr<int32_t>(),
                              lengths.data_ptr<int32_t>() + lengths.numel());
    std::vector<GpuDecoder::StreamPartial> res;
    {
      py::gil_scoped_release release;
      res = decoder_->AdvanceChunk(channels, log_probs.data_ptr(), log_probs.size(1),
                                   log_probs.size(2), lens, partial);
    }
    py::list out;
    for (const auto& p : res) {
      py::dict d;
      d["channel"] = p.channel;
      d["words"] = p.words;
      d["overflow"] = p.overflow;
      out.append(d);
    }
    return out;
  }

  py::dict FinalizeStream(int32_t channel) {
    DecodeResult r;
    {
      py::gil_scoped_release release;
      r = decoder_->FinalizeStream(channel);
    }
    py::dict d;
    d["ok"] = r.ok;
    d["reached_final"] = r.reached_final;
    d["score"] = r.score;
    d["arc_path"] = r.arc_path;
    d["words"] = r.words;
    d["overflow"] = r.overflow;
    return d;
  }

  torch::Tensor LastLattice() {
    const std::vector<int32_t>& rec = decoder_->LastLatticeRecords();
    auto t = torch::empty({static_cast<int64_t>(rec.size() / 8), 8}, torch::kInt32);
    std::memcpy(t.data_ptr<int32_t>(), rec.data(), rec.size() * sizeof(int32_t));
    return t;
  }

  py::list DecodeBatch(torch::Tensor log_probs, torch::Tensor lengths) {
    TORCH_CHECK(log_probs.dim() == 3, "log_probs must be [B, T, V]");
    TORCH_CHECK(log_probs.is_cuda(), "log_probs must be on GPU");
    const auto want = fp16_ ? torch::kFloat16 : torch::kFloat32;
    TORCH_CHECK(log_probs.scalar_type() == want, "log_probs dtype must match the ",
                fp16_ ? "fp16" : "fp32", " decoder mode");
    log_probs = log_probs.contiguous();
    lengths = lengths.to(torch::kInt32).cpu().contiguous();
    const int64_t batch = log_probs.size(0);
    TORCH_CHECK(lengths.numel() == batch);
    std::vector<int32_t> frames(lengths.data_ptr<int32_t>(),
                                lengths.data_ptr<int32_t>() + batch);
    std::vector<DecodeResult> res;
    {
      py::gil_scoped_release release;
      res = decoder_->DecodeBatch(log_probs.data_ptr(), batch, log_probs.size(1),
                                  log_probs.size(2), frames);
    }
    py::list out;
    for (const DecodeResult& r : res) {
      py::dict d;
      d["ok"] = r.ok;
      d["reached_final"] = r.reached_final;
      d["score"] = r.score;
      d["arc_path"] = r.arc_path;
      d["words"] = r.words;
      d["overflow"] = r.overflow;
      if (!r.snapshots.empty()) {
        py::list frames_py;
        for (const auto& frame : r.snapshots) {
          py::list f;
          for (const auto& [state, score] : frame) f.append(py::make_tuple(state, score));
          frames_py.append(f);
        }
        d["frames"] = frames_py;
      }
      out.append(d);
    }
    return out;
  }

 private:
  std::shared_ptr<GraphImage> graph_;
  std::unique_ptr<GpuDecoder> decoder_;
  bool fp16_ = false;
};

}  // namespace wfst_decoder

// Bind the GPU WFST decoder into the `decoder` submodule.  Names mirror the
// standalone _wfst module: Graph, load_graph, cpu_decode, GpuDecoder.
inline void registerWfstDecoderBindings(py::module_& decoder_mod) {
  using namespace oasr::wfst;
  using wfst_decoder::CpuDecodeWrapper;
  using wfst_decoder::PyGpuDecoder;

  py::class_<GraphImage, std::shared_ptr<GraphImage>>(decoder_mod, "Graph")
      .def_readonly("num_states", &GraphImage::num_states)
      .def_readonly("num_arcs", &GraphImage::num_arcs)
      .def_readonly("vocab_size", &GraphImage::vocab_size)
      .def_readonly("start_state", &GraphImage::start_state)
      .def_readonly("finals_at_end", &GraphImage::finals_at_end);

  decoder_mod.def(
      "load_graph",
      [](const std::string& path) { return std::shared_ptr<GraphImage>(LoadGraphImage(path)); },
      py::arg("path"), "Load an hlg.img graph image for the GPU WFST decoder.");

  decoder_mod.def("cpu_decode", &CpuDecodeWrapper, py::arg("graph"), py::arg("log_probs"),
                  py::arg("search_beam") = 20.0f, py::arg("output_beam") = 8.0f,
                  py::arg("min_active") = 30, py::arg("max_active") = 10000,
                  py::arg("allow_partial") = true, py::arg("online") = false,
                  py::arg("eps_iterations") = 3,
                  "Reference CPU WFST decode (exact-semantics oracle).");

  py::class_<PyGpuDecoder>(decoder_mod, "GpuDecoder")
      .def(py::init<std::shared_ptr<GraphImage>, float, float, int32_t, int32_t, bool, int32_t,
                    int32_t, int, bool, int32_t, int32_t, bool, bool, bool, bool, int32_t,
                    int32_t>(),
           py::arg("graph"), py::arg("search_beam") = 20.0f, py::arg("output_beam") = 8.0f,
           py::arg("min_active") = 30, py::arg("max_active") = 10000,
           py::arg("allow_partial") = true, py::arg("max_lanes") = 32, py::arg("max_frames") = 1024,
           py::arg("device") = 0, py::arg("debug_snapshots") = false, py::arg("main_q_factor") = 4,
           py::arg("cand_factor") = 6, py::arg("use_cuda_graphs") = true, py::arg("lattice") = false,
           py::arg("fp16_logprobs") = false, py::arg("streaming") = false,
           py::arg("lat_prune_interval") = 0, py::arg("eps_iterations") = 3)
      .def("decode_batch", &PyGpuDecoder::DecodeBatch, py::arg("log_probs"), py::arg("lengths"))
      .def("last_lattice", &PyGpuDecoder::LastLattice)
      .def("create_stream", &PyGpuDecoder::CreateStream)
      .def("release_stream", &PyGpuDecoder::ReleaseStream, py::arg("channel"))
      .def("advance_chunk", &PyGpuDecoder::AdvanceChunk, py::arg("channels"), py::arg("log_probs"),
           py::arg("lengths"), py::arg("partial") = false)
      .def("finalize_stream", &PyGpuDecoder::FinalizeStream, py::arg("channel"));
}

}  // namespace pybind
}  // namespace oasr
