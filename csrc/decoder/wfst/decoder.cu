// Batched GPU WFST beam-search decoder: host orchestration only -- workspace allocation,
// per-frame kernel launches, whole-batch CUDA-graph capture/replay, streaming, and host
// backtrack. The device kernels live in framework-agnostic headers under
// include/wfst/kernels/ and are #included below into this single translation unit
// (CUDA_SEPARABLE_COMPILATION stays OFF), so device codegen is identical to the pre-split
// monolith. Semantics contract: identical to csrc/cpu_reference.cc (k2
// intersect_dense_pruned forward pass); see docs/DESIGN.md and the add-cuda-kernel skill.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

#include "decoder/wfst/decoder.h"

#include "oasr/wfst/kernels/backtrack.cuh"
#include "oasr/wfst/kernels/common.cuh"
#include "oasr/wfst/kernels/epsilon.cuh"
#include "oasr/wfst/kernels/frame.cuh"
#include "oasr/wfst/kernels/init.cuh"
#include "oasr/wfst/kernels/lattice.cuh"

#define WFST_CUDA_CHECK(expr)                                                       \
  do {                                                                              \
    cudaError_t err__ = (expr);                                                     \
    if (err__ != cudaSuccess) {                                                     \
      throw std::runtime_error(std::string("CUDA error: ") +                        \
                               cudaGetErrorString(err__) + " at " __FILE__ ":" +    \
                               std::to_string(__LINE__));                           \
    }                                                                               \
  } while (0)

namespace oasr::wfst {

using namespace kernels;

// ---------------------------------------------------------------------------
struct GpuDecoder::Impl {
  Options opts;
  const GraphImage* host_graph;  // borrowed; aux lookups for word emission
  DeviceGraph dg{};
  Workspace ws{};
  Sizes sz{};
  cudaStream_t stream = nullptr;
  std::vector<void*> allocs;
  int32_t* d_batch = nullptr;
  void* d_lp_staging = nullptr;  // [lanes, max_frames, lp_stride] fixed for graph capture
  int64_t lp_stride = 0;         // set on first DecodeBatch
  size_t lp_elem() const { return sz.lp_half ? sizeof(uint16_t) : sizeof(float); }
  std::vector<std::pair<int32_t, cudaGraphExec_t>> graph_execs;  // bucket -> exec
  std::vector<std::pair<int32_t, cudaGraphExec_t>> chunk_execs;  // streaming
  std::vector<std::pair<int32_t, cudaGraphExec_t>> steps_execs;  // interval segments
  std::vector<bool> channel_used;  // streaming lane occupancy
  int32_t lat_out_cap = 0;
  std::vector<int32_t> lat_records;  // host copy of the last batch's lattice records

  template <typename T>
  T* Alloc(int64_t count) {
    void* p = nullptr;
    WFST_CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
    allocs.push_back(p);
    return static_cast<T*>(p);
  }

  // n_steps frame-steps (no init). Shared by offline decode, chunk advance, and graph
  // capture; all launch configurations are capacity-sized (capture-stable). Even n_steps
  // preserves the global frontier-buffer parity (required across streaming chunks).
  void LaunchFrameSteps(cudaStream_t s, const void* lp, int64_t lane_stride,
                        int64_t frame_stride, int32_t n_steps) {
    const DecoderConfig& cfg = opts.cfg;
    const dim3 expand_grid(160, static_cast<unsigned>(sz.lanes));
    Workspace w = ws;
    for (int32_t step = 0; step < n_steps; ++step) {
      ScanKernel<<<static_cast<unsigned>(sz.lanes), 1024, 0, s>>>(w, sz, dg, cfg, 0);
      MaxKernel<<<expand_grid, 256, 0, s>>>(w, sz, dg, lp, lane_stride, frame_stride);
      ExpandKernel<<<expand_grid, 256, 0, s>>>(w, sz, dg, lp, lane_stride, frame_stride);
      FinalizeKernel<<<static_cast<unsigned>(sz.lanes), 1024, 0, s>>>(w, sz, dg, cfg);
      std::swap(w.tok_state[0], w.tok_state[1]);
      std::swap(w.tok_score[0], w.tok_score[1]);
      std::swap(w.tok_winner[0], w.tok_winner[1]);
      LaunchStepTail(s, w, lp, lane_stride, frame_stride);
    }
  }

  // Post-swap step tail: epsilon-closure passes, canonical lattice persistence, and the
  // end-of-step claim cleanup (claims stay live through K3 in eps/lattice modes).
  void LaunchStepTail(cudaStream_t s, const Workspace& w, const void* lp,
                      int64_t lane_stride, int64_t frame_stride) {
    const DecoderConfig& cfg = opts.cfg;
    const bool eps = dg.eps_count != nullptr;
    if (eps) {
      const dim3 expand_grid(160, static_cast<unsigned>(sz.lanes));
      for (int32_t it = 0; it < cfg.eps_iterations; ++it) {
        ScanKernel<<<static_cast<unsigned>(sz.lanes), 1024, 0, s>>>(w, sz, dg, cfg, 1);
        MaxKernel<<<expand_grid, 256, 0, s>>>(w, sz, dg, lp, lane_stride, frame_stride);
        ExpandKernel<<<expand_grid, 256, 0, s>>>(w, sz, dg, lp, lane_stride, frame_stride);
        EpsResolveKernel<<<static_cast<unsigned>(sz.lanes), 256, 0, s>>>(w, sz, dg);
      }
    }
    if (sz.lat_cap > 0)
      LatPersistKernel<<<static_cast<unsigned>(sz.lanes), 1024, 0, s>>>(w, sz, dg);
    if (eps || sz.lat_cap > 0)
      ClearClaimsKernel<<<static_cast<unsigned>(sz.lanes), 1024, 0, s>>>(w, sz);
  }

  // Offline pass: init (+ initial epsilon closure of the start frontier, persisted as
  // lattice segment 0) + steps.
  void LaunchSteps(cudaStream_t s, const void* lp, int64_t lane_stride,
                   int64_t frame_stride, int32_t last_step) {
    InitKernel<<<(sz.lanes + 255) / 256, 256, 0, s>>>(ws, sz, dg, d_T, d_batch,
                                                      opts.cfg.search_beam);
    if (dg.eps_count != nullptr || sz.lat_cap > 0)
      LaunchStepTail(s, ws, lp, lane_stride, frame_stride);
    LaunchFrameSteps(s, lp, lane_stride, frame_stride, last_step + 1);
  }

  void EnsureStaging(int64_t vocab_stride) {
    if (d_lp_staging == nullptr) {
      lp_stride = vocab_stride;
      WFST_CUDA_CHECK(cudaMalloc(&d_lp_staging, static_cast<int64_t>(sz.lanes) *
                                                    opts.cfg.max_frames * vocab_stride *
                                                    lp_elem()));
      allocs.push_back(d_lp_staging);
    }
    if (vocab_stride != lp_stride)
      throw std::runtime_error("log_probs vocab stride changed between calls");
  }

  int32_t BucketFor(int32_t T_max) const {
    int32_t need = T_max + 1;  // steps 0..T_max inclusive
    int32_t b = (need <= 256) ? (need + 31) / 32 * 32 : (need + 63) / 64 * 64;
    return std::min<int32_t>(b, opts.cfg.max_frames + 1);
  }

  cudaGraphExec_t CaptureGraph(const std::function<void(cudaStream_t)>& body) {
    cudaStream_t cs = nullptr;
    WFST_CUDA_CHECK(cudaStreamCreate(&cs));
    WFST_CUDA_CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
    body(cs);
    cudaGraph_t graph = nullptr;
    WFST_CUDA_CHECK(cudaStreamEndCapture(cs, &graph));
    cudaGraphExec_t exec = nullptr;
    WFST_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    WFST_CUDA_CHECK(cudaGraphDestroy(graph));
    WFST_CUDA_CHECK(cudaStreamDestroy(cs));
    return exec;
  }

  cudaGraphExec_t GraphExecForBucket(int32_t T_max) {
    const int32_t bucket = BucketFor(T_max);
    for (auto& [b, e] : graph_execs) {
      if (b == bucket) return e;
    }
    cudaGraphExec_t exec = CaptureGraph([&](cudaStream_t cs) {
      LaunchSteps(cs, d_lp_staging,
                  static_cast<int64_t>(opts.cfg.max_frames) * lp_stride, lp_stride,
                  bucket - 1);
      BacktrackKernel<<<(sz.lanes + 63) / 64, 64, 0, cs>>>(ws, sz);
      HashSanitizeKernel<<<dim3(64, static_cast<unsigned>(sz.lanes)), 256, 0, cs>>>(ws, sz);
    });
    graph_execs.emplace_back(bucket, exec);
    return exec;
  }

  // Bare frame-step graphs (no init/backtrack), for the interval-pruned segment loop.
  cudaGraphExec_t StepsExec(int32_t n_steps) {
    for (auto& [b, e] : steps_execs) {
      if (b == n_steps) return e;
    }
    cudaGraphExec_t exec = CaptureGraph([&](cudaStream_t cs) {
      LaunchFrameSteps(cs, d_lp_staging,
                       static_cast<int64_t>(opts.cfg.max_frames) * lp_stride, lp_stride,
                       n_steps);
    });
    steps_execs.emplace_back(n_steps, exec);
    return exec;
  }

  // Streaming chunk graphs: even step counts keep the frontier-buffer parity invariant.
  cudaGraphExec_t ChunkExecForSteps(int32_t max_len) {
    const int32_t bucket =
        std::min<int32_t>((max_len + 7) / 8 * 8, opts.cfg.max_frames + (opts.cfg.max_frames & 1));
    for (auto& [b, e] : chunk_execs) {
      if (b == bucket) return e;
    }
    cudaGraphExec_t exec = CaptureGraph([&](cudaStream_t cs) {
      ChunkBeginKernel<<<(sz.lanes + 63) / 64, 64, 0, cs>>>(ws, sz, d_T);
      LaunchFrameSteps(cs, d_lp_staging,
                       static_cast<int64_t>(opts.cfg.max_frames) * lp_stride, lp_stride,
                       bucket);
      HashSanitizeKernel<<<dim3(64, static_cast<unsigned>(sz.lanes)), 256, 0, cs>>>(ws, sz);
    });
    chunk_execs.emplace_back(bucket, exec);
    return exec;
  }

  int32_t* d_T = nullptr;

  Impl(const GraphImage& g, const Options& o) : opts(o), host_graph(&g) {
    WFST_CUDA_CHECK(cudaSetDevice(o.device));
    WFST_CUDA_CHECK(cudaStreamCreate(&stream));

    const DecoderConfig& cfg = o.cfg;
    if (cfg.max_active_states <= 0 || cfg.max_active_states > (1 << 23))
      throw std::runtime_error("max_active_states out of supported range [1, 8M]");
    sz.lanes = cfg.max_lanes;
    sz.main_q = cfg.main_q_capacity();
    sz.claims_cap = 2 * sz.main_q;
    sz.cand_cap = cfg.cand_capacity();
    int32_t hc = 1;
    while (hc < 2 * sz.claims_cap) hc <<= 1;
    sz.hash_cap = hc;
    // Shared winners arena: statistical sizing (per-lane worst case is unaffordable);
    // typical usage is ~1-2M entries/lane, hard utterances borrow from the pool.
    // Pools scale with lane count so small rescue instances stay cheap.
    const int64_t want =
        static_cast<int64_t>(sz.lanes) * (cfg.max_frames + 2) * sz.main_q;
    const int64_t arena_budget =
        std::min<int64_t>(512LL << 20, std::max<int64_t>(64LL << 20, (16LL << 20) * sz.lanes));
    sz.arena_cap = static_cast<int32_t>(std::min<int64_t>(want, arena_budget));
    sz.path_cap = cfg.max_frames + 2;
    if (cfg.lat_prune_interval < 0 || (cfg.lat_prune_interval & 1))
      throw std::runtime_error("lat_prune_interval must be even (buffer-parity invariant)");
    // Interval mode holds only ~one window of candidates — a much smaller arena works.
    sz.lat_cap = !cfg.lattice ? 0
                 : cfg.lat_prune_interval > 0
                     ? static_cast<int32_t>(std::min<int64_t>(
                           64LL << 20, std::max<int64_t>(16LL << 20, (1LL << 20) * sz.lanes)))
                     : static_cast<int32_t>(std::min<int64_t>(
                           128LL << 20, std::max<int64_t>(32LL << 20, (4LL << 20) * sz.lanes)));
    sz.lp_half = cfg.fp16_logprobs ? 1 : 0;
    sz.stream_log_cap = cfg.streaming ? sz.arena_cap / sz.lanes : 0;
    if (cfg.streaming && cfg.lattice)
      throw std::runtime_error("lattice mode is not supported with streaming (v1)");
    channel_used.assign(sz.lanes, false);
    sz.snap_frames = o.debug_snapshots ? cfg.max_frames + 2 : 0;

    // Graph to device.
    dg.num_states = static_cast<int32_t>(g.num_states);
    dg.vocab_size = g.vocab_size;
    dg.start_state = g.start_state;
    dg.finals_at_end = g.finals_at_end;
    dg.eps_first = g.eps_first;
    dg.eps_count = nullptr;
    if (g.has_eps) {
      auto* ec = Alloc<int32_t>(g.num_states);
      WFST_CUDA_CHECK(
          cudaMemcpy(ec, g.eps_count.data(), g.num_states * 4, cudaMemcpyHostToDevice));
      dg.eps_count = ec;
    }
    auto* rs = Alloc<int32_t>(g.num_states + 1);
    auto* fc = Alloc<int32_t>(g.num_states);
    auto* di = Alloc<int2>(g.num_arcs);
    auto* w = Alloc<float>(g.num_arcs);
    WFST_CUDA_CHECK(cudaMemcpy(rs, g.row_splits.data(), (g.num_states + 1) * 4,
                               cudaMemcpyHostToDevice));
    WFST_CUDA_CHECK(
        cudaMemcpy(fc, g.final_count.data(), g.num_states * 4, cudaMemcpyHostToDevice));
    WFST_CUDA_CHECK(cudaMemcpy(di, g.arc_dest_ilabel.data(), g.num_arcs * 8,
                               cudaMemcpyHostToDevice));
    WFST_CUDA_CHECK(
        cudaMemcpy(w, g.arc_weight.data(), g.num_arcs * 4, cudaMemcpyHostToDevice));
    dg.row_splits = rs;
    dg.final_count = fc;
    dg.dest_ilabel = di;
    dg.weight = w;

    const int64_t L = sz.lanes;
    for (int b = 0; b < 2; ++b) {
      ws.tok_state[b] = Alloc<int32_t>(L * sz.main_q);
      ws.tok_score[b] = Alloc<float>(L * sz.main_q);
      ws.tok_winner[b] = Alloc<int32_t>(L * sz.main_q);
    }
    ws.tok_emit_begin = Alloc<int32_t>(L * sz.main_q);
    ws.arc_offsets = Alloc<int32_t>(L * (sz.main_q + 1));
    ws.next_claims = Alloc<int2>(L * sz.claims_cap);
    ws.cand = Alloc<int2>(L * sz.cand_cap);
    ws.hash_key = Alloc<uint32_t>(L * sz.hash_cap);
    ws.hash_payload = Alloc<unsigned long long>(L * sz.hash_cap);
    if (g.has_eps || cfg.lattice) {
      ws.hash_pos = Alloc<int32_t>(L * sz.hash_cap);
      WFST_CUDA_CHECK(cudaMemset(ws.hash_pos, 0xFF, L * sz.hash_cap * sizeof(int32_t)));
    }
    ws.winners = Alloc<int2>(sz.arena_cap);
    ws.arena_cursor = Alloc<int32_t>(1);
    ws.arc_out = Alloc<int32_t>(L * sz.path_cap);
    ws.arc_out_len = Alloc<int32_t>(L);
    ws.lanes = Alloc<LaneCounters>(L);
    d_T = Alloc<int32_t>(L);
    d_batch = Alloc<int32_t>(1);
    if (sz.lat_cap > 0) {
      ws.cand_end = Alloc<float>(L * sz.cand_cap);
      ws.tok_fwd = Alloc<float>(sz.arena_cap);
      ws.tok_bwd = Alloc<uint32_t>(sz.arena_cap);
      ws.lat = Alloc<int4>(sz.lat_cap);
      ws.lat_cursor = Alloc<int32_t>(1);
      ws.lat_seg = Alloc<int2>(L * sz.path_cap);
      if (cfg.lat_prune_interval > 0) {
        ws.lat2 = Alloc<int4>(sz.lat_cap);
        ws.lat2_cursor = Alloc<int32_t>(1);
      }
      lat_out_cap = static_cast<int32_t>(
          std::min<int64_t>(32LL << 20, std::max<int64_t>(8LL << 20, (1LL << 20) * L)));
      ws.lat_out = Alloc<int32_t>(static_cast<int64_t>(lat_out_cap) * 8);
      ws.lat_out_cursor = Alloc<int32_t>(1);
      ws.lat_out_len = Alloc<int32_t>(L);
    }
    if (sz.snap_frames > 0) {
      ws.snap = Alloc<int2>(L * sz.snap_frames * static_cast<int64_t>(sz.main_q));
      ws.snap_len = Alloc<int32_t>(L * sz.snap_frames);
    }
    WFST_CUDA_CHECK(cudaMemset(ws.hash_key, 0xFF,
                               L * sz.hash_cap * sizeof(uint32_t)));
    WFST_CUDA_CHECK(
        cudaMemset(ws.hash_payload, 0, L * sz.hash_cap * sizeof(unsigned long long)));
  }

  ~Impl() {
    for (auto& [b, e] : graph_execs) cudaGraphExecDestroy(e);
    for (void* p : allocs) cudaFree(p);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }
};

GpuDecoder::GpuDecoder(const GraphImage& graph, const Options& opts)
    : impl_(new Impl(graph, opts)) {}
GpuDecoder::~GpuDecoder() = default;

const std::vector<int32_t>& GpuDecoder::LastLatticeRecords() const {
  return impl_->lat_records;
}

// ---------------------------------------------------------------------------
// Streaming.

int32_t GpuDecoder::CreateStream() {
  Impl& im = *impl_;
  if (!im.opts.cfg.streaming) throw std::runtime_error("decoder not in streaming mode");
  WFST_CUDA_CHECK(cudaSetDevice(im.opts.device));
  for (int32_t lane = 0; lane < im.sz.lanes; ++lane) {
    if (im.channel_used[lane]) continue;
    im.channel_used[lane] = true;
    StreamCreateKernel<<<1, 1, 0, im.stream>>>(im.ws, im.sz, im.dg, lane,
                                               im.opts.cfg.search_beam);
    if (im.dg.eps_count != nullptr) {
      // Initial closure of the start frontier. Touches only this lane's state (other
      // lanes gate on phase), and launches no buffer swap — parity is preserved.
      im.LaunchStepTail(im.stream, im.ws, im.d_lp_staging,
                        static_cast<int64_t>(im.opts.cfg.max_frames) * im.lp_stride,
                        im.lp_stride);
    }
    return lane;
  }
  return -1;
}

void GpuDecoder::ReleaseStream(int32_t channel) {
  Impl& im = *impl_;
  if (channel >= 0 && channel < im.sz.lanes) im.channel_used[channel] = false;
}

std::vector<GpuDecoder::StreamPartial> GpuDecoder::AdvanceChunk(
    const std::vector<int32_t>& channels, const void* d_log_probs, int64_t chunk_frames,
    int64_t vocab_stride, const std::vector<int32_t>& lens, bool want_partial) {
  Impl& im = *impl_;
  const Sizes& sz = im.sz;
  const DecoderConfig& cfg = im.opts.cfg;
  if (!cfg.streaming) throw std::runtime_error("decoder not in streaming mode");
  if (channels.size() != lens.size()) throw std::runtime_error("channels/lens mismatch");
  if (chunk_frames > cfg.max_frames)
    throw std::runtime_error("chunk_frames > max_frames (max chunk length)");
  WFST_CUDA_CHECK(cudaSetDevice(im.opts.device));
  im.EnsureStaging(vocab_stride);
  const size_t elem = im.lp_elem();

  std::vector<int32_t> h_len(sz.lanes, 0);
  int32_t max_len = 0;
  for (size_t i = 0; i < channels.size(); ++i) {
    const int32_t lane = channels[i];
    if (lane < 0 || lane >= sz.lanes || !im.channel_used[lane])
      throw std::runtime_error("invalid channel");
    if (lens[i] < 0 || lens[i] > chunk_frames) throw std::runtime_error("bad chunk len");
    h_len[lane] = lens[i];
    max_len = std::max(max_len, lens[i]);
    // Stage this channel's rows into its lane slot.
    if (lens[i] > 0) {
      WFST_CUDA_CHECK(cudaMemcpyAsync(
          static_cast<char*>(im.d_lp_staging) +
              static_cast<int64_t>(lane) * cfg.max_frames * im.lp_stride * elem,
          static_cast<const char*>(d_log_probs) +
              static_cast<int64_t>(i) * chunk_frames * vocab_stride * elem,
          static_cast<int64_t>(lens[i]) * vocab_stride * elem, cudaMemcpyDeviceToDevice,
          im.stream));
    }
  }
  WFST_CUDA_CHECK(cudaMemcpyAsync(im.d_T, h_len.data(), sz.lanes * sizeof(int32_t),
                                  cudaMemcpyHostToDevice, im.stream));

  if (max_len > 0) {
    if (im.opts.use_cuda_graphs && !im.opts.debug_snapshots) {
      WFST_CUDA_CHECK(cudaGraphLaunch(im.ChunkExecForSteps(max_len), im.stream));
    } else {
      const int32_t steps = (max_len + 7) / 8 * 8;  // even: parity invariant
      ChunkBeginKernel<<<(sz.lanes + 63) / 64, 64, 0, im.stream>>>(im.ws, sz, im.d_T);
      im.LaunchFrameSteps(im.stream, im.d_lp_staging,
                          static_cast<int64_t>(cfg.max_frames) * im.lp_stride,
                          im.lp_stride, steps);
      HashSanitizeKernel<<<dim3(64, static_cast<unsigned>(sz.lanes)), 256, 0, im.stream>>>(
          im.ws, sz);
    }
  }

  std::vector<StreamPartial> out(channels.size());
  if (want_partial) {
    PartialBacktrackKernel<<<static_cast<unsigned>(sz.lanes), 256, 0, im.stream>>>(
        im.ws, sz, im.d_T);
  }
  std::vector<LaneCounters> lanes(sz.lanes);
  std::vector<int32_t> arc_out(want_partial ? sz.lanes * sz.path_cap : 0);
  std::vector<int32_t> arc_len(want_partial ? sz.lanes : 0);
  WFST_CUDA_CHECK(cudaMemcpyAsync(lanes.data(), im.ws.lanes,
                                  sz.lanes * sizeof(LaneCounters), cudaMemcpyDeviceToHost,
                                  im.stream));
  if (want_partial) {
    WFST_CUDA_CHECK(cudaMemcpyAsync(arc_out.data(), im.ws.arc_out,
                                    arc_out.size() * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost, im.stream));
    WFST_CUDA_CHECK(cudaMemcpyAsync(arc_len.data(), im.ws.arc_out_len,
                                    arc_len.size() * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost, im.stream));
  }
  WFST_CUDA_CHECK(cudaStreamSynchronize(im.stream));

  const GraphImage& g = *im.host_graph;
  for (size_t i = 0; i < channels.size(); ++i) {
    const int32_t lane = channels[i];
    out[i].channel = lane;
    out[i].overflow = lanes[lane].overflow;
    if (want_partial && arc_len[lane] > 0) {
      const int32_t* path = arc_out.data() + static_cast<int64_t>(lane) * sz.path_cap;
      for (int32_t k = arc_len[lane] - 1; k >= 0; --k) {
        const int32_t a = path[k];
        for (int32_t j = g.aux_row_splits[a]; j < g.aux_row_splits[a + 1]; ++j) {
          if (g.aux_pool[j] > 0) out[i].words.push_back(g.aux_pool[j]);
        }
      }
    }
  }
  return out;
}

DecodeResult GpuDecoder::FinalizeStream(int32_t channel) {
  Impl& im = *impl_;
  const Sizes& sz = im.sz;
  if (!im.opts.cfg.streaming) throw std::runtime_error("decoder not in streaming mode");
  if (channel < 0 || channel >= sz.lanes || !im.channel_used[channel])
    throw std::runtime_error("invalid channel");
  WFST_CUDA_CHECK(cudaSetDevice(im.opts.device));
  im.EnsureStaging(im.lp_stride == 0 ? im.host_graph->vocab_size : im.lp_stride);

  FinalizePrepKernel<<<1, 1, 0, im.stream>>>(im.ws, channel);
  // Two steps (even, parity invariant): step 1 = the final step for this channel; step 2
  // idles every lane. Other channels sit outside their chunk_end and just carry over.
  im.LaunchFrameSteps(im.stream, im.d_lp_staging,
                      static_cast<int64_t>(im.opts.cfg.max_frames) * im.lp_stride,
                      im.lp_stride, 2);
  BacktrackKernel<<<(sz.lanes + 63) / 64, 64, 0, im.stream>>>(im.ws, sz);

  LaneCounters lc{};
  std::vector<int32_t> path(sz.path_cap);
  int32_t plen = 0;
  WFST_CUDA_CHECK(cudaMemcpyAsync(&lc, im.ws.lanes + channel, sizeof(LaneCounters),
                                  cudaMemcpyDeviceToHost, im.stream));
  WFST_CUDA_CHECK(cudaMemcpyAsync(path.data(),
                                  im.ws.arc_out + static_cast<int64_t>(channel) * sz.path_cap,
                                  sz.path_cap * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                  im.stream));
  WFST_CUDA_CHECK(cudaMemcpyAsync(&plen, im.ws.arc_out_len + channel, sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, im.stream));
  WFST_CUDA_CHECK(cudaStreamSynchronize(im.stream));

  DecodeResult r;
  r.overflow = lc.overflow;
  r.ok = (lc.status == 1);
  r.reached_final = lc.reached_final != 0;
  r.score = lc.final_score;
  if (r.ok) {
    r.arc_path.assign(path.begin(), path.begin() + plen);
    std::reverse(r.arc_path.begin(), r.arc_path.end());
    const GraphImage& g = *im.host_graph;
    for (int32_t a : r.arc_path) {
      for (int32_t j = g.aux_row_splits[a]; j < g.aux_row_splits[a + 1]; ++j) {
        if (g.aux_pool[j] > 0) r.words.push_back(g.aux_pool[j]);
      }
    }
  }
  return r;
}

std::vector<DecodeResult> GpuDecoder::DecodeBatch(const void* d_log_probs, int64_t batch,
                                                  int64_t max_frames, int64_t vocab_stride,
                                                  const std::vector<int32_t>& frames) {
  Impl& im = *impl_;
  const Sizes& sz = im.sz;
  const DecoderConfig& cfg = im.opts.cfg;
  if (cfg.streaming) throw std::runtime_error("use AdvanceChunk/FinalizeStream in streaming mode");
  if (batch > sz.lanes) throw std::runtime_error("batch > max_lanes");
  if (max_frames > cfg.max_frames) throw std::runtime_error("T > max_frames");
  WFST_CUDA_CHECK(cudaSetDevice(im.opts.device));

  // Lane lengths + batch size to device (zero-pad unused lanes).
  std::vector<int32_t> t_host(sz.lanes, 0);
  std::copy(frames.begin(), frames.end(), t_host.begin());
  WFST_CUDA_CHECK(cudaMemcpyAsync(im.d_T, t_host.data(), sz.lanes * sizeof(int32_t),
                                  cudaMemcpyHostToDevice, im.stream));
  const int32_t batch32 = static_cast<int32_t>(batch);
  WFST_CUDA_CHECK(cudaMemcpyAsync(im.d_batch, &batch32, sizeof(int32_t),
                                  cudaMemcpyHostToDevice, im.stream));

  int32_t T_max = 0;
  for (int32_t t : frames) T_max = std::max(T_max, t);

  const bool use_graphs = im.opts.use_cuda_graphs && !im.opts.debug_snapshots;
  const bool interval_mode = cfg.lattice && cfg.lat_prune_interval > 0;
  if (interval_mode) {
    // Segmented loop with periodic lattice-arena pruning (long-form audio).
    const size_t elem = im.lp_elem();
    im.EnsureStaging(vocab_stride);
    WFST_CUDA_CHECK(cudaMemcpy2DAsync(
        im.d_lp_staging, static_cast<size_t>(cfg.max_frames) * im.lp_stride * elem,
        d_log_probs, static_cast<size_t>(max_frames) * vocab_stride * elem,
        static_cast<size_t>(max_frames) * vocab_stride * elem,
        static_cast<size_t>(batch), cudaMemcpyDeviceToDevice, im.stream));
    InitKernel<<<(sz.lanes + 255) / 256, 256, 0, im.stream>>>(im.ws, sz, im.dg, im.d_T,
                                                              im.d_batch, cfg.search_beam);
    const int64_t lane_stride = static_cast<int64_t>(cfg.max_frames) * im.lp_stride;
    const int32_t total_steps = T_max + 1;
    const dim3 bwd_grid(64, static_cast<unsigned>(batch));
    int32_t done = 0;
    while (done < total_steps) {
      int32_t n = std::min(cfg.lat_prune_interval, ((total_steps - done) + 1) / 2 * 2);
      if (use_graphs) {
        WFST_CUDA_CHECK(cudaGraphLaunch(im.StepsExec(n), im.stream));
      } else {
        im.LaunchFrameSteps(im.stream, im.d_lp_staging, lane_stride, im.lp_stride, n);
      }
      done += n;
      if (done < total_steps) {
        WFST_CUDA_CHECK(cudaMemsetAsync(im.ws.lat2_cursor, 0, sizeof(int32_t), im.stream));
        LatInitBwdKernel<<<512, 256, 0, im.stream>>>(im.ws);
        LatSeedFrontierKernel<<<static_cast<unsigned>(batch), 1024, 0, im.stream>>>(im.ws,
                                                                                    sz);
        // Segments [0, done]: seg 0 = initial closure, seg d = frame d-1. Epsilon arcs
        // chain within a segment, so repeat each step to propagate through the chains.
        const int32_t reps = 1 + (im.dg.eps_count != nullptr ? cfg.eps_iterations : 0);
        for (int32_t seg = done; seg >= 0; --seg)
          for (int32_t r = 0; r < reps; ++r)
            LatBackwardStepKernel<<<bwd_grid, 256, 0, im.stream>>>(im.ws, sz, seg);
        LatIntervalCompactKernel<<<static_cast<unsigned>(batch), 1024, 0, im.stream>>>(
            im.ws, sz, cfg.output_beam, done + 1);
        WFST_CUDA_CHECK(cudaMemcpyAsync(im.ws.lat, im.ws.lat2,
                                        static_cast<int64_t>(sz.lat_cap) * sizeof(int4),
                                        cudaMemcpyDeviceToDevice, im.stream));
        LatSwapBackKernel<<<1, 1, 0, im.stream>>>(im.ws);
      }
    }
    BacktrackKernel<<<(sz.lanes + 63) / 64, 64, 0, im.stream>>>(im.ws, sz);
    HashSanitizeKernel<<<dim3(64, static_cast<unsigned>(sz.lanes)), 256, 0, im.stream>>>(
        im.ws, sz);
  } else if (use_graphs) {
    // Stage log-probs into the decoder-owned buffer so kernel pointers are capture-
    // stable. One strided D2D copy: rows of T*V floats into max_frames*V-stride slots.
    const size_t elem = im.lp_elem();
    im.EnsureStaging(vocab_stride);
    WFST_CUDA_CHECK(cudaMemcpy2DAsync(
        im.d_lp_staging, static_cast<size_t>(cfg.max_frames) * im.lp_stride * elem,
        d_log_probs, static_cast<size_t>(max_frames) * vocab_stride * elem,
        static_cast<size_t>(max_frames) * vocab_stride * elem,
        static_cast<size_t>(batch), cudaMemcpyDeviceToDevice, im.stream));
    WFST_CUDA_CHECK(cudaGraphLaunch(im.GraphExecForBucket(T_max), im.stream));
  } else {
    im.LaunchSteps(im.stream, d_log_probs, max_frames * vocab_stride, vocab_stride, T_max);
    BacktrackKernel<<<(sz.lanes + 63) / 64, 64, 0, im.stream>>>(im.ws, sz);
    HashSanitizeKernel<<<dim3(64, static_cast<unsigned>(sz.lanes)), 256, 0, im.stream>>>(
        im.ws, sz);
  }

  if (cfg.lattice) {
    // Backward output-beam prune + arc emission (eager; not latency-critical).
    LatInitBwdKernel<<<512, 256, 0, im.stream>>>(im.ws);
    LatSeedKernel<<<(sz.lanes + 63) / 64, 64, 0, im.stream>>>(im.ws, sz);
    const dim3 bwd_grid(64, static_cast<unsigned>(batch));
    const int32_t reps = 1 + (im.dg.eps_count != nullptr ? cfg.eps_iterations : 0);
    for (int32_t seg = T_max + 1; seg >= 0; --seg)
      for (int32_t r = 0; r < reps; ++r)
        LatBackwardStepKernel<<<bwd_grid, 256, 0, im.stream>>>(im.ws, sz, seg);
    LatEmitKernel<<<static_cast<unsigned>(batch), 1024, 0, im.stream>>>(
        im.ws, sz, im.dg, cfg.output_beam, im.lat_out_cap);
  }
  std::vector<LaneCounters> lanes(batch);
  std::vector<int32_t> arc_out(batch * sz.path_cap);
  std::vector<int32_t> arc_len(batch);
  WFST_CUDA_CHECK(cudaMemcpyAsync(lanes.data(), im.ws.lanes, batch * sizeof(LaneCounters),
                                  cudaMemcpyDeviceToHost, im.stream));
  WFST_CUDA_CHECK(cudaMemcpyAsync(arc_out.data(), im.ws.arc_out,
                                  batch * sz.path_cap * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, im.stream));
  WFST_CUDA_CHECK(cudaMemcpyAsync(arc_len.data(), im.ws.arc_out_len,
                                  batch * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                  im.stream));
  int32_t n_lat_records = 0;
  if (cfg.lattice) {
    WFST_CUDA_CHECK(cudaMemcpyAsync(&n_lat_records, im.ws.lat_out_cursor, sizeof(int32_t),
                                    cudaMemcpyDeviceToHost, im.stream));
  }
  WFST_CUDA_CHECK(cudaStreamSynchronize(im.stream));

  bool lat_truncated = false;
  if (cfg.lattice) {
    lat_truncated = n_lat_records > im.lat_out_cap;
    n_lat_records = std::min(n_lat_records, im.lat_out_cap);
    im.lat_records.resize(static_cast<int64_t>(n_lat_records) * 8);
    if (n_lat_records > 0) {
      WFST_CUDA_CHECK(cudaMemcpy(im.lat_records.data(), im.ws.lat_out,
                                 static_cast<int64_t>(n_lat_records) * 8 * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost));
    }
  }

  std::vector<DecodeResult> results(batch);
  const GraphImage& g = *im.host_graph;
  for (int64_t b = 0; b < batch; ++b) {
    DecodeResult& r = results[b];
    const LaneCounters& lc = lanes[b];
    r.overflow = lc.overflow | (lat_truncated ? kOverflowArena : 0u);
    r.ok = (lc.status == 1);
    r.reached_final = lc.reached_final != 0;
    r.score = lc.final_score;
    if (!r.ok) continue;
    const int32_t* path = arc_out.data() + b * sz.path_cap;
    r.arc_path.assign(path, path + arc_len[b]);
    std::reverse(r.arc_path.begin(), r.arc_path.end());
    for (int32_t a : r.arc_path) {
      for (int32_t j = g.aux_row_splits[a]; j < g.aux_row_splits[a + 1]; ++j) {
        if (g.aux_pool[j] > 0) r.words.push_back(g.aux_pool[j]);
      }
    }
  }

  if (im.opts.debug_snapshots) {
    std::vector<int32_t> snap_len(sz.snap_frames);
    std::vector<int2> snap_row(sz.main_q);
    for (int64_t b = 0; b < batch; ++b) {
      const int32_t T = frames[b];
      results[b].snapshots.resize(T + 2);
      WFST_CUDA_CHECK(cudaMemcpy(snap_len.data(), im.ws.snap_len + b * sz.snap_frames,
                                 sz.snap_frames * 4, cudaMemcpyDeviceToHost));
      for (int32_t f = 0; f <= T + 1 && f < sz.snap_frames; ++f) {
        const int32_t n = std::min(snap_len[f], sz.main_q);
        WFST_CUDA_CHECK(cudaMemcpy(
            snap_row.data(),
            im.ws.snap + (b * static_cast<int64_t>(sz.snap_frames) + f) * sz.main_q,
            static_cast<int64_t>(n) * sizeof(int2), cudaMemcpyDeviceToHost));
        auto& frame = results[b].snapshots[f];
        frame.reserve(n);
        for (int32_t i = 0; i < n; ++i) {
          float s;
          std::memcpy(&s, &snap_row[i].y, 4);
          frame.push_back({snap_row[i].x, s});
        }
        std::sort(frame.begin(), frame.end());
      }
    }
  }

  return results;
}

}  // namespace oasr::wfst
