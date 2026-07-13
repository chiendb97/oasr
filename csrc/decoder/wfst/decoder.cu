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
#include "decoder/wfst/lazy_region.h"

#include "oasr/wfst/backtrack.cuh"
#include "oasr/wfst/common.cuh"
#include "oasr/wfst/epsilon.cuh"
#include "oasr/wfst/frame.cuh"
#include "oasr/wfst/init.cuh"
#include "oasr/wfst/lattice.cuh"

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

// 1-D grid for the flattened K2 kernels: enough blocks to fill the device even when a
// single lane holds all the work, and to keep multi-lane aggregate steps as deeply
// pipelined as the old (160, lanes) grid; excess blocks read one int and exit.
inline int32_t ExpandBlocksFor(int32_t lanes) {
  return std::min(8192, std::max(1024, 160 * lanes));
}

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

  // Lazily-committed regions + host mirrors of the device-resident capacity limits.
  static constexpr int64_t kInitArenaEntries = 8LL << 20;   // 64 MiB of winners
  static constexpr int64_t kInitLatEntries = 4LL << 20;     // 64 MiB of lattice records
  static constexpr int64_t kInitLatOutRecords = 1LL << 20;  // 32 MiB of emitted arcs
  static constexpr int32_t kMaxGrowAttempts = 12;
  LazyRegion winners_mem, tok_fwd_mem, tok_bwd_mem, lat_mem, lat2_mem, lat_out_mem;
  LazyRegion staging_mem;
  int32_t arena_limit_host = 0;
  int32_t lat_limit_host = 0;
  int32_t lat_out_limit_host = 0;
  int64_t fixed_bytes = 0;           // eager cudaMalloc workspace + graph image
  int32_t arena_high_water = 0;      // max end-of-batch arena cursor observed

  template <typename T>
  T* Alloc(int64_t count) {
    void* p = nullptr;
    WFST_CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
    allocs.push_back(p);
    fixed_bytes += count * sizeof(T);
    return static_cast<T*>(p);
  }

  // Commits the winners arena (and the parallel per-token score arrays in lattice mode)
  // up to `entries` and publishes the committed capacity to the device-side limit.
  void SetArenaCommit(int64_t entries) {
    winners_mem.EnsurePrefix(static_cast<size_t>(entries) * sizeof(int2));
    int64_t lim = std::min<int64_t>(
        sz.arena_cap, static_cast<int64_t>(winners_mem.committed() / sizeof(int2)));
    if (sz.lat_cap > 0) {
      tok_fwd_mem.EnsurePrefix(static_cast<size_t>(entries) * 4);
      tok_bwd_mem.EnsurePrefix(static_cast<size_t>(entries) * 4);
      lim = std::min<int64_t>(lim, static_cast<int64_t>(tok_fwd_mem.committed() / 4));
      lim = std::min<int64_t>(lim, static_cast<int64_t>(tok_bwd_mem.committed() / 4));
    }
    arena_limit_host = static_cast<int32_t>(lim);
    WFST_CUDA_CHECK(
        cudaMemcpy(ws.arena_limit, &arena_limit_host, 4, cudaMemcpyHostToDevice));
  }

  void SetLatCommit(int64_t entries) {
    lat_mem.EnsurePrefix(static_cast<size_t>(entries) * sizeof(int4));
    int64_t lim = std::min<int64_t>(
        sz.lat_cap, static_cast<int64_t>(lat_mem.committed() / sizeof(int4)));
    if (opts.cfg.lat_prune_interval > 0) {
      lat2_mem.EnsurePrefix(static_cast<size_t>(entries) * sizeof(int4));
      lim = std::min<int64_t>(lim, static_cast<int64_t>(lat2_mem.committed() / sizeof(int4)));
    }
    lat_limit_host = static_cast<int32_t>(lim);
    WFST_CUDA_CHECK(cudaMemcpy(ws.lat_limit, &lat_limit_host, 4, cudaMemcpyHostToDevice));
  }

  void SetLatOutCommit(int64_t records) {
    lat_out_mem.EnsurePrefix(static_cast<size_t>(records) * 8 * sizeof(int32_t));
    lat_out_limit_host = static_cast<int32_t>(std::min<int64_t>(
        lat_out_cap, static_cast<int64_t>(lat_out_mem.committed() / (8 * sizeof(int32_t)))));
  }

  // Doubles a region's commitment toward its full capacity; false when already there.
  bool GrowArena() {
    if (arena_limit_host >= sz.arena_cap) return false;
    SetArenaCommit(std::max<int64_t>(kInitArenaEntries,
                                     static_cast<int64_t>(arena_limit_host) * 2));
    return true;
  }
  bool GrowLat() {
    if (sz.lat_cap == 0 || lat_limit_host >= sz.lat_cap) return false;
    SetLatCommit(std::max<int64_t>(kInitLatEntries, static_cast<int64_t>(lat_limit_host) * 2));
    return true;
  }
  bool GrowLatOut(int64_t needed) {
    if (sz.lat_cap == 0 || lat_out_limit_host >= lat_out_cap) return false;
    SetLatOutCommit(std::max<int64_t>(needed, static_cast<int64_t>(lat_out_limit_host) * 2));
    return true;
  }

  // n_steps frame-steps (no init). Shared by offline decode, chunk advance, and graph
  // capture; all launch configurations are capacity-sized (capture-stable). Even n_steps
  // preserves the global frontier-buffer parity (required across streaming chunks).
  void LaunchFrameSteps(cudaStream_t s, const void* lp, int64_t lane_stride,
                        int64_t frame_stride, int32_t n_steps) {
    const DecoderConfig& cfg = opts.cfg;
    const unsigned xb = static_cast<unsigned>(ExpandBlocksFor(sz.lanes));
    Workspace w = ws;
    for (int32_t step = 0; step < n_steps; ++step) {
      ScanKernel<<<static_cast<unsigned>(sz.lanes), 1024, 0, s>>>(w, sz, dg, cfg, 0, lp,
                                                                  lane_stride, frame_stride);
      LaneScanKernel<<<1, 32, 0, s>>>(w, sz);
      MaxKernel<<<xb, 256, 0, s>>>(w, sz, dg, lp, lane_stride, frame_stride);
      ExpandKernel<<<xb, 256, 0, s>>>(w, sz, dg, lp, lane_stride, frame_stride);
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
      const unsigned xb = static_cast<unsigned>(ExpandBlocksFor(sz.lanes));
      for (int32_t it = 0; it < cfg.eps_iterations; ++it) {
        ScanKernel<<<static_cast<unsigned>(sz.lanes), 1024, 0, s>>>(
            w, sz, dg, cfg, 1, lp, lane_stride, frame_stride);
        LaneScanKernel<<<1, 32, 0, s>>>(w, sz);
        MaxKernel<<<xb, 256, 0, s>>>(w, sz, dg, lp, lane_stride, frame_stride);
        ExpandKernel<<<xb, 256, 0, s>>>(w, sz, dg, lp, lane_stride, frame_stride);
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
      const size_t bytes = static_cast<size_t>(sz.lanes) * opts.cfg.max_frames *
                           vocab_stride * lp_elem();
      // Minimum-granularity chunks: the per-lane row prefixes are strided through the
      // region, so large chunks would end up committing the padding holes too.
      staging_mem.Reserve(bytes, opts.device, 1);
      // Streaming chunk slots are small and rewritten constantly: commit them all.
      // Batch mode commits per-lane row prefixes as batches actually need them.
      if (opts.cfg.streaming) staging_mem.EnsurePrefix(bytes);
      d_lp_staging = staging_mem.ptr();
    }
    if (vocab_stride != lp_stride)
      throw std::runtime_error("log_probs vocab stride changed between calls");
  }

  // Commits each of the first `batch` lanes' staging prefix for `rows` frame rows and
  // stages the caller's [batch, rows, vocab] block into the per-lane slots. Per-lane 1-D
  // copies (not one 2-D copy): each copy must stay inside its lane's committed prefix —
  // the driver rejects copies whose validated span crosses unmapped holes.
  void StageLogProbs(const void* d_log_probs, int64_t batch, int64_t rows, cudaStream_t s) {
    const size_t elem = lp_elem();
    const size_t row_bytes = static_cast<size_t>(lp_stride) * elem;
    const size_t lane_bytes = static_cast<size_t>(opts.cfg.max_frames) * row_bytes;
    for (int64_t b = 0; b < batch; ++b) {
      staging_mem.EnsureRange(static_cast<size_t>(b) * lane_bytes,
                              static_cast<size_t>(rows) * row_bytes);
      WFST_CUDA_CHECK(cudaMemcpyAsync(
          static_cast<char*>(d_lp_staging) + static_cast<size_t>(b) * lane_bytes,
          static_cast<const char*>(d_log_probs) + static_cast<size_t>(b) * rows * row_bytes,
          static_cast<size_t>(rows) * row_bytes, cudaMemcpyDeviceToDevice, s));
    }
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
    if (cfg.arena_budget_entries < 0 || cfg.stream_log_entries < 0)
      throw std::runtime_error("arena_budget_entries / stream_log_entries must be >= 0");
    // Entry indices are int32 device-side; leave headroom for the streaming per-chunk
    // rounding below (< one chunk per lane).
    constexpr int64_t kMaxArenaEntries = 1LL << 30;
    const int64_t arena_budget =
        cfg.arena_budget_entries > 0
            ? std::min<int64_t>(cfg.arena_budget_entries, kMaxArenaEntries)
            : std::min<int64_t>(512LL << 20,
                                std::max<int64_t>(64LL << 20, (16LL << 20) * sz.lanes));
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
    sz.stream_log_cap = 0;
    if (cfg.streaming) {
      // Per-channel slices commit at CreateStream and unmap at ReleaseStream, so each
      // slice is rounded up to whole physical mapping chunks (no chunk straddles two
      // channels; the reservation grows by < one chunk per lane).
      int64_t cap = cfg.stream_log_entries > 0
                        ? std::min<int64_t>(cfg.stream_log_entries,
                                            (1LL << 30) / sz.lanes)
                        : sz.arena_cap / sz.lanes;
      const size_t chunk = LazyRegion::ChunkBytesFor(o.device);
      if (chunk > 0) {
        const int64_t per_chunk = static_cast<int64_t>(chunk / sizeof(int2));
        cap = (cap + per_chunk - 1) / per_chunk * per_chunk;
      }
      sz.stream_log_cap = static_cast<int32_t>(cap);
      sz.arena_cap = static_cast<int32_t>(cap * sz.lanes);
    }
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
    auto* w = Alloc<float>(g.num_arcs);
    WFST_CUDA_CHECK(cudaMemcpy(rs, g.row_splits.data(), (g.num_states + 1) * 4,
                               cudaMemcpyHostToDevice));
    WFST_CUDA_CHECK(
        cudaMemcpy(fc, g.final_count.data(), g.num_states * 4, cudaMemcpyHostToDevice));
    WFST_CUDA_CHECK(
        cudaMemcpy(w, g.arc_weight.data(), g.num_arcs * 4, cudaMemcpyHostToDevice));
    // De-interleave arc columns (SoA): dest as i32, ilabel as u16 (-1 -> 0xFFFF). The
    // kernels then stream only what each pass needs. Per-state max EMIT weight feeds the
    // K2 upper-bound skip tests.
    {
      std::vector<int32_t> dest(g.num_arcs);
      std::vector<uint16_t> il(g.num_arcs);
      for (int64_t a = 0; a < g.num_arcs; ++a) {
        dest[a] = g.arc_dest_ilabel[2 * a];
        const int32_t l = g.arc_dest_ilabel[2 * a + 1];
        if (l < -1 || l >= 0xFFFF)
          throw std::runtime_error("arc ilabel out of the u16 range [-1, 65534]");
        il[a] = static_cast<uint16_t>(l);
      }
      std::vector<float> maxw(g.num_states, -INFINITY);
      for (int32_t s = 0; s < static_cast<int32_t>(g.num_states); ++s) {
        for (int32_t a = g.EmitBegin(s); a < g.EmitEnd(s); ++a)
          maxw[s] = std::max(maxw[s], g.arc_weight[a]);
      }
      auto* dd = Alloc<int32_t>(g.num_arcs);
      auto* dl = Alloc<uint16_t>(g.num_arcs);
      auto* mw = Alloc<float>(g.num_states);
      WFST_CUDA_CHECK(cudaMemcpy(dd, dest.data(), g.num_arcs * 4, cudaMemcpyHostToDevice));
      WFST_CUDA_CHECK(cudaMemcpy(dl, il.data(), g.num_arcs * 2, cudaMemcpyHostToDevice));
      WFST_CUDA_CHECK(cudaMemcpy(mw, maxw.data(), g.num_states * 4, cudaMemcpyHostToDevice));
      dg.arc_dest = dd;
      dg.arc_ilabel = dl;
      dg.emit_maxw = mw;
    }
    dg.row_splits = rs;
    dg.final_count = fc;
    dg.weight = w;

    const int64_t L = sz.lanes;
    for (int b = 0; b < 2; ++b) {
      ws.tok_state[b] = Alloc<int32_t>(L * sz.main_q);
      ws.tok_score[b] = Alloc<float>(L * sz.main_q);
      ws.tok_winner[b] = Alloc<int32_t>(L * sz.main_q);
    }
    ws.tok_emit_begin = Alloc<int32_t>(L * sz.main_q);
    ws.tok_ub = Alloc<float>(L * sz.main_q);
    ws.arc_offsets = Alloc<int32_t>(L * (sz.main_q + 1));
    ws.lane_arc_offsets = Alloc<int32_t>(L + 1);
    ws.next_claims = Alloc<int2>(L * sz.claims_cap);
    ws.cand = Alloc<int2>(L * sz.cand_cap);
    ws.hash_key = Alloc<uint32_t>(L * sz.hash_cap);
    ws.hash_payload = Alloc<unsigned long long>(L * sz.hash_cap);
    if (g.has_eps || cfg.lattice) {
      ws.hash_pos = Alloc<int32_t>(L * sz.hash_cap);
      WFST_CUDA_CHECK(cudaMemset(ws.hash_pos, 0xFF, L * sz.hash_cap * sizeof(int32_t)));
    }
    // Winners / lattice arenas are lazily-committed regions: the full capacity is
    // reserved as stable virtual addresses (captured graphs stay valid) but physical
    // memory is mapped on demand, growing between idempotent decode attempts.
    winners_mem.Reserve(static_cast<size_t>(sz.arena_cap) * sizeof(int2), o.device);
    ws.winners = static_cast<int2*>(winners_mem.ptr());
    ws.arena_cursor = Alloc<int32_t>(1);
    ws.arena_limit = Alloc<int32_t>(1);
    ws.arc_out = Alloc<int32_t>(L * sz.path_cap);
    ws.arc_out_len = Alloc<int32_t>(L);
    ws.lanes = Alloc<LaneCounters>(L);
    d_T = Alloc<int32_t>(L);
    d_batch = Alloc<int32_t>(1);
    ws.lat_limit = ws.arena_limit;  // aliased when lattice mode is off (never read)
    if (sz.lat_cap > 0) {
      ws.cand_end = Alloc<float>(L * sz.cand_cap);
      tok_fwd_mem.Reserve(static_cast<size_t>(sz.arena_cap) * 4, o.device);
      tok_bwd_mem.Reserve(static_cast<size_t>(sz.arena_cap) * 4, o.device);
      ws.tok_fwd = static_cast<float*>(tok_fwd_mem.ptr());
      ws.tok_bwd = static_cast<uint32_t*>(tok_bwd_mem.ptr());
      lat_mem.Reserve(static_cast<size_t>(sz.lat_cap) * sizeof(int4), o.device);
      ws.lat = static_cast<int4*>(lat_mem.ptr());
      ws.lat_cursor = Alloc<int32_t>(1);
      ws.lat_limit = Alloc<int32_t>(1);
      ws.lat_seg = Alloc<int2>(L * sz.path_cap);
      if (cfg.lat_prune_interval > 0) {
        lat2_mem.Reserve(static_cast<size_t>(sz.lat_cap) * sizeof(int4), o.device);
        ws.lat2 = static_cast<int4*>(lat2_mem.ptr());
        ws.lat2_cursor = Alloc<int32_t>(1);
      }
      lat_out_cap = static_cast<int32_t>(
          std::min<int64_t>(32LL << 20, std::max<int64_t>(8LL << 20, (1LL << 20) * L)));
      lat_out_mem.Reserve(static_cast<size_t>(lat_out_cap) * 8 * sizeof(int32_t), o.device);
      ws.lat_out = static_cast<int32_t*>(lat_out_mem.ptr());
      ws.lat_out_cursor = Alloc<int32_t>(1);
      ws.lat_out_len = Alloc<int32_t>(L);
    }
    // Initial physical commitment. Streaming commits nothing here: each channel's slice
    // is mapped at CreateStream and unmapped at ReleaseStream, so the footprint tracks
    // ACTIVE channels. Batch mode starts small and grows on demand. Streaming lane
    // counters must be explicitly deadened — no InitKernel runs in streaming mode, and
    // a never-created lane would otherwise carry recycled-allocation garbage into the
    // shared per-step kernels (BacktrackKernel walks any lane with status == 1).
    if (cfg.streaming) {
      arena_limit_host = sz.arena_cap;  // streaming appends bound against stream_log_cap
      WFST_CUDA_CHECK(
          cudaMemcpy(ws.arena_limit, &arena_limit_host, 4, cudaMemcpyHostToDevice));
      StreamResetKernel<<<(sz.lanes + 63) / 64, 64, 0, stream>>>(ws, sz, -1);
      WFST_CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
      SetArenaCommit(std::min<int64_t>(sz.arena_cap, kInitArenaEntries));
    }
    if (sz.lat_cap > 0) {
      SetLatCommit(std::min<int64_t>(sz.lat_cap, kInitLatEntries));
      SetLatOutCommit(std::min<int64_t>(lat_out_cap, kInitLatOutRecords));
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

  size_t StreamSliceBytes() const {
    return static_cast<size_t>(sz.stream_log_cap) * sizeof(int2);
  }

  ~Impl() {
    for (auto& [b, e] : graph_execs) cudaGraphExecDestroy(e);
    for (auto& [b, e] : chunk_execs) cudaGraphExecDestroy(e);
    for (auto& [b, e] : steps_execs) cudaGraphExecDestroy(e);
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

GpuDecoder::MemStats GpuDecoder::GetMemStats() const {
  const Impl& im = *impl_;
  MemStats s;
  for (const LazyRegion* r : {&im.winners_mem, &im.tok_fwd_mem, &im.tok_bwd_mem,
                              &im.lat_mem, &im.lat2_mem, &im.lat_out_mem, &im.staging_mem}) {
    s.reserved_bytes += static_cast<int64_t>(r->reserved());
    s.committed_bytes += static_cast<int64_t>(r->committed());
  }
  s.fixed_bytes = im.fixed_bytes;
  s.arena_high_water = im.arena_high_water;
  return s;
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
    // Map this channel's winners slice (unmapped while the channel was closed).
    im.winners_mem.EnsureRange(static_cast<size_t>(lane) * im.StreamSliceBytes(),
                               im.StreamSliceBytes());
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
  if (channel < 0 || channel >= im.sz.lanes || !im.channel_used[channel]) return;
  WFST_CUDA_CHECK(cudaSetDevice(im.opts.device));
  // Deaden the lane BEFORE unmapping its winners slice, and drain the stream so no
  // in-flight kernel still reads the slice: the shared per-step kernels gate on
  // status / frontier_size, so nothing walks the released chain afterwards.
  StreamResetKernel<<<(im.sz.lanes + 63) / 64, 64, 0, im.stream>>>(im.ws, im.sz, channel);
  WFST_CUDA_CHECK(cudaStreamSynchronize(im.stream));
  im.winners_mem.ReleaseRange(static_cast<size_t>(channel) * im.StreamSliceBytes(),
                              im.StreamSliceBytes());
  im.channel_used[channel] = false;
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
  std::vector<LaneCounters> lanes(batch);
  std::vector<int32_t> arc_out(batch * sz.path_cap);
  std::vector<int32_t> arc_len(batch);
  int32_t n_lat_records = 0;
  bool lat_truncated = false;

  // Decode is idempotent (InitKernel resets all state), so an arena overflow against the
  // lazily-committed capacity is handled by growing the physical backing and re-running.
  // The device pointers never change, so captured graphs stay valid across attempts.
  for (int32_t attempt = 0;; ++attempt) {
    if (interval_mode) {
      // Segmented loop with periodic lattice-arena pruning (long-form audio).
      im.EnsureStaging(vocab_stride);
      im.StageLogProbs(d_log_probs, batch, max_frames, im.stream);
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
          WFST_CUDA_CHECK(cudaMemcpyAsync(
              im.ws.lat, im.ws.lat2, static_cast<int64_t>(im.lat_limit_host) * sizeof(int4),
              cudaMemcpyDeviceToDevice, im.stream));
          LatSwapBackKernel<<<1, 1, 0, im.stream>>>(im.ws);
        }
      }
      BacktrackKernel<<<(sz.lanes + 63) / 64, 64, 0, im.stream>>>(im.ws, sz);
      HashSanitizeKernel<<<dim3(64, static_cast<unsigned>(sz.lanes)), 256, 0, im.stream>>>(
          im.ws, sz);
    } else if (use_graphs) {
      // Stage log-probs into the decoder-owned buffer so kernel pointers are capture-
      // stable: per-lane D2D copies of T*V rows into max_frames*V-stride slots.
      im.EnsureStaging(vocab_stride);
      im.StageLogProbs(d_log_probs, batch, max_frames, im.stream);
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
          im.ws, sz, im.dg, cfg.output_beam, im.lat_out_limit_host);
    }
    int32_t cursor_end = 0;
    WFST_CUDA_CHECK(cudaMemcpyAsync(lanes.data(), im.ws.lanes, batch * sizeof(LaneCounters),
                                    cudaMemcpyDeviceToHost, im.stream));
    WFST_CUDA_CHECK(cudaMemcpyAsync(arc_out.data(), im.ws.arc_out,
                                    batch * sz.path_cap * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost, im.stream));
    WFST_CUDA_CHECK(cudaMemcpyAsync(arc_len.data(), im.ws.arc_out_len,
                                    batch * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                    im.stream));
    WFST_CUDA_CHECK(cudaMemcpyAsync(&cursor_end, im.ws.arena_cursor, sizeof(int32_t),
                                    cudaMemcpyDeviceToHost, im.stream));
    n_lat_records = 0;
    if (cfg.lattice) {
      WFST_CUDA_CHECK(cudaMemcpyAsync(&n_lat_records, im.ws.lat_out_cursor, sizeof(int32_t),
                                      cudaMemcpyDeviceToHost, im.stream));
    }
    WFST_CUDA_CHECK(cudaStreamSynchronize(im.stream));
    im.arena_high_water = std::max(im.arena_high_water, cursor_end);

    bool arena_ovf = false;
    for (int64_t b = 0; b < batch; ++b) arena_ovf |= (lanes[b].overflow & kOverflowArena) != 0;
    lat_truncated = cfg.lattice && n_lat_records > im.lat_out_limit_host;
    if ((!arena_ovf && !lat_truncated) || attempt >= Impl::kMaxGrowAttempts) break;
    bool grew = false;
    if (arena_ovf) {
      grew |= im.GrowArena();
      grew |= im.GrowLat();
    }
    if (lat_truncated) grew |= im.GrowLatOut(n_lat_records);
    if (!grew) break;  // at full reserved capacity: report overflow as before
  }

  if (cfg.lattice) {
    lat_truncated = n_lat_records > im.lat_out_limit_host;
    n_lat_records = std::min(n_lat_records, im.lat_out_limit_host);
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
