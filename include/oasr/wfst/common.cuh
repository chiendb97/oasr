#pragma once

// Shared device-side infrastructure for the WFST decoder kernels: score-ordering helpers,
// the device graph view + arc-range accessors, per-lane counters, and the Workspace/Sizes
// PODs passed by value into every kernel. Framework-agnostic (no Torch); #included into
// the single decoder translation unit (csrc/decoder.cu) together with the per-family
// kernel headers. See .claude/skills/add-cuda-kernel/SKILL.md.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace oasr::wfst {
namespace kernels {

constexpr uint32_t kEmptyKey = 0xFFFFFFFFu;
constexpr float kNegInf = -INFINITY;
constexpr int kMaxProbes = 256;

// Overflow bits (DecodeResult::overflow).
constexpr uint32_t kOverflowCand = 1u;      // candidate buffer full
constexpr uint32_t kOverflowHash = 2u;      // probe bound exceeded
constexpr uint32_t kOverflowArena = 4u;     // shared winners arena full
constexpr uint32_t kOverflowClaims = 8u;    // raw distinct-state claims list full
constexpr uint32_t kOverflowKept = 16u;     // surviving frontier > main_q

// Lane phase for the current step.
enum Phase : int32_t {
  kPhaseReal = 0,      // emitting expansion of frame t
  kPhaseFinal = 1,     // final step, real -1 arcs
  kPhaseRedirect = 2,  // final step, allow_partial redirect (no reachable final arc)
  kPhaseDone = 3,
  kPhaseEps = 4,       // epsilon-closure pass over the just-built frontier (TLG)
};

// Monotonic float->uint mapping: uint compare == float compare.
__host__ __device__ inline uint32_t BitsOfFloat(float f) {
#ifdef __CUDA_ARCH__
  return __float_as_uint(f);
#else
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
#endif
}
__host__ __device__ inline float FloatOfBits(uint32_t u) {
#ifdef __CUDA_ARCH__
  return __uint_as_float(u);
#else
  float f;
  std::memcpy(&f, &u, 4);
  return f;
#endif
}
__host__ __device__ inline uint32_t FloatToOrderedUint(float f) {
  const uint32_t u = BitsOfFloat(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__host__ __device__ inline float OrderedUintToFloat(uint32_t u) {
  u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
  return FloatOfBits(u);
}

struct LaneCounters {
  int32_t frontier_size;
  int32_t next_raw;      // #next-frontier entries appended by hash claimers this step
  int32_t cand_count;    // #candidates appended this step
  int32_t t;             // current step (== frame index; == T means final step)
  int32_t T;             // real frames for this lane (INT32_MAX while streaming = k2
                         // online beam semantics; FinalizeStream sets T = t)
  int32_t chunk_start;   // first frame of the current chunk (log-prob row 0)
  int32_t chunk_end;     // decode while t < chunk_end (offline: T + 1)
  int32_t log_len;       // streaming: LOGICAL winners appended in this lane (monotonic;
                         // the ring maps logical -> physical, see WinnersEntry)
  int32_t cand_consumed; // eps mode: candidates already resolved this frame
  int32_t cand_emit;     // candidates [0, cand_emit) are emitting/final; rest are eps
  int32_t phase;
  int32_t has_valid_final;
  int32_t total_arcs;    // arcs to expand this step
  float dyn_beam;
  float max_lp;          // kPhaseReal: max log-prob of this lane's frame row (else 0)
  uint32_t running_max;  // ordered-uint of the step's best end score
  uint32_t overflow;
  int32_t status;        // 0 running, 1 ok-finished, 2 dead
  int32_t reached_final;
  float final_score;
  int32_t final_tok;     // winners-arena index (GLOBAL) of the accepted final token
  float lat_best;        // interval prune: reference score for the loose keep rule
  // redirect two-pass helpers
  int32_t redirect_claimed;
  int32_t gc_root;       // streaming GC: logical id of the lane's finalized sentinel;
                         // [gc_root, log_len) is the live ring window (tail position:
                         // keeps the hot fields above at their pre-GC offsets)
  // Absolute pointer to this lane's CURRENT log-prob row, computed by K1 from the
  // device-resident LpDesc (0 outside kPhaseReal). K2a/K2b read it off the lane-counter
  // line they already touch — the hot kernels never load the descriptor.
  unsigned long long lp_row_ptr;
};

struct DeviceGraph {
  const int32_t* row_splits;
  const int32_t* final_count;
  const int32_t* eps_count;  // nullptr on epsilon-free graphs
  // Arc columns are split (SoA) so each kernel loads only what it needs: the max
  // pre-pass never touches dest, and expand touches dest only for beam survivors.
  const int32_t* arc_dest;
  const uint16_t* arc_ilabel;  // 0xFFFF encodes ilabel -1 (final arcs)
  const float* weight;
  const float* emit_maxw;  // per state: max weight over its EMITTING arcs (-inf if none)
  int32_t num_states;
  int32_t vocab_size;
  int32_t start_state;
  bool finals_at_end;
  bool eps_first;
};

constexpr uint16_t kIlabelFinal = 0xFFFFu;

__device__ inline int32_t ArcDest(const DeviceGraph& g, int32_t arc) {
  return g.arc_dest[arc];
}
__device__ inline int32_t ArcIlabel(const DeviceGraph& g, int32_t arc) {
  const uint16_t v = g.arc_ilabel[arc];
  return v == kIlabelFinal ? -1 : static_cast<int32_t>(v);
}

__device__ inline int32_t EpsCountOf(const DeviceGraph& g, int32_t s) {
  return g.eps_count != nullptr ? g.eps_count[s] : 0;
}
__device__ inline int32_t RestBegin(const DeviceGraph& g, int32_t s) {
  return g.row_splits[s] + (g.finals_at_end ? 0 : g.final_count[s]);
}
__device__ inline int32_t RestEnd(const DeviceGraph& g, int32_t s) {
  return g.row_splits[s + 1] - (g.finals_at_end ? g.final_count[s] : 0);
}
__device__ inline int32_t EmitBegin(const DeviceGraph& g, int32_t s) {
  return RestBegin(g, s) + (g.eps_first ? EpsCountOf(g, s) : 0);
}
__device__ inline int32_t EmitCount(const DeviceGraph& g, int32_t s) {
  return RestEnd(g, s) - RestBegin(g, s) - EpsCountOf(g, s);
}
__device__ inline int32_t EpsBegin(const DeviceGraph& g, int32_t s) {
  return g.eps_first ? RestBegin(g, s) : RestEnd(g, s) - EpsCountOf(g, s);
}
__device__ inline int32_t FinalBegin(const DeviceGraph& g, int32_t s) {
  return g.finals_at_end ? g.row_splits[s + 1] - g.final_count[s] : g.row_splits[s];
}

// Log-prob access descriptor, device-resident: captured graphs bake kernel parameters,
// so the lp base pointer and strides live in device memory and are re-written (one 24 B
// H2D) before every launch — offline batches decode the CALLER's [B, T, V] tensor in
// place (no staging buffer, no per-lane copies); streaming points it at the fixed
// per-channel staging slots once. Strides are in elements.
struct LpDesc {
  unsigned long long base;
  long long lane_stride;
  long long frame_stride;  // == the row length (vocab stride) for the K1 row max
};

struct Workspace {
  // Frontier (double-buffered), per lane, capacity main_q.
  int32_t* tok_state[2];
  float* tok_score[2];
  int32_t* tok_winner[2];      // GLOBAL winners-arena index of the token
  int32_t* tok_emit_begin;     // first arc to expand per token (phase-dependent), cur only
  float* tok_ub;               // per token: score + emit_maxw[state] (kPhaseReal), cur only
  int32_t* arc_offsets;        // per lane: exclusive prefix sums, [main_q + 1]
  int32_t* lane_arc_offsets;   // [lanes + 1]: exclusive scan of total_arcs (K2 flattening)
  // Next-frontier raw claims {state, hash_slot}.
  int2* next_claims;
  // Per-step candidates {prev_local, arc}.
  int2* cand;
  // Recombination hash.
  uint32_t* hash_key;
  unsigned long long* hash_payload;
  int32_t* hash_pos;  // eps mode: state -> frontier position map (per claimed slot)
  // Shared winners arena {prev_global, arc}: all lanes allocate blocks from one pool via
  // the global cursor. Statistical sizing — per-lane worst case would need 10s of GB.
  // The arena (and the lattice arenas) are lazily-committed regions; the device-resident
  // limits below carry the currently committed capacity so the host can grow the
  // physical backing between (idempotent) decode attempts without re-capturing graphs.
  int2* winners;
  int32_t* arena_cursor;  // single global counter
  int32_t* arena_limit;   // committed winners-arena entries (== sz.arena_cap when eager)
  int32_t* lat_limit;     // committed lattice-arena entries (== sz.lat_cap when eager)
  // Backtracked best-path arcs per lane (reversed), filled by BacktrackKernel.
  int32_t* arc_out;      // [lane][path_cap]
  int32_t* arc_out_len;  // [lane]
  // Lattice mode: per-token forward scores + backward scores (ordered-uint; 0 == -inf),
  // parallel to the winners arena; candidate arena {src_tok, dest_tok, arc|redirect_bit,
  // end_bits} with per-(lane,frame) segments.
  float* cand_end;  // [lane][cand_cap], end scores of this step's candidates
  float* tok_fwd;
  uint32_t* tok_bwd;
  int4* lat;
  int4* lat2;         // interval-prune scratch (compaction bounce buffer)
  int32_t* lat_cursor;
  int32_t* lat2_cursor;
  int2* lat_seg;  // [lane][step] = {base, count}
  // Backward-pruned lattice arc records, 6 x i32 each:
  // {src_tok, dst_tok, label, arc_map, score_bits, t}; per-lane counts in lat_out_len.
  int32_t* lat_out;
  int32_t* lat_out_cursor;
  int32_t* lat_out_len;  // [lane]
  LaneCounters* lanes;
  // Winners-log GC (offline long-form, cfg.gc_interval > 0): per-lane convergence
  // point of the current frontier's chains, and the finalized golden-prefix arcs the
  // host drains between segment graphs (newest -> oldest as walked).
  int32_t* gc_conv;   // [lanes]; INT32_MAX none, -1 aborted (defensive)
  int32_t* fin_arcs;  // [lane][path_cap]
  int32_t* fin_len;   // [lanes]
  LpDesc* lp_desc;  // [1]; see LpDesc
  // First winners index still physically mapped (0 unless GC released a prefix).
  // Fault shield for every winners chain walk: clean lanes stop at their finalized
  // sentinel (always >= the floor), but an arena-overflow-degraded lane can carry a
  // stale frontier pointer below it (its frontier counts appends the full arena
  // dropped) — such a walk must stop rather than touch unmapped memory.
  int32_t* gc_floor;  // [1], always allocated
  // Debug snapshots.
  int2* snap;         // {state, score-bits}, [lane][frame][main_q]
  int32_t* snap_len;  // [lane][frame]
};

struct Sizes {
  int32_t lanes;
  int32_t main_q;      // surviving-frontier capacity
  int32_t claims_cap;  // raw distinct-state claims per step (loose-admit superset)
  int32_t cand_cap;
  int32_t hash_cap;   // power of two
  int32_t lp_half;    // log-probs stored as fp16
  int32_t arena_cap;  // shared winners-arena entries (global)
  int32_t path_cap;   // max best-path arcs per lane (max_frames + 2)
  int32_t lat_cap;    // shared lattice-candidate arena entries (0 = 1-best mode)
  int32_t stream_log_cap;  // streaming: per-lane winners RING entries (0 = batch mode)
  int32_t fin_cap;    // GC: per-lane finalized-arc staging entries (0 = GC off)
  int32_t snap_frames;  // 0 = disabled
};

// Winners access: batch mode uses global indices verbatim; streaming stores LOGICAL
// per-lane monotonic ids in every pointer field (tok_winner, final_tok, entry.x) and
// maps them onto the lane's fixed ring here. Ids stay monotonic so chain order,
// sentinels (-1 start, INT32_MIN finalized root) and log_len semantics survive wraps;
// the writer never laps the live window (K3/EpsResolve guard against gc_root).
__device__ inline int2& WinnersEntry(const Workspace& ws, const Sizes& sz, int32_t lane,
                                     int32_t idx) {
  const int64_t phys =
      sz.stream_log_cap > 0
          ? static_cast<int64_t>(lane) * sz.stream_log_cap + (idx % sz.stream_log_cap)
          : static_cast<int64_t>(idx);
  return ws.winners[phys];
}

constexpr int32_t kRedirectArcBit = 1 << 30;  // lattice arc labeled -1 (allow_partial)
constexpr int32_t kEpsArcBit = 1 << 29;       // lattice arc from an epsilon hop (label 0;
                                              // src and dst tokens share a frame)
constexpr int32_t kGcStampBit = 1 << 30;      // winners-GC convergence stamp, set on the
                                              // anchor chain's arc field during a GC round
                                              // and cleared before the next decode step
                                              // (requires graph arc ids < 2^30)
constexpr int32_t kGcDoneTok = -2;            // final_tok value of a fully-finalized lane
                                              // (whole path emitted host-side; Backtrack's
                                              // w >= 0 loop no-ops without reading winners)

// Log-prob load: fp32 or fp16 storage, f32 accumulation everywhere.
__device__ inline float LoadLp(const void* lp, int64_t idx, bool half) {
  return half ? __half2float(__ldg(reinterpret_cast<const __half*>(lp) + idx))
              : __ldg(reinterpret_cast<const float*>(lp) + idx);
}

__device__ inline uint32_t HashState(uint32_t s, uint32_t mask) {
  s ^= s >> 16;
  s *= 0x85ebca6bu;
  s ^= s >> 13;
  s *= 0xc2b2ae35u;
  s ^= s >> 16;
  return s & mask;
}

// ---------------------------------------------------------------------------
// Block-wide inclusive scan of one int32 per thread (warp shuffles + one shared
// round). blockDim.x must be a multiple of 32 and <= 1024; warp_scratch needs 32
// int32 slots. Contains __syncthreads (call from uniform control flow); safe to
// call in a loop back-to-back (trailing sync protects the scratch).
__device__ inline int32_t BlockInclusiveScan(int32_t x, int32_t* warp_scratch) {
  const int32_t lane = threadIdx.x & 31;
  const int32_t wid = threadIdx.x >> 5;
  int32_t v = x;
#pragma unroll
  for (int32_t d = 1; d < 32; d <<= 1) {
    const int32_t y = __shfl_up_sync(0xFFFFFFFFu, v, d);
    if (lane >= d) v += y;
  }
  if (lane == 31) warp_scratch[wid] = v;
  __syncthreads();
  if (wid == 0) {
    const int32_t nw = static_cast<int32_t>(blockDim.x) >> 5;
    int32_t w = (threadIdx.x < nw) ? warp_scratch[threadIdx.x] : 0;
#pragma unroll
    for (int32_t d = 1; d < 32; d <<= 1) {
      const int32_t y = __shfl_up_sync(0xFFFFFFFFu, w, d);
      if (lane >= d) w += y;
    }
    warp_scratch[threadIdx.x] = w;
  }
  __syncthreads();
  if (wid > 0) v += warp_scratch[wid - 1];
  __syncthreads();
  return v;
}

// ---------------------------------------------------------------------------
// Largest i in [0, n) with offsets[i] <= slot (offsets nondecreasing, n >= 1).
__device__ inline int32_t UpperTokenGlobal(const int32_t* offsets, int32_t n, int32_t slot) {
  int32_t lo = 0, hi = n - 1;
  while (lo < hi) {
    const int32_t mid = (lo + hi + 1) >> 1;
    if (offsets[mid] <= slot) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

__device__ inline uint32_t WarpMaxU32(uint32_t v) {
#if __CUDA_ARCH__ >= 800
  return __reduce_max_sync(0xFFFFFFFFu, v);
#else
#pragma unroll
  for (int32_t d = 16; d > 0; d >>= 1)
    v = max(v, __shfl_down_sync(0xFFFFFFFFu, v, d));
  return __shfl_sync(0xFFFFFFFFu, v, 0);
#endif
}

}  // namespace kernels
}  // namespace oasr::wfst
