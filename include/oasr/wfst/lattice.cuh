#pragma once

// Lattice kernels: eager GPU backward output-beam prune, interval pruning for long-form
// audio, flat arc-record emission, and post-closure canonical candidate persistence.
// Active only in lattice mode (Sizes::lat_cap > 0).
#include "oasr/wfst/common.cuh"

namespace oasr::wfst {
namespace kernels {

// ---------------------------------------------------------------------------
// Lattice backward pass (eager, after the forward loop). k2 semantics: keep arc iff
// forward[src] + arc + backward[dst] >= best - output_beam, backward seeded 0 at the
// final token. tok_bwd stores ordered-uint scores with 0 == unreachable.
__global__ void LatInitBwdKernel(Workspace ws) {
  // Clamp: the cursor keeps counting past the committed capacity on overflow (the
  // failed appends are flagged, not written).
  const int32_t n = min(*ws.arena_cursor, *ws.arena_limit);
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    ws.tok_bwd[i] = 0u;
}

__global__ void LatSeedKernel(Workspace ws, Sizes sz) {
  const int32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= sz.lanes) return;
  const LaneCounters& lc = ws.lanes[lane];
  if (lc.status == 1) ws.tok_bwd[lc.final_tok] = FloatToOrderedUint(0.0f);
}

__global__ void LatBackwardStepKernel(Workspace ws, Sizes sz, int32_t t) {
  const int32_t lane = blockIdx.y;
  const int2 seg = ws.lat_seg[static_cast<int64_t>(lane) * sz.path_cap + t];
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < seg.y;
       i += gridDim.x * blockDim.x) {
    const int4 e = ws.lat[seg.x + i];
    if (e.y < 0) continue;
    const uint32_t bd = ws.tok_bwd[e.y];
    if (bd == 0u) continue;
    const float end = FloatOfBits(static_cast<uint32_t>(e.w));
    const float b = (end - ws.tok_fwd[e.x]) + OrderedUintToFloat(bd);
    atomicMax(&ws.tok_bwd[e.x], FloatToOrderedUint(b));
  }
}

// ---------------------------------------------------------------------------
// Interval pruning (long-form lattice decoding). k2-style window-loose rule: backward
// scores seeded 0 at the CURRENT frontier (and at final tokens of finished lanes); an
// entry survives iff end + bwd(dst) >= lane_best - output_beam. Survivors are compacted
// per (lane, frame) into the scratch arena, then copied back so captured-graph pointers
// stay valid.

__global__ void LatSeedFrontierKernel(Workspace ws, Sizes sz) {
  const int32_t lane = blockIdx.x;
  LaneCounters& lc = ws.lanes[lane];
  __shared__ uint32_t sh_max[1024];
  if (lc.status == 2 || lc.phase == kPhaseDone) {
    if (threadIdx.x == 0) lc.lat_best = -INFINITY;
    return;
  }
  if (lc.status == 1) {
    if (threadIdx.x == 0) {
      ws.tok_bwd[lc.final_tok] = FloatToOrderedUint(0.0f);
      lc.lat_best = lc.final_score;
    }
    return;
  }
  const float* score = ws.tok_score[0] + static_cast<int64_t>(lane) * sz.main_q;
  const int32_t* winner = ws.tok_winner[0] + static_cast<int64_t>(lane) * sz.main_q;
  uint32_t local = 0;
  for (int32_t i = threadIdx.x; i < lc.frontier_size; i += blockDim.x) {
    ws.tok_bwd[winner[i]] = FloatToOrderedUint(0.0f);
    local = max(local, FloatToOrderedUint(score[i]));
  }
  sh_max[threadIdx.x] = local;
  __syncthreads();
  for (int32_t d = blockDim.x >> 1; d > 0; d >>= 1) {
    if (threadIdx.x < d) sh_max[threadIdx.x] = max(sh_max[threadIdx.x], sh_max[threadIdx.x + d]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    lc.lat_best = (lc.frontier_size > 0) ? OrderedUintToFloat(sh_max[0]) : -INFINITY;
}

// CTA per lane: loose keep test + per-(lane,frame) compaction into the scratch arena.
__global__ void LatIntervalCompactKernel(Workspace ws, Sizes sz, float output_beam,
                                         int32_t t_now) {
  const int32_t lane = blockIdx.x;
  const LaneCounters& lc = ws.lanes[lane];
  const float best = lc.lat_best;
  const int32_t lat_lim = *ws.lat_limit;
  __shared__ int32_t sh_scan[1024];
  __shared__ int32_t sh_base;
  __shared__ int32_t sh_frame_total;
  for (int32_t t = 0; t < t_now; ++t) {
    int2* seg = &ws.lat_seg[static_cast<int64_t>(lane) * sz.path_cap + t];
    const int2 s = *seg;
    if (s.y == 0) continue;
    // Pass A: count this frame's survivors.
    int32_t local = 0;
    for (int32_t j = threadIdx.x; j < s.y; j += blockDim.x) {
      const int4 e = ws.lat[s.x + j];
      if (e.y >= 0) {
        const uint32_t bd = ws.tok_bwd[e.y];
        if (bd != 0u &&
            FloatOfBits(static_cast<uint32_t>(e.w)) + OrderedUintToFloat(bd) >=
                best - output_beam)
          local += 1;
      }
    }
    sh_scan[threadIdx.x] = local;
    __syncthreads();
    for (int32_t d = 1; d < blockDim.x; d <<= 1) {
      int32_t v = (threadIdx.x >= d) ? sh_scan[threadIdx.x - d] : 0;
      __syncthreads();
      sh_scan[threadIdx.x] += v;
      __syncthreads();
    }
    const int32_t my_prefix = sh_scan[threadIdx.x] - local;  // exclusive
    if (threadIdx.x == blockDim.x - 1) {
      sh_frame_total = sh_scan[blockDim.x - 1];
      sh_base = (sh_frame_total > 0) ? atomicAdd(ws.lat2_cursor, sh_frame_total) : 0;
    }
    __syncthreads();
    // Pass B: write survivors in frame order (stable within a thread's stride walk).
    int32_t written = 0;
    for (int32_t j = threadIdx.x; j < s.y; j += blockDim.x) {
      const int4 e = ws.lat[s.x + j];
      bool keep = false;
      if (e.y >= 0) {
        const uint32_t bd = ws.tok_bwd[e.y];
        keep = (bd != 0u &&
                FloatOfBits(static_cast<uint32_t>(e.w)) + OrderedUintToFloat(bd) >=
                    best - output_beam);
      }
      if (keep && sh_base + my_prefix + written < lat_lim) {
        ws.lat2[sh_base + my_prefix + written] = e;
        ++written;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) *seg = make_int2(sh_base, sh_frame_total);
    __syncthreads();
  }
}

__global__ void LatSwapBackKernel(Workspace ws) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *ws.lat_cursor = *ws.lat2_cursor;
}

// CTA per lane: output-beam keep test + compaction into the flat record buffer.
// Record = 8 x i32 {src_tok, dst_tok, label, arc_map, score_bits, t, lane, 0}.
__global__ void LatEmitKernel(Workspace ws, Sizes sz, DeviceGraph g, float output_beam,
                              int32_t out_cap) {
  const int32_t lane = blockIdx.x;
  const LaneCounters& lc = ws.lanes[lane];
  __shared__ int32_t sh_scan[1024];
  __shared__ int32_t sh_base;
  int32_t emitted = 0;
  if (lc.status == 1) {
    const float best = lc.final_score;
    const int32_t last_seg = min(lc.T + 1, sz.path_cap - 1);
    for (int32_t t = 0; t <= last_seg; ++t) {
      const int2 seg = ws.lat_seg[static_cast<int64_t>(lane) * sz.path_cap + t];
      for (int32_t base = 0; base < seg.y; base += blockDim.x) {
        const int32_t j = base + threadIdx.x;
        int32_t keep = 0;
        int4 e{};
        float end = 0.0f, bwd = 0.0f;
        if (j < seg.y) {
          e = ws.lat[seg.x + j];
          if (e.y >= 0) {
            const uint32_t bd = ws.tok_bwd[e.y];
            if (bd != 0u) {
              end = FloatOfBits(static_cast<uint32_t>(e.w));
              bwd = OrderedUintToFloat(bd);
              if (end + bwd >= best - output_beam) keep = 1;
            }
          }
        }
        sh_scan[threadIdx.x] = keep;
        __syncthreads();
        for (int32_t d = 1; d < blockDim.x; d <<= 1) {
          int32_t v = (threadIdx.x >= d) ? sh_scan[threadIdx.x - d] : 0;
          __syncthreads();
          sh_scan[threadIdx.x] += v;
          __syncthreads();
        }
        if (threadIdx.x == 0) {
          const int32_t round = sh_scan[blockDim.x - 1];
          sh_base = (round > 0) ? atomicAdd(ws.lat_out_cursor, round) : 0;
        }
        __syncthreads();
        if (keep) {
          const int32_t idx = sh_base + sh_scan[threadIdx.x] - 1;
          if (idx < out_cap) {
            const bool redirect = (e.z & kRedirectArcBit) != 0;
            const bool is_eps = (e.z & kEpsArcBit) != 0;
            const int32_t arc = e.z & ~(kRedirectArcBit | kEpsArcBit);
            int32_t* rec = ws.lat_out + static_cast<int64_t>(idx) * 8;
            rec[0] = e.x;
            rec[1] = e.y;
            rec[2] = redirect ? -1 : (is_eps ? 0 : ArcIlabel(g, arc));
            rec[3] = arc;
            rec[4] = static_cast<int32_t>(BitsOfFloat(end - ws.tok_fwd[e.x]));
            rec[5] = t;  // segment index: frame t persists as segment t+1; seg 0 = the
                         // initial closure
            rec[6] = lane;
            rec[7] = is_eps ? 1 : 0;
          }
        }
        __syncthreads();
        if (threadIdx.x == 0) emitted += sh_scan[blockDim.x - 1];
        __syncthreads();
      }
    }
  }
  if (threadIdx.x == 0) ws.lat_out_len[lane] = emitted;
}

// Lattice persistence: one segment per step, written AFTER the epsilon closure converges
// so token ids are canonical (an epsilon improvement replaces a state's winner token;
// resolving src/dst through the live frontier at this point sees the final ids).
// Segment index == lc.t (already incremented by K3): frame t persists as segment t+1,
// the initial closure as segment 0. Emitting candidates' sources live in the PREVIOUS
// frontier buffer (pre-swap view = tok_winner[1]); epsilon candidates' sources in the
// current one. Epsilon entries carry kEpsArcBit (same-frame arcs, label 0 on export).
__global__ void LatPersistKernel(Workspace ws, Sizes sz, DeviceGraph g) {
  const int32_t lane = blockIdx.x;
  LaneCounters& lc = ws.lanes[lane];
  const int32_t phase = lc.phase;
  if (phase == kPhaseDone || (lc.next_raw == 0 && lc.cand_count == 0)) return;
  const int64_t lane64 = lane;
  const int32_t total = min(lc.cand_count, sz.cand_cap);
  const int32_t seg_idx = min(lc.t, sz.path_cap - 1);
  const int2* cand = ws.cand + lane64 * sz.cand_cap;
  const float* cand_end = ws.cand_end + lane64 * sz.cand_cap;
  const uint32_t* hkey = ws.hash_key + lane64 * sz.hash_cap;
  const unsigned long long* hpay = ws.hash_payload + lane64 * sz.hash_cap;
  const int32_t* hpos = ws.hash_pos + lane64 * sz.hash_cap;
  const int32_t* cur_winner = ws.tok_winner[0] + lane64 * sz.main_q;   // post-swap
  const int32_t* prev_winner = ws.tok_winner[1] + lane64 * sz.main_q;  // pre-swap
  const uint32_t mask = sz.hash_cap - 1;

  __shared__ int32_t sh_base;
  const int32_t lat_lim = *ws.lat_limit;  // committed lattice-arena capacity
  if (threadIdx.x == 0) {
    sh_base = (total > 0) ? atomicAdd(ws.lat_cursor, total) : 0;
    int32_t count = total;
    if (sh_base + count > lat_lim) {
      atomicOr(&lc.overflow, kOverflowArena);
      count = max(0, lat_lim - sh_base);
    }
    ws.lat_seg[lane64 * sz.path_cap + seg_idx] = make_int2(sh_base, count);
  }
  __syncthreads();
  const int32_t emit_boundary = lc.cand_emit;
  for (int32_t j = threadIdx.x; j < total; j += blockDim.x) {
    if (sh_base + j >= lat_lim) break;
    const int2 c = cand[j];
    const bool is_eps = j >= emit_boundary;
    int32_t dest_tok = -1;
    int32_t arc_field = c.y | (is_eps ? kEpsArcBit : 0);
    if (phase == kPhaseRedirect) {
      dest_tok = cur_winner[0];  // the single super-final token
      arc_field |= kRedirectArcBit;
    } else {
      const uint32_t key = static_cast<uint32_t>(ArcDest(g, c.y));
      uint32_t h = HashState(key, mask);
      for (int probe = 0; probe < kMaxProbes; ++probe) {
        const uint32_t k = hkey[h];
        if (k == key) {
          const int32_t pos = hpos[h];
          if (pos >= 0) dest_tok = cur_winner[pos];
          // Epsilon candidates: keep only the state's FINAL winning eps arc (the
          // closure's improvement forest). Keeping every in-beam eps alternative can
          // create same-frame cycles the k2 lattice tooling rejects; emitting
          // alternatives remain fully complete.
          if (is_eps && static_cast<int32_t>(hpay[h] & 0xFFFFFFFFull) != j) dest_tok = -1;
          break;
        }
        if (k == kEmptyKey) break;
        h = (h + 1) & mask;
      }
    }
    const int32_t src_tok = is_eps ? cur_winner[c.x] : prev_winner[c.x];
    ws.lat[sh_base + j] = make_int4(src_tok, dest_tok, arc_field,
                                    __float_as_int(cand_end[j]));
  }
}

}  // namespace kernels
}  // namespace oasr::wfst
