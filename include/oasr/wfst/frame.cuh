#pragma once

// Per-step decode pipeline: K1 Scan (phase decision + k2 dynamic-beam update + degree
// prefix sums), K2a Max (exact per-step cutoff), K2b Expand (admit + hash recombine),
// K3 Finalize (exact filter + winner resolution + frontier build). DecoderConfig drives
// the beam/phase logic.
#include "decoder/wfst/config.h"
#include "oasr/wfst/common.cuh"

namespace oasr::wfst {
namespace kernels {

// ---------------------------------------------------------------------------
// K1: per-lane CTA — phase decision, k2 dynamic-beam update, degree prefix sums.
// Simple sequential block scan with running carry (1024 threads, chunked).
__global__ void ScanKernel(Workspace ws, Sizes sz, DeviceGraph g, DecoderConfig cfg,
                           int32_t eps_pass) {
  const int32_t lane = blockIdx.x;
  LaneCounters& lc = ws.lanes[lane];
  if (eps_pass) {
    // Epsilon-closure pass over the frontier just built by K3 (post-swap view).
    if (lc.status != 0 || (lc.phase != kPhaseReal && lc.phase != kPhaseEps)) {
      if (threadIdx.x == 0) lc.total_arcs = 0;
      return;
    }
    __shared__ int32_t sh_carry_e;
    if (threadIdx.x == 0) {
      lc.phase = kPhaseEps;
      lc.running_max = 0;
      sh_carry_e = 0;
    }
    __syncthreads();
    const int32_t n = lc.frontier_size;
    const int32_t* tok_state = ws.tok_state[0] + static_cast<int64_t>(lane) * sz.main_q;
    int32_t* offsets = ws.arc_offsets + static_cast<int64_t>(lane) * (sz.main_q + 1);
    int32_t* emit_begin = ws.tok_emit_begin + static_cast<int64_t>(lane) * sz.main_q;
    __shared__ int32_t sh_scan_e[1024];
    for (int32_t base = 0; base < n; base += blockDim.x) {
      const int32_t i = base + threadIdx.x;
      int32_t deg = 0, eb = 0;
      if (i < n) {
        const int32_t s = tok_state[i];
        deg = EpsCountOf(g, s);
        eb = EpsBegin(g, s);
      }
      sh_scan_e[threadIdx.x] = deg;
      __syncthreads();
      for (int32_t d = 1; d < blockDim.x; d <<= 1) {
        int32_t v = (threadIdx.x >= d) ? sh_scan_e[threadIdx.x - d] : 0;
        __syncthreads();
        sh_scan_e[threadIdx.x] += v;
        __syncthreads();
      }
      if (i < n) {
        offsets[i] = sh_carry_e + sh_scan_e[threadIdx.x] - deg;
        emit_begin[i] = eb;
      }
      __syncthreads();
      if (threadIdx.x == 0) sh_carry_e += sh_scan_e[blockDim.x - 1];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      offsets[n] = sh_carry_e;
      lc.total_arcs = sh_carry_e;
    }
    return;
  }
  if (lc.status != 0 || lc.t >= lc.chunk_end) {
    if (threadIdx.x == 0) {
      lc.total_arcs = 0;
      lc.phase = kPhaseDone;  // gates the eps passes of this step
    }
    return;
  }

  __shared__ int32_t sh_phase;
  __shared__ int32_t sh_has_final;
  __shared__ int32_t sh_carry;

  const int32_t n = lc.frontier_size;
  const int32_t* tok_state = ws.tok_state[0] + static_cast<int64_t>(lane) * sz.main_q;

  // Phase + has_valid_final (block OR-reduce over the frontier).
  if (threadIdx.x == 0) {
    sh_has_final = 0;
    sh_carry = 0;
  }
  __syncthreads();
  const bool final_step = (lc.t >= lc.T);
  if (final_step) {
    int32_t local = 0;
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
      if (g.final_count[tok_state[i]] > 0) local = 1;
    }
    if (local) atomicOr(&sh_has_final, 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (!final_step) {
      sh_phase = kPhaseReal;
    } else if (sh_has_final || !cfg.allow_partial) {
      sh_phase = kPhaseFinal;
    } else {
      sh_phase = kPhaseRedirect;
    }
    lc.phase = sh_phase;
    lc.has_valid_final = sh_has_final;

    // k2 dynamic-beam update (verbatim; see cpu_reference.cc UpdateBeam). 64-bit
    // final_t: streaming lanes carry T = INT32_MAX (k2 online semantics) until finalize.
    float dyn = lc.dyn_beam;
    const float def = cfg.search_beam;
    const int64_t final_t = static_cast<int64_t>(lc.T) + 1;
    const int64_t t64 = lc.t;
    float cur_min = static_cast<float>(cfg.min_active_states);
    if (t64 + 5 >= final_t) {
      cur_min = static_cast<float>(max(cfg.min_active_states, cfg.max_active_states / 2));
    }
    if (n <= cfg.max_active_states) {
      if (n >= cur_min || n == 0) {
        dyn = 0.8f * dyn + 0.2f * def;
      } else {
        if (dyn < def) dyn = def;
        dyn *= 1.25f;
      }
    } else if (t64 + 5 < final_t) {
      if (dyn > def) dyn = def;
      dyn *= 0.8f;
    }
    if (t64 == final_t - 1) dyn = 1.0e10f;
    lc.dyn_beam = dyn;

    lc.running_max = 0;
    lc.next_raw = 0;
    lc.cand_count = 0;
    lc.cand_consumed = 0;
    lc.redirect_claimed = 0;
  }
  __syncthreads();
  const int32_t phase = sh_phase;

  // Degree exclusive prefix sums over the frontier, chunked block scan.
  int32_t* offsets = ws.arc_offsets + static_cast<int64_t>(lane) * (sz.main_q + 1);
  int32_t* emit_begin = ws.tok_emit_begin + static_cast<int64_t>(lane) * sz.main_q;
  __shared__ int32_t sh_scan[1024];
  for (int32_t base = 0; base < n; base += blockDim.x) {
    const int32_t i = base + threadIdx.x;
    int32_t deg = 0, eb = 0;
    if (i < n) {
      const int32_t s = tok_state[i];
      if (phase == kPhaseReal) {
        deg = EmitCount(g, s);
        eb = EmitBegin(g, s);
      } else if (phase == kPhaseFinal) {
        deg = g.final_count[s];
        eb = FinalBegin(g, s);
      } else {  // redirect: every arc of the state
        deg = g.row_splits[s + 1] - g.row_splits[s];
        eb = g.row_splits[s];
      }
    }
    // inclusive block scan (Hillis-Steele in shared memory)
    sh_scan[threadIdx.x] = deg;
    __syncthreads();
    for (int32_t d = 1; d < blockDim.x; d <<= 1) {
      int32_t v = (threadIdx.x >= d) ? sh_scan[threadIdx.x - d] : 0;
      __syncthreads();
      sh_scan[threadIdx.x] += v;
      __syncthreads();
    }
    if (i < n) {
      offsets[i] = sh_carry + sh_scan[threadIdx.x] - deg;  // exclusive
      emit_begin[i] = eb;
    }
    __syncthreads();
    if (threadIdx.x == 0) sh_carry += sh_scan[blockDim.x - 1];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    offsets[n] = sh_carry;
    lc.total_arcs = sh_carry;
  }
}

// ---------------------------------------------------------------------------
// K2a: exact per-step max over all expanded arcs (kPhaseReal / kPhaseFinal). Block-local
// reduction, one atomicMax per CTA. Running on the full arc set before K2b makes the
// admit cutoff exact, so the admitted candidate/claim sets equal the true in-beam sets.
__global__ void MaxKernel(Workspace ws, Sizes sz, DeviceGraph g, const void* log_probs,
                          int64_t lp_lane_stride, int64_t lp_frame_stride) {
  const int32_t lane = blockIdx.y;
  LaneCounters& lc = ws.lanes[lane];
  const int32_t phase = lc.phase;
  if (lc.status != 0 || phase == kPhaseDone) return;
  const int32_t total = lc.total_arcs;
  if (total == 0) return;

  const int64_t lane64 = lane;
  const int32_t n = lc.frontier_size;
  const int32_t* offsets = ws.arc_offsets + lane64 * (sz.main_q + 1);
  const int32_t* emit_begin = ws.tok_emit_begin + lane64 * sz.main_q;
  const float* tok_score = ws.tok_score[0] + lane64 * sz.main_q;
  const int64_t lp_row = lane64 * lp_lane_stride +
                         static_cast<int64_t>(lc.t - lc.chunk_start) * lp_frame_stride;
  const bool lp_half = sz.lp_half != 0;

  uint32_t local = 0;
  for (int32_t slot = blockIdx.x * blockDim.x + threadIdx.x; slot < total;
       slot += gridDim.x * blockDim.x) {
    int32_t lo = 0, hi = n - 1;
    while (lo < hi) {
      const int32_t mid = (lo + hi + 1) >> 1;
      if (offsets[mid] <= slot) lo = mid; else hi = mid - 1;
    }
    const int32_t arc = emit_begin[lo] + (slot - offsets[lo]);
    const int2 dest_il = g.dest_ilabel[arc];
    float end = tok_score[lo] + g.weight[arc];
    if (phase == kPhaseReal) end += LoadLp(log_probs, lp_row + dest_il.y, lp_half);
    local = max(local, FloatToOrderedUint(end));
  }
  __shared__ uint32_t sh_max[256];
  sh_max[threadIdx.x] = local;
  __syncthreads();
  for (int32_t d = blockDim.x >> 1; d > 0; d >>= 1) {
    if (threadIdx.x < d) sh_max[threadIdx.x] = max(sh_max[threadIdx.x], sh_max[threadIdx.x + d]);
    __syncthreads();
  }
  if (threadIdx.x == 0 && sh_max[0] > 0) atomicMax(&lc.running_max, sh_max[0]);
}

// K2b: admit + hash recombine with the exact cutoff from K2a. The redirect final step
// (allow_partial dead-end) is handled here too: with the exact max known from K2a, the
// first thread whose end score equals it claims the single super-final candidate.
__global__ void ExpandKernel(Workspace ws, Sizes sz, DeviceGraph g, const void* log_probs,
                             int64_t lp_lane_stride, int64_t lp_frame_stride) {
  const int32_t lane = blockIdx.y;
  LaneCounters& lc = ws.lanes[lane];
  const int32_t phase = lc.phase;
  if (lc.status != 0 || phase == kPhaseDone) return;
  const int32_t total = lc.total_arcs;
  if (total == 0) return;

  if (phase == kPhaseRedirect) {
    const int64_t lane64 = lane;
    const int32_t n = lc.frontier_size;
    const int32_t* offsets = ws.arc_offsets + lane64 * (sz.main_q + 1);
    const int32_t* emit_begin = ws.tok_emit_begin + lane64 * sz.main_q;
    const float* tok_score = ws.tok_score[0] + lane64 * sz.main_q;
    const uint32_t target = lc.running_max;
    for (int32_t slot = blockIdx.x * blockDim.x + threadIdx.x; slot < total;
         slot += gridDim.x * blockDim.x) {
      int32_t lo = 0, hi = n - 1;
      while (lo < hi) {
        const int32_t mid = (lo + hi + 1) >> 1;
        if (offsets[mid] <= slot) lo = mid; else hi = mid - 1;
      }
      const int32_t arc = emit_begin[lo] + (slot - offsets[lo]);
      const float end = tok_score[lo] + g.weight[arc];
      if (FloatToOrderedUint(end) == target) {
        if (atomicCAS(&lc.redirect_claimed, 0, 1) == 0) {
          ws.cand[lane64 * sz.cand_cap] = make_int2(lo, arc);
          lc.cand_count = 1;
          ws.next_claims[lane64 * sz.claims_cap] = make_int2(g.num_states - 1, -1);
          lc.next_raw = 1;
        }
        break;
      }
    }
    return;
  }

  const int64_t lane64 = lane;
  const int32_t n = lc.frontier_size;
  const int32_t* offsets = ws.arc_offsets + lane64 * (sz.main_q + 1);
  const int32_t* emit_begin = ws.tok_emit_begin + lane64 * sz.main_q;
  const float* tok_score = ws.tok_score[0] + lane64 * sz.main_q;
  const int64_t lp_row = lane64 * lp_lane_stride +
                         static_cast<int64_t>(lc.t - lc.chunk_start) * lp_frame_stride;
  const bool lp_half = sz.lp_half != 0;
  const float beam = lc.dyn_beam;
  uint32_t* hkey = ws.hash_key + lane64 * sz.hash_cap;
  unsigned long long* hpay = ws.hash_payload + lane64 * sz.hash_cap;
  const uint32_t mask = sz.hash_cap - 1;

  for (int32_t slot = blockIdx.x * blockDim.x + threadIdx.x; slot < total;
       slot += gridDim.x * blockDim.x) {
    // binary search: largest i with offsets[i] <= slot
    int32_t lo = 0, hi = n - 1;
    while (lo < hi) {
      const int32_t mid = (lo + hi + 1) >> 1;
      if (offsets[mid] <= slot) lo = mid; else hi = mid - 1;
    }
    const int32_t arc = emit_begin[lo] + (slot - offsets[lo]);
    const int2 dest_il = g.dest_ilabel[arc];
    float end = tok_score[lo] + g.weight[arc];
    if (phase == kPhaseReal) end += LoadLp(log_probs, lp_row + dest_il.y, lp_half);
    // final phase: acoustic 0 for -1 arcs

    const uint32_t ordered_end = FloatToOrderedUint(end);
    const float cur_max = OrderedUintToFloat(lc.running_max);  // exact after K2a
    if (!(end > cur_max - beam)) continue;

    // hash claim keyed by dest state
    const uint32_t key = static_cast<uint32_t>(dest_il.x);
    uint32_t h = HashState(key, mask);
    bool placed = false;
    for (int probe = 0; probe < kMaxProbes; ++probe) {
      const uint32_t prev = atomicCAS(&hkey[h], kEmptyKey, key);
      if (prev == kEmptyKey || prev == key) {
        if (prev == kEmptyKey) {  // first claimer registers the state
          const int32_t r = atomicAdd(&lc.next_raw, 1);
          if (r >= sz.claims_cap) {
            atomicOr(&lc.overflow, kOverflowClaims);
          } else {
            ws.next_claims[lane64 * sz.claims_cap + r] =
                make_int2(dest_il.x, static_cast<int32_t>(h));
          }
        }
        // Early dominance check (1-best only): a candidate strictly below the state's
        // current best can never be resolved as the winner — skip the append entirely.
        // Lattice mode keeps every in-beam candidate (they are lattice arcs).
        if (sz.lat_cap == 0 && ordered_end < static_cast<uint32_t>(hpay[h] >> 32)) {
          placed = true;
          break;
        }
        const int32_t j = atomicAdd(&lc.cand_count, 1);
        if (j >= sz.cand_cap) {
          atomicOr(&lc.overflow, kOverflowCand);
          placed = true;
          break;
        }
        ws.cand[lane64 * sz.cand_cap + j] = make_int2(lo, arc);
        if (sz.lat_cap > 0) ws.cand_end[lane64 * sz.cand_cap + j] = end;
        const unsigned long long packed =
            (static_cast<unsigned long long>(ordered_end) << 32) |
            static_cast<uint32_t>(j);
        atomicMax(&hpay[h], packed);
        placed = true;
        break;
      }
      h = (h + 1) & mask;
    }
    if (!placed) atomicOr(&lc.overflow, kOverflowHash);
  }
}

// ---------------------------------------------------------------------------
// K3: per-lane CTA — exact filter, winner resolution, frontier build, targeted hash
// clear, bookkeeping. Sequentialized per lane (single CTA loop) for v0 clarity.
__global__ void FinalizeKernel(Workspace ws, Sizes sz, DeviceGraph g, DecoderConfig cfg) {
  const int32_t lane = blockIdx.x;
  LaneCounters& lc = ws.lanes[lane];
  if (lc.status != 0) return;
  const int64_t lane64 = lane;
  if (lc.t >= lc.chunk_end) {
    // Idle this step (streaming lane between chunks, or bucket padding): the buffers
    // still swap globally, so carry the frontier across to keep every lane's state in
    // the same parity.
    const int32_t n = lc.frontier_size;
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
      ws.tok_state[1][lane64 * sz.main_q + i] = ws.tok_state[0][lane64 * sz.main_q + i];
      ws.tok_score[1][lane64 * sz.main_q + i] = ws.tok_score[0][lane64 * sz.main_q + i];
      ws.tok_winner[1][lane64 * sz.main_q + i] = ws.tok_winner[0][lane64 * sz.main_q + i];
    }
    return;
  }
  const int32_t phase = lc.phase;
  const int32_t raw = min(lc.next_raw, sz.claims_cap);
  const float exact_max = OrderedUintToFloat(lc.running_max);
  const float cutoff = (phase == kPhaseReal) ? exact_max - lc.dyn_beam : -INFINITY;

  uint32_t* hkey = ws.hash_key + lane64 * sz.hash_cap;
  unsigned long long* hpay = ws.hash_payload + lane64 * sz.hash_cap;
  const int2* cand = ws.cand + lane64 * sz.cand_cap;
  const int32_t* cur_winner = ws.tok_winner[0] + lane64 * sz.main_q;
  int32_t* nxt_state = ws.tok_state[1] + lane64 * sz.main_q;
  float* nxt_score = ws.tok_score[1] + lane64 * sz.main_q;
  int32_t* nxt_winner = ws.tok_winner[1] + lane64 * sz.main_q;

  __shared__ int32_t sh_kept;
  __shared__ int32_t sh_base;  // arena base for this round's block
  __shared__ int32_t sh_scan[1024];
  if (threadIdx.x == 0) sh_kept = 0;
  __syncthreads();

  for (int32_t base = 0; base < raw; base += blockDim.x) {
    const int32_t r = base + threadIdx.x;
    int32_t keep = 0;
    int32_t state = -1, hslot = -1, prev_local = -1, arc = -1;
    float score = 0.0f;
    if (r < raw) {
      const int2 claim = ws.next_claims[lane64 * sz.claims_cap + r];
      state = claim.x;
      hslot = claim.y;
      if (phase == kPhaseRedirect) {
        score = OrderedUintToFloat(lc.running_max);
        prev_local = cand[0].x;
        arc = cand[0].y;
        keep = 1;
      } else {
        const unsigned long long packed = hpay[hslot];
        score = OrderedUintToFloat(static_cast<uint32_t>(packed >> 32));
        const int32_t j = static_cast<int32_t>(packed & 0xFFFFFFFFull);
        if (packed != 0 && j < sz.cand_cap && score > cutoff) {
          prev_local = cand[j].x;
          arc = cand[j].y;
          keep = 1;
        }
      }
    }
    // block scan to compact
    sh_scan[threadIdx.x] = keep;
    __syncthreads();
    for (int32_t d = 1; d < blockDim.x; d <<= 1) {
      int32_t v = (threadIdx.x >= d) ? sh_scan[threadIdx.x - d] : 0;
      __syncthreads();
      sh_scan[threadIdx.x] += v;
      __syncthreads();
    }
    // Allocate this round's winners block: shared arena (batch mode) or the lane's own
    // region (streaming mode — per-lane monotonic ids survive across chunks).
    if (threadIdx.x == 0) {
      const int32_t round_kept = sh_scan[blockDim.x - 1];
      if (sz.stream_log_cap > 0) {
        sh_base = lane * sz.stream_log_cap + lc.log_len;
        if (round_kept > 0 && lc.log_len + round_kept > sz.stream_log_cap) {
          atomicOr(&lc.overflow, kOverflowArena);
        }
        lc.log_len = min(lc.log_len + round_kept, sz.stream_log_cap);
      } else {
        sh_base = (round_kept > 0) ? atomicAdd(ws.arena_cursor, round_kept) : 0;
      }
    }
    __syncthreads();
    const bool lattice = sz.lat_cap > 0;
    int32_t widx_or_dead = -1;
    // Lattice persistence and epsilon closure both need the claims to stay live past K3
    // (state -> frontier-position map via hash_pos; ClearClaimsKernel runs at step end).
    const bool keep_hash = lattice || g.eps_count != nullptr;
    if (keep) {
      const int32_t out = sh_kept + sh_scan[threadIdx.x] - 1;
      const int32_t widx = sh_base + sh_scan[threadIdx.x] - 1;
      const int32_t widx_cap =
          (sz.stream_log_cap > 0) ? (lane + 1) * sz.stream_log_cap : sz.arena_cap;
      if (widx < widx_cap && out < sz.main_q) {
        ws.winners[widx] = make_int2(cur_winner[prev_local], arc);
        nxt_state[out] = state;
        nxt_score[out] = score;
        nxt_winner[out] = widx;
        if (lattice) ws.tok_fwd[widx] = score;
        if (keep_hash) ws.hash_pos[lane64 * sz.hash_cap + hslot] = out;
        widx_or_dead = widx;
      } else {
        atomicOr(&lc.overflow, widx >= sz.arena_cap ? kOverflowArena : kOverflowKept);
      }
    }
    if (r < raw && hslot >= 0) {
      if (keep_hash) {
        // A filtered-out claim's payload resets so later epsilon candidates compete
        // fresh for that state (CPU-reference revival semantics).
        if (widx_or_dead < 0) hpay[hslot] = 0ull;
      } else {
        hkey[hslot] = kEmptyKey;
        hpay[hslot] = 0ull;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) sh_kept += sh_scan[blockDim.x - 1];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const int32_t kept = min(sh_kept, sz.main_q);
    lc.frontier_size = kept;
    if (g.eps_count != nullptr) lc.cand_consumed = min(lc.cand_count, sz.cand_cap);
    lc.cand_emit = min(lc.cand_count, sz.cand_cap);  // emitting/eps candidate boundary

    if (phase == kPhaseFinal || phase == kPhaseRedirect) {
      // Final step just executed.
      if (kept > 0) {
        lc.status = 1;
        lc.reached_final = (phase == kPhaseFinal) ? 1 : 0;
        lc.final_score = nxt_score[0];
        lc.final_tok = nxt_winner[0];
        // All final-step arcs land in the super-final state, but keep the max anyway.
        for (int32_t i = 1; i < kept; ++i) {
          if (nxt_score[i] > lc.final_score) {
            lc.final_score = nxt_score[i];
            lc.final_tok = nxt_winner[i];
          }
        }
      } else {
        lc.status = 2;
      }
    }
    lc.t += 1;
  }
  __syncthreads();

  // Debug snapshot of the new frontier at index lc.t (post-increment: frame t's result
  // lands at snapshot[t+1] as in the CPU reference).
  if (sz.snap_frames > 0 && lc.t < sz.snap_frames) {
    const int32_t kept = lc.frontier_size;
    int2* snap = ws.snap + (lane64 * sz.snap_frames + lc.t) * sz.main_q;
    for (int32_t i = threadIdx.x; i < kept; i += blockDim.x) {
      snap[i] = make_int2(nxt_state[i], __float_as_int(nxt_score[i]));
    }
    if (threadIdx.x == 0) ws.snap_len[lane * sz.snap_frames + lc.t] = kept;
  }
}

}  // namespace kernels
}  // namespace oasr::wfst
