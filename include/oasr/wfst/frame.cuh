#pragma once

// Per-step decode pipeline: K1 Scan (phase decision + k2 dynamic-beam update + degree
// prefix sums), K2a Max (exact per-step cutoff), K2b Expand (admit + hash recombine),
// K3 Finalize (exact filter + winner resolution + frontier build). DecoderConfig drives
// the beam/phase logic.
//
// Hot-loop structure (K2a/K2b): arc slots resolve to tokens via the block-cooperative
// shared-memory lookup (BlockTokenLookup) instead of a per-slot global binary search,
// and per-token upper bounds (tok_ub + max_lp) skip arcs that provably cannot beat the
// running max (K2a) or make the beam (K2b). Both are exact: fp32 addition is monotone,
// so ub >= every end score of the token, and skipped arcs are exactly the ones the
// original code would have computed and then discarded.
#include <cooperative_groups.h>

#include "decoder/wfst/config.h"
#include "oasr/wfst/common.cuh"

namespace oasr::wfst {
namespace kernels {

// ---------------------------------------------------------------------------
// K1: per-lane CTA — phase decision, k2 dynamic-beam update, degree prefix sums,
// per-token upper bounds, and the frame's max log-prob (for the K2 skip tests).
__global__ void ScanKernel(Workspace ws, Sizes sz, DeviceGraph g, DecoderConfig cfg,
                           int32_t eps_pass, const void* log_probs, int64_t lp_lane_stride,
                           int64_t lp_frame_stride) {
  const int32_t lane = blockIdx.x;
  LaneCounters& lc = ws.lanes[lane];
  __shared__ int32_t sh_ws[32];  // warp scratch for BlockInclusiveScan
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
    for (int32_t base = 0; base < n; base += blockDim.x) {
      const int32_t i = base + threadIdx.x;
      int32_t deg = 0, eb = 0;
      if (i < n) {
        const int32_t s = tok_state[i];
        deg = EpsCountOf(g, s);
        eb = EpsBegin(g, s);
      }
      const int32_t carry = sh_carry_e;
      const int32_t incl = BlockInclusiveScan(deg, sh_ws);
      if (i < n) {
        offsets[i] = carry + incl - deg;
        emit_begin[i] = eb;
      }
      if (threadIdx.x == blockDim.x - 1) sh_carry_e = carry + incl;
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
  __shared__ float sh_fmax[32];

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

  // Frame max log-prob (kPhaseReal): upper-bound term for the K2 skip tests. Reducing
  // over the full row (a superset of the labels arcs index) keeps the bound safe.
  if (phase == kPhaseReal) {
    const int64_t lp_row = static_cast<int64_t>(lane) * lp_lane_stride +
                           static_cast<int64_t>(lc.t - lc.chunk_start) * lp_frame_stride;
    const bool lp_half = sz.lp_half != 0;
    float m = kNegInf;
    for (int64_t i = threadIdx.x; i < lp_frame_stride; i += blockDim.x) {
      m = max(m, LoadLp(log_probs, lp_row + i, lp_half));
    }
#pragma unroll
    for (int32_t d = 16; d > 0; d >>= 1) m = max(m, __shfl_down_sync(0xFFFFFFFFu, m, d));
    if ((threadIdx.x & 31) == 0) sh_fmax[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
      const int32_t nw = static_cast<int32_t>(blockDim.x) >> 5;
      for (int32_t w = 1; w < nw; ++w) m = max(m, sh_fmax[w]);
      lc.max_lp = m;
    }
  } else if (threadIdx.x == 0) {
    lc.max_lp = 0.0f;  // final/redirect arcs carry no acoustic term
  }
  __syncthreads();

  // Degree exclusive prefix sums over the frontier (+ per-token upper bounds).
  int32_t* offsets = ws.arc_offsets + static_cast<int64_t>(lane) * (sz.main_q + 1);
  int32_t* emit_begin = ws.tok_emit_begin + static_cast<int64_t>(lane) * sz.main_q;
  float* tok_ub = ws.tok_ub + static_cast<int64_t>(lane) * sz.main_q;
  const float* tok_score = ws.tok_score[0] + static_cast<int64_t>(lane) * sz.main_q;
  for (int32_t base = 0; base < n; base += blockDim.x) {
    const int32_t i = base + threadIdx.x;
    int32_t deg = 0, eb = 0;
    float ub = kNegInf;
    if (i < n) {
      const int32_t s = tok_state[i];
      if (phase == kPhaseReal) {
        deg = EmitCount(g, s);
        eb = EmitBegin(g, s);
        ub = tok_score[i] + g.emit_maxw[s];
      } else if (phase == kPhaseFinal) {
        deg = g.final_count[s];
        eb = FinalBegin(g, s);
      } else {  // redirect: every arc of the state
        deg = g.row_splits[s + 1] - g.row_splits[s];
        eb = g.row_splits[s];
      }
    }
    const int32_t carry = sh_carry;
    const int32_t incl = BlockInclusiveScan(deg, sh_ws);
    if (i < n) {
      offsets[i] = carry + incl - deg;  // exclusive
      emit_begin[i] = eb;
      if (phase == kPhaseReal) tok_ub[i] = ub;
    }
    if (threadIdx.x == blockDim.x - 1) sh_carry = carry + incl;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    offsets[n] = sh_carry;
    lc.total_arcs = sh_carry;
  }
}

// ---------------------------------------------------------------------------
// K2 flattening: lanes' arc ranges concatenate into one global slot space so any lane
// mix load-balances over the whole grid (a single hot lane would otherwise be capped at
// its slice of a static (blocks, lane) grid — measured as the dominant tail). The tiny
// scan below publishes the per-lane segment starts; K2a/K2b run 1-D grids over the
// grand total.
__global__ void LaneScanKernel(Workspace ws, Sizes sz) {
  if (blockIdx.x != 0 || threadIdx.x >= 32) return;
  constexpr uint32_t kFull = 0xFFFFFFFFu;
  const int32_t lane = static_cast<int32_t>(threadIdx.x);
  int32_t carry = 0;
  for (int32_t base = 0; base < sz.lanes; base += 32) {
    const int32_t l = base + lane;
    const int32_t deg = (l < sz.lanes) ? ws.lanes[l].total_arcs : 0;
    int32_t v = deg;
#pragma unroll
    for (int32_t d = 1; d < 32; d <<= 1) {
      const int32_t y = __shfl_up_sync(kFull, v, d);
      if (lane >= d) v += y;
    }
    if (l < sz.lanes) ws.lane_arc_offsets[l] = carry + v - deg;
    carry += __shfl_sync(kFull, v, 31);
  }
  if (lane == 0) ws.lane_arc_offsets[sz.lanes] = carry;
}

// Per-lane view for the flattened K2 kernels (built once per warp-window on the fast
// path, per slot on lane-straddling windows; the loads are warp-uniform either way).
struct K2Lane {
  LaneCounters* lc;
  const int32_t* offsets;
  const int32_t* emit_begin;
  const float* tok_score;
  const float* tok_ub;
  int64_t lp_row;
  int32_t n;
  int32_t total;
  int32_t phase;
  float max_lp;
};

__device__ inline K2Lane MakeK2Lane(const Workspace& ws, const Sizes& sz, int32_t lane,
                                    int64_t lp_lane_stride, int64_t lp_frame_stride) {
  K2Lane v;
  const int64_t lane64 = lane;
  v.lc = &ws.lanes[lane];
  v.offsets = ws.arc_offsets + lane64 * (sz.main_q + 1);
  v.emit_begin = ws.tok_emit_begin + lane64 * sz.main_q;
  v.tok_score = ws.tok_score[0] + lane64 * sz.main_q;
  v.tok_ub = ws.tok_ub + lane64 * sz.main_q;
  v.n = v.lc->frontier_size;
  v.total = v.lc->total_arcs;
  v.phase = v.lc->phase;
  v.max_lp = v.lc->max_lp;
  v.lp_row = lane64 * lp_lane_stride +
             static_cast<int64_t>(v.lc->t - v.lc->chunk_start) * lp_frame_stride;
  return v;
}

// One arc slot of the max pass; returns the ordered end score (0 when skipped).
__device__ inline uint32_t MaxSlotEnd(const K2Lane& v, const DeviceGraph& g, const void* lp,
                                      bool lp_half, int32_t slot, int32_t tok,
                                      int32_t tok_off) {
  const bool real = (v.phase == kPhaseReal);
  // Upper-bound skip: fp add is monotone, so ub >= every end score of this token; a
  // skipped arc is dominated by an already-posted end score and cannot change the max.
  if (real && !(FloatToOrderedUint(v.tok_ub[tok] + v.max_lp) > v.lc->running_max)) return 0;
  const int32_t arc = v.emit_begin[tok] + (slot - tok_off);
  float end = v.tok_score[tok] + g.weight[arc];
  if (real) end += LoadLp(lp, v.lp_row + ArcIlabel(g, arc), lp_half);
  return FloatToOrderedUint(end);
}

// Warp walker over a contiguous span [begin, end) of the flattened slot space. The lane
// resolution and the token cursor persist across 32-slot strips, so lane 0's global
// binary search runs once per lane segment and every offsets entry is streamed (loaded
// coalesced) about once per warp. Per strip, every lane calls slot_body (valid flags the
// in-range lanes) and then strip_end runs once (both must be warp-uniform call sites:
// they may use warp intrinsics).
template <typename SlotBody, typename StripEnd>
__device__ inline void WarpWalkSlots(const Workspace& ws, const Sizes& sz, int32_t begin,
                                     int32_t end, int64_t lp_lane_stride,
                                     int64_t lp_frame_stride, SlotBody&& slot_body,
                                     StripEnd&& strip_end) {
  constexpr uint32_t kFull = 0xFFFFFFFFu;
  const int32_t lane_id = static_cast<int32_t>(threadIdx.x) & 31;
  int32_t pos = begin;
  while (pos < end) {
    const int32_t l = UpperTokenGlobal(ws.lane_arc_offsets, sz.lanes, pos);
    const int32_t seg_begin = ws.lane_arc_offsets[l];
    const int32_t seg_end = min(end, ws.lane_arc_offsets[l + 1]);
    const K2Lane v = MakeK2Lane(ws, sz, l, lp_lane_stride, lp_frame_stride);
    int32_t s = pos - seg_begin;  // local slot cursor (warp-uniform)
    const int32_t s_end = seg_end - seg_begin;
    // Token cursor: invariant offsets[base_tok] == off_prev <= every remaining slot.
    int32_t base_tok = 0, off_prev = 0;
    if (lane_id == 0) {
      base_tok = UpperTokenGlobal(v.offsets, v.n, s);
      off_prev = v.offsets[base_tok];
    }
    base_tok = __shfl_sync(kFull, base_tok, 0);
    off_prev = __shfl_sync(kFull, off_prev, 0);
    while (s < s_end) {
      const int32_t slot = s + lane_id;
      int32_t tok = base_tok, off = off_prev;
      bool resolved = false;
      for (;;) {
        // Boundary batch: lane k holds offsets[base_tok + 1 + k] (coalesced load).
        const int32_t cand_tok = base_tok + 1 + lane_id;
        const int32_t b = (cand_tok <= v.n) ? v.offsets[cand_tok] : INT32_MAX;
        // cnt = #boundaries <= slot in this batch (binary count over the sorted batch).
        int32_t cnt = 0;
#pragma unroll
        for (int32_t step = 16; step > 0; step >>= 1) {
          const int32_t bv = __shfl_sync(kFull, b, cnt + step - 1);
          if (bv <= slot) cnt += step;
        }
        const int32_t bfin = __shfl_sync(kFull, b, cnt & 31);  // cnt in [0, 31] here
        if (bfin <= slot) ++cnt;
        const int32_t off_cand = __shfl_sync(kFull, b, (cnt - 1) & 31);
        if (!resolved && cnt < 32) {
          tok = base_tok + cnt;
          off = (cnt == 0) ? off_prev : off_cand;
          resolved = true;
        }
        if (__all_sync(kFull, resolved)) {
          // Restart the next strip from the last lane's token (monotone cursor).
          base_tok = __shfl_sync(kFull, tok, 31);
          off_prev = __shfl_sync(kFull, off, 31);
          break;
        }
        base_tok += 32;
        off_prev = __shfl_sync(kFull, b, 31);
      }
      slot_body(l, v, slot, tok, off, slot < s_end);
      strip_end(v);
      s += 32;
    }
    pos = seg_end;
  }
}

// K2a: exact per-step max over all expanded arcs (kPhaseReal / kPhaseFinal). The running
// max is posted once per warp-strip so later strips skip tokens whose upper bound is
// already dominated — the final max is still exact (every skipped arc is <= an already
// posted end score). Running on the full arc set before K2b makes the admit cutoff
// exact, so the admitted candidate/claim sets equal the true in-beam sets.
__global__ void MaxKernel(Workspace ws, Sizes sz, DeviceGraph g, const void* log_probs,
                          int64_t lp_lane_stride, int64_t lp_frame_stride) {
  const int32_t grand = ws.lane_arc_offsets[sz.lanes];
  if (grand == 0) return;
  const bool lp_half = sz.lp_half != 0;
  const int32_t nwarps = static_cast<int32_t>(gridDim.x * blockDim.x) >> 5;
  const int32_t wid = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int32_t span = (((grand + nwarps - 1) / nwarps) + 31) & ~31;  // 32-aligned
  const int32_t begin = wid * span;
  if (begin >= grand) return;
  const int32_t end = min(begin + span, grand);

  uint32_t strip_max = 0;
  WarpWalkSlots(
      ws, sz, begin, end, lp_lane_stride, lp_frame_stride,
      [&](int32_t, const K2Lane& v, int32_t slot, int32_t tok, int32_t off, bool valid) {
        if (valid) strip_max = max(strip_max, MaxSlotEnd(v, g, log_probs, lp_half, slot,
                                                         tok, off));
      },
      [&](const K2Lane& v) {
        // Post per strip so later strips (and other warps) can skip dominated tokens.
        const uint32_t wm = WarpMaxU32(strip_max);
        if ((threadIdx.x & 31) == 0 && wm > v.lc->running_max)
          atomicMax(&v.lc->running_max, wm);
        strip_max = 0;
      });
}

// One arc slot of the expand pass (admit + hash recombine; also the redirect claim).
__device__ inline void ExpandSlot(const Workspace& ws, const Sizes& sz, const DeviceGraph& g,
                                  const K2Lane& v, int32_t lane, const void* lp, bool lp_half,
                                  int32_t slot, int32_t tok, int32_t tok_off) {
  LaneCounters& lc = *v.lc;
  const int64_t lane64 = lane;
  const int32_t arc = v.emit_begin[tok] + (slot - tok_off);
  if (v.phase == kPhaseRedirect) {
    // Redirect final step (allow_partial dead-end): with the exact max known from K2a,
    // the first thread whose end score equals it claims the single super-final candidate.
    const float end = v.tok_score[tok] + g.weight[arc];
    if (FloatToOrderedUint(end) == lc.running_max) {
      if (atomicCAS(&lc.redirect_claimed, 0, 1) == 0) {
        ws.cand[lane64 * sz.cand_cap] = make_int2(tok, arc);
        lc.cand_count = 1;
        ws.next_claims[lane64 * sz.claims_cap] = make_int2(g.num_states - 1, -1);
        lc.next_raw = 1;
      }
    }
    return;
  }
  const bool real = (v.phase == kPhaseReal);
  const float cur_max = OrderedUintToFloat(lc.running_max);  // exact after K2a
  const float beam = lc.dyn_beam;
  // Token-level skip: no arc of this token can pass the (exact) beam test.
  if (real && !(v.tok_ub[tok] + v.max_lp > cur_max - beam)) return;
  float end = v.tok_score[tok] + g.weight[arc];
  if (real) end += LoadLp(lp, v.lp_row + ArcIlabel(g, arc), lp_half);
  // final phase: acoustic 0 for -1 arcs

  const uint32_t ordered_end = FloatToOrderedUint(end);
  if (!(end > cur_max - beam)) return;
  const int32_t dest = ArcDest(g, arc);  // loaded only for beam survivors

  // hash claim keyed by dest state
  uint32_t* hkey = ws.hash_key + lane64 * sz.hash_cap;
  unsigned long long* hpay = ws.hash_payload + lane64 * sz.hash_cap;
  const uint32_t mask = sz.hash_cap - 1;
  const uint32_t key = static_cast<uint32_t>(dest);
  uint32_t h = HashState(key, mask);
  bool placed = false;
  for (int probe = 0; probe < kMaxProbes; ++probe) {
    // Plain read first: within a step a slot transitions EMPTY -> key exactly once, so
    // matched/mismatched slots need no RMW at all (the CAS runs only on EMPTY).
    uint32_t cur = hkey[h];
    if (cur == kEmptyKey) {
      const uint32_t prev = atomicCAS(&hkey[h], kEmptyKey, key);
      if (prev == kEmptyKey) {  // first claimer registers the state
        const int32_t r = atomicAdd(&lc.next_raw, 1);
        if (r >= sz.claims_cap) {
          atomicOr(&lc.overflow, kOverflowClaims);
        } else {
          ws.next_claims[lane64 * sz.claims_cap + r] = make_int2(dest, static_cast<int32_t>(h));
        }
        cur = key;
      } else {
        cur = prev;
      }
    }
    if (cur == key) {
      // Early dominance check (1-best only): a candidate strictly below the state's
      // current best can never be resolved as the winner — skip the append entirely.
      // (A stale/lower payload read only makes the check conservative.) Lattice mode
      // keeps every in-beam candidate (they are lattice arcs).
      if (sz.lat_cap == 0 && ordered_end < static_cast<uint32_t>(hpay[h] >> 32)) {
        placed = true;
        break;
      }
      // Aggregated append: converged threads here share this strip's decode lane, so
      // one atomicAdd per coalesced group replaces per-thread adds on the hot counter.
      const cooperative_groups::coalesced_group grp =
          cooperative_groups::coalesced_threads();
      int32_t j0 = 0;
      if (grp.thread_rank() == 0)
        j0 = atomicAdd(&lc.cand_count, static_cast<int32_t>(grp.size()));
      const int32_t j = grp.shfl(j0, 0) + static_cast<int32_t>(grp.thread_rank());
      if (j >= sz.cand_cap) {
        atomicOr(&lc.overflow, kOverflowCand);
        placed = true;
        break;
      }
      ws.cand[lane64 * sz.cand_cap + j] = make_int2(tok, arc);
      if (sz.lat_cap > 0) ws.cand_end[lane64 * sz.cand_cap + j] = end;
      const unsigned long long packed =
          (static_cast<unsigned long long>(ordered_end) << 32) | static_cast<uint32_t>(j);
      atomicMax(&hpay[h], packed);
      placed = true;
      break;
    }
    h = (h + 1) & mask;
  }
  if (!placed) atomicOr(&lc.overflow, kOverflowHash);
}

// K2b: admit + hash recombine with the exact cutoff from K2a, over the flattened slot
// space (same walker as K2a).
__global__ void ExpandKernel(Workspace ws, Sizes sz, DeviceGraph g, const void* log_probs,
                             int64_t lp_lane_stride, int64_t lp_frame_stride) {
  const int32_t grand = ws.lane_arc_offsets[sz.lanes];
  if (grand == 0) return;
  const bool lp_half = sz.lp_half != 0;
  const int32_t nwarps = static_cast<int32_t>(gridDim.x * blockDim.x) >> 5;
  const int32_t wid = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int32_t span = (((grand + nwarps - 1) / nwarps) + 31) & ~31;  // 32-aligned
  const int32_t begin = wid * span;
  if (begin >= grand) return;
  const int32_t end = min(begin + span, grand);

  WarpWalkSlots(
      ws, sz, begin, end, lp_lane_stride, lp_frame_stride,
      [&](int32_t l, const K2Lane& v, int32_t slot, int32_t tok, int32_t off, bool valid) {
        if (valid) ExpandSlot(ws, sz, g, v, l, log_probs, lp_half, slot, tok, off);
      },
      [](const K2Lane&) {});
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
  const int32_t arena_lim = *ws.arena_limit;  // committed winners capacity
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
  __shared__ int32_t sh_base;   // arena base for this round's block
  __shared__ int32_t sh_round;  // this round's kept count
  __shared__ int32_t sh_ws[32];
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
    const int32_t incl = BlockInclusiveScan(keep, sh_ws);
    if (threadIdx.x == blockDim.x - 1) sh_round = incl;
    __syncthreads();
    // Allocate this round's winners block: shared arena (batch mode) or the lane's own
    // ring (streaming — LOGICAL monotonic ids survive across chunks and ring wraps; the
    // live window [gc_root, log_len) must never exceed the ring, so an overrunning
    // round is dropped-and-flagged rather than lapping live entries).
    if (threadIdx.x == 0) {
      const int32_t round_kept = sh_round;
      if (sz.stream_log_cap > 0) {
        sh_base = lc.log_len;
        // Window-form arithmetic: the live window stays tiny even when the logical ids
        // approach the int32 wall (~2^31 appends per channel). The clamp also pulls an
        // EpsResolve counter overshoot back to the window edge, as the absolute-min
        // form did.
        const int32_t window = lc.log_len - lc.gc_root;
        if (round_kept > 0 && window + round_kept > sz.stream_log_cap) {
          atomicOr(&lc.overflow, kOverflowArena);
        }
        lc.log_len = lc.gc_root + min(window + round_kept, sz.stream_log_cap);
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
      const int32_t out = sh_kept + incl - 1;
      const int32_t widx = sh_base + incl - 1;
      const int32_t widx_cap =
          (sz.stream_log_cap > 0) ? lc.gc_root + sz.stream_log_cap : arena_lim;
      if (widx < widx_cap && out < sz.main_q) {
        WinnersEntry(ws, sz, lane, widx) = make_int2(cur_winner[prev_local], arc);
        nxt_state[out] = state;
        nxt_score[out] = score;
        nxt_winner[out] = widx;
        if (lattice) ws.tok_fwd[widx] = score;
        if (keep_hash) ws.hash_pos[lane64 * sz.hash_cap + hslot] = out;
        widx_or_dead = widx;
      } else {
        atomicOr(&lc.overflow, widx >= widx_cap ? kOverflowArena : kOverflowKept);
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
    if (threadIdx.x == 0) sh_kept += sh_round;
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
