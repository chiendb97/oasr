#pragma once

// Epsilon-closure kernels (TLG graphs): resolve a closure pass's payload winners into the
// live frontier, and the end-of-step claim cleanup that keeps hash claims live across the
// closure passes. No-ops on epsilon-free graphs (DeviceGraph::eps_count == nullptr).
#include "oasr/wfst/common.cuh"

namespace oasr::wfst {
namespace kernels {

// Epsilon-closure resolution (post-swap view; runs after Max+Expand of an eps pass).
// This pass's payload winners either update their state's existing frontier entry or
// append a new one; every improvement gets a fresh winners-log entry so backtracks and
// later arcs see the updated chain.
__global__ void EpsResolveKernel(Workspace ws, Sizes sz, DeviceGraph g) {
  const int32_t lane = blockIdx.x;
  LaneCounters& lc = ws.lanes[lane];
  if (lc.status != 0 || lc.phase != kPhaseEps) return;
  const int64_t lane64 = lane;
  const int32_t c0 = lc.cand_consumed;
  const int32_t c1 = min(lc.cand_count, sz.cand_cap);
  const int2* cand = ws.cand + lane64 * sz.cand_cap;
  uint32_t* hkey = ws.hash_key + lane64 * sz.hash_cap;
  unsigned long long* hpay = ws.hash_payload + lane64 * sz.hash_cap;
  int32_t* hpos = ws.hash_pos + lane64 * sz.hash_cap;
  int32_t* tok_state = ws.tok_state[0] + lane64 * sz.main_q;
  float* tok_score = ws.tok_score[0] + lane64 * sz.main_q;
  int32_t* tok_winner = ws.tok_winner[0] + lane64 * sz.main_q;
  const uint32_t mask = sz.hash_cap - 1;

  for (int32_t j = c0 + threadIdx.x; j < c1; j += blockDim.x) {
    const int2 c = cand[j];
    const uint32_t key = static_cast<uint32_t>(g.dest_ilabel[c.y].x);
    uint32_t h = HashState(key, mask);
    int32_t slot = -1;
    for (int probe = 0; probe < kMaxProbes; ++probe) {
      const uint32_t k = hkey[h];
      if (k == key) {
        slot = static_cast<int32_t>(h);
        break;
      }
      if (k == kEmptyKey) break;
      h = (h + 1) & mask;
    }
    if (slot < 0) continue;
    const unsigned long long packed = hpay[slot];
    if (static_cast<int32_t>(packed & 0xFFFFFFFFull) != j) continue;  // not the winner
    const float score = OrderedUintToFloat(static_cast<uint32_t>(packed >> 32));

    int32_t widx;
    if (sz.stream_log_cap > 0) {
      const int32_t off = atomicAdd(&lc.log_len, 1);
      if (off >= sz.stream_log_cap) {
        atomicOr(&lc.overflow, kOverflowArena);
        continue;
      }
      widx = lane * sz.stream_log_cap + off;
    } else {
      widx = atomicAdd(ws.arena_cursor, 1);
      if (widx >= sz.arena_cap) {
        atomicOr(&lc.overflow, kOverflowArena);
        continue;
      }
    }
    ws.winners[widx] = make_int2(tok_winner[c.x], c.y);
    if (sz.lat_cap > 0) ws.tok_fwd[widx] = score;

    const int32_t pos = hpos[slot];
    if (pos >= 0) {
      tok_score[pos] = score;
      tok_winner[pos] = widx;
    } else {
      const int32_t out = atomicAdd(&lc.frontier_size, 1);
      if (out < sz.main_q) {
        tok_state[out] = key;
        tok_score[out] = score;
        tok_winner[out] = widx;
        hpos[slot] = out;
      } else {
        atomicOr(&lc.overflow, kOverflowKept);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    lc.cand_consumed = c1;
    lc.frontier_size = min(lc.frontier_size, sz.main_q);
  }
}

// End-of-step claim cleanup in eps mode (claims stay live through the closure passes).
__global__ void ClearClaimsKernel(Workspace ws, Sizes sz) {
  const int32_t lane = blockIdx.x;
  const LaneCounters& lc = ws.lanes[lane];
  const int64_t lane64 = lane;
  const int32_t n = min(lc.next_raw, sz.claims_cap);
  uint32_t* hkey = ws.hash_key + lane64 * sz.hash_cap;
  unsigned long long* hpay = ws.hash_payload + lane64 * sz.hash_cap;
  int32_t* hpos = ws.hash_pos + lane64 * sz.hash_cap;
  for (int32_t r = threadIdx.x; r < n; r += blockDim.x) {
    const int32_t hslot = ws.next_claims[lane64 * sz.claims_cap + r].y;
    if (hslot >= 0) {
      hkey[hslot] = kEmptyKey;
      hpay[hslot] = 0ull;
      hpos[hslot] = -1;
    }
  }
}

}  // namespace kernels
}  // namespace oasr::wfst
