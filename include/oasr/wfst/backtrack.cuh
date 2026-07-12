#pragma once

// Path-extraction and table-hygiene kernels: streaming partial-hypothesis backtrack, the
// final best-path backtrack (one thread per lane walks the winners chain), and the
// end-of-batch hash sanitize that wipes lanes whose claim/probe lists overflowed.
#include "oasr/wfst/common.cuh"

namespace oasr::wfst {
namespace kernels {

// Partial hypothesis: backtrack from the best CURRENT frontier token of each requested
// lane (d_chunk_len marks the lanes advanced this call).
__global__ void PartialBacktrackKernel(Workspace ws, Sizes sz, const int32_t* d_chunk_len) {
  const int32_t lane = blockIdx.x;
  const LaneCounters& lc = ws.lanes[lane];
  int32_t* out = ws.arc_out + static_cast<int64_t>(lane) * sz.path_cap;
  __shared__ unsigned long long sh_best[256];
  if (d_chunk_len[lane] <= 0 || lc.status != 0 || lc.frontier_size == 0) {
    if (threadIdx.x == 0) ws.arc_out_len[lane] = (lc.status != 0) ? -1 : 0;
    return;
  }
  const float* score = ws.tok_score[0] + static_cast<int64_t>(lane) * sz.main_q;
  const int32_t* winner = ws.tok_winner[0] + static_cast<int64_t>(lane) * sz.main_q;
  unsigned long long best = 0;
  for (int32_t i = threadIdx.x; i < lc.frontier_size; i += blockDim.x) {
    const unsigned long long packed =
        (static_cast<unsigned long long>(FloatToOrderedUint(score[i])) << 32) |
        static_cast<uint32_t>(i);
    best = max(best, packed);
  }
  sh_best[threadIdx.x] = best;
  __syncthreads();
  for (int32_t d = blockDim.x >> 1; d > 0; d >>= 1) {
    if (threadIdx.x < d) sh_best[threadIdx.x] = max(sh_best[threadIdx.x], sh_best[threadIdx.x + d]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    int32_t w = winner[static_cast<int32_t>(sh_best[0] & 0xFFFFFFFFull)];
    int32_t len = 0;
    while (w >= 0 && len < sz.path_cap) {
      const int2 e = ws.winners[w];
      if (e.y >= 0) out[len++] = e.y;
      if (e.x == w) break;
      w = e.x;
    }
    ws.arc_out_len[lane] = len;
  }
}

// Hash sanitize: a claims-list overflow means K2b claimed hash slots that were never
// recorded, so K3's targeted clear missed them — the lane's table would silently poison
// every subsequent batch decoded in that lane. Wipe flagged lanes' tables at batch end.
__global__ void HashSanitizeKernel(Workspace ws, Sizes sz) {
  const int32_t lane = blockIdx.y;
  const LaneCounters& lc = ws.lanes[lane];
  if ((lc.overflow & (kOverflowClaims | kOverflowHash)) == 0) return;
  uint32_t* hkey = ws.hash_key + static_cast<int64_t>(lane) * sz.hash_cap;
  unsigned long long* hpay = ws.hash_payload + static_cast<int64_t>(lane) * sz.hash_cap;
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < sz.hash_cap;
       i += gridDim.x * blockDim.x) {
    hkey[i] = kEmptyKey;
    hpay[i] = 0ull;
  }
}

// Backtrack: one thread per lane walks the global winners chain, emitting the arc path
// in reverse into arc_out (host reverses + maps words; a few hundred bytes per lane).
__global__ void BacktrackKernel(Workspace ws, Sizes sz) {
  const int32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= sz.lanes) return;
  const LaneCounters& lc = ws.lanes[lane];
  int32_t* out = ws.arc_out + static_cast<int64_t>(lane) * sz.path_cap;
  int32_t len = 0;
  if (lc.status == 1) {
    int32_t w = lc.final_tok;
    while (w >= 0 && len < sz.path_cap) {
      const int2 e = ws.winners[w];
      if (e.y >= 0) out[len++] = e.y;
      if (e.x == w) break;  // safety
      w = e.x;
    }
  }
  ws.arc_out_len[lane] = len;
}

}  // namespace kernels
}  // namespace oasr::wfst
