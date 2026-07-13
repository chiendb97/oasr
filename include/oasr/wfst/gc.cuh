#pragma once

// Winners-log garbage collection for offline long-form decoding (host-orchestrated,
// launched BETWEEN segment graphs, never inside a captured graph).
//
// Insight: the winners log is append-only and chains only point backwards, so the live
// set is exactly the entries reachable from the current frontier; beam chains converge
// to one common ancestor within a short window, making everything below it per lane a
// finalized "golden prefix" the host can drain and the arena can unmap. Three phases:
//
//   1. GcStampKernel     — per lane, stamp the anchor chain (frontier token 0, or the
//                          accepted final token) by setting kGcStampBit in each entry's
//                          arc field. Chains are strictly decreasing in index, so the
//                          walk is bounded by the live window.
//   2. GcConvergeKernel  — every other frontier token walks its chain until it hits a
//                          stamped entry (that hit lies ON the anchor chain); the
//                          deepest hit over all tokens is the lane's convergence point
//                          (any common ancestor is sound — exactness not required).
//   3. GcFinalizeKernel  — clear the stamps above the convergence point (those entries
//                          stay live), cut the chain there (prev = INT32_MIN sentinel;
//                          -1 already means "start token"), and emit the golden prefix
//                          below it into fin_arcs for the host to prepend. Fully
//                          decoded lanes (status == 1, 1-best) emit their whole
//                          remaining chain and set final_tok = kGcDoneTok so they stop
//                          pinning the watermark (BacktrackKernel then emits nothing).
//
// The host then unmaps whole chunks below min-over-lanes(convergence) and re-commits
// ahead of the global cursor (see decoder.cu RunWinnersGc).
#include "oasr/wfst/common.cuh"

namespace oasr::wfst {
namespace kernels {

// Anchor of a lane's live chain set: the accepted final token once the lane finished,
// else frontier token 0. -1 when the lane holds no live chain (dead / fully finalized).
__device__ inline int32_t GcAnchor(const Workspace& ws, const Sizes& sz, int32_t lane) {
  const LaneCounters& lc = ws.lanes[lane];
  if (lc.status == 1) return lc.final_tok;  // kGcDoneTok (< 0) once fully finalized
  if (lc.status == 0 && lc.frontier_size > 0)
    return ws.tok_winner[0][static_cast<int64_t>(lane) * sz.main_q];
  return -1;
}

__global__ void GcStampKernel(Workspace ws, Sizes sz) {
  const int32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= sz.lanes) return;
  ws.fin_len[lane] = 0;
  // An arena-overflow lane's frontier counts appends that were dropped, so its slots
  // beyond the written range are stale — pointers there may reference long-dead (even
  // unmapped) entries. Skip the lane AND veto this round's release (conv = -1): its
  // stale chains must keep resolving against mapped memory.
  if ((ws.lanes[lane].overflow & kOverflowArena) != 0) {
    ws.gc_conv[lane] = -1;
    return;
  }
  const int32_t anchor = GcAnchor(ws, sz, lane);
  ws.gc_conv[lane] = anchor < 0 ? INT32_MAX : anchor;
  if (anchor < 0) return;
  // Stamp the anchor chain down to the root / previous sentinel. The hop cap is
  // defensive (a live chain segment is <= (1 + eps_iterations) * gc_interval hops);
  // an unstamped tail only makes walkers report a deeper hit, never a wrong one.
  const int32_t floor = *ws.gc_floor;
  int32_t w = anchor;
  for (int32_t hops = 0; w >= floor && hops < sz.fin_cap; ++hops) {
    const int2 e = ws.winners[w];
    if (e.y >= 0) ws.winners[w].y = e.y | kGcStampBit;
    if (e.x == w) break;
    w = e.x;
  }
}

__global__ void GcConvergeKernel(Workspace ws, Sizes sz) {
  const int32_t lane = blockIdx.x;
  const LaneCounters& lc = ws.lanes[lane];
  // Finished lanes carry a single live chain (final_tok); nothing to converge.
  // gc_conv < 0 marks an overflow-degraded lane (see GcStampKernel) — don't walk it.
  if (lc.status != 0 || lc.frontier_size <= 1 || ws.gc_conv[lane] < 0) return;
  const int32_t floor = *ws.gc_floor;
  const int32_t* winner = ws.tok_winner[0] + static_cast<int64_t>(lane) * sz.main_q;
  for (int32_t i = 1 + threadIdx.x; i < lc.frontier_size; i += blockDim.x) {
    int32_t w = winner[i];
    int32_t hit = -1;  // stays -1 on a hop-cap abort: host skips the release round
    for (int32_t hops = 0; w >= floor && hops < sz.fin_cap; ++hops) {
      const int2 e = ws.winners[w];
      hit = w;  // terminal fallback: the shared root is on the anchor chain too
      if (e.y >= 0 && (e.y & kGcStampBit)) break;
      if (e.x == w) break;
      w = e.x;
      if (hops == sz.fin_cap - 1) hit = -1;  // cap hit without a stamp: abort lane
    }
    atomicMin(&ws.gc_conv[lane], hit);
  }
}

__global__ void GcFinalizeKernel(Workspace ws, Sizes sz) {
  const int32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= sz.lanes) return;
  LaneCounters& lc = ws.lanes[lane];
  const int32_t conv = ws.gc_conv[lane];
  const int32_t anchor = GcAnchor(ws, sz, lane);
  if (anchor < 0 || conv == INT32_MAX) return;    // no live chain
  if (conv < 0) return;                           // converge aborted (defensive)
  // 1-best finished lanes finalize COMPLETELY: emit from final_tok itself and mark the
  // lane done — its chain no longer bounds the unmap watermark. (Lattice mode keeps
  // final_tok: LatSeed/LatEmit still index tok_bwd with it.)
  const bool full = (sz.lat_cap == 0 && lc.status == 1);
  // Clear the stamps on [anchor, conv] — these entries stay live and their arc fields
  // are read by later steps and the final backtrack.
  const int32_t floor = *ws.gc_floor;
  int32_t w = anchor;
  for (int32_t hops = 0; w >= floor && hops < sz.fin_cap; ++hops) {
    const int2 e = ws.winners[w];
    if (e.y >= 0 && (e.y & kGcStampBit)) ws.winners[w].y = e.y & ~kGcStampBit;
    if (w == conv || e.x == w) break;
    w = e.x;
  }
  if (w != conv) {  // conv off the anchor chain (defensive) — skip this lane's round
    ws.gc_conv[lane] = -1;
    return;
  }
  int32_t start;
  if (full) {
    start = conv;  // emit conv's own arc too; nothing below stays reachable
    lc.final_tok = kGcDoneTok;
    ws.gc_conv[lane] = INT32_MAX;  // finished lane no longer bounds the watermark
  } else {
    const int32_t old_prev = ws.winners[conv].x;
    if (old_prev == INT32_MIN || old_prev == -1) return;  // already at a root
    ws.winners[conv].x = INT32_MIN;  // finalized-root sentinel (backtrack stops here)
    start = old_prev;
  }
  // Emit the golden prefix, newest -> oldest (host reverses). Entries below conv are on
  // the stamped anchor chain; mask the stamp on emit (the memory itself is garbage).
  int32_t* fin = ws.fin_arcs + static_cast<int64_t>(lane) * sz.fin_cap;
  int32_t len = 0;
  w = start;
  while (w >= floor) {
    const int2 e = ws.winners[w];
    if (e.y >= 0) {
      if (len >= sz.fin_cap) {  // defensive; matches Backtrack's own truncation cap
        atomicOr(&lc.overflow, kOverflowArena);
        break;
      }
      fin[len++] = e.y & ~kGcStampBit;
    }
    if (e.x == w) break;
    w = e.x;
  }
  ws.fin_len[lane] = len;
}

// ---------------------------------------------------------------------------
// Streaming (per-lane ring) GC — one round per AdvanceChunk. The ring never unmaps
// while a channel is open, so this is not about physical memory: it advances gc_root
// (the writer's wrap guard) and drains finalized golden-prefix arcs to the host BEFORE
// the ring laps them, making stream_log_cap a live-window size instead of a hard cap
// on stream length (it also lifts the path_cap truncation of long-stream backtracks:
// device tails stay within the window; the host owns everything older).

__global__ void StreamGcStampKernel(Workspace ws, Sizes sz) {
  const int32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= sz.lanes) return;
  ws.fin_len[lane] = 0;
  const LaneCounters& lc = ws.lanes[lane];
  // Only live, clean lanes: finished lanes already delivered their result through
  // FinalizeStream (their window stops growing), overflow-degraded lanes may carry
  // stale pointers (walk them read-only at most — ring memory is always mapped, but a
  // bogus root advance would degrade them further).
  if (lc.status != 0 || lc.frontier_size <= 0 || (lc.overflow & kOverflowArena) != 0) {
    ws.gc_conv[lane] = -1;
    return;
  }
  const int32_t anchor = ws.tok_winner[0][static_cast<int64_t>(lane) * sz.main_q];
  ws.gc_conv[lane] = anchor;
  // Stamp depth only needs to cover walker merge points (recombination is shallow —
  // a few frames), not the whole window; deeper walkers abort the round harmlessly.
  int32_t w = anchor;
  for (int32_t hops = 0; w >= 0 && hops < sz.fin_cap; ++hops) {
    int2& e = WinnersEntry(ws, sz, lane, w);
    if (e.y >= 0) e.y = e.y | kGcStampBit;
    if (e.x == w) break;
    w = e.x;
  }
}

__global__ void StreamGcConvergeKernel(Workspace ws, Sizes sz) {
  const int32_t lane = blockIdx.x;
  const LaneCounters& lc = ws.lanes[lane];
  if (ws.gc_conv[lane] < 0 || lc.status != 0 || lc.frontier_size <= 1) return;
  const int32_t* winner = ws.tok_winner[0] + static_cast<int64_t>(lane) * sz.main_q;
  for (int32_t i = 1 + threadIdx.x; i < lc.frontier_size; i += blockDim.x) {
    int32_t w = winner[i];
    int32_t hit = -1;
    for (int32_t hops = 0; w >= 0 && hops < sz.fin_cap; ++hops) {
      const int2 e = WinnersEntry(ws, sz, lane, w);
      hit = w;
      if (e.y >= 0 && (e.y & kGcStampBit)) break;
      if (e.x == w) break;
      w = e.x;
      if (hops == sz.fin_cap - 1) hit = -1;  // merge deeper than the stamp range
    }
    atomicMin(&ws.gc_conv[lane], hit);
  }
}

// Clear stamps, then walk the segment below the convergence point ONCE, emitting arcs
// into fin_arcs as a phase ring keyed by the emit counter: the ring retains the OLDEST
// min(total, fin_cap) arcs, a cut pointer lagging fin_cap emits behind the walk lands
// on the entry just above them, and fin_len reports the TOTAL so the host unscrambles
// the phase ((total-1) % fin_cap backwards) and detects how much was drained. Bounded
// drain per round; a backlog (convergence stall) catches up at fin_cap arcs per round.
__global__ void StreamGcFinalizeKernel(Workspace ws, Sizes sz) {
  const int32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= sz.lanes) return;
  LaneCounters& lc = ws.lanes[lane];
  const int32_t conv = ws.gc_conv[lane];
  if (conv < 0) return;
  const int32_t anchor = ws.tok_winner[0][static_cast<int64_t>(lane) * sz.main_q];
  // Clear the stamps on [anchor, conv] (same bounded traversal as the stamp pass).
  int32_t w = anchor;
  for (int32_t hops = 0; w >= 0 && hops < sz.fin_cap; ++hops) {
    int2& e = WinnersEntry(ws, sz, lane, w);
    if (e.y >= 0 && (e.y & kGcStampBit)) e.y = e.y & ~kGcStampBit;
    if (w == conv || e.x == w) break;
    w = e.x;
  }
  if (w != conv) return;  // conv off the stamped range (defensive): skip the round
  int32_t* fin = ws.fin_arcs + static_cast<int64_t>(lane) * sz.fin_cap;
  int32_t emits = 0;
  int32_t cut = conv;      // lags fin_cap emits behind the walk
  int32_t lag = 0;
  w = WinnersEntry(ws, sz, lane, conv).x;
  while (w >= 0) {
    const int2 e = WinnersEntry(ws, sz, lane, w);
    if (e.y >= 0) {
      fin[emits % sz.fin_cap] = e.y & ~kGcStampBit;
      ++emits;
      if (lag >= sz.fin_cap) {
        // advance the cut one chain hop (skipping nothing: every hop below conv in a
        // streaming ring carries an arc except the terminal start token)
        cut = WinnersEntry(ws, sz, lane, cut).x;
      } else {
        ++lag;
      }
    }
    if (e.x == w) break;
    w = e.x;
  }
  ws.fin_len[lane] = emits;
  if (emits == 0) return;
  // Sentinel the cut and advance the writer's wrap guard: [gc_root, log_len) is live.
  WinnersEntry(ws, sz, lane, cut).x = INT32_MIN;
  lc.gc_root = cut;
}

}  // namespace kernels
}  // namespace oasr::wfst
