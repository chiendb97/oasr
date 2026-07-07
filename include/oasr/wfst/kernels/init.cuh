#pragma once

// Setup kernels: batch init (K0, batch size device-resident so captured graphs are
// batch-agnostic) and the streaming lane-lifecycle kernels (stream create, chunk begin,
// finalize-prep).
#include "oasr/wfst/kernels/common.cuh"

namespace oasr::wfst {
namespace kernels {

// ---------------------------------------------------------------------------
// K0: batch init. Batch size is device-resident so captured graphs are batch-agnostic.
__global__ void InitKernel(Workspace ws, Sizes sz, DeviceGraph g, const int32_t* T_per_lane,
                           const int32_t* d_batch, float search_beam) {
  const int32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= sz.lanes) return;
  const int32_t batch = *d_batch;
  LaneCounters& lc = ws.lanes[lane];
  lc = LaneCounters{};
  if (lane >= batch) {
    lc.status = 2;
    lc.phase = kPhaseDone;
    return;
  }
  lc.T = T_per_lane[lane];
  lc.chunk_start = 0;
  lc.chunk_end = lc.T + 1;  // includes the final step
  lc.dyn_beam = search_beam;
  lc.frontier_size = 1;
  lc.phase = kPhaseReal;
  ws.tok_state[0][lane * sz.main_q] = g.start_state;
  ws.tok_score[0][lane * sz.main_q] = 0.0f;
  // Arena slot `lane` holds the lane's start token; the cursor starts past them.
  ws.tok_winner[0][lane * sz.main_q] = lane;
  ws.winners[lane] = make_int2(-1, -1);
  if (lane == 0) *ws.arena_cursor = sz.lanes;
  if (sz.lat_cap > 0) {
    ws.tok_fwd[lane] = 0.0f;
    for (int32_t i = 0; i < sz.path_cap; ++i)
      ws.lat_seg[static_cast<int64_t>(lane) * sz.path_cap + i] = make_int2(0, 0);
    if (lane == 0) {
      *ws.lat_cursor = 0;
      *ws.lat_out_cursor = 0;
    }
  }
  if (sz.snap_frames > 0) {
    ws.snap[(static_cast<int64_t>(lane) * sz.snap_frames) * sz.main_q] =
        make_int2(g.start_state, __float_as_int(0.0f));
    ws.snap_len[lane * sz.snap_frames] = 1;
  }
}

// ---------------------------------------------------------------------------
// Streaming kernels. A channel == a lane; T = INT32_MAX encodes k2's online beam
// semantics in the shared formula (all offline-only clauses compare against final_t).

__global__ void StreamCreateKernel(Workspace ws, Sizes sz, DeviceGraph g, int32_t lane,
                                   float search_beam) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  LaneCounters& lc = ws.lanes[lane];
  lc = LaneCounters{};
  lc.T = INT32_MAX;
  lc.dyn_beam = search_beam;
  lc.frontier_size = 1;
  lc.phase = kPhaseReal;
  lc.log_len = 1;
  ws.tok_state[0][static_cast<int64_t>(lane) * sz.main_q] = g.start_state;
  ws.tok_score[0][static_cast<int64_t>(lane) * sz.main_q] = 0.0f;
  ws.tok_winner[0][static_cast<int64_t>(lane) * sz.main_q] = lane * sz.stream_log_cap;
  ws.winners[static_cast<int64_t>(lane) * sz.stream_log_cap] = make_int2(-1, -1);
}

// d_chunk_len[lane] > 0 opens a chunk for the lane; 0 leaves it idle this call.
__global__ void ChunkBeginKernel(Workspace ws, Sizes sz, const int32_t* d_chunk_len) {
  const int32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  if (lane >= sz.lanes) return;
  LaneCounters& lc = ws.lanes[lane];
  const int32_t len = d_chunk_len[lane];
  if (lc.status != 0 || len <= 0) return;
  lc.chunk_start = lc.t;
  lc.chunk_end = lc.t + len;
}

// Arms the final step: the next decoded step sees t == T and runs the k2 final-frame
// rules (final/redirect phase, beam 1e10).
__global__ void FinalizePrepKernel(Workspace ws, int32_t lane) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  LaneCounters& lc = ws.lanes[lane];
  if (lc.status != 0) return;
  lc.T = lc.t;
  lc.chunk_start = lc.t;
  lc.chunk_end = lc.t + 1;
}

}  // namespace kernels
}  // namespace oasr::wfst
