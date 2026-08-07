# GPU WFST Beam-Search Decoder

The WFST decoder (`include/oasr/wfst/*.cuh` + `csrc/decoder/wfst/`) is OASR's
GPU-resident WFST beam-search decoder for CTC ASR: it takes per-frame
log-posteriors `log_prob[B, T, V]` and a k2-style decoding graph (epsilon-free
HLG, or TLG with epsilon arcs) and produces the best word sequence per
utterance — optionally a pruned lattice — entirely on the device. The full
per-utterance decode (hundreds of frames × 5 kernels) replays as **one CUDA
graph launch** with **zero host synchronisation inside the decode loop**; the
only D2H transfer is the batch-end result readback.

Its semantics contract is **k2 `intersect_dense_pruned`, exactly**: the
dynamic-beam update, final-frame rules, `allow_partial` handling and
`output_beam` backward pruning are implemented verbatim, so word/score output
is interchangeable with the k2 CPU path at identical settings (this is tested,
see §13). Measured against k2 itself on the primary HLG the decoder is 6–10×
faster at equal WER; the standalone project's `docs/{RESEARCH,DESIGN,REPORT}.md`
hold that original analysis.

The decoder is a stateful C++ object (`GpuDecoder`: persistent workspace +
uploaded graph + CUDA-graph caches + streaming channels) compiled on first use
via TVM-FFI JIT (`oasr/jit/wfst_decoder.py`, cached in `~/.cache/oasr/jit/`)
and driven from Python through opaque `int64` handles — the same packaging as
the GPU CTC decoder. There is no build flag; `pip install` never compiles it.

## 1. Purpose and Responsibilities

1. **Frontier management**: the set of alive graph states per lane
   (= utterance slot), with Viterbi scores and backpointers.
2. **Arc expansion**: one relaxation per frame over all out-arcs of the
   frontier — `end = score[src] + arc_weight + log_prob[t][ilabel]`.
3. **Exact pruning**: k2's dynamic beam against the exact per-step maximum
   (never a stale or sampled cutoff), soft `max_active` via beam decay.
4. **Recombination**: dedupe candidates by destination state, keeping the
   best incoming arc (open-addressing hash, one u64 `atomicMin`-style resolve).
5. **Backtrack**: winners-log chains → graph arc ids → words via the aux CSR.
6. **Lattice mode**: persist all in-beam candidates per frame; GPU backward
   `output_beam` prune; export flat records that
   `oasr/decoder/wfst/lattice.py` assembles into k2 `Fsa` lattices.
7. **Streaming**: chunked decoding with k2 *online* beam semantics
   (channel-per-lane, partial hypotheses per chunk).

## 2. High-Level Architecture

```
              log_probs [B, T, V] (fp32 or fp16, CUDA, packed)
                              │  read in place: device-resident LpDesc carries
                              ▼  base+strides (captured graphs stay valid)
      ┌──────────────────────────────────────────────────────┐
      │ K0  InitKernel      batch init; d_batch device-      │
      │                     resident → graphs batch-agnostic │
      └──────────────────────────┬───────────────────────────┘
                                 ▼
   ┌───────────────────── per-step loop (captured) ─────────────────────┐
   │ K1  ScanKernel      CTA/lane: phase decision, k2 dyn-beam update,  │
   │                     degree prefix-sums, per-token upper bounds,    │
   │                     frame max log-prob                             │
   │     LaneScanKernel  1 warp: exclusive scan of per-lane total_arcs  │
   │                     → the flattened global slot space              │
   │ K2a MaxKernel       1-D grid over ALL lanes' arcs: exact per-step  │
   │                     max (warp-strip posting + upper-bound skip)    │
   │ K2b ExpandKernel    same walk: exact beam admit + hash recombine   │
   │ K3  FinalizeKernel  CTA/lane: exact post-filter, winner resolve,   │
   │                     next-frontier build, targeted hash clear       │
   │     (swap frontier buffers; TLG: eps-closure passes; lattice:      │
   │      LatPersistKernel + ClearClaimsKernel)                         │
   └────────────────────────────┬────────────────────────────────────--┘
                                ▼
      BacktrackKernel (1 thread/lane walks the winners chain)
      HashSanitizeKernel (wipes lanes whose claim lists overflowed)
      [lattice: backward output-beam prune + LatEmitKernel]
                                │  one async D2H: lane counters, arc
                                ▼  paths, cursors
                    words / scores / overflow  (+ lattice records)
```

Two boundaries are load-bearing and must not be fused: K1→K2 needs the
completed degree scan, and K2a→K2b needs global visibility of the exact
maximum (a kernel boundary is the cheapest device-wide fence). The exact
two-pass max is a *capacity* measure, not just a semantic one: admitting
against a stale running max produces unbounded loose supersets on hard
utterances.

## 3. Decoding Semantics (k2 parity)

Per step, per lane (see `cpu_reference.cc` for the executable spec):

- **Dynamic beam** (K1, verbatim k2): within `[min_active, max_active]` (or
  empty) relax `b ← 0.8·b + 0.2·search_beam`; under `min_active` widen
  `b ← max(b, search_beam)·1.25`; over `max_active` tighten
  `b ← min(b, search_beam)·0.8`; in the last ~5 frames
  `min_active ← max(min_active, max_active/2)`; on the final step `b = 1e10`.
  `max_active` is **soft** — enforced only through next-step decay, never a
  hard top-k, so transient frontiers can far exceed it (measured worst on the
  primary HLG: ~217k states at beam 20).
- **Admit test**: `end > exact_max − b`, where `exact_max` is the true maximum
  over *all* expanded arcs of this step (K2a). K3 re-applies the same filter
  on resolved winners, so the surviving state set equals k2's arc-level
  pruning exactly.
- **Final step** (`t == T`): expand only ilabel = −1 arcs, acoustic 0, beam
  1e10. If no frontier state has a final arc and `allow_partial` is set, the
  step *redirects*: the single best-scoring arc claims a synthetic super-final
  token (`kPhaseRedirect`).
- **`output_beam` never acts in the forward pass** — it is applied by the
  lattice backward prune only (`forward[src] + arc + backward[dst] ≥ best −
  output_beam`), matching k2.
- **Blank-skip stays outside the decoder** (WeNet-style frame pre-slicing in
  `WfstDecoderSearch._prepare`, threshold 0.98 by default). In-kernel skipping
  would be wrong on this graph class: blank self-loops cover only ~50% of the
  primary HLG's states.
- **Determinism**: scores are deterministic (max/min are order-independent).
  Exact-score ties break by candidate-append order, which depends on atomic
  scheduling — the same property k2 has via its insert races. Ties can
  therefore pick a different (equal-score) backpointer between runs/builds.
- **Streaming** uses k2's *online* beam semantics: `T = INT32_MAX` until
  `FinalizeStream`, which makes every offline-only clause of the beam formula
  (final-window `min_active`, final-step beam) inert until the caller ends
  the stream.

## 4. Graph Image

Graphs load from a binary `hlg.img` (mmap-friendly, versioned):

| Section | Type | Notes |
|---|---|---|
| header (256 B) | magic `WFSTIMG1`, version 1/2, flags, counts | flags: finals-at-end, has-eps, eps-first |
| `row_splits[N+1]` | i32 | CSR arc offsets per state, **k2 arc order preserved** |
| `final_count[N]` | i32 | #ilabel=−1 arcs at one end of each state's range |
| `arc_dest_ilabel[2A]` | i32 interleaved | `{dest, ilabel}` per arc |
| `arc_weight[A]` | f32 | |
| `aux_row_splits[A+1]`, `aux_pool` | i32 | ragged word ids per arc (CSR) |
| `eps_count[N]` | i32 | v2 + has-eps only (TLG) |

Export once, offline (k2 is imported *only* here):

```bash
python -m oasr.decoder.wfst.graph_export --hlg HLG.pt --out hlg.img
# TLG graphs: add --epsilon-id 0
```

**Identity arc order is load-bearing**: our arc index == k2 graph arc index,
so lattice `arc_map_a` is the identity and `aux_labels` attach by direct
indexing — zero translation, exact parity. Never reorder arcs.

On upload (`GpuDecoder` ctor) the interleaved arc column is **split SoA**:

- `arc_ilabel` as **u16** (`0xFFFF` encodes ilabel −1) — requires
  `vocab_size < 65535`, checked with a hard error at load;
- `arc_dest` as i32 — loaded by K2b only for arcs that pass the beam test;
- `emit_maxw[N]` (f32) — per-state max weight over its *emitting* arcs,
  computed host-side at load; feeds the K2 upper-bound skip (§5.3).

The split matters because the two hot kernels read different columns: the max
pass never needs `dest`, and most expanded arcs never survive to need it in
the expand pass either. Device graph footprint for the primary HLG
(4.68M states / 88.2M arcs): ~938 MiB (was 1097 MiB interleaved).

## 5. Per-Step Kernel Pipeline

### 5.1 K1 — ScanKernel (CTA per lane, 1024 threads)

Decides the phase (`real | final | redirect | eps`), applies the k2 beam
update, then computes, per frontier token: the arc degree for this phase, the
first arc to expand (`tok_emit_begin`), the exclusive degree prefix-sum
(`arc_offsets`, warp-shuffle `BlockInclusiveScan` — see `common.cuh`), and
two upper-bound ingredients used by K2:

- `tok_ub[i] = tok_score[i] + emit_maxw[state]`;
- `lc.max_lp` = max over the lane's current log-prob row (block reduce; 0 for
  final/redirect phases, whose arcs carry no acoustic term).

Finished/idle lanes publish `total_arcs = 0` and drop out of the step.

### 5.2 The flattened slot space (LaneScanKernel)

Per-lane frontier sizes are wildly imbalanced (a single lane can transiently
hold ~200k states while its batchmates hold hundreds), and profiling showed
those steps *are* the runtime: with a static `(blocks, lane)` grid the hot
lane is capped at its slice of the GPU while every other lane's blocks exit.
Instances >100 µs carried ~66–70% of total K2 time.

`LaneScanKernel` (one warp) therefore exclusive-scans `total_arcs` across
lanes into `lane_arc_offsets[lanes+1]`, concatenating all lanes' arc ranges
into one **global slot space**. K2a/K2b run 1-D grids
(`ExpandBlocksFor(lanes) = clamp(160·lanes, 1024, 8192)` blocks × 256) over
it, so any lane mix — including one hot lane — load-balances over the whole
device. Excess blocks read one int and exit.

### 5.3 K2a/K2b — the warp walker (`WarpWalkSlots`)

Each warp owns a **contiguous, 32-aligned span** of the global slot space and
streams through it in 32-slot strips:

1. Resolve the decode lane once per lane segment (5-step binary search over
   the 33-entry `lane_arc_offsets`, L1-resident).
2. Lane 0 runs **one** global binary search into the lane's `arc_offsets` at
   the segment start; from there the token cursor only moves forward.
3. Per strip, the warp loads the next 32 token boundaries **coalesced** (one
   transaction), and every lane resolves its own token with a 5-shuffle
   binary count over the register-held sorted batch — no shared memory, no
   `__syncthreads`, warps stay fully independent.

This replaces the original ~18 dependent global loads *per arc slot* (the
per-slot binary search) that made both kernels latency-bound (0.35 issued
warps/scheduler at ~90% occupancy). A block-cooperative shared-memory variant
was tried first and **regressed**: two serialized scalar searches plus block
syncs per 256-slot window cost more than they saved at median frontier sizes.
Warp granularity with zero synchronisation is the right shape.

**Upper-bound skip** (both kernels): fp32 addition is monotone, so
`ub = tok_ub[tok] + max_lp` bounds every end score of the token *exactly* —
`(s+w)+lp ≤ (s+w)+maxlp ≤ (s+maxw)+maxlp` holds in IEEE arithmetic, not just
in ℝ. Therefore:

- K2a skips arcs whose `ub ≤ running_max`: the skipped arc is dominated by an
  already-posted end score, so the final max is unchanged, bit-exactly. Each
  warp posts its strip max to `lc.running_max` so later strips (and other
  warps) skip more. This cut MaxKernel 3.5×.
- K2b skips whole tokens with `ub ≤ exact_max − beam`: every arc of the token
  fails the admit test it would have been given. The admitted candidate set
  is unchanged.

**K2b hash recombination**, keyed by destination state, per admitted arc:

- probe with a **plain read first**; the CAS runs only on `EMPTY` slots
  (within a step a slot transitions `EMPTY → key` exactly once, so matched
  and mismatched slots need no RMW at all);
- first claimer appends `{state, hash_slot}` to the claims list — it does
  **not** write cost or backpointer (the stale-winner race: another thread
  may still improve the payload; resolution happens in K3 after the kernel
  boundary);
- dominance check (1-best only): a candidate strictly below the slot's
  current best is dropped without appending — a stale (lower) payload read
  only makes this conservative, never wrong;
- surviving candidates append `{prev_local, arc}` to the candidate buffer
  with a **warp-aggregated** counter add (`cg::coalesced_threads()`; all
  converged threads of a strip share the same decode lane by construction),
  then `atomicMax(hash_payload, {ordered_end : cand_idx})` resolves cost and
  argmax in one u64 op.
- linear-probe bound (`kMaxProbes = 256`) → `kOverflowHash`; capacity trips
  set their own overflow bits (§7).

The redirect phase short-circuits in `ExpandSlot`: the thread whose end score
equals the exact max CASes the single super-final candidate.

### 5.4 K3 — FinalizeKernel (CTA per lane)

Walks the claims list in 1024-wide rounds: reads each claimed slot's resolved
payload, applies the exact cutoff, compacts survivors (warp-shuffle block
scan) into the next frontier, allocates their winners-log entries (one
`atomicAdd` block per round on the shared arena cursor — or the lane's own
region in streaming mode), writes `{prev_global_winner, arc}` backpointers,
and clears exactly the claimed hash slots (k2's targeted-delete trick; full
clears would cost ~MBs of writes per step). In eps/lattice modes the claims
stay live for the step tail and `ClearClaimsKernel` runs instead. The final
step additionally resolves lane status/score/`final_tok`.

### 5.5 Epsilon closure (TLG) and lattice persistence

On graphs with epsilon arcs, each step appends `eps_iterations` closure passes
(ScanKernel(eps) → LaneScan → Max → Expand → `EpsResolveKernel`): payload
winners either improve their state's live frontier entry (fresh winners-log
entry, so later arcs and backtracks see the updated chain) or append a new
one. Lattice mode persists each step's full candidate set *after* the closure
(`LatPersistKernel`, token ids canonical), tagging epsilon arcs (`kEpsArcBit`)
and redirect arcs (`kRedirectArcBit`).

## 6. Memory Model

### 6.1 Workspace (per decoder instance, allocated once)

Capacities derive from `DecoderConfig`:

```
main_q     = main_q_factor · max_active_states     # surviving frontier / lane
claims_cap = 2 · main_q                            # distinct dest states / step
cand_cap   = cand_factor · main_q                  # admitted candidates / step
hash_cap   = next_pow2(2 · claims_cap)             # per-lane slots
arena_cap  = min(512 Mi, max(64 Mi, 16 Mi · lanes))   # winners entries (shared)
path_cap   = max_frames + 2
```

| Structure | Layout | Sizing | Allocation |
|---|---|---|---|
| Frontier ×2 (`tok_state/score/winner`) | SoA i32/f32/i32 | `lanes × main_q` | eager |
| `tok_emit_begin`, `tok_ub`, `arc_offsets` | i32/f32 | `lanes × main_q(+1)` | eager |
| Claims `{state, hash_slot}` | int2 | `lanes × claims_cap` | eager |
| Candidates `{prev, arc}` (+`cand_end` f32 in lattice) | int2 | `lanes × cand_cap` | eager |
| Hash `key u32 + payload u64` (+`hash_pos` i32 eps/lattice) | SoA | `lanes × hash_cap` | eager |
| **Winners log** `{prev, arc}` | int2, global cursor | `arena_cap` | **lazy (VMM)** |
| Lattice arenas (`lat`, `lat2`, `tok_fwd`, `tok_bwd`, `lat_out`) | — | `lat_cap` / `arena_cap` | **lazy (VMM)** |
| Log-prob staging (STREAMING only) | `[lanes, max_frames, V]` | fp32/fp16 | **lazy (VMM)**; offline decodes the caller's tensor in place via the device-resident `LpDesc` |

At the offline defaults the Python wrapper uses (`main_q_factor=32`,
`cand_factor=3`, `max_active=10000`): `main_q` 320k, `hash_cap` 2Mi slots
(≈25 MiB/lane), ≈48 MiB of eager workspace per lane.

### 6.2 Lazily-committed regions (`csrc/decoder/wfst/lazy_region.h`)

The winners log and lattice arenas must be *reserved* at worst-case size
(capacity-sized launches, capture-stable pointers) but their realistic use is
1–2 orders of magnitude smaller — the measured winners high-water for a full
B=32 LJSpeech batch is ~44 MiB against the 4 GiB worst case. `LazyRegion`
resolves this with CUDA VMM:

- `cuMemAddressReserve` the full capacity once — kernel argument pointers and
  captured CUDA graphs stay valid for the decoder's lifetime;
- `cuMemCreate`/`cuMemMap` physical chunks (32 MiB units; driver-granularity
  units for the staging buffer) only where the region is actually used;
- eager `cudaMalloc` fallback when the device/driver lacks VMM support
  (identical behaviour and footprint to the pre-paging decoder).

Committed capacity is published device-side (`ws.arena_limit`,
`ws.lat_limit`); kernels bound their appends against it and flag
`kOverflowArena` when exceeded. `DecodeBatch` then **doubles the commitment
and re-runs**: decode is idempotent (`InitKernel` resets all state), pointers
never move, no graph re-capture — the growth costs one re-decode of the
triggering batch, once per high-water regime. Initial commits: 64 MiB
winners, 64 MiB lattice, 32 MiB lattice-out.

Two placement rules follow from VMM semantics:

- offline decodes have **no staging at all**: the kernels read the log-prob
  base pointer and strides from a device-resident descriptor (`LpDesc`,
  rewritten with one 24 B H2D before every decode — captured graphs read it
  at replay), so the caller's packed `[B, T, V]` tensor is consumed in place.
  K1 publishes each lane's absolute row pointer into its `LaneCounters`, so
  the hot K2 kernels never touch the descriptor. Streaming keeps a small
  fully-committed staging block (`AdvanceChunk` scatters caller rows into
  fixed per-channel slots) and points the descriptor at it once;
- streaming winners regions are interleaved per channel, not a prefix:
  `stream_log_cap` (default `arena_cap / lanes`, override with
  `DecoderConfig::stream_log_entries`) is rounded up to whole mapping chunks,
  each channel's slice is committed at `CreateStream` and unmapped at
  `ReleaseStream` (`LazyRegion::ReleaseRange`), so the streaming footprint
  tracks **active** channels instead of `max_lanes` (32 idle channels: 4 GiB →
  0). `ReleaseStream` also deadens the lane's device counters
  (`StreamResetKernel`) before unmapping — the shared per-step kernels and
  `BacktrackKernel` gate on lane status, so nothing walks a released chain.
  With per-chunk GC (§8.1) the slice is a ring and `stream_log_entries` sizes
  the live window (one 32 MiB chunk ≈ 4Mi entries suffices), not the stream.

`DecoderConfig::arena_budget_entries` (0 = the `min(512Mi, max(64Mi,
16Mi·lanes))` formula above) caps the winners-arena reservation for
memory-constrained deployments; both budget knobs are trailing arguments of
`wfst_create_decoder`.

`wfst_decoder_mem_stats(handle, out_i64[4])` reports
`{reserved, committed, fixed, arena_high_water}`. Measured after the first
decode (primary HLG, fp32, T≈250):

| Config | Before (eager) | After (lazy) |
|---|---|---|
| bench: 32 lanes, mqf 32, max_frames 1024 | 7,260 MiB | **2,712 MiB** (fixed 2,405 + committed 280 of 4,722 reserved) |
| production offline: 8 lanes, mqf 32 | ~2.6–3.1 GiB | **1,410 MiB** |

### 6.3 Winners-log GC for long-form audio (`gc_interval`)

Even lazily committed, the winners log grows O(T·lanes): every step appends the
surviving frontier and nothing is ever reclaimed. For long-form audio
(`T ≳ 2000`) that is hundreds of MiB of physical memory holding almost
entirely dead entries — only entries reachable from the current frontier are
live, and beam chains converge to a single common ancestor within a short
window, so per lane everything below that ancestor is a finalized "golden
prefix" and, log-wide, garbage is a *prefix* of the append order.

`DecoderConfig::gc_interval = N` (> 0, even; trailing arg of
`wfst_create_decoder`, `WfstDecoderOptions.gc_interval`) enables a GC round
every N offline steps. 1-best mode swaps the whole-batch graph for a segmented
host loop of cached `StepsExec` graphs (same kernels, same order — results are
bit-identical, verified against the external decoder at 576/576); interval-
lattice mode piggybacks on its existing prune points. GC never runs inside a
captured graph. Each round:

1. **Convergence find** — per lane, stamp the anchor chain (frontier token 0,
   or `final_tok` once finished) via bit 30 of the arc field (`kGcStampBit`;
   arc ids < 2^30 enforced at construction), then walk every other frontier
   token until it hits a stamp; the deepest hit is the lane's convergence
   point (chains are strictly index-decreasing, so walks are short and exact).
2. **Prefix finalization** — cut the chain at the convergence point
   (`prev = INT32_MIN` sentinel; `-1` still means start token), emit the arcs
   below it into a per-lane staging buffer the host drains and prepends to the
   final backtrack. Fully decoded lanes (1-best) emit their entire remaining
   chain and set `final_tok = kGcDoneTok`, so early finishers stop pinning the
   watermark in mixed-length batches.
3. **Window slide** — the committed winners window becomes
   `[chunk_floor(min-over-lanes convergence), cursor + headroom)`:
   `ReleaseRange` unmaps whole chunks behind the watermark, `EnsureRange`
   tops up ahead of the global cursor. The headroom (init 64 MiB) doubles via
   the normal grow-and-retry path if a segment outruns it. A device-resident
   `gc_floor` bounds every winners walk — clean lanes stop at their sentinel
   anyway; an arena-overflow-degraded lane (whose frontier can carry stale
   pointers, and which GC therefore skips and stops releasing for) must fail
   soft rather than touch unmapped memory.

Lattice-mode caveat: only the winners log is windowed — surviving lattice
records reference arbitrarily old token ids, so `tok_fwd`/`tok_bwd` stay
prefix-committed (token renumbering during interval compaction would be
required to window them; see §14). In pure lattice mode (no interval prune)
GC is rejected: every token stays reachable by construction.

Measured (tiled LJSpeech, T 1800–3000, 32 lanes, RTX 5090): words + scores
identical GC on/off; winners committed drops from the 264 MiB high-water +
growth curve to a sliding ~64–128 MiB window (whole-decoder committed 2346 →
96 MiB together with the in-place log-prob descriptor); decode +4.2% at N=64,
+2.3% at N=128, +1.6% at N=256. Recommended: `gc_interval=128` for T > 2000;
leave 0 for short utterances (the segmented loop and window bookkeeping only
pay off when the log outgrows its window).

### 6.4 Why lazy commitment and not block-table paging

The batch-mode winners log is append-only through a single global cursor —
it is already perfectly packed, so a PagedAttention-style block table would
add an indirection per write/backtrack hop and reclaim nothing; the waste was
worst-case *provisioning*, which VMM commitment eliminates at zero access
cost. Block-table paging **does** fit the streaming winners regions (fixed
per-channel slices strand memory under short streams and cap long ones —
KV-cache economics); that design (per-channel logical indices + device block
allocator + translation in K3/EpsResolve/backtrack, mirroring
`oasr/cache/BlockPool`) is the recommended follow-up if streaming WFST scales
to high channel counts.

## 7. Outputs and Overflow Semantics

**1-best**: `BacktrackKernel` (1 thread/lane) walks the winners chain from
`final_tok`, emitting graph arc ids in reverse; the host maps arcs → words
through the aux CSR. Results marshal through caller-allocated CPU tensors;
`word_lens` reports the TRUE length so callers detect truncation and re-issue
with a bigger buffer (`decode_batch` is idempotent).

**Lattice**: flat records `{src_tok, dst_tok, label, arc_map, score_bits,
seg, lane, eps} × i32[8]` after the GPU backward `output_beam` prune;
`oasr/decoder/wfst/lattice.py` assembles k2 `FsaVec` lattices with exact
`aux_labels` via the identity `arc_map`. Long-form audio uses
`lat_prune_interval=N` (even): a k2-style window-loose interval prune bounds
the arena; the exact final pass is unchanged. Note the JIT binding currently
exposes 1-best only; lattice mode runs but records are reachable only through
the C++ API (`LastLatticeRecords`).

**Overflow bits** (`DecodeResult::overflow` / `out_meta[:,2]`):

| bit | name | meaning | handling |
|---|---|---|---|
| 1 | `kOverflowCand` | candidate buffer full | caller: rescue decode (bigger factors) |
| 2 | `kOverflowHash` | probe bound exceeded | caller: rescue decode |
| 4 | `kOverflowArena` | winners/lattice arena full | **automatic**: grow-commit + re-run; reported only at full reserved capacity |
| 8 | `kOverflowClaims` | distinct-state claims list full | caller: rescue; `HashSanitizeKernel` wipes the lane's table at batch end (a claims overflow leaves CAS-claimed slots the targeted clear can't see) |
| 16 | `kOverflowKept` | surviving frontier > `main_q` | caller: rescue |

Any non-zero overflow means the lane's result may be degraded; the standard
pattern (see `benchmarks/bench_wfst.py`) is to re-decode flagged lanes on a
small rescue instance with larger factors. At the wrapper defaults this is
rare (6 / 2000 LJSpeech utterances, unchanged from the pre-optimization
decoder).

## 8. Streaming Lifecycle

```
dec = wfst_create_decoder(..., streaming=1)     # max_frames = max CHUNK length
ch  = wfst_create_stream(dec)                   # -1 when all lanes busy
wfst_advance_chunk(dec, [ch], lp[C,Tc,V], lens, want_partial, ...)   # repeat
wfst_finalize_stream(dec, ch, ...)              # k2 final-frame step + backtrack
wfst_release_stream(dec, ch)                    # channel reusable afterwards
```

A channel occupies one lane for its lifetime. `AdvanceChunk` batches any
subset of channels per call: each channel's rows stage into its lane slot,
`ChunkBeginKernel` opens the per-lane window, and one captured chunk graph
(step count rounded up to a multiple of 8 — **even**, preserving the global
frontier-buffer parity across chunks) advances every requested lane; lanes
outside their window idle-carry their frontier. Winners live in per-lane
regions addressed by LOGICAL monotonic ids (`WinnersEntry` maps them onto the
lane's fixed ring); partial hypotheses (`want_partial`) backtrack from the
best current-frontier token. `FinalizeStream` arms `T = t` and runs two steps
(the final step + one idle step, keeping parity), then backtracks. Lattice
mode is not supported with streaming.

### 8.1 Per-chunk GC: unbounded streams (`gc_interval > 0`)

With `gc_interval = 0` a channel's region is a hard cap on stream length
(`kOverflowArena` once `log_len` hits `stream_log_cap`) **and** long-stream
backtracks silently truncate to the newest `path_cap = max_frames + 2` arcs —
a T=251 utterance streamed in 128-frame chunks returns only the suffix of its
words. `gc_interval > 0` (any even value; the streaming Python wrapper always
enables it) fixes both: after every `AdvanceChunk`, a lane-local GC round
(`StreamGc*` kernels, outside the captured chunk graph) finds each live
channel's convergence point, cuts the chain there, and drains the finalized
golden-prefix arcs to a host-side per-channel list. The region becomes a
**ring** over logical ids — `gc_root` (the finalized sentinel) is the
writer's wrap guard, so `stream_log_cap` / `stream_log_entries` turns into a
live-**window** size (32 MiB default granularity) instead of a stream-length
cap. Partial and final results are the host prefix + the device tail (the
tail walk stops at the sentinel; `path_cap` grows to `fin_cap ≥ 4096` so a
lagging tail fits). The per-round drain is bounded by `fin_cap` arcs via a
phase-ring emit (a backlog from a convergence stall catches up at `fin_cap`
per round); a window overrun (convergence never found) keeps the flagged
`kOverflowArena` degrade-not-corrupt semantics.

Measured: an hour-long synthetic stream (90k frames, LJS tiled) decodes with
constant 42 MiB committed and results bit-identical between a minimal ring
(4Mi entries, wrapped ~3×) and a 32Mi-entry ring — and identical scores to
the gc-off path, whose word list is exactly the truncated suffix.

## 9. Orchestration and CUDA-Graph Replay

- All launch configurations are **capacity-sized** and all loop state is
  device-resident, so a whole decode captures into one `cudaGraphExec_t`.
  Offline graphs are cached per T-bucket (multiples of 32 up to 256 steps,
  else 64); streaming chunk graphs per 8-multiple; interval-mode segment
  graphs per segment length. Idle-step waste at bucket edges is bounded by
  the early-exit checks.
- `use_cuda_graphs=0` (or `debug_snapshots`) falls back to plain per-step
  launches — same kernels, same numerics; this is what you profile under
  `ncu`.
- Frontier buffers swap by pointer per step *in the captured sequence*; every
  code path therefore preserves step-count parity (see the even-step rules
  above).

## 10. Public C API (TVM-FFI)

`oasr.jit.wfst_decoder.gen_wfst_decoder_module().build_and_load()` exports:

| function | purpose |
|---|---|
| `wfst_load_graph(path) -> h` / `wfst_free_graph(h)` | load/free an `hlg.img` |
| `wfst_graph_info(h, out_i64[5])` | `{states, arcs, vocab, start, finals_at_end}` |
| `wfst_create_decoder(graph_h, search_beam, output_beam, min_active, max_active, allow_partial, max_lanes, max_frames, device, main_q_factor, cand_factor, use_cuda_graphs, lattice, fp16_logprobs, streaming, lat_prune_interval, eps_iterations) -> h` | build a decoder instance |
| `wfst_free_decoder(h)` | destroy (frees workspace, unmaps regions) |
| `wfst_decoder_mem_stats(h, out_i64[4])` | `{reserved, committed, fixed, arena_high_water}` bytes/entries |
| `wfst_decode_batch(h, lp[B,T,V] cuda, lengths cpu, out_words[B,cap], out_word_lens, out_scores f64, out_meta[B,3])` | offline batched decode; `out_meta = {ok, reached_final, overflow}` |
| `wfst_create_stream / wfst_advance_chunk / wfst_finalize_stream / wfst_release_stream` | streaming lifecycle (§8) |

The exact-semantics CPU oracle (`wfst_cpu_decode(graph_path, lp[T,V] cpu, ...)`) is
**not** exported by this module. It lives in a separate, test-only JIT module
(`oasr.jit.wfst_decoder.gen_wfst_cpu_reference_module`, sources under `csrc/tests/wfst/`)
so the production decoder library carries no reference-decoder code; it takes a graph-image
*path* (loads the image itself) rather than a decoder handle.

A decoder handle co-owns its graph image (word lookups during backtrack), so
freeing the graph handle first is safe. Instances are not thread-safe; one
instance per stream/GPU.

## 11. Python API and Engine Integration

`oasr/decoder/wfst_decoder.py` provides the duck-typed searcher
(`WfstDecoderSearch` + `WfstDecoderOptions`) selected by
`DecoderConfig.wfst_backend = "gpu"`. **WFST decoding is GPU-only** — CUDA is a
hard requirement and the CPU `"k2"` value is an unsupported legacy path, built
only under `OASR_USE_K2=1` and not exercised by CI.  Graph handles and decoder
instances are shared process-wide,
keyed on `(fst, beams, actives, device, …)`; passing an `HLG.pt` exports and
caches the `.img` next to it on first use.

- **Offline**: `decode_offline(logp)` / `decode_offline_batch(enc_out,
  lengths)` — rows are length-clipped and blank-skipped exactly as the
  single-utterance path, then decoded in `≤ max_offline_lanes` padded
  sub-batches (defaults: 8 lanes, `main_q_factor=32`, `cand_factor=3`,
  `max_frames=4096`). The engine's `CtcWfstDecodeStrategy`
  (`oasr/engine/decode/ctc_wfst.py`) calls the batched path and raises
  `wfst_max_offline_lanes` to `max_batch_size`.
- **Streaming**: the searcher protocol (`reset`/`search`/`finalize_search`)
  borrows a channel of one shared 32-channel streaming decoder
  (`main_q_factor=16`, `cand_factor=4`, 128-frame chunks) and releases it on
  finalize or GC.
- **fp16**: `fp16_logprobs=1` consumes half-precision log-probs (f32
  accumulation inside; word output identical to fp32 on the benchmark sets).

## 12. Performance

The decoder scales with batch: per-utterance cost grows far slower than `B`,
because the flattened slot space keeps lanes load-balanced. Kernel time is
dominated by Expand, then Max, then Finalize and Scan.

Two structural properties drive the memory profile:

- **Lazily-committed regions** (§6.2) mean an idle streaming channel holds
  essentially nothing, and the workspace grows on overflow rather than being
  provisioned for the worst case.
- **Winners-log GC** (§6.3, §8.1) bounds long-form and unbounded-stream memory,
  which would otherwise grow with utterance length.

Measured throughput, VRAM figures and the comparison against k2:
`.artifacts/decoder_perf.md` Part 2.

Reproduce:

```bash
# throughput (add --compare-external for an A/B vs the standalone build)
CUDA_VISIBLE_DEVICES=<gpu> python benchmarks/bench_wfst.py --batch 1 8 32

# bit-exact output parity vs the external _wfst.so
python scripts/wfst_parity_check.py --num-utts 192 --batch 1 8 32

# kernel timeline (graph nodes need node-level tracing)
nsys profile --cuda-graph-trace=node -t cuda python benchmarks/bench_wfst.py ...
# per-kernel metrics: build the decoder with use_cuda_graphs=0, then ncu
```

## 13. Testing

- `tests/test_wfst_decoder.py` — self-contained: toy-FST GPU vs the CPU
  reference oracle (exact costs + backpointer chains, via debug snapshots),
  batched-offline equivalence, wrapper offline/streaming smoke, channel
  lifecycle.
- `scripts/wfst_parity_check.py` — the migration/optimization gate: identical
  GPU batches through the in-tree JIT decoder and the standalone project's
  `_wfst.so`; asserts every lane's words, score and overflow match. Any
  kernel change must keep this at 0 mismatches (score ties excepted, §3).
- `csrc/tests/wfst/cpu_reference.cc` — the executable k2-semantics spec
  (offline + online beam modes), reachable from Python via `wfst_cpu_decode`
  in the test-only `gen_wfst_cpu_reference_module` (kept out of the production
  decoder library).

## 14. Constraints

These are properties of the design, not a to-do list:

- **Vocab < 65,535** — u16 ilabel encoding; checked at graph upload.
- **Streaming logical ids hit the int32 wall** after ~2³¹ appends per channel
  (~35 h of continuous audio). Recycle channels at utterance boundaries.
- **Exact-score ties are run-nondeterministic** — the same class of
  nondeterminism as k2. `scripts/wfst_parity_check.py` excludes score ties for
  this reason.
- **Overflow rescue is the caller's job.** The decoder reports overflow through
  the `ok` flag and the per-lane bits; the engine wrapper surfaces them but does
  not auto-rescue.

Open follow-up levers, with the measurement behind each:
`.artifacts/decoder_perf.md` §2.3.
