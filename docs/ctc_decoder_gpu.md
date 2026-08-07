# GPU CTC Prefix Beam Search Decoder

The CTC prefix beam search decoder (`include/oasr/ctc_decoder.cuh` +
`csrc/decoder/ctc/ctc_decoder.cu`) is OASR's GPU-resident CTC decoder. It performs
batched offline decoding and per-frame streaming decoding entirely on the
device, with no host loop or per-frame H2D/D2H synchronisation in the
hot path.

The kernel pipeline is derived from torchaudio's
`ctc_prefix_decoder_kernel_v2.cu` (BSD 3-Clause, NVIDIA Corp.) but
extended with:

- A **single-launcher streaming step** (`streaming_step`) that doesn't
  read back state-buffer headers from the GPU.
- A **chunk launcher** (`ctc_beam_search_chunk`) that iterates over a
  whole chunk of frames in one C++ call, eliminating per-frame Python
  overhead.
- A **paged-memory mode** that stores variable-length token sequences
  in fixed-size pages with reference-counted prefix sharing across
  beams — substantially less memory at large `beam × max_seq_len`.

## 1. Purpose and Responsibilities

The decoder converts CTC-trained log-probabilities `log_prob[B, T, V]`
into the N-best token sequences per utterance, scored by their CTC
posterior. It is responsible for:

1. **Beam state management**: per-beam (blank-end / non-blank-end)
   probabilities, current sequence length, last token, accumulated
   score.
2. **Probability matrix updates**: for every step, build the
   `(beam × vocab)` extension table.
3. **Prefix merging**: collapse beams that produce identical sequences
   after extension.
4. **Top-K selection**: pick the surviving `beam` extensions across
   `beam × vocab` candidates per batch.
5. **Sequence storage**: keep the actual decoded token list for each
   beam (flat double-buffered or paged with prefix sharing).
6. **Frame skipping** (offline only): pre-filter frames whose blank
   probability exceeds a configurable threshold.

The decoder is exposed through three TVM-FFI launchers
(offline, streaming-init, streaming-step / chunk / read-state) which
the Python `oasr.ctc_decode` module wraps.

## 2. High-Level Architecture

```
                 log_prob[B,T,V] (float32, CUDA)
                            │
                            ▼
           ┌─────────────────────────────────┐
           │  init_select_kernel             │  (offline only)
           │  filter blank-dominant frames   │
           └────────────────┬────────────────┘
                            ▼
                ┌───────────────────────┐
                │   first_step_kernel   │  step 0
                │ block-wide radix-sort │
                │ top-K of vocab        │
                └───────────┬───────────┘
                            │
                            ▼
        ┌───────────── per-step loop ─────────────┐
        │  prob_matrix_kernel       (non-blank)   │
        │  prob_space_blank_kernel  (blank/space) │
        │  merge_kernel             (dedupe)      │
        │  topk_phase1_kernel       (per-batch    │
        │                            block-Rsort) │
        │  topk_phase2_kernel       (reduce +     │
        │                            beam update) │
        └─────────────────────────────────────────┘
                            │
                            ▼
              fixup_parity + cudaMemcpy2DAsync
              (or gather_paged_results_kernel)
                            │
                            ▼
            out_tokens, out_lengths, out_scores
```

The same pipeline is used in flat and paged modes; the paged mode
swaps `merge_kernel` / `topk_phase2_kernel` /
`first_step_kernel` for their `_paged` variants and an extra
`gather_paged_results_kernel` to materialise the final flat output.

## 3. CTC Prefix Beam Search Algorithm

This section walks through the algorithm in the same order the kernels
execute it. The notation matches the source: arrays in `InternalData`
keep their names (`pprev`, `ptable`, `ptablen`, `clast`, `clen`,
`clist`, `score`); operations on log-probabilities use `logsumexp`
defined as

```
logsumexp(a, b) := max(a, b) + log(1 + exp(-|a - b|))
```

with `NEG_INF = -FLT_MAX` as the additive identity (encoded as bit
pattern `0xcc` for `cudaMemset`).

### 3.1 Beam state

A beam represents a *CTC prefix* — a sequence of distinct emission
labels with no consecutive duplicates and no blanks. CTC's collapse
rule means many alignments map to the same prefix, so each beam keeps
**two** posterior probabilities depending on whether the most recent
acoustic frame emitted a blank or a non-blank:

```
pprev[bid, k] = float2(blank_score, nonblank_score)

clist[bid, k, 0..L-1] = decoded label sequence (length L = clen[bid, k])
clast[bid, k]         = clist[bid, k, L - 1]      (or blank_id when L = 0)
score[bid, k]         = logsumexp(blank_score, nonblank_score)
```

`clen` and `clist` are **double-buffered**: two physical copies indexed
by `parity = step % 2`. Each step reads parity `(step - 1) % 2` and
writes `step % 2`, eliminating the need for an in-place barrier
between source and destination.

### 3.2 The two extension paths

When a new frame `t` arrives with vocabulary log-probabilities
`log_prob[bid, t, :]`, every beam can extend via two operations:

1. **Emit `c`** for any non-blank label `c`. Two sub-cases govern how
   the previous probabilities flow into the extension:

   - **`c ≠ clast[beam]`** (different label): both blank-ending and
     non-blank-ending paths can extend, because either way the
     emission is a fresh, non-repeated label. Contribution is
     `cur_prob + logsumexp(pprev.blank, pprev.nonblank)`.

   - **`c == clast[beam]`** (same label as last). CTC requires a
     blank between two equal labels, otherwise they collapse to one.
     Therefore only the blank-ending path can extend with `c`:
     contribution is `cur_prob + pprev.blank`. The same-label path
     *without* a separating blank is the "stay" path that stays in
     the current beam — its contribution `cur_prob + pprev.nonblank`
     is folded into **this** beam's blank slot (see 3.4).

2. **Emit blank or stay on the same label** (the "no new label"
   path). The prefix is unchanged; only `pprev` updates.

These rules drive the per-step build of `ptable` (blank-ending
extensions) and `ptablen` (non-blank-ending extensions) — see 3.4.

### 3.3 Step 0 — initialisation (`first_step_kernel`)

Step 0 has no prior beam state, so it short-circuits the merge / top-K
path and seeds the beams directly from the vocabulary at the first
selected frame.

Configuration: `<<<batch, 128>>>` with `BLOCK_SIZE=128`,
`ITEMS_PER_THREAD=4` → 512 vocabulary items per radix-sort iteration.

```
1. first_t = select_seqs[bid, 0]                # first surviving frame
   need_add_blank = (first_t > 0)               # leading blanks were skipped
   nb_beams = beam - 1   if beam > 1 else beam  # reserve one slot for empty prefix

2. # Block-wide streaming top-(nb_beams) over the vocabulary.
   keys[..]   = log_prob[bid, first_t, c]     for c != blank_id else NEG_INF
   values[..] = c                              for c != blank_id else -1
   while items remain:
       BlockRadixSort.SortDescendingBlockedToStriped(keys, values)
       overwrite striped positions [nb_beams ..) with the next batch of items
   # Final top-nb_beams sit at striped positions [0, nb_beams).

3. for k in [0, nb_beams):
       token = top_values[k]
       key   = top_keys[k]
       if need_add_blank:                       # leading blanks → blank slot
           pprev[bid, k] = (key, NEG_INF)
       else:                                    # direct emission → nonblank slot
           pprev[bid, k] = (NEG_INF, key)
       clist[bid, k, 0] = token
       clen[bid, k]     = 1
       clast[bid, k]    = token
       score[bid, k]    = key

4. if beam > 1:                                 # explicit empty-prefix beam
       blank_prob = log_prob[bid, first_t, blank_id]
       pprev[bid, beam - 1] = (blank_prob, NEG_INF)
       clast[bid, beam - 1] = blank_id
       clen[bid,  beam - 1] = 0
       score[bid, beam - 1] = blank_prob
```

The empty-prefix slot prevents spurious initial tokens during
streaming when early frames have moderate (but sub-threshold) blank
probability and no confident non-blank.

### 3.4 Per-step main loop

> **Fused fast path (production default).** For `beam ≤ 32`
> (`FUSED_MAX_BEAM`) the five kernels described in §3.4.1–3.4.5 are
> replaced by a **single fused kernel per step**
> (`fused::fused_step_kernel`, one block per batch row) — see §3.4.6.
> The multi-kernel pipeline below remains the reference semantics, the
> `beam > 32` fallback, and the A/B baseline
> (`OASR_CTC_FUSED=0` selects it explicitly).

For step `s ≥ 1`, the legacy host launcher
`ctc_prefix_beam_search_step` (or `_paged`) runs five kernels in
order. Let `t = select_seqs[bid, s]` denote the current acoustic
frame; let `t_prev = select_seqs[bid, s - 1]`. When intervening
blank-dominant frames were skipped (`t > t_prev + 1`), we apply a
**collapsed-blank correction** to the previous probabilities:

```
if need_add_blank:
    prev_blank    = logsumexp(pprev.blank, pprev.nonblank)
    prev_nonblank = NEG_INF
else:
    prev_blank    = pprev.blank
    prev_nonblank = pprev.nonblank
```

The collapse reflects that one or more blank frames between
selections leave only the blank-ending path alive.

#### 3.4.1 Step 1 — `prob_matrix_kernel` (non-blank extensions)

Grid `<<<(bx, batch), 256>>>`. Each block iterates over a stride of
`(beam, c)` pairs from the flat space `[beam × ldc)`:

```
for (beam_idx, c) assigned to this thread:
    if c == blank_id:                continue  # handled by next kernel
    if space_id >= 0 and c == space_id: continue
    last = clast[bid, beam_idx]
    prev = (prev_blank, prev_nonblank)
    cur  = log_prob[bid, t, c]

    # Output slot index in the flat per-batch matrix:
    idout = c + (beam_idx + bid * beam) * ldc

    if last == c:
        # Same-as-last: only blank-ending path may extend (see 3.2).
        ptablen[idout] = cur + prev.blank

        # The same-label "stay" contribution from the non-blank path
        # folds into THIS beam's blank slot — it represents an
        # alignment that keeps the prefix unchanged but reaches it
        # via a non-blank-ending route.
        blank_slot = blank_id + (bid * beam + beam_idx) * ldc
        ptablen[blank_slot] = cur + prev.nonblank
    else:
        # Different label: extend from either previous path.
        ptablen[idout] = cur + logsumexp(prev.blank, prev.nonblank)

    ptable[idout]  = NEG_INF        # non-blank label → no blank-ending prob
```

After this kernel, every entry of `ptable` is `NEG_INF` and every
entry of `ptablen` for non-blank labels holds a candidate extension
score; the `blank_slot` column of `ptablen` is partially populated
with the same-label-stay contribution from beams whose `clast == c`
matched.

#### 3.4.2 Step 2 — `prob_space_blank_kernel` (blank/space)

Grid `<<<batch, ldbeam>>>`. One thread per beam writes the blank slot
(and, optionally, a "space" slot if the model treats space as a
distinct word boundary):

```
beam_idx = threadIdx.x
last     = clast[bid, beam_idx]
prev     = (prev_blank, prev_nonblank)            # with collapsed-blank fix
blank    = log_prob[bid, t, blank_id]
slot_b   = blank_id + (bid * beam + beam_idx) * ldc

ptable[slot_b] = blank + logsumexp(prev.blank, prev.nonblank)

if need_add_blank or last == blank_id:
    ptablen[slot_b] = NEG_INF      # collapse leaves no nonblank-ending residue

# else: ptablen[slot_b] retains the same-label-stay value written by 3.4.1.

if space_id >= 0 and space_id != blank_id:
    space  = log_prob[bid, t, space_id]
    slot_s = space_id + (bid * beam + beam_idx) * ldc
    ptablen[slot_s] = space + logsumexp(prev.blank, prev.nonblank)
    ptable [slot_s] = NEG_INF
```

After steps 1 + 2, every `(beam_idx, c)` entry of `(ptable, ptablen)`
holds the log-probability of that specific extension path. Total per
beam: `ldc` candidates; total per batch: `beam × ldc`.

#### 3.4.3 Step 3 — `merge_kernel` (prefix deduplication)

Grid `<<<(beam, batch), ldbeam>>>`. Every block holds the source-side
`clen[bid, :]` in shared memory, then each thread processes one
`(shorter, longer)` pair:

```
shorter_beam = blockIdx.x         # i
longer_beam  = threadIdx.x        # j

if (clen[longer] - 1) == clen[shorter] and
   clist[bid, longer, 0..clen[shorter]-1] == clist[bid, shorter, *]:

    # j is exactly one token longer than i, sharing i's prefix.
    # Therefore i extended by clast[j] is the SAME prefix as j.
    # Fold i's "extend by clast[j]" candidate into j's blank slot:

    src_slot = clast[longer] + (shorter + bid * beam) * ldc   # i extends by clast[j]
    dst_slot = blank_id      + (longer  + bid * beam) * ldc   # j's blank slot

    ptable [dst_slot] = logsumexp(ptable [dst_slot], ptable [src_slot])
    ptablen[dst_slot] = logsumexp(ptablen[dst_slot], ptablen[src_slot])
    ptable [src_slot] = NEG_INF
    ptablen[src_slot] = NEG_INF
```

Folding into `j`'s **blank slot** is the trick that keeps the
post-merge candidate space unchanged — `j`'s blank slot would
otherwise have stayed unique, so we co-opt it as the canonical home
for the merged prefix and zero out the now-redundant `i + clast[j]`
candidate.

In paged mode (`merge_paged_kernel`) the only difference is that
`seq_compare` is replaced by `paged_seq_compare`, which walks the
two beams' `block_table` rows. When two logical pages map to the
same physical page (because of prior CoW sharing) the per-token loop
is skipped — a fast path that grows in importance as beams diverge
late in long utterances.

#### 3.4.4 Step 4 — `topk_phase1_kernel` (per-batch partial top-K)

Grid `<<<(bxs, batch), 128>>>`. The flat candidate space `beam × ldc`
is partitioned into `bxs ≤ MAX_BLOCKS_PER_BATCH` contiguous chunks;
each block computes the **local** top-`beam` of its chunk using a
*streaming* `cub::BlockRadixSort`:

```
chunk = items in [bx * chunk_size, min((bx + 1) * chunk_size, beam*ldc))
keys[..]   = logsumexp(ptable[idx], ptablen[idx])
values[..] = idx                                    # flat index into beam*ldc
SortDescendingBlockedToStriped(keys, values)        # first iteration

while more items remain:
    overwrite striped positions [beam, ...) with next items_per_iter - beam values
    SortDescendingBlockedToStriped(keys, values)    # re-sort

# After the loop, striped positions [0, beam) hold this chunk's top-beam.
topk_key_buffer  [(bid * bxs + bx) * beam + k] = keys[k]
topk_value_buffer[(bid * bxs + bx) * beam + k] = values[k]
```

The streaming sort is more efficient than a full descending sort
because each iteration sorts the **entire** SM-resident array, but
only the bottom `items_per_iter - beam` positions accept new
candidates; the existing top-`beam` survives unchanged. With
`BLOCK_SIZE = 128, ITEMS_PER_THREAD = 4`, each iteration consumes
512 candidates while preserving the running top-`beam`. The kernel
keeps key sentinels at `NEG_INF` outside `chunk_end` so out-of-range
positions never displace real candidates.

#### 3.4.5 Step 5 — `topk_phase2_kernel` (reduce + state update)

Grid `<<<batch, 128>>>`. One block per batch reduces the
`bxs × beam` partial winners to the global top-`beam`, then writes
the new beam state.

```
items_per_batch = bxs * beam
keys/values     = topk_key_buffer / topk_value_buffer for this batch
                  → streaming SortDescendingBlockedToStriped (same pattern as phase 1)

# top-beam now in shared memory smem.topk.{keys, vals}.
# Cache src state to prevent write-before-read races (src_beam may also be a dst_beam).
smem.topk.src_clast[k] = clast[bid, k]
smem.topk.src_clen[k]  = clen_src[bid, k]

# Sub-warp parallel write: each output beam owned by WRITE_THREADS=8 threads.
sub_warp_id = tx / 8
tid_in_sub  = tx % 8

for out_beam in range(beam) striding by sub_warps:
    flat_id  = smem.topk.vals[out_beam]
    src_beam = flat_id / ldc
    char_id  = flat_id - src_beam * ldc
    new_score = smem.topk.keys[out_beam]
    prevlen   = smem.topk.src_clen[src_beam]

    # 1) Copy parent prefix in parallel.
    for s in range(tid_in_sub, prevlen, WRITE_THREADS):
        clist_dst[bid, out_beam, s] = clist_src[bid, src_beam, s]

    if tid_in_sub == 0:
        # 2) Append (or not) the new label.
        if char_id == blank_id:
            clast   [bid, out_beam] = src_clast[src_beam]   # unchanged prefix
            clen_dst[bid, out_beam] = prevlen
        else:
            clast   [bid, out_beam] = char_id               # extended prefix
            clen_dst[bid, out_beam] = prevlen + 1
            if prevlen < ldseq_len:
                clist_dst[bid, out_beam, prevlen] = char_id

        score[bid, out_beam] = new_score

        # 3) New (blank, nonblank) split for the next step.
        p  = ptable [bid, flat_id]
        pn = ptablen[bid, flat_id]
        if need_add_blank:                # collapsed-blank correction
            pprev[bid, out_beam] = (new_score, NEG_INF)
        else:
            pprev[bid, out_beam] = (p, pn)
```

Caching `src_clast` / `src_clen` in shared memory matters because
top-K phase 2 can pick the same `src_beam` for **multiple** output
beams (a beam may "win" with several different extensions); without
the snapshot, an `out_beam` writing to its own `clast[bid, dst_base]`
slot might overwrite values another `out_beam` is still reading.

##### Paged variant (`topk_phase2_paged_kernel`)

The paged kernel keeps the same top-K logic but replaces the parent-
prefix copy with two passes separated by `__syncthreads()`:

**Pass 1 — Fork**: each output beam copies its source's
`block_table` row entry by entry, releasing whatever pages
`block_table_dst[out_beam]` previously referenced and acquiring
references to the source pages (`atomicAdd(ref_counts, 1)`):

```
for p in range(num_pages_used):
    old_phys = block_table_dst[bk_dst * max_lp + p]
    if old_phys != INVALID_PAGE:
        free_page(old_phys)             # decrement; push to free_pool if rc -> 0
    phys = block_table_src[bk_src * max_lp + p]
    block_table_dst[bk_dst * max_lp + p] = phys
    atomicAdd(&ref_counts[phys], 1)     # acquire
```

**Pass 2 — Append / CoW** (one thread per output beam): write the
new token into the last logical page. Three cases:

- **Blank extension** — no token written; just propagate `clast`.
- **Non-blank, fresh page** (`prevlen % page_size == 0`):
  `alloc_page` (LIFO from `free_pool` if not empty, else
  `next_free_page++`), set `ref_counts[new_phys] = 1`, write the
  token at offset 0.
- **Non-blank, existing page**: if `ref_counts[phys] > 1` the page is
  shared with a sibling beam — copy the page contents to a fresh
  `new_phys`, redirect `block_table_dst`, decrement the old
  reference, set `ref_counts[new_phys] = 1`, then write. If
  `ref_counts == 1`, it's already exclusive — write in place.

The `__syncthreads()` between Pass 1 and Pass 2 ensures every
`free_page` push is visible before any `alloc_page` pop, otherwise an
allocator could miss a page that just became free.

#### 3.4.6 Fused single-kernel step (`fused::fused_step_kernel`, beam ≤ 32)

Profiling on RTX 5090 showed `topk_phase1_kernel` alone at ~75 % of
decoder GPU time: every frame materialises the full
`[beam × vocab]` `ptable`/`ptablen` score matrix to global memory and
extracts the top `beam` with a streaming block radix sort over all
`beam × vocab` items. But the score matrix is **separable**: for an
"ordinary" char `c` of beam `k`

```
score(k, c) = lp[c] + A_k,     A_k = logsumexp(prev_blank, prev_nonblank)
```

so the per-beam ranking of ordinary chars *is* the ranking of `lp`.
The only per-beam exceptions are the blank slot, the repeated
last-char slot, the optional space slot, and slots zeroed by the
duplicate-prefix merge. The global top-`beam` over all `(k, c)` pairs
is therefore contained in

```
{ top-K_all chars of lp } × beams  ∪  per-beam special slots
K_all = beam + 3 + max_patches      (≤ 2·beam + 2)
```

`fused_step_kernel` (one 256-thread block per batch row, flat and
paged variants via a `PAGED` template flag) does the whole step:

1. **Snapshot** per-beam state (`pprev`, `clast`, `clen`) into shared
   memory, applying the collapsed-blank correction.
2. **Merge map**: the §3.4.3 pair condition over `beam²` pairs,
   recorded as per-row patch lists (chars zeroed) and per-row merge
   sources (for the blank-slot fold).
3. **Vocab top-K_all** via a block-wide byte-radix select
   (`block_topk_select`): per-warp 256-bin histograms, a warp-shuffle
   suffix-scan to find the threshold bin, refining one byte at a time.
   Keys are `(sortable_f32(lp), ~char)` composites, so ties resolve by
   ascending char id and termination is guaranteed. Sortable keys are
   cached in shared memory (`VKEY_CACHE`) when the vocab fits.
4. **Score candidates** (`≈ beam × (K_all + 3)`) with the exact
   §3.4.1/3.4.2 formulas; merge folds accumulate in ascending
   source-beam order (the legacy cross-block fold was racy).
5. **Select + rank** the global top-`beam` with the same radix select
   over `(score, ~id)` keys plus a warp-shuffle bitonic sort.
6. **State update** identical to §3.4.5 (sub-warp `clist` copy, or the
   paged fork/CoW passes). `pprev` for the winners is recomputed
   analytically — bit-identical to the legacy table read since
   `ptable == NEG_INF` for every non-blank slot.

`ptable`/`ptablen` and the phase-1 top-k buffers are **not allocated**
when the fused path is active (`layout_has_prob_tables`), shrinking a
`B=1, beam=10, V=5000` streaming state from ~440 KB to ~40 KB.

Determinism notes (differences from the legacy pipeline, validated by
`tests/test_ctc_decoder_fused_parity.py`):

- Exact score ties may rank in a different beam order (legacy radix
  stability vs `(score desc, id asc)` composites). Scores are
  bit-identical; only the order of equal-scored beams can differ.
- The merge fold into a beam's blank slot is ordered (ascending source
  beam); the legacy fold was a racy cross-block read-modify-write.
- The paged allocator is **partitioned per batch row** (row `b` owns
  pages `[b·ppr, (b+1)·ppr)`), pass-2 CoW pool pushes are deferred
  past the last pop, and self-forks skip the free/re-acquire cycle.
  The legacy shared pool could pop a `free_pool` slot before the
  concurrent push wrote it and could push still-referenced pages,
  which made large paged offline decodes nondeterministic (corrupted
  low-rank beams).

The offline launchers also pass the step index by value
(`step_const`) so no per-step device-counter update is needed; the
streaming path passes `-1` and reads the device-resident counter so
captured CUDA graphs stay valid across frames (§8).

#### 3.4.7 Chunk-level vocab top-K pre-pass (`fused::fused_topk_prepass_kernel`)

After §3.4.6 the fused step is **latency-bound** (<2 % SM utilisation):
one block per batch row, serialised frame-to-frame by the beam-state
dependency. Its dominant critical-path component was step 3 above —
the multi-pass byte-radix select over the whole vocab. But that
ranking depends only on the frame's `lp`, not on beam state, and
`K_all = beam + 3 + max_patches ≤ 2·beam + 2` for *any* reachable
state. So the ranking is hoisted out of the sequential loop:

- `fused_topk_prepass_kernel` runs once per tile of `PREPASS_TILE = 128`
  frames with grid `(tile_len, batch)` — full GPU parallelism — and
  emits each frame's top-`(2·beam + 2)` chars + log-probs into the
  `pre_chars` / `pre_lp` / `pre_cnt` workspace buffers (layout
  `[row][batch][C_BUF]`). Bit `r` of a by-value 128-bit mask
  (`mask_lo`/`mask_hi`) gates row `r`, so blank-skip frames are not
  ranked at all.
- The step body (`fused_step_body`) in pre-pass mode replaces its
  Phase-3 vocab scan with a load of ≤ `C_BUF` precomputed candidates.
  The precomputed list is a *superset* of the in-kernel top-`k_all`
  selection, and every scored candidate is an exact slot of the
  conceptual score matrix, so the global top-beam — and hence the
  decode — is unchanged (validated by the §3.4.6 parity suite plus
  exact old-vs-new output comparisons across tile boundaries and
  masks).

Offline indexes frames through `select_seqs`; streaming ranks chunk
positions directly. The single-frame `ctc_beam_search_step` API keeps
the in-kernel select (`fused_step_kernel` wrapper, `pre_cnt_row = -1`).

#### 3.4.8 Multi-frame chunk kernel (`fused::fused_chunk_kernel`)

With the vocab scan hoisted, the remaining per-frame cost was the
launch chain itself: a counter kernel + a `d_lp_frame_buf` D2D copy +
a CUDA-graph replay per decoded frame (~12 µs/frame at V=5000).
`fused_chunk_kernel` removes the chain: one launch decodes a whole
pre-pass tile, each block owning one batch row and looping the tile's
frames in-kernel with the beam-state dependency carried across
iterations (a block's global-state writes are visible to its own reads
after `__syncthreads()`).

- Frame gating, counters and parity are all in-kernel: the blank-skip
  mask travels as the same two by-value 64-bit words as the pre-pass
  (no H2D copy), `step`/`frame_idx` advance from by-value starting
  values, the `select_seqs` ring write (formerly
  `set_select_seq_step_kernel`) is inlined, and the
  clen/clist/block_table double-buffer parity is a pointer swap per
  iteration.
- `step == 0` runs the first-step initialisation inline from the same
  pre-pass candidates (a superset of the top-`(nb_beams + 1)` chars the
  dedicated first-step kernels select, so the greedy non-blank pick is
  identical).
- Per tile the whole decode is **two launches** (pre-pass + chunk
  kernel) per stream. The offline loops use the same kernel in
  `select_seqs` mode: `ceil(max_select / 128) × 2` launches total.
  CUDA graphs are no longer used on the fused path (`use_cuda_graphs`
  is ignored there); the per-frame graph machinery remains only for the
  legacy / `beam > 32` / `OASR_CTC_FUSED=0` chunk path.

RTX 5090, fused path across the optimisation series (§3.4.7 → §3.4.10):
offline `(1, 200, 5000, 10)` 2.50 → 1.60 → 1.42 → 1.11 ms, `(16, 200,
5000, 10)` 3.93 → 3.19 → 2.87 → 2.55 ms; streaming `(V=5000, beam=10)`
16.9 → 12.4 → 8.2 → 6.6 µs/frame (2.6× total). Fused-vs-legacy speedup
at vocab 5000 rises from ~2.3–3× to ~3.5–6.8×.

#### 3.4.9 Multi-stream batched chunk decode (`fused::fused_chunk_batched_kernel`)

`ctc_beam_search_chunk_batched` (the engine's `decode_chunk_batch` hot
path) used to launch the §3.4.8 pair once per stream on one CUDA
stream — N independent streams ran as a serial chain N deep. The
batched kernels fold a **group** of up to `FusedStreamGroup::CAP = 64`
streams into a single launch pair per tile: the batched pre-pass runs
grid `(tile_len, group × batch)` and the batched chunk kernel grid
`(group × batch)`, each block decoding its own stream's frames
concurrently.

All streams share one config (engine invariant) and therefore one
workspace layout, so per-stream state is reached as `first_stream_ptr +
delta[slot]` — the group's byte deltas, per-stream `step`/`frame_idx`
counters and blank-skip bitmaps travel **by value** in one ~2 KB
kernel-parameter struct (no device pointer table, no H2D upload).
Streams are fully independent (own buffers, own per-row paged-allocator
partitions), so concurrent blocks cannot interact and results are
bit-identical to the sequential path. Deltas must be 128-byte-aligned
for the shared-offset trick (always true for torch-allocated states);
misaligned groups fall back to per-stream launches.

RTX 5090, `decode_chunk_batch` (V=5000, beam=10, 16-frame chunks,
threshold 0.95, paged): N=8 streams 0.86 → 0.10 ms/engine-step
(8.6×), N=64 6.7 → 0.14 ms (47×), N=128 13.5 → 0.26 ms (51×, incl.
§3.4.10) — decode cost per engine step is now nearly flat in the
number of streams.

#### 3.4.10 Intra-step latency (barrier-bound phases)

With launches gone, warp-stall sampling showed the chunk kernel >50 %
**barrier-stalled**: one block per row, so every `__syncthreads()`
costs the slowest warp's segment latency. Per-frame `clock64()` phase
timing (`-DOASR_CTC_PHASE_PROF`, prints per-frame cycles from the
chunk kernel) attributed ~22 % to the Phase-5 byte-radix select, ~39 %
to the Phase-6 state update. Changes, all bit-exact by construction:

- **Phase 5** — per-warp streaming top-32 over a contiguous candidate
  slice using register bitonic networks (`warp_bitonic_sort32` /
  `warp_bitonic_merge32_desc`, no block barriers), then one
  rank-by-count merge of the `warps × beam` survivors. Two barriers
  replace ~4 per radix level; measured 4.3k → 1.6k cycles/frame
  (+1.5k in the merge).
- **Phase 6** — each sub-warp owns at most one output beam
  (`beam ≤ 32`), so the winner's `lp_row[char_id]` load is issued
  before the paged fork pass (latency hidden behind the fork's global
  RMWs), and the copy-on-write page copy runs on all 8 sub-warp lanes
  instead of a lane-0 serial `page_size`-long chain. 7.5k → 5.4k
  cycles/frame.
- **Phases 1–4** — candidates are scored into fixed slots (skipped
  slots write unique low dummy keys that sort below every real key)
  instead of `atomicAdd`-compacted ones; the merge-source list +
  insertion sort became a per-beam bitmask iterated in ascending-bit
  order; the pre-pass candidate load rides in the Phase-1 snapshot
  pass; `k_all` is only computed on the in-kernel-select fallback.

Frame time 19.3k → 15.4k cycles (−20 %); the remaining profile is
~30 % Phase-4 scoring, ~35 % Phase-6 global update, ~20 % select+rank.

### 3.5 Frame selection (offline only)

Before the main loop, the offline launcher runs
`init_select_kernel<128, 4>` with one block per batch. It reads
`log_prob[bid, t, blank_id]` for every `t` and emits a packed list of
indices `t` where `log_prob < log_threshold`:

```
log_threshold = log(blank_threshold)        # blank_threshold defaults to 0.98

selected[ITEM] = 1 if log_prob[bid, t, blank_id] < log_threshold else 0
exclusive_scan = cub::BlockScan(selected, +)
if selected[ITEM]:
    select_seqs[bid, exclusive_scan + block_aggregate] = t
select_seq_lens[bid] = total selected
```

The CUB block-scan provides ordered output without atomics. After
the kernel a single `cudaMemcpyAsync(select_seq_lens → host)` +
`cudaStreamSynchronize` reads `max_select` so the host can bound the
main step loop. This is the **only** D→H sync in the offline path.

In streaming mode the launcher writes the identity mapping
`select_seqs[bid, t] = t` and never filters — frame skipping (if
desired) is the caller's responsibility.

### 3.6 Parity fixup and result extraction

Because each batch element has its own `select_seq_lens[bid]`, two
batches in the same call may run a different number of steps. The
host's loop bound is `max_select`, but `bid_short` may have stopped
at step `max_select - k`, leaving its valid output in
`clen[(max_select - k - 1) % 2]` while the global `final_parity =
(max_select - 1) % 2` points at the other buffer. `fixup_parity_kernel`
copies the "frozen" results forward to the global parity for those
batches.

In paged mode the analogous `fixup_parity_paged_kernel` copies block-
table entries instead of `clist` data, incrementing reference counts
on the carried-forward physical pages.

Finally:

- **Flat**: three `cudaMemcpy2DAsync` calls strip `ldbeam` /
  `ldseq_len` padding and produce `out_lengths[B, beam]`,
  `out_tokens[B, beam, max_out_len]`, `out_scores[B, beam]`.
- **Paged**: `gather_paged_results_kernel` iterates each beam's
  block table and copies `clen[bid, k]` tokens out of `page_storage`
  into the user's flat output.

### 3.7 End-to-end pseudocode

```
# Offline:
init_internal_data(workspace, B, beam, V, max_seq)
memset(clast, 0); memset(clen[0/1], 0); memset(clist[0/1], 0xff)
memset(ptable, 0xcc); memset(ptablen, 0xcc); memset(select_seq_lens, 0)

init_select_kernel<<<B, ...>>>(log_prob, ...)
max_select = host-read max(select_seq_lens)

for step in [0, max_select):
    if step == 0:
        first_step_kernel<<<B, 128>>>(log_prob, ...)
    else:
        prob_matrix_kernel       <<<(bx, B),  256   >>>(...)
        prob_space_blank_kernel  <<<B,        ldbeam>>>(...)
        merge_kernel             <<<(beam, B), ldbeam>>>(...)
        topk_phase1_kernel<128,4><<<(bxs, B), 128   >>>(...)
        topk_phase2_kernel<128,2><<<B,        128   >>>(...)

fixup_parity_kernel<<<B, 32>>>(...)
cudaMemcpy2DAsync × 3   # strip alignment padding
```

Streaming and paged variants follow the same shape; the streaming
launcher reuses the per-step pipeline frame-by-frame
(`ctc_beam_search_chunk` does this loop in C++), while paged
substitutes the merge / phase-2 / first-step kernels and replaces
the final `cudaMemcpy2DAsync` with `gather_paged_results_kernel`.

## 4. Internal Structure

### 4.1 `InternalData` (one per active call)

A bump-allocated workspace that holds every device-side buffer the
pipeline needs. It is not stored in the workspace; the host
reconstructs pointers on each call from the known dimensions
(`setup_internal_data_pointers`), so no D→H read of the header is
required between streaming steps.

| Field | Shape | Purpose |
|-------|-------|---------|
| `pprev` | `[B, ldbeam]` `float2` | `(blank, nonblank)` log-probs per beam. |
| `ptable`, `ptablen` | `[B, beam, ldc]` `float` | Per-step extension probability matrices (blank-ending, non-blank-ending). |
| `clast` | `[B, ldbeam]` `int` | Last token in each beam. |
| `clen[0/1]` | `[B, ldbeam]` `int` | Decoded length per beam, double-buffered. |
| `clist[0/1]` | `[B, beam, ldseq_len]` `int` | Decoded token sequences (flat mode only). |
| `score` | `[B, ldbeam]` `float` | Accumulated beam log-probability. |
| `topk_key_buffer` | `[B, MAX_BLOCKS_PER_BATCH, beam]` `float` | Phase 1 top-K keys. |
| `topk_value_buffer` | `[B, MAX_BLOCKS_PER_BATCH, beam]` `int` | Phase 1 top-K values. |
| `select_seqs` | `[B, max_seq_len]` `int` | Frame indices that survive blank-skip. Identity in streaming. |
| `select_seq_lens` | `[B]` `int` | Number of valid entries in `select_seqs[b]`. |
| `pre_chars`, `pre_lp` | `[PREPASS_TILE, B, C_BUF]` `int` / `float` | Per-frame vocab top-K candidates from the chunk-level pre-pass (§3.4.7; fused layout only). |
| `pre_cnt` | `[PREPASS_TILE, B]` `int` | Valid candidates per pre-pass row. |
| `paged` | `PagedSequenceState` | Paged-mode memory descriptor (null pointers in flat mode). |

`ldbeam = align16(beam)` and `ldseq_len = align16(max_seq_len)` — 16-element
alignment so subsequent strided memcpys hit aligned offsets.

`FastDivmod(vocab_size)` is used inside kernels to recover `(beam, c)`
from a flat top-K index without an integer divide.

### 4.2 `PagedSequenceState`

Stores token sequences in fixed-size pages with a per-beam block table;
analogous to vLLM-style paged attention memory but applied to discrete
token IDs.

| Field | Shape | Purpose |
|-------|-------|---------|
| `page_storage` | `[num_pages, page_size]` `int` | Token pool. |
| `block_table[2]` | `[B, beam, max_logical_pages]` `int`, double-buffered | Logical→physical page map. |
| `ref_counts` | `[num_pages]` `int` | One per physical page. Atomic dec on free. |
| `next_free_page` | `int` | Atomic counter for fresh page allocation. |
| `free_pool` | `[num_pages]` `int` | LIFO stack of recycled physical pages. |
| `free_pool_size` | `int` | Top of the recycle stack. |

Sizing: `num_pages = batch * beam * (max_logical_pages + 1)` by default
(steady-state plus one full beam slab to absorb the step-1 CoW burst
before the recycle pool warms up). `page_size = 16` (one cache line)
fits the device-side `paged_read` / `paged_seq_compare` helpers.

### 4.3 `StreamingState`

A small header that *used* to be read between streaming steps. The
launcher now reconstructs `InternalData` from passed-in dimensions and
treats the first `STATE_HEADER_SIZE` bytes of the buffer as reserved
padding for layout compatibility — no D→H sync per step.

```
state_buffer layout:
  [0,  STATE_HEADER_SIZE)               reserved
  [STATE_HEADER_SIZE, end)              workspace (InternalData buffers
                                        + PagedSequenceState region)
```

## 5. Workspace Layout

Two queries answer "how big does the buffer need to be?":

```c
size_t calculate_workspace_size      (B, beam, V, max_seq_len);
size_t calculate_paged_workspace_size(B, beam, V, max_seq_len, page_size, num_pages=0);

size_t calculate_state_buffer_size      (B, beam, V, max_seq_len);
size_t calculate_paged_state_buffer_size(B, beam, V, max_seq_len, page_size, num_pages=0);
```

`init_internal_data` / `init_internal_data_paged` carve the workspace
into 128-byte aligned slabs in a fixed order (matching
`setup_internal_data_pointers` / `setup_internal_data_paged_pointers`).
Both halves of the API agree on the layout so a state buffer can be
re-set up cheaply on the host side.

The flat workspace is dominated by two buffers:

```
ptable  ≈ B * beam * vocab_size * 4 B
ptablen ≈ B * beam * vocab_size * 4 B
```

Paged mode saves memory on `clist` (which is dropped) but adds
`page_storage + 2 × block_tables + ref_counts + free_pool` for variable-
length sequences.

## 6. Per-Step Kernel Pipeline

For `beam ≤ 32` both the offline decode loops and the chunk launchers
dispatch the fused fast path instead: per `PREPASS_TILE`-frame tile,
one `fused::fused_topk_prepass_kernel` (§3.4.7) followed by one
`fused::fused_chunk_kernel<256, BEAM_CAP ∈ {16, 32}, PAGED>` that loops
the tile's frames in-kernel — first step included (§3.4.8).  The
single-frame `ctc_beam_search_step` API still launches
`fused::fused_first_step_kernel` / `fused::fused_step_kernel` per frame
(§3.4.6).  The table below is the legacy / `beam > 32` /
`OASR_CTC_FUSED=0` pipeline.

`ctc_prefix_beam_search_step` (flat) and
`ctc_prefix_beam_search_step_paged` (paged) launch the same kernel
sequence each step (except step 0):

| # | Kernel | Grid / block | Purpose |
|---|--------|--------------|---------|
| 0 | `first_step_kernel` (paged: `_paged_kernel`) | `<<<batch, 128>>>` | Block-wide radix sort top-K of the vocabulary at the first surviving frame. Fills `pprev`, `clast`, `clen[0]`, `clist[0]`, `score`. |
| 1 | `prob_matrix_kernel` | `<<<(bx, batch), 256>>>` | Build `ptablen[beam, c]` for non-blank chars. |
| 2 | `prob_space_blank_kernel` | `<<<batch, ldbeam>>>` | Update `pprev` blank/nonblank for next step. |
| 3 | `merge_kernel` (paged: `merge_paged_kernel`) | `<<<(beam, batch), ldbeam>>>` | Detect duplicate prefixes after extension and consolidate into the blank slot of the canonical beam. |
| 4 | `topk_phase1_kernel` | `<<<(bxs, batch), 128>>>` | Block-wide partial top-K of `ptable ∪ ptablen` per batch. `bxs ≤ MAX_BLOCKS_PER_BATCH`. |
| 5 | `topk_phase2_kernel` (paged: `_paged_kernel`) | `<<<batch, 128>>>` | Reduce phase-1 outputs to the final `beam` survivors and write the new `clen` / `clist` (or paged block tables + `ref_counts`). |

Step 0 takes the special `first_step_kernel` path because there is no
previous beam state to extend; subsequent steps take the merge → top-K
path.

After the loop:

- **Flat**: `fixup_parity_kernel` corrects any batches whose last
  active step has a different parity from the global `final_parity`,
  then three `cudaMemcpy2DAsync` calls strip `ldbeam` / `ldseq_len`
  padding.
- **Paged**: `gather_paged_results_kernel` walks each beam's
  block_table and copies `clen` tokens out of `page_storage` into the
  user-supplied flat `out_tokens`.

### Frame selection (offline only)

`init_select_kernel` reads `log_prob[..., blank_id]` and emits frame
indices where it stays below a user-configurable threshold (using a
CUB block-scan, no atomics). A short sync reads `select_seq_lens` to
compute `max_select`, the host-side step bound for the main loop.

In streaming mode no filtering is done — every frame the caller gives
us is treated as active (`init_streaming_select_kernel` writes the
identity mapping `select_seqs[b, t] = t`).

## 7. Flat vs Paged Mode

| | Flat | Paged |
|---|------|-------|
| Token storage | `clist[2][B, beam, ldseq_len]` ints (~8 B / token / beam × 2) | `page_storage[num_pages, 16]` + 2 × `block_table` |
| Sequence growth | Copy parent prefix into child every step | Atomic `alloc_page` / share with parent until the page fills, then CoW |
| Dedup-driven prefix sharing | None | `paged_seq_compare` walks the block tables |
| Final output | `cudaMemcpy2DAsync` (DtoD strided) | `gather_paged_results_kernel` |
| Memory at large `beam × max_seq_len` | Quadratic in beam × seq | Roughly linear in distinct prefixes |
| Best for | Short utterances, small beams | Long utterances or large beams |

Flat mode is the fastest path on small problems because it avoids the
indirection of paged reads and the CoW bookkeeping; it remains the
default for the engine. Paged mode is selected via
`use_paged_memory=True` on `GpuDecoderConfig`.

## 8. Streaming Lifecycle

```
init_streaming_state          ──▶ allocate beam state + identity select_seqs
init_streaming_state_paged    ──▶ same, plus PagedSequenceState

per chunk:
    ctc_beam_search_chunk     ──▶ iterates chunk_T frames in C++
                                  (or per-frame ctc_beam_search_step)

end of utterance:
    ctc_beam_search_read_state  ──▶ flat memcpy or gather_paged_results
```

`ctc_beam_search_chunk` is the single-stream hot path. On the fused
path (`beam ≤ 32`) it issues two launches per `PREPASS_TILE` tile —
vocab top-K pre-pass + multi-frame chunk kernel (§3.4.7/§3.4.8) — with
the optional CPU `is_speech_mask` (uint8) folded into a by-value
128-bit bitmap; on the legacy path it loops per active frame (captured
CUDA graphs or eager `streaming_step_persistent`). Returns the new
`step`; the caller maintains `frame_idx` itself.
`ctc_beam_search_chunk_batched` (the engine hot path) additionally
folds groups of up to 64 streams into each launch pair (§3.4.9).

`streaming_step` rebuilds `InternalData` from the dimensions the caller
passes (no GPU read), updates `select_seqs[b, step]` to reflect the
caller's `actual_frame_index` if it differs from `step`, and dispatches
the same per-step pipeline as the offline decoder.

`read_streaming_results` reads the current `step`, derives
`final_parity = (step ≤ 1) ? 0 : (step - 1) % 2`, and produces flat
output tensors. It does *not* touch `current_step` in the state —
`step` is owned by the Python `StreamState` (or the C++ caller).

## 9. Public C API (TVM-FFI)

| Function | Purpose |
|----------|---------|
| `ctc_decoder_workspace_size(batch, beam, vocab, max_seq)` | Flat workspace bytes. |
| `ctc_decoder_paged_workspace_size(..., page_size)` | Paged workspace bytes. |
| `ctc_decoder_state_size(...)` | Flat streaming-state bytes. |
| `ctc_decoder_paged_state_size(..., page_size)` | Paged streaming-state bytes. |
| `ctc_beam_search_decode(out_tokens, out_lengths, out_scores, log_prob, seq_lengths, workspace, beam, blank_id, blank_threshold)` | Full offline decode (flat). |
| `ctc_beam_search_decode_paged(..., page_size)` | Full offline decode (paged). |
| `ctc_beam_search_init_state(state_buffer, batch, beam, vocab, max_seq, blank_id)` | Flat streaming init. |
| `ctc_beam_search_init_state_paged(..., page_size)` | Paged streaming init. |
| `ctc_beam_search_step(state_buffer, log_prob_frame, beam, blank_id, step, blank_threshold, actual_frame_index, batch, vocab, max_seq, use_paged_memory, page_size)` | One streaming frame. |
| `ctc_beam_search_chunk(state_buffer, log_prob_chunk, is_speech_mask, ..., start_step, blank_threshold, start_frame_idx, ...) → new_step` | Whole-chunk loop in C++. |
| `ctc_beam_search_read_state(out_tokens, out_lengths, out_scores, state_buffer, step, ...)` | Read streaming results into flat tensors. |

The launchers validate that input tensors are CUDA-contiguous, have
the expected dtypes (`log_prob = float32`, `seq_lengths = int32`), and
have the expected dimensionality (`CHECK_DIM` macros). On failure
they raise via `TVM_FFI_ICHECK`.

## 10. Python API (`oasr.ctc_decode`)

The Python layer is a thin wrapper around the TVM-FFI launchers.

```python
@dataclass
class GpuDecoderConfig:
    beam_size: int = 10
    blank_id: int = 0
    blank_threshold: float = 0.98          # offline-only frame skip
    max_seq_len: int = 200
    use_paged_memory: bool = False
    page_size: int = 16

@dataclass
class GpuDecoderResult:
    tokens: List[List[List[int]]]          # [B][beam][token_ids]
    lengths: torch.Tensor                  # (B, beam) int32
    scores:  torch.Tensor                  # (B, beam) float32


# --- Offline ---
def ctc_beam_search_decode(
    log_prob: Tensor,                # (B, T, V) float32 cuda
    seq_lengths: Tensor,             # (B,)      int32   cuda
    *,
    beam_size: int = 10,
    blank_id:  int = 0,
    blank_threshold: float = 0.98,
    max_seq_len: int = 200,
    use_paged_memory: bool = False,
    page_size: int = 16,
) -> GpuDecoderResult: ...


# --- Streaming ---
class GpuStreamingDecoder:
    def __init__(self, config: GpuDecoderConfig | None = None): ...

    # Multi-request mode
    def create_state(self, batch, vocab_size, device=None) -> StreamState: ...
    def reset_state(self, state, batch, vocab_size, device=None): ...

    # Single-request mode
    def init_stream(self, batch, vocab_size, device=None): ...

    def decode_chunk(self, log_prob, *, state=None) -> None: ...
    def finalize_stream(self, *, state=None) -> GpuDecoderResult: ...

    @property
    def step(self) -> int: ...
    @property
    def config(self) -> GpuDecoderConfig: ...


class StreamHandle:                  # binds (decoder, state) pair
    def decode_chunk(self, log_prob): ...
    def finalize_stream(self) -> GpuDecoderResult: ...
    @property
    def step(self) -> int: ...
    @property
    def config(self) -> GpuDecoderConfig: ...
```

`StreamState` is a thin dataclass holding the GPU buffer and the
host-side `(step, actual_frame_idx, batch, vocab_size)` counters.
Pooling these in `CtcStateCacheManager` lets the engine reuse buffers
across requests without `cudaMalloc`.

## 11. Configuration Options

| Field | Default | Notes |
|-------|---------|-------|
| `beam_size` | 10 | Beam width. Workspace scales linearly. ≤ 128 is supported by `first_step_kernel` shared-memory layout. |
| `blank_id` | 0 | CTC blank index. Hard-coded in token filtering. |
| `blank_threshold` | 0.98 | Probability above which an offline frame is skipped. Use `1.0` to disable, `0.0` to skip all (no-op). Streaming ignores this — pre-filter in Python if needed. |
| `max_seq_len` | 200 | Hard cap on decoded sequence length. Both workspace and paged page count derive from it. |
| `use_paged_memory` | `False` | Switch to paged token storage. |
| `page_size` | 16 | Tokens per page. 16 is one cache line. Smaller pages improve sharing but increase page-table overhead. |

Workspace size is a pure function of these knobs plus `(batch,
vocab_size)`. There is no autotuning: the kernels use compile-time
template parameters tuned for the typical shape (`BLOCK_SIZE=128`,
`ITEMS_PER_THREAD={2, 4}`; fused path `BLOCK_SIZE=256`,
`BEAM_CAP ∈ {16, 32}`).

The environment variable `OASR_CTC_FUSED=0` selects a separately
compiled module (`ctc_decoder_legacy`, built with
`-DOASR_CTC_DISABLE_FUSED`) that forces the legacy multi-kernel step
pipeline for every beam size — an emergency rollback and the A/B
baseline for `tests/test_ctc_decoder_fused_parity.py`. The two
variants use different workspace layouts; never share a decoder or
`StreamState` across a mid-process flag flip.

## 12. Usage Examples

### 12.1 Offline batch decode

```python
import torch
from oasr import ctc_beam_search_decode, GpuDecoderConfig

# log_prob from your encoder, shape (B, T, V), float32, CUDA.
seq_lengths = torch.tensor([T, T, T], dtype=torch.int32, device="cuda")

result = ctc_beam_search_decode(
    log_prob, seq_lengths,
    beam_size=10, blank_id=0, blank_threshold=0.98,
    max_seq_len=200,
)

print(result.tokens[0][0])      # 1-best token IDs for utterance 0
print(result.scores[:, 0])      # 1-best score per utterance
```

### 12.2 Streaming, single request

```python
from oasr import GpuStreamingDecoder, GpuDecoderConfig

dec = GpuStreamingDecoder(GpuDecoderConfig(beam_size=10, max_seq_len=400))
dec.init_stream(batch=1, vocab_size=5000, device="cuda")

for chunk_log_prob in encoder_chunks:        # (1, chunk_T, V) float32 CUDA
    dec.decode_chunk(chunk_log_prob)

result = dec.finalize_stream()
text_tokens = result.tokens[0][0]
```

### 12.3 Streaming, many concurrent requests (engine-style)

```python
dec = GpuStreamingDecoder(GpuDecoderConfig(beam_size=10))
states = {sid: dec.create_state(batch=1, vocab_size=V) for sid in active_sids}

# interleave chunks from any request:
for sid, chunk in incoming_chunks:
    dec.decode_chunk(chunk, state=states[sid])

for sid in finished_sids:
    out = dec.finalize_stream(state=states[sid])
    # ... return states[sid] to a pool for reuse
```

### 12.4 Paged mode

```python
dec = GpuStreamingDecoder(GpuDecoderConfig(
    beam_size=20, max_seq_len=2000,
    use_paged_memory=True, page_size=16,
))
```

Memory cost drops dramatically once `beam × max_seq_len` is large; the
end-to-end interface is unchanged.

### 12.5 Direct C++ usage

```cpp
#include <oasr/ctc_decoder.cuh>
using namespace oasr::ctc_decoder;

size_t ws_bytes = calculate_workspace_size(B, beam, V, T);
void* workspace; cudaMalloc(&workspace, ws_bytes);

ctc_beam_search_decode_batch(
    log_prob, b_stride, t_stride, v_stride,
    seq_lengths,
    out_tokens, out_lengths, out_scores,
    workspace, B, beam, V, T, max_out_len,
    /*blank_id=*/0, /*space_id=*/-1,
    /*blank_threshold=*/0.98f, stream);
```

## 13. Error Handling and Edge Cases

| Condition | Where | Behaviour |
|-----------|-------|-----------|
| `log_prob` not float32 / not 3-D | TVM-FFI launcher | `TVM_FFI_ICHECK` → exception. |
| `seq_lengths` not int32 / not 1-D | launcher | `TVM_FFI_ICHECK`. |
| Output tensors wrong rank | launcher | `CHECK_DIM` failure. |
| `max_out_len < beam_count` of best beam | offline | Tokens truncated by `cudaMemcpy2DAsync` to `max_out_len`. |
| `step ≥ max_seq_len` reached mid-chunk | `ctc_beam_search_chunk` | Loop breaks early; returns the saturated `step`. |
| `select_seq_lens[b] == 0` (all frames are blank-dominant) | `first_step_kernel` | Early return; that batch's beams stay zero-init. |
| `blank_threshold ≥ 1.0` | offline | All frames selected (no skipping). |
| `blank_threshold ≤ 0.0` | offline | `log_threshold = NEG_INF` — no frame ever passes; output is empty. |
| Paged `num_pages` exhausted | runtime | `alloc_page` falls through to `atomicAdd(next_free_page, 1)`; if it exceeds `num_pages` the page index is invalid. **Caller must size correctly.** Default sizing covers worst-case CoW burst plus steady state. |
| `is_speech_mask` length ≠ `chunk_T` | `ctc_beam_search_chunk` | `TVM_FFI_ICHECK`. |
| `is_speech_mask` not uint8 | same | rejected. |
| `actual_frame_index != step` | `streaming_step` | Tiny `set_select_seq_step_kernel` patches `select_seqs[b, step]`. |
| `beam > 128` | `first_step_kernel` | The `smem.topk.{keys, vals}[128]` arrays cap beam at 128. |

The Python wrapper is purposely thin: errors raised by `TVM_FFI_ICHECK`
propagate as `RuntimeError`. There is no soft-failure path.

Sequences are **double-buffered**: callers must use the parity returned
by `final_parity = (step ≤ 1) ? 0 : (step - 1) % 2`. Reading from the
wrong buffer returns stale state from a previous step.

## 14. Performance Considerations

1. **Hot path stays on the GPU.** `streaming_step` reconstructs
   `InternalData` from passed-in dimensions, so no D→H sync is needed
   between frames. The only blocking op left is the offline frame
   selection (`cudaStreamSynchronize` after `init_select_kernel` to
   read `max_select`). Streaming has no such sync.
2. **`ctc_beam_search_chunk` ≫ Python loop.** The chunk launcher
   removes ~10 μs of Python overhead per frame (Python→C++ trip +
   tensor slice bookkeeping) which dominates wall time on chunked
   streaming workloads. Always prefer it over per-frame
   `decode_chunk` calls when you have ≥ 2 frames.
3. **`is_speech_mask` filters cheaply.** Compute the mask once on the
   GPU (e.g. `log_prob[:, blank_id] < threshold`), copy it to host
   uint8, and pass it through. Skipped frames bypass the kernel
   pipeline entirely.
4. **Frame selection (offline).** With `blank_threshold = 0.98` the
   typical reduction is 30–60 % of frames, halving step count and
   thus halving the cost of every per-step kernel. Tune to match
   your acoustic model's blank distribution.
5. **Top-K is two-phase.** Phase 1 launches up to
   `MAX_BLOCKS_PER_BATCH = 16` blocks per batch to partial-sort
   `ldc × beam` candidates; phase 2 reduces. The split keeps each
   block's shared memory bounded for large vocabularies.
6. **Workspace allocation.** Both `ptable` and `ptablen` are
   `B × beam × vocab × 4 B`. For `B=8, beam=10, V=5000` that's
   ~1.6 MiB each, doubled for double-buffered `clist` in flat mode
   (per token sequence). Size `max_seq_len` for your longest
   utterance, no more.
7. **Paged mode trade-off.** Paged storage saves memory roughly
   linearly in the number of *unique prefix tails* across beams. If
   your beams diverge quickly (high entropy) the win is small; on
   long utterances with high beam similarity (low entropy) it can be
   2–10×.
8. **Alignment.** `align16(beam)` and `align16(max_seq_len)` keep
   strided memcpys aligned. Don't pass tiny `max_seq_len`; you'll pay
   alignment overhead for no benefit.
9. **The fused step kernel is the default** (§3.4.6) for `beam <= 32`,
   collapsing the 5-kernel per-step pipeline into one. It is
   latency-bound rather than throughput-bound, so further gains need
   restructuring *across* steps — e.g. hoisting the vocab top-K out of
   the sequential loop as a chunk-level pre-pass — not more
   micro-optimisation. `OASR_CTC_FUSED=0` restores the legacy
   multi-kernel step as an A/B and rollback switch (set before process
   start). Measured speedups, per workload and end to end:
   `.artifacts/decoder_perf.md` Part 1.

   > **Parity note for refactors.** Compare **fused-old vs fused-new**.
   > Fused-vs-legacy has pre-existing boundary-tie differences and is
   > not a clean oracle.

## 15. Extension Points

- **Custom frame selection.** Replace `init_select_kernel` (offline)
  or pre-compute `is_speech_mask` (streaming) to inject your own
  filter. The rest of the pipeline only consumes `select_seqs` /
  `select_seq_lens`, so any monotone filter works.
- **External LM integration.** Currently no LM rescoring is fused.
  The cleanest extension point is between merge and top-K: add a
  kernel that reads `ptable` / `ptablen`, looks up an LM score for
  `(clist[src][beam] ++ char)`, and adds it before phase 1.
- **Different page size.** Override `page_size` in the config. 8 and
  16 are tested; small values (≤ 4) over-tax `block_table` and
  `paged_read`, large values (≥ 64) reduce sharing.
- **Hot-pluggable token store.** `merge_kernel` and
  `topk_phase2_kernel` interact with `clist` only via integer
  indexing; replacing them with a different paged scheme is
  mechanical (the paged kernels are a worked example).
- **Paged + offline.** Already supported via
  `ctc_beam_search_decode_batch_paged`. The offline path does the
  same gather as streaming `read_streaming_results`.

## 16. Quick Reference

```text
# Workspace queries
calculate_workspace_size      (B, beam, V, max_seq)
calculate_paged_workspace_size(B, beam, V, max_seq, page_size, num_pages=0)
calculate_state_buffer_size      (...)        # streaming-state size
calculate_paged_state_buffer_size(...)

# Offline
ctc_beam_search_decode_batch       (logp, ..., out, ws, B, beam, V, T, ...)
ctc_beam_search_decode_batch_paged (logp, ..., out, ws, ..., page_size, num_pages)

# Streaming
init_streaming_state[_paged]   (state_buffer, ...)
streaming_step                 (state, logp_frame, step, frame_idx, ...)
ctc_beam_search_chunk          (state, logp_chunk, mask?, start_step, ...)
read_streaming_results         (out, state, step, ...)

# Python
oasr.ctc_beam_search_decode(...) -> GpuDecoderResult
GpuStreamingDecoder + StreamState + StreamHandle
```
