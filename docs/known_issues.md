# Known issues

Tracked defects with reproductions. Each entry is written so it can be pasted
into the issue tracker verbatim. Fixes for kernel-level items belong on their
own branch, not on feature/refactor branches that merely surface them.

---

## GPU-DEC-1 — `ctc_gpu` `blank_threshold` frame-skip garbled transcripts  ✅ FIXED

**Severity:** high (wrong transcripts, affected offline *and* streaming)
**Component:** `include/oasr/ctc_decoder.cuh` GPU CTC prefix-beam kernels
**Status:** **fixed** on `bugfix/ctc_cuda_decode` (commit pending)
**Surfaced by:** `refactor/gpu-only-engine` (engine switched to `decoder_type="ctc_gpu"`)

### Symptom (pre-fix)
With the default `GpuDecoderConfig.blank_threshold = 0.98`, the GPU prefix-beam
decoder duplicated subword pieces even on **clean, full-sequence offline**
log-probs:

| reference (CPU prefix-beam / GPU @ `blank_threshold=1.0`) | GPU @ `blank_threshold=0.98` (pre-fix) |
|---|---|
| `... EXHIBITION` | `... EXHIBIT EXHIBITIONION` |
| `... THE CHINESE TOOK ... WOOD BLOCKS ... WOOD CUTTERS ...` | `... THE CHIN CHINESESE TOOK ... WOOD BLOCKSS ... WOOD CUTTERTERS ...` |

On LJ001-0001 the tail token ids went `[..., 1565, 2365]` (correct) →
`[..., 1565, 1565, 2365, 2365]` — each emitted token **duplicated**.

### Root cause
The duplication was **not** at the skipped-blank gaps; it was at consecutive
same-token frames *following* a skipped-blank step. When a step had
`need_add_blank` set (one or more blank-dominant frames were skipped before
it), `topk_phase2_kernel` (and `topk_phase2_paged_kernel`) overrode the
**outgoing** beam state to `{new_score, NEG_INF}` — i.e. it relabelled the
prefix as "ends in **blank**". But that step had just emitted a **non-blank**
token, so the prefix actually ends in non-blank (`pnb` carries the mass,
`pb = NEG_INF`). With the prefix mislabelled "ends in blank", the very next
identical frame took the CTC *repeat-after-blank* path and **extended** the
prefix (`X` → `X X`) instead of collapsing (`X` → `X`), duplicating the token.

The blank-frame collapse for skipped frames *before* a step is already applied
correctly to the **incoming** `prev` inside `prob_matrix_kernel` /
`prob_space_blank_kernel` (they collapse it to `{logsumexp(pb,pn), NEG_INF}`),
so `ptable`/`ptablen` already account for the skip. The extra override of the
*outgoing* state was the bug. `first_step_kernel` /
`first_step_paged_kernel` had the same wrong pattern for the leading-blank case
(`need_add_blank ? {key, NEG_INF} : {NEG_INF, key}`).

Both only fired when frames were skipped, which is why `blank_threshold=1.0`
(no skip) stayed bit-exact with the CPU `prefix_beam` oracle and only `0.98`
garbled.

### Fix
`include/oasr/ctc_decoder.cuh`: the outgoing `pprev` for the next step is now
always taken straight from `ptable`/`ptablen` (`make_float2(p, pn)`), and the
first-step token always lands in the non-blank slot (`make_float2(NEG_INF,
key)`), in both the flat and paged kernels. For a *blank* winner
`ptablen[blank_slot]` is already `NEG_INF` and `p == new_score`, so the no-skip
path is unchanged (still bit-exact with CPU); only the skip path is corrected.

### Verification
Isolated decode of encoder log-probs on 80 LJSpeech utterances: GPU @ `0.98`
(flat **and** `use_paged_memory=True`) is now bit-identical to the CPU
`prefix_beam` oracle on every utterance (was 8/12 garbled before), and GPU @
`1.0` is unchanged. `tests/test_engine.py::test_streaming_matches_offline_single_stream`
passes bit-exactly (the `xfail` marker was removed).

---

## GPU-DEC-2 — streaming `step` budget is capped by `max_seq_len` (output cap)

**Severity:** medium (drops the end of *long* streaming transcripts; does not
affect normal-length utterances at the default `blank_threshold=0.98`)
**Component:** `csrc/ctc_decoder.cu::ctc_beam_search_chunk[_batched]` +
`include/oasr/ctc_decoder.cuh` streaming `select_seqs` sizing
**Status:** open — design change (decouple the two budgets)
**Independent of GPU-DEC-1.**

### Symptom
The streaming chunk loop counts one `step` per **decoded (non-blank) frame**
and breaks at `step >= max_seq_len`, and the streaming `select_seqs` buffer is
sized `batch * max_seq_len`. But `GpuDecoderConfig.max_seq_len` (default 200) is
meant to be the **output-token** cap. So `max_seq_len` is overloaded as *both*
the per-frame step budget *and* the output-token cap. Offline does not have this
problem — it sizes `select_seqs` to the input length and only caps the output.

Without blank-skip (`blank_threshold=1.0`) every frame is a step, so any
utterance with more than `max_seq_len` frames truncates. Measured on
LJ001-0001 (T=240 encoder frames), `max_seq_len=200`:

| config | `step` reached | output tokens | matches offline? |
|---|---|---|---|
| stream `thr=1.0`, `max_seq_len=200` | 200 (capped) | 28 | ❌ truncated |
| stream `thr=1.0`, `max_seq_len=248` | 240 (all)   | 34 | ✅ |
| stream `thr=0.98`, `max_seq_len=200` | 47 (non-blank only) | 34 | ✅ |

So the incremental state machine is **correct** — given enough budget the
streaming output is bit-exact with offline. The earlier guess that the
truncation lived in `decode_chunk` / `finalize_stream` was wrong; it is purely
the `max_seq_len` step cap.

### Why it usually does not bite
At the default `blank_threshold=0.98`, `step` only counts non-blank frames
(~20 % of frames for typical speech), so an utterance needs roughly
`max_seq_len / 0.2` ≈ 1000 encoder frames (tens of seconds) before `step`
reaches 200. Short/medium utterances and the engine's default streaming config
are unaffected, which is why the engine tests pass.

### Repro
```python
on = ASREngine(EngineConfig(
    ckpt_dir=CKPT, device="cuda", dtype=torch.float16, decoder_type="ctc_gpu",
    chunk_size=16, num_left_chunks=-1, max_batch_size=1,
    gpu_decoder_config=GpuDecoderConfig(blank_threshold=1.0)))  # forces no skip
print(on.transcribe([LONG_WAV]))  # tail dropped once frames > max_seq_len
```

### Fix options
- **Decouple the budgets:** give the streaming state a separate frame/step
  budget for `select_seqs` (and the chunk-loop cap) independent of the
  output-token cap that sizes `clist`/`ldseq_len`. Note `topk_phase2` no longer
  reads `select_seqs` after the GPU-DEC-1 fix, so only
  `set_select_seq_step_kernel`, `prob_matrix_kernel`, `prob_space_blank_kernel`
  and `first_step_kernel` touch it; `need_add_blank` only needs `select_seqs[step]`
  and `select_seqs[step-1]`.
- **Workaround (no code change):** size `GpuDecoderConfig.max_seq_len` ≥ the
  expected non-blank-frame count for the longest stream.
