# Shipped tuning database (system tier)

One JSON file per `(arch family, kernel family)`: `sm120/gemm.json`,
`sm120/conv1d.json`, `sm120/scaled_mm.json` (the int8 W8A8 GEMM of quantized
layers), `sm120/scaled_mm_fp8.json` (the fp8 e4m3 one), … Read by `oasr.tune.database`; the format, the tier rules and the
validator policy are documented there. The user tier
(`~/.cache/oasr/tune/v2/…`, written by tuning runs on the machine that measured
them) wins over these files per signature.

These files change **only by review**, like `ci/wer-reference.json`. Produce a
new one with `oasr tune build` on the target card and review the diff with
`oasr tune diff`; do not hand-edit timings.

## Provenance of the files here

### `sm120/scaled_mm_fp8.json` — Conformer fp8 widths (2026-10-21)

The fp8 family's first file (lane `sm89_fp8_mma`), built by `oasr tune build
--family scaled_mm_fp8` on an otherwise idle RTX 5090 from the same four census
shapes as `scaled_mm.json` below, in fp8: the U2++ Conformer quantized online
with `fp8_w8a8`, and the `smoothquant_fp8_w8a8_fused` checkpoint, each offline
(`max_batch_size=32`) and streaming (chunk 16, widths 1–64).  Every fp8 variant
computes the same bits (one summation order for every tile), so the bitwise
numerics gate holds as for int8.  The heuristic keeps small and mid M; at large M
(above M = 2040–7136 depending on the width) `b64x64x64_w32x32x64_s6_e4m3` wins
every signature where the heuristic picks a 128-wide tile -- the 64×64 tile keeps
two CTAs per SM at fp8's half-rate fp32 accumulation.  Regenerate as below with
`--quantization fp8_w8a8`, `--recipe smoothquant_fp8_w8a8_fused` and
`--family scaled_mm_fp8`.

### `sm120/scaled_mm.json` — Conformer int8 widths (2026-10-20; re-measured 2026-10-21)

Re-measured on 2026-10-21 after the scaled-mm headers became element-generic
(the `impl_hash` validator marked the file stale): the same decisions, region
for region.

The int8 `scaled_mm` family's first file, built by `oasr tune build --family
scaled_mm` on an otherwise idle RTX 5090 (`provenance.contended` false) from four
censuses of the WeNet U2++ Conformer (`$CKPT_DIR`): quantized online with
`int8_w8a8`, and the `smoothquant_w8a8_fused` checkpoint (whose FFN `w_1` is a
quantizing `scaled_mm_quant` launch feeding `w_2`), each offline (LJSpeech-200
durations through the length-sorted batching replay, `max_batch_size=32`) and
streaming (chunk 16, widths 1–64) -- the same censuses `gemm.json`'s Conformer
rows were measured from.  Six signatures: `op=scaled_mm` at `(N, K)` = `(256,
256)`, `(256, 2048)`, `(512, 256)`, `(768, 256)`, `(2048, 256)`, and
`op=scaled_mm_quant` at `(2048, 256)`; census points plus ladder fill, M = 16–8608,
every production variant measured at every point (protocol 3; the quant op
data-parallel only), numerics gated bitwise against the heuristic's variant.

Five signatures are `[[null, "heuristic"]]`: the shape heuristic stays the pick
in every region (an entry still records that it was measured, so `tuning.strict`
passes and the report can tell measured from unseen).  `(768, 256)` takes
`b128x128x64_w64x64x64_s5` above M = 7136, where the heuristic's 128×256 tile
spills into a second wave (192 CTAs on 170 SMs; 1.09–1.10× at M = 8160).  No
Stream-K tile won anywhere (K ≤ 2048: 16 int8 K-tiles, too few to amortise the
barrier memset and the partial-tile fixup).  Regenerate in one go:

```bash
set -a; source .env; set +a
export CUDA_VISIBLE_DEVICES=0 OASR_TUNE_USER_DB=off
oasr quantize $CKPT_DIR /tmp/u2pp_fused --recipe smoothquant_w8a8_fused \
    --calib-manifest $LIBRISPEECH_DEV_DIR/manifest.jsonl
for CK in "--ckpt-dir $CKPT_DIR --quantization int8_w8a8" "--ckpt-dir /tmp/u2pp_fused"; do
  oasr tune census $CK --service-mode offline --max-batch-size 32 \
      --manifest benchmarks/manifests/ljspeech_200.jsonl --audio-root $WAV_DIR \
      --out /tmp/census_$((n++))_offline.json
  oasr tune census $CK --service-mode streaming --max-batch-size 64 --chunk-size 16 \
      --out /tmp/census_$((n++))_streaming.json
done
oasr tune build $(printf -- '--census %s ' /tmp/census_*.json) --family scaled_mm \
    --out system --fresh --log /tmp/scaled_mm_measurements.jsonl
oasr tune diff <previous scaled_mm.json> oasr/tune/db/sm120/scaled_mm.json
```

(`oasr tune …` / `oasr quantize …` are `python -m oasr.tune.cli …` /
`python -m oasr.cli quantize …` where the console script is not installed.)
`--family scaled_mm` leaves `gemm.json` alone; `--note "…"` records free text in
the file's provenance.

### `sm120/gemm.json` — Conformer widths re-measured (2026-09-29)

The seven Conformer-CTC signatures (`(256,256)`, `(256,2048)`, `(256,4864)`,
`(512,256)`, `(768,256)`, the swish FF `(2048,256)` and the CTC head
`(5008,256)`) were re-measured with `oasr tune build` (protocol 2, top-16 +
cross-evaluation, cover partition) from an offline census (LJSpeech-200 durations
through the length-sorted batching replay, `max_batch_size=32`) and a streaming
census (chunk 16, widths 1–64). Regret against the measured oracle on those 149
points: converted table 8.8% geomean / 41% p95, re-measured 3.0% / 13.9%. End to
end (RTX 5090): offline Conformer 400 utterances **1.023×** over the converted
table (3 interleaved rounds, disjoint ranges), streaming neutral (~1.009×,
overlapping), LJSpeech-200 WER identical (4.085%, 108S 21D 10I). The file also
carries the device model (`model`) the resolver's model tier ranks uncovered
widths with, and a data-derived coverage basis.

### `sm120/gemm.json` (other signatures), `sm120/conv1d.json`

Converted on 2026-09-29 from the Python literals `_GEMM_HEURISTIC_RULES_SM120`
(`oasr/jit/gemm.py`) and `_CONV1D_HEURISTIC_RULES_SM120` /
`_CONV1D_ACTIVATION_HEURISTIC_RULES_SM120` (`oasr/jit/conv.py`) at `3a06738`.
The conversion is lossless: all 161,448 GEMM selections (7 SM families × 4 ops ×
62 `(N, K)` × 31 M × 3 dtypes) and 1,890 Conv1D selections were compared against
the literal selectors and are identical, including `GEMM_DEFAULT` identity and
the miss / inactive-arch counters. Comments that sat inside the literals are
each entry's `notes`.

Measured on an RTX 5090 (sm_120, 170 SMs) with `scripts/tune_asr_gemm.py`
(protocol 1: 64-call CUDA-graph loop, single-replay self-overlap gate,
`--min-speedup 1.05`), across several sweeps:

- Conformer-CTC widths — the original table; later re-tuned over the expanded
  candidate space (thin-N tiles, working serial split-K, parallel split-K,
  Stream-K).
- whisper-tiny `K=384` widths (2026-08-03): +2.2–3.8% encoder at batch ≥ 4, WER
  identical. The `(384, 1536)` small-M rows deliberately avoid cuBLAS: the
  encoder is CPU-issue-bound at batch 1–2 and the cuBLAS branch costs ~4.9 µs
  more per call to issue.
- The LSTM/RNN gate projection `(2560, 640)` (2026-08-23), measured GPU-only in a
  captured 100-call loop — a back-to-back launch loop cannot resolve it.
- **The 2026-08-24 capture-driven Zipformer sweep** (165 representative shapes
  captured from real checkpoints). A coverage census found Zipformer running
  `GEMM_DEFAULT` on 49 distinct `(op, N, K)` keys against 1 for Conformer; these
  are those keys. Every arm cleared the self-overlap gate (faster both
  back-to-back in one graph *and* on a single replay). Kernel-level 1.07–8.63×;
  end-to-end offline batch 64, 64 LJSpeech utterances, 6 interleaved arms of 25
  reps: 1.0192× min / 1.0151× median, transcripts identical. The sweep's other
  16 keys were measured and **not** kept (conformer 1.003×, paraformer 1.006×,
  whisper 1.002×, nemotron 0.993–1.004×, all inside their own σ).
- The Zipformer ConvNeXt pointwise contraction `(128, 384)`: a `block_n=16` rule
  once returned an empty transcript (CUTLASS's epilogue cannot address the tile);
  see that entry's notes.

Conv1D: exact production shapes only (Whisper's fixed-window frontend, widths 384
and 1280, and the Paraformer predictor conv), per dtype.
