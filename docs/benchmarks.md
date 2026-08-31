# Benchmarks — engine, service, accuracy

This document covers the three top-level harnesses:

| Script | Measures |
|---|---|
| `benchmarks/bench_engine.py` | In-process `ASREngine` — pure GPU + Python overhead, no IPC, no HTTP/WS. |
| `benchmarks/bench_service.py` | End-to-end `oasr-server` (Rust + HTTP + PyO3 dispatcher + engine). |
| `benchmarks/bench_accuracy.py` | WER/CER **and** speed in the same CSV row. |

Run the first two back-to-back on the same machine for an apples-to-apples
comparison — the engine number is the ceiling, the service number is what real
clients see.

Kernel-level benchmarking is a separate harness,
`benchmarks/oasr_benchmark.py` — see [`benchmarks/README.md`](../benchmarks/README.md)
and the `/benchmark-kernel` skill.

## Measurement protocol

Three rules, each of which has produced a wrong conclusion when skipped:

1. **Interleave the arms.** A single-order A/B lets the second arm benefit from a
   warm allocator. Report a σ over several iterations, not one run.
2. **Watch issue time against wall time.** At small batch the encoders are
   CPU-issue-bound, and a change that removes GPU work can still make them
   slower.
3. **Verify a fresh JIT hash directory** before trusting a kernel comparison —
   `rm -rf ~/.cache/oasr/jit` after editing a header that the cache key does not
   cover.

## Setup

1. Install the Python package — this also builds the Rust serving core into
   `oasr._core` and installs the `oasr-server` console script (needs a Rust
   toolchain + `protobuf-compiler` on `PATH`):
   ```bash
   pip install -e .[serving]
   ```

2. Copy `.env.example` to `.env`, edit the paths, then source it:
   ```bash
   cp .env.example .env
   $EDITOR .env
   set -a; source .env; set +a
   ```

   The recipes below assume these env vars are exported:

   | Variable | Purpose |
   |---|---|
   | `CKPT_DIR` | WeNet checkpoint directory (expanded into `--ckpt-dir`) |
   | `AUDIO_DIR` | Directory of mono 16 kHz `.wav` files (expanded into `--audio-dir`) |
   | `OASR_RS_BIN` | Optional override for the `oasr-server` path (`bench_service.py` reads it directly); defaults to the `oasr-server` console script on `PATH`, then `rust/target/release/oasr-server` |
   | `NUM_UTTERANCES` | Default for `--num-utterances` (both scripts) |
   | `MAX_BATCH_SIZE` | Default for `--max-batch-size` (both scripts) |
   | `CONCURRENCY` | Default for `--concurrency` (`bench_service.py`) |
   | `CHUNK_MS` | Default for `--chunk-ms` (`bench_service.py`) |

   The CLI flag still wins when both are given — `.env` sets the default, the
   flag overrides it for a single run.

## Engine benchmark

`bench_engine.py` runs the engine directly — no HTTP, no IPC, just `transcribe(...)` /
`transcribe_offline(...)` calls. Use it to characterise the GPU + Python ceiling.

Template — substitute the bracketed placeholders, or drop the flag entirely
to pick up the matching `.env` default (CLI flag still wins when both are
given):

```bash
python benchmarks/bench_engine.py \
    --ckpt-dir [CKPT_DIR] \
    --audio-dir [AUDIO_DIR] \
    --subroutines [offline|streaming|offline_wfst|streaming_wfst] \
    --max-batch-size [MAX_BATCH_SIZE] \
    --num-utterances [NUM_UTTERANCES] \
    --chunk-size [CHUNK_SIZE] \
    --dtype [float16|bfloat16|float32] \
    --cuda-graphs [on|off]
```

Concrete invocations with `.env` sourced:

```bash
# Offline — length-bucketed batches
python benchmarks/bench_engine.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --subroutines offline \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --num-utterances "$NUM_UTTERANCES"

# Streaming — interleaved chunk-by-chunk decode, paged KV cache
python benchmarks/bench_engine.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --subroutines streaming \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --num-utterances "$NUM_UTTERANCES"

# WFST decoding (in-tree GPU decoder) — pass the lang dir (contains HLG.pt;
# words.txt beside it provides the word table) or a direct .img/.pt path.
# The HLG.pt is exported to a cached .img next to it on first use.
python benchmarks/bench_engine.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --wfst-path /path/to/lang_bpe \
    --subroutines offline_wfst streaming_wfst \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --num-utterances "$NUM_UTTERANCES"

# CUDA-Graph toggle — captured (default) vs eager replay for profiling
python benchmarks/bench_engine.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --subroutines streaming --cuda-graphs off

# Export per-subroutine results to CSV
python benchmarks/bench_engine.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --output-path engine_results.csv
```

The output is one block per `--subroutines` value:

```
[PERF] offline      :: median time 1024 ms; std 87 ms
         RTF=0.0001  throughput=1951 utts/s  total_audio=13362 s
```

Throughput / RTF here represent the **GPU + scheduler ceiling** for this batch /
chunk-size config.

## Service benchmark

`bench_service.py` auto-spawns `oasr-server` (resolved via `$OASR_RS_BIN`, then
the `oasr-server` console script on `PATH`, then `rust/target/release/oasr-server`),
waits for `/readyz`, drives it with the chosen subroutines, then shuts it
down on exit. Use it to measure the gap closed (or not) by Rust + HTTP/WS.

Template — substitute the bracketed placeholders, or drop the flag entirely
to pick up the matching `.env` default (CLI flag still wins when both are
given):

```bash
python benchmarks/bench_service.py \
    --ckpt-dir [CKPT_DIR] \
    --audio-dir [AUDIO_DIR] \
    --subroutines [offline|streaming|grpc_offline|grpc_streaming|whisper] \
    --num-utterances [NUM_UTTERANCES] \
    --concurrency [CONCURRENCY] \
    --max-batch-size [MAX_BATCH_SIZE] \
    --chunk-ms [CHUNK_MS] \
    --wire-encoding [f32_le|i16_le] \
    --realtime [0|1] \
    --dtype [float16|bfloat16|float32]
```

Concrete invocations with `.env` sourced:

```bash
# Offline (HTTP POST /v1/transcriptions) — i16_le default (halves wire bytes)
python benchmarks/bench_service.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --subroutines offline \
    --num-utterances "$NUM_UTTERANCES" \
    --concurrency "$CONCURRENCY" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --wire-encoding i16_le

# Same, explicit f32_le baseline for comparison
python benchmarks/bench_service.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --subroutines offline \
    --num-utterances "$NUM_UTTERANCES" \
    --concurrency "$CONCURRENCY" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --wire-encoding f32_le

# WFST decoding end to end (server runs the in-tree GPU WFST decoder):
# --fst-path takes a prebuilt .img or a k2 HLG.pt (exported + cached on first
# use); the words.txt beside it provides the word table. Works for the
# grpc_offline / grpc_streaming subroutines the same way.
python benchmarks/bench_service.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --subroutines offline \
    --decoder-type ctc_wfst \
    --fst-path /path/to/lang_bpe/HLG.pt \
    --num-utterances "$NUM_UTTERANCES" \
    --concurrency "$CONCURRENCY" \
    --max-batch-size "$MAX_BATCH_SIZE"

# Streaming (WS /v1/stream) — no realtime pacing for max-rate test
python benchmarks/bench_service.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --subroutines streaming --realtime 0 \
    --num-utterances "$NUM_UTTERANCES" \
    --concurrency "$CONCURRENCY" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --chunk-ms "$CHUNK_MS" \
    --wire-encoding i16_le

# Streaming under live-mic pacing (each chunk waits chunk-ms wall-time)
python benchmarks/bench_service.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --subroutines streaming --realtime 1 \
    --num-utterances "$NUM_UTTERANCES" \
    --concurrency "$CONCURRENCY" \
    --chunk-ms "$CHUNK_MS"

# gRPC variants — same args, different subroutine
python benchmarks/bench_service.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --subroutines grpc_offline \
    --num-utterances "$NUM_UTTERANCES" --concurrency "$CONCURRENCY"

python benchmarks/bench_service.py \
    --ckpt-dir "$CKPT_DIR" --audio-dir "$AUDIO_DIR" \
    --subroutines grpc_streaming --realtime 0 \
    --num-utterances "$NUM_UTTERANCES" --concurrency "$CONCURRENCY" \
    --chunk-ms "$CHUNK_MS"
```

The summary block reports `requests`, `wall`, `audio`, RTF (audio_s / wall_s),
throughput, latency percentiles, and (streaming only) `first-partial` time +
partials per request. The numbers below illustrate the **format** — they came
from one run on one box and are not a target:

```
streaming (WS /v1/stream, chunk=640ms, i16_le):
  requests   ok=2000  rejected=0  fail=0
  wall       27.71 s
  audio      13362.03 s
  RTF        482.29x   (audio_seconds / wall_seconds — higher is faster)
  throughput 72.19 req/s
  latency    mean=871 ms  p50=604  p90=1647  p95=2475  p99=5412  max=6044
  first-partial   mean=270 ms  p50=132  p95=599
  partials/req    mean=10.8
```

## Accuracy benchmark

`benchmarks/bench_accuracy.py` is the accuracy counterpart: manifest-driven
WER/CER with RTFx, throughput, p50 and p99 **in the same CSV row**, so accuracy
and speed are visible together.

```bash
# Build a manifest from a corpus (audio is not shipped with the repo)
python benchmarks/bench_accuracy.py --build-manifest \
    --audio-dir "$AUDIO_DIR" --out benchmarks/manifests/my_corpus.jsonl

# Measure
python benchmarks/bench_accuracy.py \
    --ckpt-dir "$CKPT_DIR" --manifest benchmarks/manifests/my_corpus.jsonl \
    --output_path accuracy.csv
```

- WER is the **corpus** rate (total errors / total reference words), with
  Whisper `EnglishTextNormalizer` semantics — `oasr/testing/wer.py`.
- Manifests under `benchmarks/manifests/` ship **without audio**; build one for
  your local corpus with `--build-manifest`.
- `--build-manifest` unwraps `(...)`, because the English normalizer deletes
  bracketed spans while a reader speaks them aloud. Skipping that step is pure
  measurement error.
- `oasr/testing/accuracy.py` holds the manifest and transcribe helpers shared
  with the CI accuracy gate (`tests/test_accuracy.py`, reference rates in
  `ci/wer-reference.json`).
- `--decode-option k=v` (repeatable) reaches the decode family's own knobs, and
  the CSV carries them, so two rows that differ only by an option stay tellable
  apart. This is how a storage or capture path is checked *at corpus scale*
  rather than tensor by tensor:

  ```bash
  python benchmarks/bench_accuracy.py --ckpt-dir "$SPEECH_LLM_CKPT" \
      --manifest benchmarks/manifests/ljspeech_200.jsonl --audio-root "$WAV_DIR" \
      --dtype bfloat16 --decode-option kv_storage=dense --decode-option step_graphs=0
  ```

  An entry in `ci/wer-reference.json` may pin the same options, which the `llm`
  row does for its prompt: for a speech-LLM the prompt is part of the decode
  configuration, and changing it moves the WER with no defect behind it.
- `--vad-mode` and `--vad-backend` are sweep axes too, with `--vad-model-dir`
  and a repeatable `--vad-option k=v`; the cell is a CSV column and each row
  gets its own `--save-transcripts` file, so "did the transcript change?" is a
  `diff`:

  ```bash
  python benchmarks/bench_accuracy.py --ckpt-dir "$CKPT_DIR" \
      --manifest benchmarks/manifests/ljspeech_200.jsonl --audio-root "$WAV_DIR" \
      --vad-mode off segment --vad-backend energy silero
  ```

  A corpus of **short** utterances measures almost nothing here: the splitter
  declines to cut audio one padded span already covers, so `segment` scores
  identically to `off` — a useful negative control, not evidence that
  segmentation works. Verify that on long-form audio, and read the segment
  count and the seconds dropped beside the rate.

## Notes

- `--service-mode` on `bench_service.py` is **auto-derived** from
  `--subroutines` (offline / whisper / grpc_offline → `offline`; streaming /
  grpc_streaming → `streaming`). Mixed subroutine sets are rejected.
- `--num-workers > 1` is rejected — `oasr-server` is now one-process-per-GPU.
  For multi-GPU, launch N `oasr-server` processes manually with distinct
  `--http-bind`/`--grpc-bind` + `CUDA_VISIBLE_DEVICES`.
- Wire encoding `i16_le` halves the HTTP/WS bytes vs `f32_le`. Server-side
  decode (`oasr-asr::decode_raw_pcm`) widens i16 back to f32 by dividing by
  32768, matching the bench's scale-by-32767 encode (one count short of
  saturation at ±1.0).
- For Nsight-Compute kernel profiling, see `benchmarks/oasr_benchmark.py`
  with `--profile`, and the `/benchmark-kernel` skill.
- Point-in-time results — engine, serving, decoders and kernels — live under
  `.artifacts/` (`engine_perf.md`, `serving_perf.md`, `decoder_perf.md`,
  `fmha_tuning.md`, `gemm_tuning.md`, `profiling_report.md`). Record new
  measurements there, not in this file.
