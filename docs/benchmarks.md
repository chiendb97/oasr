# Benchmarks — engine vs. service

This document covers the two top-level perf harnesses:

| Script | Measures |
|---|---|
| `benchmarks/bench_engine.py` | In-process `ASREngine` — pure GPU + Python overhead, no IPC, no HTTP/WS. |
| `benchmarks/bench_service.py` | End-to-end `oasr-server` (Rust + HTTP + PyO3 dispatcher + engine). |

Run both back-to-back on the same machine for an apples-to-apples comparison —
the engine number is the ceiling, the service number is what real clients see.

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
partials per request:

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
  with `--profile` (see CLAUDE.md → Benchmarks & Profiling).
