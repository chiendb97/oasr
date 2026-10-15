# ASR Inference Benchmark Suite (`asrbench`)

`asrbench` is OASR's suite for measuring the inference performance of speech
recognition backends: latency, throughput, batching efficiency, concurrency,
streaming behaviour and resource use. It lives in `benchmarks/asrbench/` and
runs as `python -m benchmarks.asrbench`.

The suite is built on four principles:

- **Reproducible.** Workloads are deterministic and content-addressed, sizing
  comes from a recorded profile, and every run stores the hardware, software
  and configuration it ran on.
- **Backend-neutral.** One workload runs unchanged against the in-process
  engine, a served `oasr-server` (HTTP or gRPC), any OpenAI-compatible server,
  or Hugging Face Transformers, under the same measurement protocol.
- **Auditable.** Every run keeps the raw per-request and per-session records
  its summaries were computed from; any summary can be recomputed from them.
- **Explicit about gaps.** A capability a backend lacks is reported as
  `unsupported` with a reason. It is never emulated.

Accuracy is outside the scope of performance runs. Word error rate is
available as a correctness guardrail through a separate, untimed command
(`validate`).

## Contents

- [Quick start](#quick-start)
- [Concepts](#concepts)
- [Command reference](#command-reference)
- [Datasets](#datasets)
- [Workloads](#workloads)
- [Profiles](#profiles)
- [Benchmark kinds](#benchmark-kinds)
- [Backends](#backends)
- [Measurement methodology](#measurement-methodology)
- [Streaming methodology](#streaming-methodology)
- [Metrics reference](#metrics-reference)
- [Resource monitoring](#resource-monitoring)
- [Correctness validation](#correctness-validation)
- [Results](#results)
- [Reports](#reports)
- [Regression gating](#regression-gating)
- [Limitations](#limitations)
- [Design rationale](#design-rationale)

## Quick start

Export the checkpoint and corpus paths first (`cp .env.example .env`, edit the
paths, then `set -a; source .env; set +a`).

```bash
# 1. Prepare the primary dataset (one-time, 346 MB, checksum-verified).
python -m benchmarks.asrbench prepare librispeech --download

# 2. Run the smoke profile against the in-process engine.
python -m benchmarks.asrbench run --profile smoke --backend oasr_engine -o dtype=float16

# 3. List stored runs, then render a report.
python -m benchmarks.asrbench runs
python -m benchmarks.asrbench report <run-id>

# 4. Gate a candidate against a baseline (exit 1 on a regression or a failed point).
python -m benchmarks.asrbench compare <baseline-run-id> <candidate-run-id>
```

To start without downloading anything, point a profile at data already on
disk: `prepare ljspeech` indexes a local LJSpeech copy, and `prepare manifest`
indexes any `{id, audio, text}` JSONL manifest (see
[Using your own data](#using-your-own-data)). Then add `--dataset ljspeech/all`
or `--dataset manifest/<name>` to any `run`.

### Dependencies

The core suite needs `numpy`, `soundfile` and `PyYAML`. The rest is optional
and loaded only by the feature that uses it:

| Package | Required for |
|---|---|
| `scipy` (or `torchaudio`) | Resampling audio that is not at the profile's sample rate |
| `httpx`, `websockets` (`pip install -e ".[serving]"`) | `oasr_server` and `openai_http` backends |
| `grpcio`, `grpcio-tools` | `oasr_server -o transport=grpc` |
| `psutil` | CPU and memory monitoring |
| `nvidia-ml-py` | GPU monitoring and GPU inventory |
| `pyarrow` | Earnings-21 preparation |
| `matplotlib` | `report --plots` |
| `transformers` | `transformers` backend |

A missing optional dependency disables only the feature that needs it, with a
message naming the package; missing monitoring probes leave their metrics
absent rather than zero.

## Concepts

| Term | Meaning |
|---|---|
| **Dataset** | A prepared corpus: a manifest of audio items plus a card recording provenance, licence and checksums. Referenced as `name/subset`. |
| **Workload** | An explicit, ordered list of items selected from a dataset by a seeded specification. Identified by a content hash. |
| **Profile** | A YAML plan naming workloads and the benchmarks to run on them. Built-in profiles cover smoke, standard and release evaluation. |
| **Benchmark** | One entry in a profile: a benchmark *kind* (protocol) plus its parameters. |
| **Point** | One coordinate of a benchmark's sweep, such as batch size 8 or 32 concurrent sessions. Each point is measured in its own window. |
| **Result** | The outcome of one point: status, summary metrics, attributed resources and server metrics, warnings. |
| **Run** | One execution of a profile against one backend. Stored as a directory of records and summaries. |
| **Backend** | An adapter that exposes an inference framework through the suite's contract and declares its capabilities. |
| **Service mode** | `offline` (whole-utterance requests) or `streaming` (incremental sessions). A run loads one backend instance per mode it needs. |

## Command reference

| Command | Purpose |
|---|---|
| `datasets` | List registered datasets with licence, download size and prepared subsets |
| `prepare NAME` | Index a local copy (`--source-dir`) or download (`--download`) a dataset |
| `profiles [NAME]` | List built-in profiles, or print one resolved with any overrides applied |
| `backends [-v]` | List backends, their availability and (`-v`) their options |
| `run --profile P --backend B` | Execute a profile and write a run directory; exit 1 if any point failed |
| `runs` | List stored runs, newest first |
| `validate --backend B` | Measure WER and empty-transcript rate on a labelled subset (untimed) |
| `compare BASE CAND` | Regression check between two runs or validation reports |
| `report RUN [RUN ...]` | Render Markdown and CSV (and optionally charts) from stored runs |
| `recompute RUN` | Re-derive every summary from the raw records and report disagreements |
| `save-baseline RUN --name N` | Copy a run to `<results root>/baselines/N` |

Commands that read a run (`compare`, `report`, `recompute`, `save-baseline`,
`validate --reference`) accept a directory path, a directory name under the
results root, a run id as printed by `runs`, or a baseline name.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `ASRBENCH_DATA_DIR` | Prepared datasets (also `--data-dir`) | `~/.cache/oasr/asrbench` |
| `ASRBENCH_RESULTS_DIR` | Runs, validation reports and baselines (also `run --out`) | `$XDG_DATA_HOME/oasr/asrbench/results`, i.e. `~/.local/share/oasr/asrbench/results` |
| `CKPT_DIR` | Default checkpoint for `oasr_engine` and `oasr_server` | — |
| `WHISPER_CKPT` | Default model for the `transformers` backend | — |
| `LJSPEECH_DIR`, `AUDIO_DIR`, `WAV_DIR` | Discovery of a local LJSpeech copy; audio root for relative manifest paths | — |
| `HF_ENDPOINT`, `HF_TOKEN` | Hugging Face mirror and token for Earnings-21 | public endpoint |
| `OASR_RS_BIN` | `oasr-server` executable | `oasr-server` on `PATH` |

Sizing never comes from the environment. See
[Configuration source](#configuration-source).

## Datasets

Datasets are selected for workload coverage (duration spread, volume,
continuous audio, reproducibility and access), not for acoustic or linguistic
diversity.

| Dataset | Licence | Acquisition | Durations | Role |
|---|---|---|---|---|
| `librispeech` (`test-clean`; also `test-other`, `dev-clean`, `dev-other`) | CC BY 4.0 | 346 MB archive, MD5-verified, four mirrors | 1.3–35 s, mean ≈ 7.4 s; 2,620 utterances, 5.4 h | **Primary.** Duration buckets, load and streaming sessions, long-form source, WER guardrail |
| `ljspeech` | Public domain | Indexed in place from a local copy | 1.1–10 s; 13,100 clips, 24 h | Zero-download smoke source; short-utterance latency |
| `earnings21` | CC BY-SA 4.0 | One Parquet shard at a pinned revision, SHA-256-verified (238–652 MB) | Whole calls, mean ≈ 53 min; 44 calls, 39 h | Real long-form audio and long live sessions |
| `longform` (derived) | Inherits the source | Built from a prepared dataset | Configurable targets (default 60, 300, 900 s) | Long recordings with exact transcripts |
| `manifest` | As supplied | Indexed in place from a local JSONL manifest | As supplied | Your own corpus |
| `synthetic` | n/a | Generated | As configured | Tests of the suite itself; flagged as simulated, never a measurement input |

### Selection rationale

- **LibriSpeech** is free to download, small enough to fetch whole, pinned by
  published checksums, and covers the useful utterance range. It is the
  training domain of the repository's English checkpoints, so a WER change on
  it signals a defect rather than a domain shift. Its chapter grouping provides
  the contiguous speech the long-form builder needs.
- **Earnings-21** is the most permissively licensed standard long-form corpus
  (TED-LIUM is CC BY-NC-ND; SPGISpeech and GigaSpeech are gated or
  non-commercial) and is a long-form track of the Open ASR Leaderboard. It is
  fetched per shard, so a few hours of real long-form audio cost 238 MB rather
  than 4.4 GB.
- **Derived long-form** recordings concatenate consecutive utterances of one
  chapter up to a target length, never cropping mid-utterance, so the
  transcript stays exact and no download is needed. Use Earnings-21 when the
  recording itself must be natural (pauses, speaker turns, crosstalk).
- **Not selected:** Common Voice (downloads require an account), GigaSpeech and
  SPGISpeech (gated or non-commercial), AMI (multi-channel meeting audio with
  heavier preparation and no workload shape the selected sets lack), FLEURS
  (its duration profile duplicates LibriSpeech's).

### Preparing datasets

```bash
python -m benchmarks.asrbench datasets                         # registry and prepared state
python -m benchmarks.asrbench prepare librispeech --subset test-clean --download
python -m benchmarks.asrbench prepare librispeech --source-dir /data/LibriSpeech   # existing copy
python -m benchmarks.asrbench prepare ljspeech                 # discovered via $LJSPEECH_DIR, $AUDIO_DIR, $WAV_DIR
python -m benchmarks.asrbench prepare earnings21 --download --shards 8
python -m benchmarks.asrbench prepare longform --source librispeech/test-clean --targets 60,300,900
```

Downloads are an explicit opt-in. Without `--download`, an absent corpus
fails with acquisition instructions and nothing is fetched. Downloads go
through `.part` files with checksum verification and mirror fallback, and
archives are extracted with path-traversal and link checks. `prepare --verify`
re-reads every file's header duration. Every prepared dataset records a
manifest hash, so an edited manifest is detected.

### Using your own data

The `manifest` adapter indexes an existing corpus described by a JSON Lines
manifest in the repository's format, one object per line:

```json
{"id": "utt_0001", "audio": "utt_0001.wav", "text": "reference transcript"}
```

Nothing is copied. Durations are read from the audio headers, so no
`duration` key is required. The manifest file's SHA-256 is recorded as the
dataset revision: editing the manifest changes the workload identity, and
`compare` will not match runs taken before and after the edit. The subset name
is chosen at preparation time. It defaults to the manifest's file stem, or to
its directory name when the file is called `manifest.jsonl`.

```bash
python -m benchmarks.asrbench prepare manifest --subset my_set \
    --source-dir /data/corpora/my_set/manifest.jsonl -o group_sep=__ -o language=vi
python -m benchmarks.asrbench run --profile smoke --backend oasr_engine --dataset manifest/my_set
python -m benchmarks.asrbench prepare longform --source manifest/my_set --targets 60,120
python -m benchmarks.asrbench validate --backend oasr_engine --dataset manifest/my_set \
    --normalizer basic
```

| Option (`-o KEY=VALUE`) | Meaning |
|---|---|
| `audio_root` | Directory against which relative `audio` paths resolve. By default, the first of the manifest's directory, `$WAV_DIR` and `$AUDIO_DIR` that contains **every** referenced file. A directory containing only some of the files is never used, because that would silently benchmark a subset. The chosen root is recorded. |
| `group_sep` | Derives each item's group from its id (with `__`, `lesson__07` belongs to group `lesson`). Groups mark contiguous speech, which the long-form builder concatenates. A per-line `group` key has the same effect. |
| `language`, `license` | Recorded for provenance; they do not affect measurement. |

Optional per-line keys `speaker` and `group` are preserved; other keys are
ignored. Absolute `audio` paths are used as given. For languages other than
English, run `validate` with `--normalizer basic`. The default `english`
normaliser applies English-specific rules.

### Storage layout

```
$ASRBENCH_DATA_DIR/<name>/<subset>/
    dataset.json      card: sources, revision, checksums, licence, duration statistics,
                      manifest hash, suite version
    manifest.jsonl    {id, audio, duration, sample_rate, text, speaker, group}
    audio/            present only when the adapter writes audio (derived long-form, Earnings-21)
```

### Adding a dataset

Subclass `DatasetAdapter` (`datasets/base.py`) and implement `prepare()` and
`acquisition()`. `prepare()` must raise `DatasetUnavailable`, not download,
unless `download=True`, and returns `write_prepared(...)`. Register the class
in `DATASETS` (`datasets/__init__.py`). Use the shared `download()` helper,
which handles mirrors, `.part` files and MD5/SHA-256 verification, and
`safe_extract()` for archives.

## Workloads

A workload specification in a profile, resolved against a prepared dataset,
yields an explicit list of items. Each item is a span of one audio file, so a
fixed-duration crop is an offset and a length.

| Field | Meaning |
|---|---|
| `dataset` | `name/subset`. In built-in profiles, `${primary}` and `longform/${primary_slug}` expand to the profile's primary dataset. |
| `sampling` | `stratified` (an equal share per duration bucket; a bucket's shortfall is redistributed and recorded), `random`, `first` or `all` |
| `num_items`, `duration_range` | Item count, and the `[low, high)` duration filter in seconds (`null` for no upper bound) |
| `buckets` | Duration buckets. Default: `very_short` < 2 s, `short` 2–5 s, `typical` 5–15 s, `medium` 15–30 s, `long` 30–120 s, `longform` ≥ 120 s. The 30 s boundary is Whisper's input window. |
| `crop_s` | Fixed input durations: real speech cropped to exactly these lengths at seeded offsets. Long sources yield several crops. Crops carry no transcript. |
| `order` | `shuffled` (seeded), `manifest`, `duration_asc` or `duration_desc` |
| `require_text` | Select only items with a reference transcript |
| `seed` | Seed from which every random choice is derived |
| `manifest` | A previously saved workload file. It pins the exact inputs, and the selection fields are ignored. |

**Determinism.** Every random choice draws from
`random.Random(f"{seed}:{purpose}")`. CPython seeds string inputs through
SHA-512, so selection is independent of `PYTHONHASHSEED`.

**Identity.** The workload id is a hash of *what* was selected (the dataset
manifest hash plus each item's id, offset and duration, in order), not of where
the files are stored. Two machines with the same data therefore produce the
same id, and `compare` only matches results whose workload ids are equal.

## Profiles

| Profile | Purpose | Datasets required |
|---|---|---|
| `smoke` | Verifies that every benchmark kind works against a real backend at toy sizes. Typically under a minute on one GPU, including start-up. | `librispeech/test-clean`, or any dataset via `--dataset` |
| `standard` | Regular evaluation: duration buckets, fixed-duration latency (1–30 s), batch sizes 1–32 (uniform and mixed), throughput ×3, closed-loop load at concurrency 1–64, open-loop load at 25–90 % of measured capacity, paced streaming at 160/320/640 ms chunks with 1/8/32 sessions, long-form. | `librispeech/test-clean` and `longform/librispeech-test-clean` |
| `full` | Release evaluation: wider sweeps and more repetitions, open-loop load up to 100 % of capacity, streaming up to 128 sessions, unpaced streaming, real long-form and five-minute live sessions. | As `standard`, plus `earnings21/test` |
| `selftest` | Continuous integration of the suite itself, on generated audio and the mock backend. | `synthetic/default` |

```bash
python -m benchmarks.asrbench profiles                                   # list
python -m benchmarks.asrbench profiles standard --dataset ljspeech/all   # print resolved
python -m benchmarks.asrbench run --profile standard --backend oasr_server \
    -o dtype=float16 -o max_batch_size=64 --tag main
python -m benchmarks.asrbench run --profile smoke --backend oasr_engine \
    --only single,batch --batch-sizes 1 8 32 --repetitions 3
python -m benchmarks.asrbench run --profile my_profile.yaml --backend openai_http \
    -o base_url=http://127.0.0.1:8000 -o model=openai/whisper-large-v3 \
    --set defaults.timeout_s=300
```

**Overrides** apply in a fixed order:

1. `--dataset` replaces the primary dataset.
2. Sweep flags: `--batch-sizes`, `--concurrency`, `--rates` (switches `load`
   to open loop), `--chunk-ms`, `--sessions`, `--repetitions`, `--warmup`,
   `--timeout-s`.
3. Workload flags: `--num-items`, `--duration-range`, `--seed`.
4. Generic assignments: `--set section.field=value`, where the section is
   `defaults`, `workloads.<name>` or a benchmark name.
5. Selection: `--only` and `--skip`, by benchmark name or kind.

**Profile files** have five top-level keys: `name`, `description`, `defaults`
(`primary`, `seed`, `sample_rate`, `timeout_s`, `monitor_interval_s`),
`workloads` (name to specification) and `benchmarks` (a list of
`{name, kind, workload, ...parameters}`). Unknown parameters are rejected.

```yaml
name: latency_only
description: Single-request latency over duration buckets.
defaults: {primary: librispeech/test-clean, seed: 0, timeout_s: 60}
workloads:
  utterances: {dataset: "${primary}", sampling: stratified, num_items: 200}
benchmarks:
  - {name: single, kind: single, workload: utterances, repetitions: 3}
```

## Benchmark kinds

Each sweep point of a benchmark becomes one result with its own measured
window.

| Kind | Protocol | Wall time used for `rtfx` | Key outputs |
|---|---|---|---|
| `cold_start` | The first request after initialisation, then the same request warm. Runs first in its service-mode group and warns if anything ran before it. | — | `init_s`, `first_request_ms`, `warm_request_ms` |
| `single` | One request in flight at a time; `repetitions` passes over the items | Sum of service times | Latency distribution, per-bucket latency, per-request `rtf` |
| `batch` | One native batch call of exactly *B* inputs per batch. Composition is `uniform` (similar lengths) or `mixed` (seeded); the workload is cycled to fill whole batches. | Sum of batch spans | `batch_latency_ms`, per-row completion, `padding_ratio`, allocator peak |
| `throughput` | All items submitted at once; the backend schedules them | First submission to last completion | `rtfx`, `throughput.req_per_s`, dispersion across repetitions |
| `load` (closed) | *N* clients, each sending its next request when the previous completes (optional think time) | Span | Throughput at concurrency *N*; latency is service time under load |
| `load` (open) | Requests released on a seeded Poisson or constant schedule. Rates are absolute (`rates`) or fractions of the throughput measured earlier in the same run (`rate_fractions`, `capacity_from`). | Span | Latency from the **scheduled** time, `send_lag_ms`, `achieved_ratio`, queue wait |
| `longform` | `single` over long recordings, warmed on the shortest item. Reported `unsupported` when the backend bounds input length (`allow_bounded: true` overrides). | Sum of service times | `rtfx`, memory over the window |
| `streaming` | Sessions paced like a live microphone (or unpaced, for capacity); `sessions` concurrent × `rounds` | Session span (unpaced only) | `first_partial_ms`, `final_latency_ms`, `emission_delay_ms`, `late_chunk_pct` |

Batch size and concurrency are measured separately. `batch` requires a native
batch API; concurrent single requests are measured by `load` and never
relabelled as a batch. A batch width the backend cannot serve is reported
`unsupported`, never silently reduced.

## Backends

| Backend | Description | Batch | Concurrent | Streaming |
|---|---|---|---|---|
| `oasr_engine` | `ASREngine` in process. Uses page-locked audio buffers as the server does, and reports engine queue wait, observed batch widths and per-stage host time. | Yes | Yes | Native |
| `oasr_server` | `oasr-server`, started by the suite or attached to a running instance, over HTTP/WebSocket or gRPC. Monitors the server process and scrapes `/metrics` per window. | No | Yes | Native on a streaming-mode server; buffered on an offline-mode server |
| `openai_http` | Any server exposing the OpenAI-compatible `/v1/audio/transcriptions` route (for example vLLM or faster-whisper-server), optionally with a realtime WebSocket | No | Yes | As declared (`-o realtime=true -o streaming_mode=native\|buffered`) |
| `transformers` | Hugging Face `pipeline("automatic-speech-recognition")`; the reference implementation for drift checks. `chunk_length_s` enables long-form. | Yes | No | No |
| `mock` | A simulated dynamic-batching server for the suite's own tests; flagged as simulated in every output | Yes | Yes | Native |

Backend options are passed as `-o key=value` and parsed as YAML, for example
`-o dtype=float16`, `-o "engine={max_wait_time: 0.05}"` or
`-o "server_args=[--vad-mode, segment]"`.
`python -m benchmarks.asrbench backends -v` lists every option.

### Benchmarking `oasr-server`

The `oasr_server` backend measures what a client of the service observes: the
transport, audio decoding, the dispatcher and the engine.

**Suite-managed server (default).** The suite starts one `oasr-server` per
service mode the profile needs, on free local ports, with admission limits
raised so that admission is not what a load test measures. Offline kinds and
load tests run against `--service-mode offline`; the streaming kind runs
against `--service-mode streaming`. Server settings are passed as options
(`-o dtype=`, `-o max_batch_size=`, `-o chunk_size=`, `-o decode_method=`,
`-o "server_args=[...]"`). The server log is written to
`$TMPDIR/asrbench-oasr-server-<pid>-<mode>.log`.

```bash
python -m benchmarks.asrbench run --profile smoke --backend oasr_server -o dtype=float16
```

**Attached server.** To measure a server started separately (custom flags,
another host, a deployment), pass its address with `-o base_url=`. A server
runs in one service mode. The suite reads that mode from `/v1/models` (or,
if absent, from the realtime handshake) and declares capabilities from it, not
from the mode the profile requested. Against an offline-mode server, streaming
results are labelled `buffered`; against a streaming-mode server, offline
kinds are reported `unsupported`. The served model and dtype are recorded from
`/v1/models`. Pass `-o server_pid=` so that the resource monitor samples the
server process; otherwise only the client is sampled.

```bash
oasr-server --ckpt-dir "$CKPT_DIR" --service-mode offline --dtype float16 \
    --http-bind 127.0.0.1:8080 --grpc-bind 127.0.0.1:50051 &
python -m benchmarks.asrbench run --profile smoke --backend oasr_server \
    -o base_url=http://127.0.0.1:8080 -o server_pid=$!
```

**Transports.** `oasr-server` serves HTTP and gRPC from one process, so the
transport is an option of the same backend. Run the profile once per transport
and render both runs in one report:

```bash
python -m benchmarks.asrbench run --profile smoke --backend oasr_server \
    -o dtype=float16 -o transport=http --tag http
python -m benchmarks.asrbench run --profile smoke --backend oasr_server \
    -o dtype=float16 -o transport=grpc --tag grpc
python -m benchmarks.asrbench report <http-run-id> <grpc-run-id>
```

| | `transport=http` (default) | `transport=grpc` |
|---|---|---|
| Offline request | `POST /v1/audio/transcriptions`, multipart WAV (PCM16) | Unary `Recognize`; by default the same WAV bytes (`-o grpc_encoding=linear16` or `linear32f` sends headerless PCM) |
| Streaming session | `/v1/realtime` WebSocket, PCM16 chunks, then commit | Bidirectional `StreamingRecognize`: configuration, PCM16 chunks, then half-close |
| Back-pressure (counted as rejected) | HTTP 429 or 503 | `RESOURCE_EXHAUSTED` |
| Attached server | `-o base_url=` | `-o base_url=` (mode probe and `/metrics`) and `-o grpc_addr=host:port` |

In both transports the first final result ends a streaming session. gRPC
stubs are compiled from `rust/proto/oasr_speech_v1.proto` when the backend
loads, which requires `grpcio` and `grpcio-tools`.

**Service overhead.** To quantify the cost of serving, run the same profile
against `oasr_engine` and `oasr_server` and render both runs in one report.

### Comparability

The suite holds constant what it controls: workload ids, decoded waveforms,
the transport within an adapter family, warm-up, the measurement protocol,
statistics and seeds. Backend options pin what each framework controls:
checkpoint, precision, decode method, batch width and chunk size. For an
attached server, the configuration it reports about itself is used instead.
These are recorded per run. `compare` refuses runs whose hardware or backend
configuration differ, including runs over different transports, unless told
the change is deliberate (`--allow-hardware-change`, `--allow-config-change`).
Use `report` to view different configurations side by side.

### Adding a backend

Subclass `Backend` (`backends/base.py`) and implement `load`, `info`,
`capabilities` and `transcribe`. Implement `transcribe_batch` and
`open_stream` only when the framework provides them natively, and register the
class in `backends/__init__.py`. Guidelines:

- Declare capabilities accurately. Anything undeclared is reported
  `unsupported`, which is the correct outcome. Never emulate batching with
  sequential calls or streaming with repeated offline calls.
- Encode request payloads in `prepare()` and `prepare_chunk()`, which run
  before timing starts.
- Take timestamps at the point where the result is observed.
- Override `mode_unsupported()` so the runner can skip loading a model for a
  mode the backend cannot serve.

## Measurement methodology

### Timing

All timestamps come from `time.perf_counter()` and are stored relative to the
run's origin. A backend records each timestamp at the moment it observes the
event (the last byte of a response, the engine step that emitted the output),
never after an `await` at which another coroutine's work could be billed to
it.

### Completion

A request is complete when its transcript is available as host-side text,
which implies the device work behind it has finished. In-process backends
also synchronise the device at benchmark boundaries. GPU submission time is
never reported as latency.

### Warm-up

By default (`warmup: all`), a full pass over exactly the items or batches to
be measured precedes the measured phase. For `load` it is
`warmup_requests: all` at the level's own concurrency; for `streaming` it is
`warmup: round`, one concurrent round of unpaced sessions. Engines that
capture CUDA graphs per input shape pay that cost on a shape's first
occurrence: on an OASR Conformer, about 27 ms for the first request of a shape
against about 1.8 ms once captured. A two-request warm-up left captures inside
the measured window and inflated single-request p50 roughly fifteen-fold.
Warm-up records are stored with `phase: warmup` and excluded from summaries.

### Measurement windows

Allocator peaks and server counters are reset when the measured phase starts.
Resource samples, allocator peaks and server metrics are attributed to that
window only.

### Open loop and coordinated omission

Open-loop latency is measured from each request's *scheduled* send time, so
a stalled server is charged for every request that should have been sent
during the stall. `send_lag_ms` (actual minus scheduled submission) shows when
the client, rather than the schedule, set the pace; a p99 above 50 ms raises a
warning. With an in-process backend, send lag reflects engine steps occupying
the shared event loop. Served requests wait for the same step, so the reported
latency remains correct. Closed-loop latency is service time under *N*
concurrent clients by construction; use the open loop for latency objectives.

### Failures

Percentiles are computed over successful requests. Timeouts, failures and
back-pressure rejections are counted separately and reported as rates
(`error_rate`, `timeout_rate`), so the timeout setting cannot shape the tail
percentiles. A timed-out request is cancelled on the backend. `empty_rate`
detects a backend that responds quickly with empty transcripts.

### Percentiles

Percentiles use the nearest-rank rule over `n − 1` positions (numpy's
`method="nearest"`), the rule `benchmarks/core/metrics.py` uses. Every
distribution reports its sample count `n`. A tail percentile with too few
samples to differ from the maximum (p99 requires 100) is flagged in the run
and is not judged by `compare`.

### Repetitions

`repetitions` repeats the measured phase. Latency distributions pool all
repetitions. Aggregates (`rtfx`, `throughput.req_per_s`) additionally report
`rep_mean`, `rep_std`, `rep_min` and `rep_max`, and `compare` widens its
threshold by the observed dispersion.

### Configuration source

All sizing comes from the profile, and the resolved profile is stored in
`run.json`, so a run can be reproduced from its own record. The environment
supplies only paths (checkpoints and data directories). The repository's older
harnesses take sizing defaults from `.env`, where an unset variable silently
changes the working point; this suite deliberately does not.

## Streaming methodology

Chunk *k* covers audio `[k·c, (k+1)·c)` and is due at `t_audio0 + (k+1)·c`,
the moment a live microphone would have produced it. Due times are absolute,
so pacing does not drift as it would with a fixed sleep after each send. A
chunk sent late is recorded as late (`chunk_lag_ms`, `late_chunk_pct`, with a
default tolerance of `max(10 ms, 10 % of c)`) rather than shifting every later
chunk. Concurrent sessions start staggered across one chunk period, as a
population of independent clients would.

| Metric | Definition |
|---|---|
| `first_partial_ms` | First non-empty transcript event minus audio start; includes at least one chunk of algorithmic delay |
| `final_latency_ms` | Final transcript minus end of audio (paced), i.e. how long after the speaker stops the result settles; unpaced, minus the last chunk sent |
| `final_after_last_chunk_ms` | Final transcript minus the last chunk actually sent: the server's share once the client has delivered everything |
| `emission_delay_ms` | Per word, the first event whose normalised prefix agrees with the final transcript up to that word, minus the word's end time. Requires word timestamps (`word_timestamps: true`); paced sessions only |
| `partial_interval_ms`, `partials_per_session` | Cadence of partial results |
| `rtfx` | Unpaced sessions only: total audio divided by session span |

Streaming modes are explicit: `native` (incremental decoding with partial
results while audio arrives), `buffered` (decoded after the stream ends) and
`segmented` (per detected segment). Each result records its mode, and
`compare` does not diff results with different modes. A backend without
streaming reports `unsupported`.

## Metrics reference

`benchmarks/asrbench/metrics.py` (`METRICS`) defines the unit and preferred
direction of every metric family; `report` and `compare` read it rather than
inferring direction. Summaries are flat `{"family.stat": value}` mappings,
where `.n` is the sample count.

| Family | Unit | Preferred direction |
|---|---|---|
| `latency_ms.*`, `service_ms.*`, `batch_latency_ms.*`, `bucket.<name>.latency_ms.*` | ms | Lower |
| `rtfx` (aggregate audio ÷ wall time), `throughput.req_per_s`, `throughput.audio_s_per_s` | ×, req/s | Higher |
| `rtf.*` (per-request latency ÷ audio duration) | s/s | Lower |
| `init_s`, `first_request_ms`, `warm_request_ms` | s, ms | Lower |
| `error_rate`, `timeout_rate`, `empty_rate` | ratio | Lower |
| `requests.*`, `sessions.*` | count | Informational |
| `send_lag_ms.*`, `offered_rate` | ms, req/s | Informational |
| `achieved_ratio` | ratio | Higher |
| `queue_wait_ms.*`, `batch_size_observed.*` | ms, count | Lower / informational |
| `first_partial_ms.*`, `final_latency_ms.*`, `emission_delay_ms.*`, `late_chunk_pct` | ms, % | Lower |
| `cpu_pct.*`, `gpu_util_pct.*`, `gpu_power_w.*` | %, W | Informational |
| `rss_mb.*`, `gpu_mem_mb.*`, `gpu_alloc_mb.max`, `energy_j`, `energy_j_per_audio_h` | MiB, J | Lower |

## Resource monitoring

A separate process (`python -m benchmarks.asrbench.monitor`) samples the
target process through `psutil` (CPU, where 100 % is one core, and resident
memory including child processes) and the GPU through NVML (utilisation,
memory, power and the driver's cumulative energy counter). It runs out of
process because a sampling thread would contend for the GIL inside an
interpreter-bound engine loop and perturb the measurement. Samples carry
`time.monotonic()` timestamps and are attributed to measurement windows
afterwards.

- A window shorter than one sampling period is widened by one period on each
  side, so peaks are still captured; `.n` shows how many samples support a
  value.
- Energy is reported only for windows of at least one second.
- A missing probe (no `psutil`, a non-NVIDIA device, no driver library in a
  container) yields absent metrics, never zeros, and the run records which
  probes were active.
- In-process backends also report the PyTorch allocator peak
  (`gpu_alloc_mb.max`). Server backends are monitored through the server's
  process id.
- The sampling interval is the profile's `monitor_interval_s`
  (`--monitor-interval`); `--no-monitor` disables monitoring.

**Environment capture** (`environment.py`) records the git commit and dirty
state, CPU model and core count, memory, every NVML-visible GPU (name, UUID,
memory, driver, compute capability, power limit, maximum SM clock), package
versions, `OASR_*` settings and JIT cache hashes. It does not import PyTorch,
so the client never holds a CUDA context on a device a server is using.

## Correctness validation

```bash
python -m benchmarks.asrbench validate --backend oasr_engine \
    --dataset librispeech/test-clean --num-items 200 --max-wer 8

# Drift check: validate a reference implementation of the same checkpoint,
# then compare OASR's hypotheses against it.
python -m benchmarks.asrbench validate --backend transformers -o model="$WHISPER_CKPT" \
    -o "generate_kwargs={language: en, task: transcribe}" --dataset ljspeech/all --num-items 200
python -m benchmarks.asrbench validate --backend oasr_engine -o ckpt_dir="$WHISPER_CKPT" \
    --dataset ljspeech/all --num-items 200 --reference <reference-validation-id> \
    --max-changed 20 --max-wer-delta 0.5
```

Validation is untimed and uses the fastest path the backend offers (native
batches, concurrency, or sequential requests). It writes `validation.json` and
`predictions.jsonl` (`{id, ref, hyp, error}`) under
`<results root>/validation/`, and reports:

- corpus WER (`oasr.testing.wer`; Whisper's English normaliser by default,
  `--normalizer basic` or `none` otherwise);
- the empty-transcript rate (default limit 0) and failed requests;
- against a reference: the share of utterances whose normalised hypothesis
  changed, the WER of the new hypotheses against the reference hypotheses
  (`drift_wer_pct`), and the WER delta.

Any configured check that fails produces exit code 1. A fixed manifest in the
CI accuracy gate's format is also accepted (`--manifest
benchmarks/manifests/ljspeech_200.jsonl --audio-root "$WAV_DIR"`). `compare`
accepts two validation reports and checks `wer_pct` (default limit +0.3
points).

## Results

### Location

Runs are written to the results root: `$ASRBENCH_RESULTS_DIR` if set,
otherwise `$XDG_DATA_HOME/oasr/asrbench/results` (by default
`~/.local/share/oasr/asrbench/results`). This location is persistent and
outside any source checkout, so results survive a fresh clone and never appear
in version control. `run --out DIR` writes a single run elsewhere. `run` prints
the run directory on completion, and `runs` lists stored runs with their ids,
profiles, backends, tags and completion state.

### Layout

```
<results root>/
    <YYYYmmdd-HHMMSS>_<profile>_<backend>_<run-id>/
        run.json           schema and suite versions, environment, resolved profile,
                           backend options and per-mode info/capabilities/init_s,
                           workloads, results
        requests.jsonl     one line per request: result_id, rep, phase, seq, item,
                           duration_s, bucket, batch, worker, t_sched, t_submit,
                           t_first, t_done, status, error, text
        sessions.jsonl     one line per streaming session: chunk_ms, paced, t_audio0,
                           chunk_due[], chunk_sent[], events[], words
        resources.jsonl    monitor samples (resources.<pid>.jsonl for further instances)
        workloads/*.json   every resolved workload, item by item
    validation/<stamp>_<backend>_<id>/    validate output
    baselines/<name>/                     save-baseline copies
```

### Result records

Each result carries `result_id`, `benchmark`, `kind`, `mode`, `point` (its
sweep coordinates), `workload_id`, `params`, `status` (`ok`, `unsupported` or
`failed`, with a `reason`), `window`, `summary`, `server` and `warnings`.
`run.json` is rewritten after every point, so an interrupted run keeps its
completed results.

`run.json` carries a `schema_version`. Readers access fields by name, and a
run written with a newer schema is refused rather than misread.

`recompute RUN` re-derives every summary from the raw records with the same
functions used during the run (`analysis.py`) and lists any disagreement. It
is the audit path for stored numbers, and the way to compute a metric added
after a run was taken. `save-baseline RUN --name NAME` preserves a run under
`<results root>/baselines/NAME`.

## Reports

`report RUN [RUN ...] [--out DIR] [--plots]` renders stored runs only; it
never executes a measurement. It writes:

- `report.md`: the runs and their configurations; workloads; one table per
  benchmark kind, where additional runs show deltas against the first run,
  marked better or worse according to each metric's direction; latency by
  duration bucket; resources; unsupported and failed results with reasons;
  warnings; and metric definitions.
- `summary.csv`: one row per run and result, one column per metric, for
  external tools.
- With `--plots` (requires `matplotlib`): throughput and latency against batch
  size and concurrency, latency against throughput, streaming latency against
  session count, and latency by duration bucket.

Rows are aligned only when their result keys match. Reports of simulated runs
carry a banner. The default output directory is `<first run>/report`.

## Regression gating

```bash
python -m benchmarks.asrbench compare BASELINE CANDIDATE \
    [--thresholds thresholds.yaml] [--json summary.json] [--allow-missing] [-v]
```

| Exit code | Meaning |
|---|---|
| 0 | No regression |
| 1 | A metric regressed, a point failed, or the candidate lost a result or metric the baseline measured |
| 2 | Not comparable (different hardware, backend configuration or workloads; nothing matched; unreadable input) |

What the candidate did not measure fails the gate as well as what got worse;
otherwise a candidate that crashes passes:

- **Failed points.** Any point that failed in the candidate, and any point the
  baseline measured that the candidate reports `unsupported`, are listed under
  `failed` and exit 1. A point that failed in the baseline but not in the
  candidate has nothing to be judged against, so it is listed under `unjudged`.
- **Missing results and metrics.** A result the baseline measured that is
  absent from the candidate, or a thresholded metric the baseline has that the
  candidate lacks (absent, `null` or NaN), is listed under `missing` and the
  gate exits 1. An interrupted run is the usual cause, and `compare` notes a
  run with no `finished_utc`. `--allow-missing` lists them without failing,
  for a deliberate subset such as a `--only` run or a validation without
  `--reference`.
- **Telemetry.** A missing resource metric (process memory, NVML, the
  allocator) is listed under `unjudged` and never fails, because its absence
  means the sampler did not run.

`run` itself exits 1 when any point failed. It still records the failure and
continues with the next point, and it prints the run directory first so the
run can be inspected or compared.

Results are matched by key (`benchmark[point]@workload_id`). A metric
regresses when it moves in its worse direction by more than **all** of:

- `max_regression_pct` of the baseline value;
- `min_abs`, an absolute floor that suppresses changes too small to matter;
- `noise_sigmas` × the combined across-repetition standard deviation, when
  both runs recorded one.

`max_regression_abs` is an absolute limit suited to rates. A rule can exclude
sweep points with `unless`. The default `rtfx` and throughput rules skip
open-loop points, where throughput below saturation follows the offered rate;
latency, error rates and `achieved_ratio` carry the signal there. The defaults
(`regression.DEFAULT_THRESHOLDS`) are a starting point for a quiet machine,
not acceptance criteria. A CI gate should maintain its own thresholds file:

```yaml
noise_sigmas: 2.0
rules:
  - {metric: latency_ms.p50, max_regression_pct: 10, min_abs: 1.0}
  - {metric: latency_ms.p99, max_regression_pct: 20, min_abs: 5.0}
  - {metric: rtfx, max_regression_pct: 10, unless: {mode: open}}
  - {metric: error_rate, max_regression_abs: 0.01}
  - {metric: final_latency_ms.p95, max_regression_pct: 20, min_abs: 20}
```

The JSON summary lists `regressions`, `failed`, `missing`, `improvements`,
`unchanged`, `insufficient_samples`, `unjudged`, `skipped` (streaming-mode
mismatches), results present on only one side, and blocking `problems`. A configuration problem names only
the fields that differ, for example
`served[0].service_mode: 'offline' vs 'streaming'`.

A typical CI flow keeps a baseline per machine and gates each candidate
against it:

```bash
python -m benchmarks.asrbench run --profile standard --backend oasr_engine --tag "$GIT_SHA"
python -m benchmarks.asrbench save-baseline <run-id> --name main      # on the reference build
python -m benchmarks.asrbench compare main <candidate-run-id> --thresholds ci/thresholds.yaml
```

## Limitations

- **Shapes under dynamic batching.** An engine that captures CUDA graphs
  lazily encounters new (batch, length) shapes under open-loop load that no
  warm-up can enumerate. Their capture cost appears in open-loop tail latency,
  as it would in production, until captures saturate. Longer runs amortise
  it.
- **Single-process client.** One asyncio client generates load. At very high
  request rates the client can become the bottleneck; `send_lag_ms` and the
  client-lag warning indicate when this happens.
- **No forced alignment.** Emission delay uses the backend's own word timings
  and is therefore relative to that model's alignment. LibriSpeech provides no
  reference word alignments.
- **Device-wide telemetry.** NVML energy and utilisation cover the whole
  device; on a shared GPU they include other tenants. Utilisation is NVML's
  kernel-active ratio, not SM occupancy.
- **Non-NVIDIA devices** receive CPU and memory telemetry only. Support for
  other accelerators would be a new probe in `monitor.py`.
- **Third-party frameworks** are reached through `openai_http` and
  `transformers`. Native adapters for sherpa-onnx, faster-whisper or NeMo are
  not included; the backend contract describes what one would implement.

## Design rationale

- **A dedicated package rather than extensions to `benchmarks/run.py`.** The
  per-component harnesses (`--family engine|service|accuracy`) lack
  per-request records, deterministic workloads and an open loop, and their
  output format is the established record for engine work. Extending them
  would break comparability with existing results. The suite reuses their
  percentile rule and the CI accuracy gate's WER implementation instead.
- **Profile-owned sizing.** A run records exactly what it measured; see
  [Configuration source](#configuration-source).
- **Full-pass warm-up by default.** Shape-dependent capture costs must be
  outside the measured window; see [Warm-up](#warm-up).
- **Open-loop rates as fractions of measured capacity.** An absolute rate that
  saturates one GPU leaves another idle. Fractions keep a profile portable
  across hardware, in the spirit of MLPerf's Server scenario.
- **Unsupported is a result.** No emulated batching, no simulated streaming,
  no silently reduced batch sizes: every gap is a row with a reason.
- **Correctness outside timed runs.** Scoring never shares a window with
  timing. Performance runs record only `empty_rate`, which costs nothing to
  compute.
- **Out-of-process monitoring.** Keeps telemetry from contending for the GIL
  of an interpreter-bound engine.
- **Results outside the source tree.** Run directories are machine output and
  can be large. Storing them in the user's data directory keeps the checkout
  clean, and `runs` with run-id resolution makes them as easy to reference as
  a local path.
