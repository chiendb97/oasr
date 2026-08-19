# AGENTS.md

**OASR** (Open Automatic Speech Recognition) is a high-performance CUDA inference
framework for ASR. It serves seven encoder architectures across five decode
paradigms — CTC, transducer (RNN-T), AED, non-autoregressive CIF, and speech-LLM
— from one engine, with custom CUDA/CUTLASS kernels exposed to Python via TVM-FFI
JIT compilation and a Rust HTTP/gRPC serving front-end.

- User-facing overview and quick start: [`README.md`](README.md)
- Stable technical documentation: [`docs/`](docs/README.md)
- Point-in-time results, issues and investigations: a gitignored `.artifacts/`
  working directory, never cited from here (see rule 12)

---

> If a `CLAUDE.local.md` file exists alongside this file, read and respect it — it contains developer-specific overrides that supplement this shared guidance.

---

## Rules (read first)

1. **Never edit the engine core to add a variant.** Every extension lands through
   a registry — subclass a base, register under a name, select by configuration.
   There are seven such axes; see [Architecture](#architecture).
2. **Models are built from `oasr.layers`, never from bare `nn.Linear` /
   `nn.LayerNorm` / `nn.Embedding` / `nn.Conv*`.** `tests/test_layer_waist.py`
   enforces this and fails on a newly registered architecture that has no tiny
   config to check.
3. **A missing kernel must be declared, not routed around.** `oasr.layers._backend`
   distinguishes *out of scope* (CPU/fp32), *a declared kernel gap*
   (`KERNEL_GAPS` — or the call raises), and *a performance choice*. A silent
   reroute to torch makes a missing kernel invisible.
4. **The output tensor is the first parameter** of every TVM-FFI launcher.
5. **Do not run `cargo build/test --workspace`**, and run cargo from `rust/`, not
   the repo root. See [Rust workspace](#rust-workspace).
6. **Do not add a `pull_request` trigger to `.github/workflows/test-gpu.yml`** —
   the repo is public and that workflow runs on a self-hosted runner.
7. **Never read a gating environment variable directly in a test.** Declare the
   asset in `tests/assets.py` and gate through `assets.require(...)` /
   `@pytest.mark.requires_assets(...)`, so every skip is counted and reported.
8. **Run the tests and the benchmarks with the checkpoints exported.** Start every
   session with `set -a; source .env; set +a` (copy `.env.example` if you have no
   `.env`); every command in this file assumes it. The dangerous half is silent:
   without the assets the real-checkpoint tests **skip** and the suite still exits
   0, so a green run says nothing about the architecture you changed. A benchmark
   is louder — it refuses (`--ckpt-dir is required`) — but its *sizing* comes from
   `.env` too (`$NUM_UTTERANCES`, `$MAX_BATCH_SIZE`, `$CONCURRENCY`, `$CHUNK_MS`
   are argparse defaults), so a missing one changes the measurement without
   saying so. Then *read* the `external assets:` table pytest prints at the end of
   the run: the asset covering your change must say `ok`, not `MISSING`. Use
   `--strict-assets` to make a missing one fail instead of skip. A kernel, layer
   or model change is not validated until it has been run this way.
9. **Per-decode-family knobs go on the strategy's `options_cls`, not on
   `EngineConfig`.** Engine-level knobs (tick budget, decode slots, KV budget)
   stay on `EngineConfig`. Per-*request* knobs go on `DecodingOptions`, and one
   that changes *what is decoded* or that a caller will look for in the response
   must be declared (`selective_options` / `word_timing_modes`) so a family
   without it rejects rather than ignores.
10. **Never pass `-inf` as `attn_bias`.** Use a large finite mask floor —
    mathematically identical, and the fused kernel is inaccurate above a
    moderate finite bias magnitude.
11. **Never branch kernel dispatch on CUDA-graph capture state.** A
    capture-dependent branch makes the graph pick a different kernel than eager,
    and the resulting one-ulp difference has changed decoded tokens.
12. **Keep transient material out of `CLAUDE.md` and `README.md`, and do not
    cite `.artifacts/` from them.** Benchmark numbers, known issues,
    investigations and experiment results go in the gitignored `.artifacts/`;
    a fresh clone does not have those files, so a pointer to one is a dead link.
13. **Do not commit or push unless asked.** Create the branch and the files, then
    hand off.

---

## Common commands

Every test and benchmark row below assumes the checkpoints are exported first
(rule 8). `pytest` does **not** read `.env` itself.

```bash
set -a; source .env; set +a          # cp .env.example .env first, and edit the paths
```

| Task | Command |
|---|---|
| Editable install | `pip install -e .` |
| Install everything | `pip install -e ".[all]"` |
| Run all Python tests | `pytest tests/` |
| Run them the way CI does | `pytest $(python ci/gpu_suites.py --paths <family>) --strict-assets` (`--list` for the family names) |
| One test file / function | `pytest tests/test_conv.py::TestDepthwiseConv1D -v` |
| Skip slow tests | `pytest tests/ -m "not slow"` |
| Engine concurrency stress (opt-in) | `pytest tests/test_engine_concurrent.py -m concurrent -v` |
| Format Python | `black oasr/ tests/ benchmarks/ scripts/ ci/` then `isort` the same paths |
| Lint Python | `ruff check oasr/ tests/ benchmarks/ scripts/ ci/` |
| Type check (ratchet) | `python scripts/mypy_ratchet.py` |
| Format C++/CUDA | `clang-format -i csrc/**/*.cu csrc/**/*.h csrc/**/*.cpp` |
| Rust build / test / lint | `cd rust && cargo build --release && cargo test && cargo clippy --all-targets -- -D warnings` |
| Serve a checkpoint | `oasr-server --ckpt-dir <dir> --service-mode offline --http-bind 127.0.0.1:8080 --grpc-bind 127.0.0.1:50051` |
| Convert a checkpoint | `oasr-convert <src> <dst>` |
| Transcribe (server / in-process) | `oasr transcribe audio.mp3` · `oasr transcribe audio.mp3 --ckpt-dir <dir>` |
| Word timestamps | `oasr transcribe audio.mp3 --response-format verbose_json --timestamp-granularity word` |
| Kernel benchmark | `python benchmarks/oasr_benchmark.py --list` |
| Engine / service / accuracy benchmark | `python benchmarks/bench_{engine,service,accuracy}.py` — flags default from `.env` |
| WER for one architecture | `python benchmarks/bench_accuracy.py --ckpt-dir $CKPT_DIR --manifest benchmarks/manifests/ljspeech_200.jsonl --audio-root $WAV_DIR` |
| …for an explicit-only one | as above with `--ckpt-dir $TRANSDUCER_CKPT --architecture transducer` |

---

## Installation & build

```bash
git submodule update --init          # provides 3rdparty/cutlass — CMake does not
pip install -e .                     # editable install
CUDA_ARCHITECTURES=80 pip install -e .
pip install -e ".[all]"              # audio + hub + tokenizers + attention + serving
cd rust && cargo build --release     # optional standalone oasr-server binary
```

`pip install` produces three artifacts inside the package:

| Artifact | Built by | Contains |
|---|---|---|
| `oasr/_C.so` | CMake + pybind11 | CPU-side CTC/WFST decoders, the post-decode alignment pass, legacy enums |
| `oasr/_core.so` | setuptools-rust | `oasr._core` — the Rust serving core |
| `oasr-server` | console script | forwards `sys.argv` into `oasr._core.serve` |

CUDA kernels are **not** built here: they JIT-compile on first *call* via TVM-FFI
and cache in `~/.cache/oasr/jit/`. That is why `import oasr` works on a machine
with no compiled extension, and why `test-cpu.yml` builds nothing.

**Build requirements:** CUDA ≥ 11.8, CMake ≥ 3.18, Python ≥ 3.10, a Rust
toolchain and `protobuf-compiler` on `PATH`, C++17. With `--no-build-isolation`,
`pip install "setuptools-rust>=1.10"` first.

On a filesystem whose I/O semantics break rustc's autocfg probes (an NFS mount,
for example), redirect cargo's target directory — via `CARGO_TARGET_DIR`, or via
a **gitignored** `.cargo/config.toml`. Two are needed, because cargo is invoked
from two places: `rust/.cargo/config.toml` for `cd rust && cargo build`, and one
at the repo root for `pip install`, which runs cargo from there.

Extras: `[audio]`, `[hub]` (Hub download + native checkpoint I/O), `[tokenizers]`
(kinds beyond `symbol_table`), `[attention]` (CuTeDSL fused attention; SDPA
fallback without it), `[serving]` (benchmark client libs), `[wfst]` (k2 — offline
graph export only, never at decode time).

Full details: [`docs/kernels.md`](docs/kernels.md).

### Rust workspace

```bash
cd rust                                          # always — not the repo root
cargo build --release                            # default-members (incl. oasr-server)
cargo test                                       # default-members
cargo test -p oasr-asr                           # one crate
cargo fmt                                        # before committing Rust
cargo clippy --all-targets -- -D warnings
cargo clippy -p oasr-core --lib -- -D warnings   # excluded from default-members
```

**Never `--workspace`.** `oasr-core` enables `pyo3/extension-module` while
`oasr-server` enables `pyo3/auto-initialize`; those features are mutually
exclusive and Cargo unifies them per build, so one invocation covering both fails
to compile. `oasr-core` is therefore excluded from `default-members` and is built
by setuptools-rust on its own (`pip install`, or
`python setup.py build_rust --inplace`).

---

## Architecture

```
Request → InputProcessor (GPU fbank) → Scheduler (BatchingPolicy + PartitionPolicy)
   → ModelRunner
        ├─ offline:   model.forward_offline / forward_offline_packed → enc_out
        └─ streaming: StreamingEncoderBackend.forward_step           → enc_out
   → OutputProcessor → DecodeStrategy (+ Detokenizer) → RequestOutput
```

Kernels are layered strictly:

```
Python functional API (oasr/functionals/<family>.py)  — @oasr_api
    └── JIT generator (oasr/jit/<family>.py) → JitSpec / JinjaJitSpec
            └── TVM-FFI JIT binding (csrc/<family>_jit_binding.cu)
                    └── TVM-FFI launcher (csrc/<family>.cu)
                            └── Pure CUDA kernels (include/oasr/<family>.cuh)
```

### The seven extension axes

Each is a registry. Adding a variant means subclass + register — no engine edits.

| Axis | Base class | Selected by |
|---|---|---|
| Encoder architecture | `oasr.models.BaseAsrModel` / `BaseEncoder` | native format → `architecture=` → `CheckpointConverter.detect` |
| Checkpoint format | `CheckpointConverter` (`oasr/models/converter.py`) | `detect()`, ranked by `detect_specificity` |
| Decode family | `oasr.engine.decode.DecodeStrategy` | `EngineConfig.decode_method` (validated against `model.capabilities`) |
| Streaming runtime | `oasr.engine.streaming_backend.StreamingEncoderBackend` | `model.encoder.streaming_kind` |
| Batching | `oasr.engine.batching.BatchingPolicy` / `PartitionPolicy` | `EngineConfig.schedule_policy` |
| Tokenizer | `oasr.tokenizers.Tokenizer` | converter-emitted `TokenizerSpec.kind` |
| Feature frontend | `oasr.features.ExtractorSpec` | `FeatureConfig.feature_type`, from `FeatureSpec` |

Orthogonal to all seven is the **layer waist**: `oasr.layers` is what every
architecture is built from, so a kernel improvement, CUDA-graph capture or a
future quantized path applies to all of them.

[`docs/architecture.md`](docs/architecture.md) is the authoritative map, with the
extension cookbook for each axis.

---

## Key files

| Path | Role |
|---|---|
| `oasr/engine/engine.py` | `ASREngine` — the step loop; offline + streaming in one pool |
| `oasr/engine/config.py` | `EngineConfig` — every engine-level knob |
| `oasr/engine/scheduler.py` | Batch selection and partition, starvation bounds |
| `oasr/engine/decode/` | Decode strategies (`ctc_gpu`, `ctc_wfst`, `transducer`, `aed`, `llm`, `paraformer`, `rescoring`) + `options.py` |
| `oasr/engine/decode/{alignment,ctc_align,attention_align}.py` | Word timings: the shared frames→words half, and the two per-family aligners |
| `oasr/engine/streaming_backend/` | `PagedStreamingBackend`, `StatefulStreamingBackend` |
| `oasr/engine/graph_cache.py` | CUDA-graph capture of the steady-state streaming encoder |
| `oasr/engine/memory.py` | VRAM-aware capacity derivation (paged pool, AR decoder KV) |
| `oasr/engine/decoder_graph.py` | CUDA-graph capture of one AR decoder step (needs paged decoder KV) |
| `oasr/cache/decoder_state.py` | Per-row decoder KV — dense capacity buffers and paged — that both AR decoder surfaces thread |
| `oasr/models/base.py` | `BaseAsrModel` / `BaseEncoder` / `CacheSpec` / `LoadReport` |
| `oasr/models/interfaces.py` | `CAPABILITIES` — what each decode family requires of a model |
| `oasr/models/registry.py` | `register_model`, `build_model_from_checkpoint`, entry-point discovery |
| `oasr/layers/` | The narrow waist; `_backend.py` holds the routing rules and `KERNEL_GAPS` |
| `oasr/jit/core.py`, `oasr/jit/env.py` | JIT specs, nvcc flags, the cache key |
| `oasr/functionals/gemm.py`, `oasr/functionals/attention.py` | The two families with shape-aware routing |
| `csrc/tvm_ffi_utils.h` | DLPack dispatch + the validation macros every launcher uses |
| `csrc/alignment/` | The post-decode alignment pass and the beam read-back, in C++ (`_C.alignment`) |
| `rust/crates/oasr-engine-client/` | The GIL-owning dispatcher thread |
| `rust/crates/oasr-serve/` | Mode-agnostic serving core shared by binary and extension |
| `rust/crates/oasr-server-http/src/{openai,realtime}.rs` | The OpenAI-compatible routes and the `/v1/realtime` WebSocket |
| `rust/crates/oasr-asr/src/{codec,encoding}.rs` | Container decoding (symphonia) + the one `encoding=` parser both front-ends share |
| `rust/crates/oasr-metrics/` | Every metric's name, kind, unit, help and **buckets**, declared once; `install_recorder` is the only place the exporter is built |
| `oasr/engine/metrics.py` | Engine-scope collection, drained by the dispatcher and replayed into the exporter |
| `oasr/client.py`, `oasr/cli.py` | The Python client and the `oasr` command |
| `tests/assets.py` | The single declaration point for every external test asset |
| `ci/gpu_suites.py` | The per-family GPU test split, shared by both GPU backends |

---

## Design patterns

| Pattern | Where | Note |
|---|---|---|
| Registry per extension axis | `oasr/models`, `oasr/engine/{decode,streaming_backend,batching}`, `oasr/tokenizers`, `oasr/features` | Subclass + register; selection is by configuration |
| Narrow waist | `oasr/layers` | Every architecture composes the same layers; each layer owns a kernel path **and** a torch path |
| Config / template / dispatch split | `include/oasr/<family>/` | `cutlass_*_configs.h`, `*_cutlass_template.h`, `*_cutlass.h` |
| JIT on first call | `oasr/jit/` | Cache key covers sources, `include/`, nvcc flags and the CUTLASS version stamp |
| Checkpoint-derived specs | `TokenizerSpec`, `FeatureSpec`, `DecodingDefaults` | Converters emit them; the engine materializes config from them |
| Declared capability | `CAPABILITIES` + `require_capability` | An unserviceable checkpoint fails at engine construction, naming the missing members |
| Declared cache | `streaming_state_specs`, `fixed_attention_window`, `streaming_geometry`, `ExtractorSpec.framing` | A new streaming cache is data, not a new manager |
| Per-family options | `DecodeStrategy.options_cls` + `--decode-option k=v` | Adding a family needs no new CLI flag and no `EngineConfig` field |
| Declared alignment | `word_timing_modes` + `TokenAlignment` → `word_timings` | Per-family *how*, shared *what happens next*; a family that cannot align says so per mode |
| Counted gaps | `KERNEL_GAPS`, `format_gap_report()`, `rule_miss_report()` | What is missing is measurable, not invisible |

---

## Anti-patterns & gotchas

### Build & JIT
- **Stale JIT cache.** Editing a vendored CUTLASS header without bumping `version.h` leaves
  the cache short-circuiting on an old `.so`. `rm -rf ~/.cache/oasr/jit`, and confirm a fresh
  hash directory before trusting a kernel benchmark.
- **Changing `[tool.isort]` without `[tool.ruff.lint.isort]`** (or vice versa). Both sort
  imports and mirror `known_first_party` / `combine_as_imports` /
  `force_sort_within_sections`; change them together or the two tools fight and CI flaps.

### Testing & measuring
- **Assuming a green `pytest tests/` means coverage.** Without the external assets the
  real-checkpoint tests skip. Read the `external assets:` table, or pass `--strict-assets`.
  Not hypothetical: a fused-mask kernel shipped green while returning an empty transcript for
  *every* Paraformer request, because the validating shell had no `$OASR_PARAFORMER_CKPT`.
- **Validating a kernel change with an fp32 parity oracle alone.** fp32 and CPU route to
  torch, so a launcher precondition — contiguity, alignment, a dtype dispatch — is never
  reached. Test in a served dtype (fp16/bf16, on CUDA).
- **Benchmarking an explicit-only architecture without `--architecture`.** An icefall
  pruned-RNNT dir sniffs as `zipformer`, so you measure a different branch or fail outright.
- **Trusting a single-order A/B.** Interleave the arms — the second benefits from a warm
  allocator. Report a σ, not one run.
- **Optimizing GPU time at small batch.** Encoders are CPU-issue-bound at batch 1–2, where
  removing GPU work can make them *slower*. Compare issue time against wall time.
- **Trusting a regression test that has never been seen to fail.** Revert the fix and watch it
  fail in exactly the predicted parametrisations.
- **Believing a solo test proves a stream is synchronised.** A missing cross-stream ordering
  only misbehaves under concurrent load: congest the other stream on purpose, don't rely on
  timing. And `CUDA_LAUNCH_BLOCKING=1` "fixing" it is not a diagnosis — it serialises
  everything, including the overlap that may be the actual bug.

### CUDA graphs & kernels
- **Reusing a CUDA-graph replay buffer's output.** One buffer per shape key; a returned tensor
  is live only until the next replay *or capture*. Copy when a step can hit the same key twice.
- **Retrying a capture that already failed.** Capture costs a warm-up forward before it records
  anything, then runs eager regardless — strictly worse than never trying. Remember the failed
  shape; treat an OOM as a fact about the process and stop capturing (`DecoderStepGraphCache`).
- **A paged block table whose width is not a whole number of K tiles.** The fused kernel's
  paged loader walks `N_BLOCK // block_size` pages per K tile *unpredicated*, so a short table
  reads its last tile past the tensor and dereferences whatever followed as a page id. Only
  bites when a page is smaller than a K tile — hence decoder pages, not encoder ones.
- **Adding a field to the paged region without updating both allocators.** `init_paged_state`
  and `setup_internal_data_paged_pointers` bump-allocate the *same* bytes: a missing field
  shifts every later pointer, and the illegal access lands far from the edit.
- **`LinearActivation(activation="gelu")`.** Only `gelu_tanh` exists — the CUDA epilogue is the
  tanh approximation, and fusing it under the exact-erf name is a silent accuracy change.

### Decode path
- **Per-token or per-character Python on the decode path.** The step loop is interpreter-bound
  and holds the GIL for every request the engine finishes; word grouping and the beam read-back
  both cost more than the work they decorated, hence `_C.alignment`. Threads are not a lever —
  pure Python holds the GIL, and a prefetch producer measured 0.95–0.98× against a 1.2× ceiling.
- **`torch.tensor(ids, device="cuda")` in a per-step path.** A pageable H2D copy synchronises
  the stream, so a 40-byte slot-id tensor costs whatever is queued behind it — one of these sat
  after the streaming encoder forward and the host waited out that forward every step. Use
  `oasr.utils.staging.to_device`; same for `.to(device, non_blocking=True)` from unpinned memory.
- **Reading a streaming frame index out of `select_seqs`.** It is a ring of width `max_seq_len`
  and a stream decodes past its token cap, so the value wraps. Use `device_frame_idx_ptr`;
  offline may read the ring, where step == frame.
- **Re-deriving what the decoder already knew.** A frame-synchronous decoder knows the emitting
  frame *as* it emits it; recovering it afterwards cost 10× the decode and cannot serve a
  stream. Record it in the beam.
- **A shared scalar generation offset in an AR decoder.** It is why two decode groups could not
  be merged, and merging is worth ~1.5× on trickle arrivals. Index KV per row
  (`oasr/cache/decoder_state.py`).
- **Guessing a frame rate, or rebuilding words from tokenizer pieces.** Both produce output
  that is plausible and wrong. `FrameClock.resolve` returns `None` and the request is refused;
  words are cut out of the rendered transcript so each is a literal substring of `text`.
- **Changing the alignment rule without its oracle.** The pass is C++ only (`csrc/alignment/`)
  and Python raises rather than falling back. `tests/test_alignment_cpp.py` states the same rule
  and must agree **exactly**, so a change lands in both — which is also why neither side uses
  `std::isspace` or `sum()`: both differ across implementations and the difference reaches the
  published output. See [`docs/decoding.md`](docs/decoding.md) § Word timings.

### Declare, don't ignore
- **Declaring `streaming_kind` from what the class implements** rather than what *this config's
  weights* can do. Over-claiming raises on the first request instead of failing at construction.
  Same for `word_timing_modes`, a **property** because a transducer times both modes under
  greedy and neither under beam search.
- **Accepting a per-request option a decode family cannot act on.** `task` / `language` change
  *what is decoded*, so a family without the control rejects at admission
  (`DecodeStrategy.validate_options`). Sampling knobs differ — ignoring one returns the same
  transcript.
- **An `EngineConfig` field for one decode family** (use `options_cls`), or **`nn.Linear` in a
  model file** (use `oasr.layers`).

### Serving
- **Handing the engine heap audio.** After the codec, ask
  `ASREngine.new_audio_buffer(n)` for page-locked memory and decode into that: the engine
  then DMAs each row of a micro-batch straight into the padded device batch instead of
  packing it into staging, which is a second full copy of the waveform (**1.13-1.19×**
  offline). Hand over the **tensor**, never a `numpy()` re-wrap of its pages — PyTorch can
  only event-track the in-flight copy through the tensor whose storage it allocated, and an
  anonymous re-wrap can be recycled under a live DMA. The offer is bounded
  (`max_pinned_audio_seconds`); page-locked memory is process-global.
- **Sniffing a container out of a declared `LINEAR16` body.** MP3 and AAC are identified by an
  11-bit frame sync that real PCM hits by chance; only unambiguous magic may override a
  caller's declared encoding (`oasr-asr::codec::Container::is_unambiguous`).
- **Bounding audio by request bytes once codecs are accepted.** A few MiB of MP3 is hours of
  waveform. `--max-audio-seconds` bounds the decode; `--max-audio-mib` bounds the body.

---

## Development workflow

1. Branch from `main` (see [Branching](#branching-policy-and-prs)).
2. `pip install -r requirements-dev.txt` — pinned; CI gates on these exact
   versions. Optionally `pip install pre-commit && pre-commit install`, which
   mirrors the fast half of the lint workflow locally.
3. Make the change. If it touches a registry axis, follow the cookbook in
   [`docs/architecture.md`](docs/architecture.md) rather than editing the engine.
4. Add tests. A new architecture also needs a tiny config in
   `tests/test_layer_waist.py` and, if it changes decode behaviour, an entry in
   `ci/wer-reference.json`.
5. Run locally, in this order:
   ```bash
   set -a; source .env; set +a          # rule 8 — without this the real-checkpoint tests skip
   black oasr/ tests/ benchmarks/ scripts/ ci/ && isort oasr/ tests/ benchmarks/ scripts/ ci/
   ruff check oasr/ tests/ benchmarks/ scripts/ ci/
   python scripts/mypy_ratchet.py
   python ci/gpu_suites.py --check      # a new test file in no family never runs on the matrix
   pytest tests/ -m "not slow"
   cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
   ```
   `mypy` and `cargo test` are deliberately **not** pre-commit hooks — too slow
   for a commit. Run them before opening a PR, or let CI do it.
6. **Check the `external assets:` table at the end of the pytest run**, not just
   the pass count. If the asset covering your change reads `MISSING`, the tests
   that would have exercised it skipped and you have not tested your change. Add
   `--strict-assets` to make that a failure. If your change touches an
   architecture, also measure it: `bench_accuracy.py` against that
   architecture's checkpoint, compared with `ci/wer-reference.json`.
7. Record any measurements in `.artifacts/`, not in `AGENTS.md` or `README.md`.

Style: Python is 100 characters (black + isort's black profile). C++ is Google
style, 100 characters, C++17. CUDA flags include `--expt-relaxed-constexpr`,
`--expt-extended-lambda`, `-O3`, `--use_fast_math`.

### Branching policy and PRs

- `main` is the default and merge target. Never commit to it directly.
- Branch names follow `<type>/<topic>`: `feat/`, `fix/` or `bugfix/`, `perf/`,
  `refactor/`, `chore/`, `docs/`.
- Open a GitHub PR against `main` (`gh pr create`); merges go through PRs.
- Keep a PR to one concern. Discuss substantial changes in an issue first.
- Do not commit or push on a maintainer's behalf unless explicitly asked.

---

## CI / testing

Tests live under `tests/`, flat, one file per kernel or component
(`tests/test_<thing>.py`, FlashInfer convention). `tests/conftest.py` provides
`device`, `dtype` / `dtype_all`, `batch_seq_hidden`, and the asset fixtures
`ckpt_dir` / `wav_dir` / `audio_path` / `lang_dir` (which **gate**, not return
`""`). Default options `-v --tb=short` come from `pyproject.toml`.

Markers: `slow`, `concurrent` (opt-in), `cuda`, `requires_assets(*names)`.

**External assets are declared once, in `tests/assets.py`** — checkpoints, audio
directories, decoding graphs, upstream reference source trees, each with the
marker file that proves it is really present (a dangling-LFS-symlink HF snapshot
is not a usable checkpoint). Flags: `--strict-assets` (missing asset ⇒ failure),
`--allow-missing-asset NAME` (documented exception), `--min-passed N` (coverage
floor).

**Supplying them is on you, per rule 8.** `pytest` never reads `.env`; the paths
reach it only from the environment, so `set -a; source .env; set +a` is part of
running the suite rather than an optimisation. A run without it still exits 0 —
that is the whole hazard. CI passes `--strict-assets` for exactly this reason and
names each genuinely-absent asset with `--allow-missing-asset`, so a gap lives in
the workflow file instead of hiding in a skip. Adding an architecture means
adding its checkpoint to `tests/assets.py` **and** to `.env.example`, so the next
person can run what you ran.

Three gates matter beyond the unit tests:

| Gate | File | Checks |
|---|---|---|
| Accuracy | `tests/test_accuracy.py` | WER on a fixed 200-utterance LJSpeech manifest against `ci/wer-reference.json`. The one check a numerical-parity oracle structurally *cannot* make — parity feeds identical features to both sides, so a frontend-convention bug cancels on both. |
| Structural | `tests/test_layer_waist.py` | No bare torch layer in any registered architecture; every architecture has a tiny config; kernel and torch paths agree on CUDA. |
| Contract | `tests/test_model_contract.py` | The `CAPABILITIES` table is satisfiable by every registered architecture. |

Workflows in `.github/workflows/`:

| Workflow | Runner | Trigger |
|---|---|---|
| `lint.yml` | GitHub-hosted | push to `main`, every PR |
| `test-cpu.yml` | GitHub-hosted | push to `main`, every PR (Python 3.10 + 3.12, no GPU) |
| `test-gpu.yml` | self-hosted `oasr-gpu` | nightly + manual — **do not add a `pull_request` trigger** |
| `test-gpu-modal.yml` | Modal `RTX-PRO-6000` (sm_120) | weekly + manual |

Both GPU backends run the same per-family split from `ci/gpu_suites.py` with
`--strict-assets`. `mypy` is a **per-file ratchet** against `ci/mypy-baseline.json`,
not a zero-error gate (`--update` after a cleanup). Full detail:
[`docs/ci.md`](docs/ci.md).

---

## Benchmarking

Four harnesses. All of them take their checkpoint, corpus **and sizing** defaults
from `.env` (copy `.env.example`, edit the paths, then
`set -a; source .env; set +a` — rule 8). A missing `$CKPT_DIR` stops the run, but a
missing `$NUM_UTTERANCES` or `$MAX_BATCH_SIZE` does not: it silently measures a
different working point than the number you are comparing against, which is the
same trap as an unsourced test run wearing a different hat.

| Harness | Measures |
|---|---|
| `benchmarks/oasr_benchmark.py` | Kernel level, via a routine registry (`benchmarks/routines/`) |
| `benchmarks/bench_engine.py` | In-process `ASREngine` — the GPU + Python ceiling |
| `benchmarks/bench_service.py` | End-to-end `oasr-server` — what clients see |
| `benchmarks/bench_accuracy.py` | WER/CER **and** RTFx / p50 / p99 in the same CSV row |

```bash
python benchmarks/oasr_benchmark.py --list
python benchmarks/oasr_benchmark.py --routine gemm --subroutine bmm \
    --backends cutlass torch --batch-count 256 --M 200 --N 200 --K 64 \
    --dtype float16 --refcheck -vv
python benchmarks/oasr_benchmark.py --testlist benchmarks/testlists/conformer_base.txt \
    --output_path results.csv --refcheck

ncu --set full -o gemm_profile python benchmarks/oasr_benchmark.py \
    --routine gemm --subroutine gemm --backends cutlass --profile --dry_run_iters 0
```

Backend names differ by family: `cutlass` / `torch` for GEMM and Conv2D;
`cuda` / `torch` for Norm, Conv1D, Activation and composites.

**Methodology:** interleave the arms, report a σ over several iterations, watch
issue time against wall time at small batch, and confirm a fresh JIT hash
directory. Recipes: [`docs/benchmarks.md`](docs/benchmarks.md) and the
`/benchmark-kernel` skill. Results go in `.artifacts/`.

---

## Configuration

Engine configuration is `EngineConfig` ([`docs/engine.md`](docs/engine.md) §6);
serving flags are on `oasr-server --help` ([`docs/serving.md`](docs/serving.md)).
Environment variables:

| Variable | Purpose |
|---|---|
| `CUDA_ARCHITECTURES` | SM targets for the build (e.g. `80` or `80;86`) |
| `OASR_CUDA_ARCH_LIST` | Manual override for JIT arch detection |
| `OASR_ATTN_BACKEND` | `auto` (default) / `cute` (require the kernel) / `sdpa` (force fallback) |
| `OASR_LAYERS_BACKEND` | `oasr` (default) or `torch` (parity oracles, and the "is this the kernels' fault" A/B). There is no `auto`. |
| `OASR_GEMM_HEURISTIC` | `0` disables shape-aware GEMM selection (A/B, rollback) |
| `OASR_GEMM_STREAMK`, `OASR_GEMM_SPLITK_PARALLEL` | `0` skips compiling those GEMM variants |
| `OASR_GEMM_WS_CACHE` | `0` disables the persistent split-K/Stream-K workspace cache |
| `OASR_CTC_FUSED` | `0` forces the legacy multi-kernel CTC beam-search step (A/B, rollback) |
| `OASR_FEATURE_BACKEND` | `torch` forces the reference feature frontend (A/B, parity oracle) |
| `OASR_USE_K2` | `1` builds the k2-backed WFST decoder (needs `pip install k2` + `K2_SOURCE_DIR`) |
| `OASR_METRICS` | `0` binds the engine-side metric collector to a no-op (front-end metrics are unaffected) |
| `OASR_RS_BIN` | Path to an `oasr-server` executable, for `bench_service.py` |

These `OASR_*` A/B switches exist so a regression can be attributed. Set them
before process start.

---

## Troubleshooting

| Symptom | Check |
|---|---|
| `ImportError` on `oasr._C` / `oasr._core` | The package was imported before it was built. `pip install -e .` again. |
| `AttributeError: 'NoneType' object has no attribute 'align_…'` on a word-timestamp request | Same cause: the alignment pass is C++ only and `_C` was never built. `pip install -e .`. |
| A kernel change has no effect | Stale JIT cache — `rm -rf ~/.cache/oasr/jit`. |
| CUTLASS headers not found | `git submodule update --init`. |
| Cargo fails on autocfg probes (bogus generic-argument errors in `tower` / `indexmap`) | The target dir is on a filesystem rustc's probes cannot use. Set `CARGO_TARGET_DIR` to a local path. |
| Cargo fails on conflicting pyo3 features | You used `--workspace`. Don't. |
| Green tests but no real coverage | The external assets are absent — read the `external assets:` table, or use `--strict-assets`. |
| Engine INFO logs missing under `oasr-server` | Known limitation — the front-end configures `tracing`, not Python `logging`. |
| `BlockPool exhausted` mid-stream | Size `max_num_blocks`, or set it to `None` and let the engine derive it from free VRAM. |

---

## Key documentation

| Document | Covers |
|---|---|
| [`docs/README.md`](docs/README.md) | Documentation index |
| [`docs/architecture.md`](docs/architecture.md) | The seven seams, the layer waist, the extension cookbook |
| [`docs/engine.md`](docs/engine.md) | `ASREngine` step loop, executors, `EngineConfig`, VRAM sizing |
| [`docs/scheduler.md`](docs/scheduler.md) | Batching and partition policies, starvation bounds |
| [`docs/models.md`](docs/models.md) | Model contracts, capabilities, the seven architectures |
| [`docs/decoding.md`](docs/decoding.md) | Decode families, the incremental AR protocol, beam search, decoding options |
| [`docs/kernels.md`](docs/kernels.md) | CUDA/CUTLASS layer, the JIT pipeline, the functional API |
| [`docs/features.md`](docs/features.md) | Feature frontends, `FeatureSpec`, streaming framing |
| [`docs/tokenizers.md`](docs/tokenizers.md) | Tokenizer axis and `TokenizerSpec` |
| [`docs/checkpoints.md`](docs/checkpoints.md) | Resolution precedence, converter contract, native format |
| [`docs/cache_manager.md`](docs/cache_manager.md) | Paged and slot streaming caches |
| [`docs/ctc_decoder_gpu.md`](docs/ctc_decoder_gpu.md) | GPU CTC prefix beam search |
| [`docs/wfst_decoder_gpu.md`](docs/wfst_decoder_gpu.md) | In-tree GPU WFST decoder |
| [`docs/serving.md`](docs/serving.md) | Rust front-end, HTTP/gRPC APIs, dispatcher, deployment |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Benchmark recipes and measurement protocol |
| [`docs/ci.md`](docs/ci.md) | The four workflows, the asset gate, the accuracy gate, why mypy is a ratchet |
| [`docs/autotuning.md`](docs/autotuning.md) | `oasr.tune` API and cache format |

### Skills

| Skill | Use for |
|---|---|
| `/add-cuda-kernel` | The authoritative walkthrough for a new kernel family — CUDA header → csrc launcher → JIT binding → JIT generator → Python API → layer wrapper → tests → AOT registration |
| `/benchmark-kernel` | Benchmarking and profiling with `oasr_benchmark.py`, testlists, CSV output, Nsight Compute |
