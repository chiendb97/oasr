# Autotuning and the tuning database

OASR picks a kernel configuration per shape from **measured data**, per GPU
architecture. This page covers where that data lives, how it is produced, and
how the runtime reads it.

```
 model + EngineConfig + traffic ─► oasr tune census ─► ShapeSet (what to tune)
 ArchProfile(device) ─► lane generator ─► cost model ─► top-k ∪ forced ─► bench protocol
                                                                  │  (measurement log)
                              partition (set cover + DP regions) ◄┘
                                          │
                            tuning DB  (system tier: oasr/tune/db, user tier: ~/.cache)
                                          │
 runtime: select_default_config ─► user ▸ system ▸ cost model ▸ default   (every tier counted)
```

## Quick start

```bash
set -a; source .env; set +a

# 1. What does this deployment run?  (probes the engine's own graph keys)
oasr tune census --ckpt-dir $CKPT_DIR --service-mode offline --max-batch-size 32 \
    --manifest benchmarks/manifests/ljspeech_200.jsonl --audio-root $WAV_DIR --out offline.json
oasr tune census --ckpt-dir $CKPT_DIR --service-mode streaming --max-batch-size 64 \
    --chunk-size 16 --out streaming.json

# 2. Measure it on this GPU and write this machine's user tier.
oasr tune build --census offline.json --census streaming.json --out user

# 3. Score it: regret of each selector against the measured oracle.
oasr tune evaluate --log ~/.cache/oasr/tune/v2/measurements/sm120.jsonl \
    --db new=$HOME/.cache/oasr/tune/v2/sm120/gemm.json

oasr tune status        # which tiers are loaded, which are stale
oasr tune diff oasr/tune/db/sm120/gemm.json ~/.cache/oasr/tune/v2/sm120/gemm.json
```

A deployment can also tune at engine construction, before any graph is captured:

```python
from oasr.engine import ASREngine, EngineConfig, TuningConfig

engine = ASREngine(EngineConfig(ckpt_dir=..., service_mode="offline",
                                tuning=TuningConfig(mode="prewarm", budget_s=120)))
```

## The tuning database

`oasr/tune/database.py`. One JSON file per `(arch family, kernel family)` —
`sm120/gemm.json`, `sm120/conv1d.json` — in one of two tiers:

| Tier | Where | Written by | Precedence |
|---|---|---|---|
| **system** | `oasr/tune/db/sm<family>/` (shipped) | review only (`oasr tune build --out system` + `oasr tune diff`) | second |
| **user** | `~/.cache/oasr/tune/v2/sm<family>/` | `oasr tune build --out user`, `oasr.autotune()`, `TuningConfig(mode="prewarm")` | first, per signature |

`OASR_TUNE_USER_DB=<dir>` moves the user tier; `OASR_TUNE_USER_DB=off` disables
it (the test suite does, so a developer's tuning never changes what a test sees).

An entry is keyed by the **static signature** of an operation and indexed by its
dynamic dimension:

```json
"gemm|op=gemm|dt=half|N=512|K=256": {
  "regions": [[512, "b32x64x64_w16x32x64_s3"], [2152, "b64x64x64_w32x32x64_s3"], [null, "torch"]],
  "evidence": {"M=416": {"b32x64x64_w16x32x64_s3": [0.0061, 0.0001, 9], "_l2": "cold"}},
  "source": "aot",
  "notes": ["M=416*: b32x64x64_w16x32x64_s3 (2.00x vs default)"]
}
```

- **Regions round up.** M takes the first region whose `m_hi >= M`; `null` is the
  catch-all. A config therefore only serves sizes at or below the largest it was
  measured at (FlashInfer's floor-mapping bug #5449 is what the other direction
  does). Entries written by `oasr.autotune()` have **no** catch-all: a size above
  the largest measured one falls through to the next tier.
- **Configs are explicit parameters** (`"configs": {id: {"tile": [...], "warp":
  [...], "stages": 3, ...}}`), never an import-time sentinel; `"default"`,
  `"torch"` and `"fused"` are the three sentinels.
- `dt=half` serves fp16 and bf16 alike (measured in one, verified to carry to the
  other); a `dt=fp16`/`dt=bf16` entry wins over it for its own dtype.
- Conv1D entries use exact `points` (`"1x3000"`) until they are re-measured as
  regions over `M = B * T_out`.

**Validators.** *Hard* (schema, kernel family, arch family, MMA lane): a mismatch
makes the file unusable and it is skipped with a warning. *Soft* (kernel
implementation hash, SKU, nvcc, CUTLASS stamp): a mismatch keeps the file in
service and marks it **stale** (`oasr tune status`) — a stale entry can only be
slower than a fresh tune, because every config is checked against the compiled
set and the static feasibility predicates at resolve time.

**Snapshots.** Files are read once per *epoch*; `oasr.tune.database.reload()`
(or `ASREngine.reload_tuning()`, which also releases every captured graph)
starts a new one. Nothing else changes a selection inside a process, so a
captured graph and an eager call always run the same kernel (AGENTS rule 11).

**The production compile set** is what the DB references for the arch, the
default, and a coverage basis (`jit.gemm.get_production_configs`);
`OASR_GEMM_COMPILE_SET=all` compiles the whole space.

## The cost model

`oasr/tune/cost_model.py` — waves × resident CTAs × a per-config K-loop line,
plus split-K reduction and in-graph launch terms, with a memory floor:

```
T(c; M,N,K) = W · r · (a_c + b_c · k_iters) + R_c + launches · L_graph,   T ≥ bytes / BW
```

The device numbers come from `oasr.tune.arch.ArchProfile` (queried SMs, shared
memory, L2; micro-benchmarked DRAM bandwidth, tensor throughput and launch cost,
cached per SKU). Measured on 149 Conformer points on an RTX 5090, the
*uncalibrated* structure's top-1 is within 0.4% (geomean) of the best CUTLASS arm;
per-config calibration fits the measured signatures better but does not transfer
to new widths, so the runtime tier ranks with the structure and the build uses
both models' top-k for pruning. Two rules made the difference: a split larger
than the K-tile count is infeasible, and a partial wave's CTAs do not share an
SM.

The runtime **model tier** ranks the compiled configs for an aligned
`gemm`/`gemm_activation` shape no entry covers — 11% geomean regret against 124%
for the default on held-out Conformer widths — and needs the arch's tuning file
to ship a model (`oasr tune build` writes one). `OASR_TUNE_MODEL_FALLBACK=0`
turns it off.

## The census

`oasr/tune/census.py`. The engine bounds most of its shape space before a request
arrives: offline encoder graphs are keyed `(B_bucket, T_bucket)`, streaming ones
by cohort width. The census enumerates those keys from `EngineConfig`, weights
them by traffic (a length-sorted batching replay over a duration sample; a
concurrency histogram), probes each key eagerly under the shape recorder with
inputs padded exactly as the captured path pads them, and marks every point a
captured key reaches **must-tune**. `--capture` merges an `OASR_CAPTURE_GEMM`
recording (decode paths the probes do not reach).

The census engine loads the checkpoint's own frontend. An `EngineConfig` that
names none gets a default `fbank`, which `dataclasses.replace` would otherwise
pass back in as an explicit choice. Whisper was once probed with 448 fbank frames
this way. A fixed-window frontend (`whisper_logmel`, 3000 frames) is keyed at
exactly its window, never rounded up to the graph granularity: the encoder
rejects 3008.

## Building on other architectures

`oasr tune prebuild` compiles every JIT module a build loads, without a GPU, for
the target named by `OASR_CUDA_ARCH_LIST`. The modules are compiled, never
loaded, concurrently, with the job budget split by translation-unit count.
`JitSpec`'s cache key has no device in it (sources, headers, flags, nvcc
identity), so a GPU box sharing the JIT directory finds every library:

```bash
OASR_CUDA_ARCH_LIST=9.0 OASR_JIT_DIR=/shared/jit/sm90 oasr tune prebuild --census offline.json --jobs 32
```

`ci/modal_tune.py` uses it to build a tuning file for every architecture the
shipped DB lacks. Its first stage compiles on a CPU-only container sized per
CUTLASS lane. Its second stage measures on the GPU, spawned as soon as that
architecture's own compile lands. It is a separate app from the test app, so a
tuning run never rebuilds the test image. The GPU stage builds with `--thin` at
16 points per signature. On the 11-model census that is 1234 points instead of
3422. Score a change to that rule offline against a full build on a local card
before renting one. Each finished signature is checkpointed on a Volume, so a
preempted container resumes, and so does `--run-id` after the local client dies.

```bash
modal run ci/modal_tune.py --dry-run                       # the plan; spends nothing
modal run ci/modal_tune.py                                 # A100, L40S, H100, B200
oasr tune diff oasr/tune/db/sm90/gemm.json .artifacts/tune_matrix/sm90/gemm.json
```

Results land in `.artifacts/tune_matrix/sm<family>/`: the tuning file, the
measurement log, and the compile and run reports. `run.json` names any module
the GPU still had to compile. Promoting a file into `oasr/tune/db/` is a
reviewed change, never automatic.

## The benchmark protocol

`oasr/tune/bench.py`, shared by `oasr tune build`, `scripts/tune_asr_gemm.py`,
`oasr.autotune()` and the calibration:

1. Refuses to run inside a CUDA-graph capture.
2. Each arm is a CUDA graph of N ≥ 8 calls (one call per replay ranks *worse*
   than eager timing — Inductor PR #196413), plus a single-replay gate against
   self-overlap. A tiny unrelated kernel follows every call, and its separately
   measured cost is subtracted. Back-to-back copies of one cuBLAS kernel
   overlap each other through programmatic dependent launch, but a GEMM between
   a norm and an activation cannot. On B200 the unseparated loop credited
   cuBLAS with about 0.65 µs per call that it does not get inside a model.
3. Weights are rotated through enough copies to be cold when the served model's
   weight working set exceeds L2 — and stay warm when it does not.
4. Arms are interleaved in shuffled rounds; the estimator is the median, σ is
   1.4826 × MAD; successive halving keeps the default and the incumbent to the end.
5. Every candidate passes a numerical gate first (within 4× cuBLAS's own error, no
   NaN/Inf): a tile can launch cleanly and still be wrong.
6. Records SM clock and other compute processes; a contended run is marked.

The objective is `GPU time + α · host issue time`, α being the fraction of the
shape's calls that run eagerly (from the census): on a captured shape issue cost
is free, on an eager one cuBLAS's extra ~4 µs per call counts.

## Building: partition and pruning

`oasr/tune/build.py`, `oasr/tune/partition.py`. Per signature: measure the census
points plus ladder fill points (`oasr.tune.shapes`) — only the union of the
models' top-16 plus the forced set (default, fused, cuBLAS) — then measure every
point's top-3 at every other point (cross-evaluation), choose a shared config set
by weighted greedy set cover (every point within 3% of its best), segment the M
axis by DP, and keep the fallback in any region whose config does not win by the
gate (`--min-speedup`, 1.05). On the Conformer census this took 58 s and scored
3.0% geomean regret against the measured oracle (the converted SM120 table: 8.8%;
the default: 124%); end to end, offline Conformer ran +2.3% over the shipped table.

Census points are always measured unless `--thin` is given. With it,
`--max-points` caps them too (`build.thin_census`): the smallest and largest M
stay, and each of `max_points - 2` equal log-M bins keeps its heaviest point. A
dropped point's weight, calls and must-tune flag move to the next kept point
above it, the one whose region serves it. That is for dense streaming censuses,
which key every cohort width: one signature can carry 140 Ms a few percent
apart.

## `oasr.autotune()`

The runtime autotuner profiles every registered candidate for a shape it has no
result for, under the same protocol:

```python
import oasr

with oasr.autotune(cache="oasr_tune.json"):
    output = oasr.gemm(A, B)   # profiles on first call of this shape's bucket
```

- **Dynamic dimensions key by bucket**, not by value (`oasr.tune.shapes`: step 8 to
  128, ×√2 to 1024, ×2 above), so a new M does not re-profile. Version-1 cache
  files are re-keyed on load.
- A cache file tuned on another architecture is **not loaded** (hard validator);
  other environment differences warn.
- GEMM-family winners are also published to the **user tier** (`publish=True`),
  so the production path — outside `autotune()` — serves them after the context
  exits.
- Ops that cannot be graph-captured are timed eagerly (`warmup`/`rep` apply only
  there).

Supported ops: `gemm`, `gemm_activation`, `gemm_log_softmax`, `bmm`,
`group_gemm` (each with a torch/cuBLAS candidate), `conv2d`, `conv2d_activation`,
and the recurrent layers.

## Environment

| Variable | Effect |
|---|---|
| `OASR_TUNE_USER_DB` | user-tier directory; `off` disables it |
| `OASR_TUNE_MODEL_FALLBACK` | `auto` (default) / `0`: the cost-model tier for uncovered shapes |
| `OASR_GEMM_COMPILE_SET` | `tuned` (default) / `all`: which variants production modules compile |
| `OASR_GEMM_HEURISTIC` | `0`: every shape uses `GEMM_DEFAULT` |
