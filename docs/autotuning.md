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
#    (a quantized model: add --quantization int8_w8a8 to the census; the build then
#     also writes scaled_mm.json, the int8 GEMM's file)

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
`sm120/gemm.json`, `sm120/conv1d.json`, `sm120/scaled_mm.json` (the int8 GEMM of
a quantized model; [below](#the-int8-scaled-gemm-scaled_mm)) — in one of two tiers:

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

### The int8 scaled GEMM (`scaled_mm`)

The W8A8 GEMM of a quantized layer (`oasr.scaled_mm`, [`quantization.md`](quantization.md))
is its own kernel family with its own file, keyed by op and width. A region
names a tile, or the heuristic (an illustrative entry):

```json
"scaled_mm|op=scaled_mm|N=768|K=256": {
  "regions": [[7136, "heuristic"], [null, "b128x128x64_w64x64x64_s3"]], ...
}
```

- **Two ops.** `scaled_mm` (fp16/bf16 output) and `scaled_mm_quant` (the same GEMM
  quantizing its output to int8 for the next GEMM) are measured and keyed
  separately; `select_quant_config` never serves a Stream-K tile (there is no
  quantizing Stream-K launcher), so such an entry is warned about once and skipped.
- **The `"heuristic"` sentinel.** The family's untuned selector is a shape
  heuristic. For int8 it is the ladder `_INT8_LADDER`: the largest tile that fills
  enough of the GPU for its K, within 1 % (geomean) of the best of 25 tiles
  measured over five architectures' shapes. For fp8 it is the largest
  data-parallel tile still launching half a wave of CTAs. A region names
  `"heuristic"` where nothing beat it by the gate; the build keeps an entry even
  when every region does — `[[null, "heuristic"]]` says *measured*, which
  `tuning.strict` and the coverage report need, where an absent entry would say
  *never looked*. Only a measured entry can pick a Stream-K tile (it wins only at
  deep K, ≥ 16384).
- **Resolution** (`oasr.jit.scaled_mm.select_config` / `select_quant_config`):
  user tier, then system, M rounded up to a region, a config outside this
  process's production set warned once and skipped; else the heuristic. Memoised
  per `(epoch, shape)` — one dict lookup per call — and recorded per resolution
  as tier `user` / `system` / `heuristic` in `oasr tune status` and
  `ASREngine.tuning_report()`. Every variant computes the same bits (an exact
  int32 accumulator; Stream-K reduces int32 partials), so an entry changes speed,
  never a transcript.
- **Census and build.** The shape recorder patches `oasr.scaled_mm` and
  `oasr.functionals.scaled_mm.{scaled_mm, scaled_mm_quant}` (the lanes call
  through the module), recording dtype `int8`. `oasr tune census --quantization
  int8_w8a8` probes a full-precision checkpoint quantized online; a quantized
  checkpoint needs no flag. `oasr tune build` then writes `scaled_mm.json` beside
  `gemm.json` (`--family scaled_mm` builds only it): every eligible variant is
  measured at every point (the set is nine tiles: seven data-parallel, two
  Stream-K — no pruning, no cost model),
  numerics-gated **bitwise** against the heuristic's variant, and partitioned
  against a `heuristic` pseudo-arm (`oasr/tune/build_scaled_mm.py`).
- **fp8: a family of its own.** The fp8 e4m3 GEMM (`fp8_w8a8`) keeps its entries
  in `scaled_mm_fp8.json` (family `scaled_mm_fp8`, lane `sm89_fp8_mma`), keyed
  `scaled_mm_fp8|op=scaled_mm_fp8|N=..|K=..` and `op=scaled_mm_fp8_quant`; the
  recorder files an fp8-operand call under those ops (dtype `float8_e4m3fn`),
  `oasr tune census --quantization fp8_w8a8` probes it, and `oasr tune build`
  (`--family scaled_mm_fp8`) measures its eight tiles the same way. Its tiles
  also compute the same bits as each other (one summation order, see
  [quantization.md](quantization.md#fp8)), so the bitwise numerics gate holds.
  The shipped `sm120/scaled_mm_fp8.json` covers the Conformer widths: at large
  M the 64x64 tile beats the heuristic's 128-wide ones on every signature.
- **Lanes.** A *lane* is a kernel generation with its own config space and JIT
  module: the CUTLASS 2.x `mma.sync` lanes (`sm80_int8_mma`, `sm89_fp8_mma`) and, on
  sm90 / sm100, the native CUTLASS 3.x ones (`sm90_int8_wgmma`, `sm90_fp8_wgmma`,
  `sm100_int8_umma`, `sm100_fp8_umma`; six tiles each, ping-pong / cooperative or
  1-SM / 2-SM). A file records the lane it was measured on -- the device's default
  lane, so a Hopper or Blackwell build measures the native kernels -- and its entries
  name that lane's configs (`"lane"`, `"cluster"`, `"schedule"` for a native one).
  Resolution is per lane: an entry naming another lane's config (the `mma.sync` A/B arm
  reading a native file) is warned about once and skipped, and the shape heuristic
  answers. The native lanes' heuristic is an unfitted half-wave rule until such a
  file exists.
- **The engine.** `TuningConfig(strict=True)` raises on an uncovered int8
  must-tune point as on a GEMM one, `mode="prewarm"` measures it into the user
  tier before capture (one `budget_s` for both families), and the tuning report
  ends with one line per quantized model: `scaled_mm: 96 quantized layer(s) over
  5 signature(s): 96 with a tuning entry (system=96), 0 on the shape heuristic`.

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
`JitSpec`'s cache key has no device in it, only the sources, headers, compile
commands and tool versions. A GPU box sharing the JIT directory therefore finds
every library, provided it runs the same toolchain and environment (the same
image, as `ci/modal_tune.py` uses for both stages):

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
