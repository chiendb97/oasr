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
  `"torch"` and `"fused"` are the three sentinels. Every config's `"kind"`
  names the [runner](#runners-backends-a-region-selects-among) that serves it.
- `dt=half` serves fp16 and bf16 alike (measured in one, verified to carry to the
  other); a `dt=fp16`/`dt=bf16` entry wins over it for its own dtype.
- Conv1D entries use exact `points` (`"1x3000"`) until they are re-measured as
  regions over `M = B * T_out`.

**Validators.** *Hard* (schema, kernel family, arch family, MMA lane): a mismatch
makes the file unusable and it is skipped with a warning. *Soft* (kernel
implementation hash, SKU, nvcc, CUTLASS stamp, and each runner's library
versions under `"runners"`): a mismatch keeps the file in service and marks it
**stale** (`oasr tune status`) — a stale entry can only be slower than a fresh
tune, because every config is checked against the compiled set, the static
feasibility predicates and its runner at resolve time. A runner's version change
makes stale only the entries that select that runner.

The soft validators describe the entries, not the last writer:
- **Re-stamping:** a write stamps the file with the writer's validators. Every
  entry the writer didn't measure keeps the validators it was measured under,
  as its own `"soft"` field.
- **Per-entry staleness:** `oasr tune status` reports those entries against
  their own validators.
- **Mixed entries:** an entry that gains a point measured under newer
  validators keeps the older ones, so it stays stale until it's rebuilt.

**Concurrent writers.** The user tier has three writers: `oasr.autotune()`'s
flush, `oasr tune build --out user` and the engine's `prewarm`. Each takes the
file's lock, re-reads it and applies only its own changes, so an entry another
process wrote meanwhile survives. The lock is held for the merge and the write,
never for the build. `--fresh` still replaces the file.

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

### Attention lanes and tiles (`fmha`)

`OASR_ATTN_BACKEND=auto` picks one attention lane per process -- the CuTeDSL
kernel wherever it is built -- and runs it with one tile. Neither lane is right
everywhere, and neither is its default tile: on an RTX 5090 the C++ lane's
split-KV is 1.4-3.4x on a Whisper decode step, and inside the CuTeDSL lane a
32x16 tile is 1.8x the default 64x64 on the U2++ offline encoder. So the choice is
per shape and joint over `(lane, configuration)`, and it is the `fmha` family of
the database (`oasr/jit/attention_select.py`):

```json
"fmha|op=fmha|dt=half|D=64|H=6|Hkv=6|mask=none|bias=0|ba=0|kv=paged16|ss=0|q=1|B=8": {
  "regions": [[64, "auto"], [null, "cxx_s4"]], ...
}
```

- **The signature** is what decides which kernel can run and how it behaves:
  head dim, heads, KV heads, mask kind (`none` / `causal` / `window`), bias and
  whether its rows are 4-byte aligned (`ba`: both lanes branch on it, and it flips
  the winner between neighbouring key lengths), dense or paged KV and the page
  size, left padding (`ss`), the query-length class (`1`, `16`, `32`, `64` or
  `L`), and the batch rounded up to a power of two. Regions run over the key
  length `T_k`.
- **Configurations.** `cute_m{M}n{N}t{threads}` -- a CuTeDSL tile, seven measured
  (`CUTE_TILES`); `cxx` -- the C++ lane with its own split rule, `cxx_s{n}` with
  `n`-way split-KV (`n=1`: never; a split count is cut to what the key range can
  use, a pure function of the shape); `sdpa` -- torch's SDPA, an external runner;
  and `auto`, "measured; today's routing is best here". A paged call takes no tile
  taller than 64 rows (the paged path pads to 64).
- **Resolution** (`tuned_choice`, called by `oasr.fmha`): only when nobody named
  a lane -- `OASR_ATTN_BACKEND=auto` and no `backend=` -- and only for fp16 / bf16
  dense or paged calls; `fmha_varlen` and the fp8 KV cache keep their own routing.
  User tier, then system; a choice this process cannot serve (a CuTeDSL tile for a
  windowed call, a lane not built here) is warned about once and falls through.
  With no `fmha` file the lookup returns at once (an epoch compare). Memoised per `(epoch,
  shape)`, counted by tier (`auto` = untuned) and by runner; an `sdpa` region is a
  policy hit (`tuned-sdpa:fmha.fmha`). `oasr.fmha(..., backend=AttnChoice(...))`
  names a lane *with* its configuration.
- **Numerics.** The lanes compute in a different order, so a table can change a
  transcript at the rounding level: gate it with the WER references, as any
  attention change. Graph and eager agree, because the choice is a function of
  the shape alone.
- **Census.** `oasr tune census --family fmha --manifest M --audio-root R` replays
  the manifest through the engine with every graph off -- an AR family's decoder
  steps included, with paged decoder KV as served -- because a decode step's shapes
  depend on what was said. Streaming replays it at a few concurrency levels.
- **Build.** `oasr tune build --census a.json --family fmha` measures every
  configuration that can serve each point through `oasr.fmha` itself, under the
  shared protocol, numerics-gated against an fp32 SDPA reference (within four
  times torch's own half-precision error), and partitions against the `auto`
  pseudo-arm. SDPA is always measured and logged; it may win a region only with
  `--allow-external`. The file's provenance records census-weighted attention
  time for `auto`, the table, and the table with SDPA.

### Recurrent and gated-MLP paths (`recurrent`, `gated_mlp`)

Two more layers chose their kernel by cut-offs timed once on an RTX 5090 --
counted as extrapolations on any other card (`oasr/jit/measured.py`):

- **An LSTM / RNN layer**: the scalar cohort kernel or the tensor-core path
  (`_TENSOR_CORE_MIN_BATCH_*` in `oasr/layers/recurrent.py`); at a single timestep,
  the fused recurrent step or the decomposed GEMM path (`_LSTM_BANDS`); which
  fused lane (`_AUTO_ORDER`) and tile (`_TILES` / `kRecurrentStepRoutes`); which
  GEMM tactic (`_default_recurrent_tactic`).
- **A gated MLP**: fuse only inside one m-tile (`rows <= 64`), on the C++ lane
  first, at the tile wave arithmetic picks from a hand-chosen candidate list.

Each is now a family whose regions name a *path* (`oasr/jit/recurrent_select.py`,
`oasr/jit/mlp_select.py`, sharing `oasr/jit/selection.py`):

```json
"recurrent|op=layer|kind=lstm|dt=half|H=640|I=640|T=step": {
  "regions": [[1, "auto"], [16, "cxx_t0"], [null, "auto"]], ...
},
"gated_mlp|op=gated_mlp|dt=half|act=silu|bias=0|N=11008|K=4096": {
  "regions": [[64, "auto"], [null, "cxx_t5"]], ...
}
```

- **Recurrent signature**: kind (`lstm`, `rnn_tanh`, `rnn_relu`), hidden width,
  input width, single step (`T=step`) or sequence (`T=seq`); regions over the
  batch. Configurations: `scalar`; `gemm` (today's tactic rule) or `gemm_t{t}[_s{k}]`;
  `cxx_t{i}` / `cute_t{i}` -- the fused step on a lane, tile `i` of the eight both
  lanes share (an LSTM single step only); `auto`.
- **Gated-MLP signature**: activation, bias, `N`, `K`; regions over rows.
  Configurations: `gemm` (gate GEMM with its activation, up GEMM, multiply -- each
  GEMM resolved through the GEMM database); `cxx_t{i}` / `cute_t{i}`, tile `i` of
  the six both lanes share, **at any row count** (a region may fuse past one
  m-tile where that measured faster); `auto`.
- **Resolution**: the layer (`oasr.layers.LSTM` / `RNN`: scalar or tensor cores)
  and the functionals (`oasr.lstm_gemm_layer` / `rnn_gemm_layer`: fused step, lane,
  tile, tactic; `oasr.jit.mlp.routed_gated_mlp`, hence `oasr.gated_mlp`) consult
  the family first, memoised per `(epoch, shape)`; with no file the lookup returns
  at once. A covered shape never reaches the hand-timed tables, so it is not an
  extrapolation; an uncovered one runs exactly as before. Not consulted when
  `OASR_RECURRENT_CUTE` / `OASR_GATED_MLP_CUTE` is `0` or `1`, when a
  `*_BACKEND` names a lane, or when a caller passes `_tactic`.
- **Census**: `oasr tune census --family recurrent|gated_mlp --ckpt-dir CKPT`
  reads the static dims off the model's own `LSTM` / `RNN` / `GatedMLP` modules,
  built on the CPU (nothing is served), or takes `--dims lstm:640:640` /
  `--dims silu:11008:4096`, and crosses them with a ladder (`--ladder`; batches
  1..512, rows 1..2048 by default; `--seq-lens`, default `1 32`). A decode step's
  batch is the number of live streams, so a ladder is the honest census for it.
- **Build**: `oasr tune build --census r.json --family recurrent` measures every
  path through the functionals themselves. A recurrent point is timed as a
  **dependent chain** -- call `i` reads the state call `i-1` wrote -- the protocol a
  recurrence imposes. Numerics against an fp32 formula within four times the
  error of today's path (`auto`, measured to the end); the GEMM build's partition.

## Inductor as a fusion reference (`oasr tune compare-inductor`)

`oasr/tune/inductor.py`. A benchmark, not a runner: it writes a report and never
a tuning file. The question it answers is which fusions or GEMM tiles
`torch.compile` finds that OASR's kernels miss. Each answer is kernel work for
OASR, not a backend to serve: Inductor's code depends on the torch and Triton
versions, it compiles per shape, and it can only see the layers' torch path.

```bash
set -a; source .env; set +a
oasr tune compare-inductor --ckpt-dir $CKPT_DIR --out .artifacts/inductor
oasr tune compare-inductor --ckpt-dir $OASR_PARAFORMER_CKPT --dtype bfloat16 \
    --blocks EncoderLayerSANM --batches 1,16 --seconds 8
```

**Blocks.** A block is a non-leaf module the encoder repeats: the layer and the
feed-forward, attention and convolution modules inside it. The command takes one
block per class and set of weight shapes, so a Zipformer stack of another width
is its own block. A leaf module is a single kernel, which the GEMM and conv
tables already cover. `--blocks` filters by class name or by part of the module
path.

**Arguments.** The command does not synthesise arguments; it records them. It
runs one offline encoder forward on the CPU (fp32, real weights) for each batch
size and length. A pre-hook on each block records its first call, and the
forward stops once every block has been reached. The masks, positional
embeddings and keyword arguments are what the model really passes, so any
architecture works without an adapter. A fixed-window frontend (Whisper,
Qwen2-Audio) always feeds its window, so every length gives one shape.

**Arms**, each the block in the served dtype on the current GPU:

| Arm | What runs |
|---|---|
| `oasr` | the OASR kernels: what serves |
| `torch` | the layers' torch path (`OASR_LAYERS_BACKEND=torch`), eager: Inductor's input |
| `inductor` | `torch.compile` of the torch path, default mode: the pointwise and reduction glue fused, GEMMs on ATen |
| `inductor-max` | `max-autotune-no-cudagraphs` with `--backends` (default `ATEN,TRITON,CUTLASS,CUTEDSL`): Triton GEMM templates with fused epilogues, plus whatever the CUTLASS and CuTeDSL generators offer on this architecture |

All four arms go through [the benchmark protocol](#the-benchmark-protocol):
graph-captured loops, rotated weight copies sized by the model's weight bytes
(`--max-copy-mib` bounds them), and interleaved rounds. Inductor's own CUDA
graphs stay off because the harness captures every arm itself. One compiled
graph serves every weight copy, since parameters are graph inputs. Compiles skip
Inductor's FX and autotune caches, so each compile is real, timed and recorded.

**Isolation.** Each shape is measured in a child process (`--in-process`
measures in the command's own process instead). Inductor's autotuning launches
candidate kernels, and one that faults can leave a sticky CUDA error, after
which no number from that process can be trusted. Inductor catches the exception,
so the fault surfaces at the next CUDA call. When a child dies, the block it was
on is recorded as failed, with the child's last error (all of the children's
stderr is kept in `<stem>.children.log`), and a fresh child measures the
remaining blocks. `--autotune-subproc` decides where `inductor-max` benchmarks
its candidates. In-process, a layer's few hundred candidate modules stay loaded,
which runs a shared GPU out of memory. In Inductor's tuning subprocess, one
sm120 run left a misaligned-address fault in the parent. `auto` (the default)
uses the subprocess only when the GPU has under 8 GiB free.

**What the report holds** (`<stem>.json` with every number, `<stem>.md` for
reading):

- **Timing:** per block and shape, each arm's time and sigma, OASR time divided
  by the best compiled time, and a verdict. Inductor is *ahead* when it is at
  least `--gap` faster (default 5 %) and the difference exceeds three combined
  sigmas.
- **Numerics:** each arm is compared with the block in fp32 and passes when its
  error is within four times the eager torch arm's (`bench.numerics_ok`). A
  compiled arm that fails is reported, never counted as a win.
- **Kernel inventory:** the kernels one call launches, from the profiler, each
  named by the aten op that launched it (`aten::masked_fill_`) or by its own
  demangled name (an OASR launcher, an Inductor Triton kernel). The ops fused
  into an Inductor kernel are read from the kernel's name, leaving out
  layout-only ops.
- **Kernel to-do:** the blocks where Inductor is ahead, worst shape first. Each
  entry gives OASR's kernels, the torch kernels still on the OASR path, the
  winning arm's kernels, and its graph breaks.
- **Inductor autotune:** for every autotuned op, the candidates each backend
  offered and the backend that won, recorded through Inductor's feedback hook. A
  backend that cannot generate for this GPU shows up as offering nothing.

**Reading it.** A *graph break* is a call Dynamo cannot trace, such as an OASR
kernel the torch path still reaches (the fused attention, for example). That
call runs as itself inside the compiled arm, and Inductor cannot fuse across it,
so Inductor's time is the conservative one. Only encoders are covered: offline
forward, unquantized. Decoder steps and streaming chunks carry state that a
recorded call cannot replay.

## Runners: backends a region selects among

`oasr/tune/runners/`. A **runner** is one backend that can serve a kernel
family's operations, and a **tactic** is one configuration inside it. Tuning is
a joint search over `(runner, tactic)` pairs under the one benchmark protocol,
and a region names the winner. The file format already carries the runner: a
config's `"kind"` is owned by exactly one runner of its family.

| Family | Runner | Kinds | External | Dispatch |
|---|---|---|---|---|
| `gemm` | `oasr_cutlass` | `cutlass`, `default` | no | the compiled variant, by name |
| `gemm` | `oasr_fused` | `fused` | no | the CTC head's single-call launcher |
| `gemm` | `cublas` | `torch` | **yes** | `Runner.bind` → `oasr.functionals.gemm_torch` |
| `conv1d` | `oasr_cutlass` | `cutlass`, `default` | no | the family's own resolver |
| `scaled_mm`, `scaled_mm_fp8` | `oasr_cutlass` | `cutlass`, `heuristic` | no | the family's own resolver |

**What a runner declares.** `RunnerTraits`: `capturable` (may serve a shape a
captured graph reaches), `batch_invariant` (a runner-wide guarantee that rows do
not depend on M), `numerics` (`bitwise` or `tolerance`), `external` (not an OASR
kernel), and `compiled` (its tactics are OASR JIT variants, so only these count
against the compile budget and the coverage basis). `versions()` names the
library versions its timings depend on.

**Adding one** is a subclass and a `register_runner` call, with no edit to the
tuner, the database or the dispatcher (AGENTS rule 1):

```python
from oasr.tune.runners import Runner, RunnerTraits, register_runner

class MyGemm(Runner):
    name, family = "mylib", "gemm"
    kinds = ("mylib",)            # the config kind its tactics are stored under
    backends = ("mylib",)         # its Tactic.backend in the oasr.autotune() registry
    ops = ("gemm", "bmm")
    traits = RunnerTraits(capturable=True, external=True)

    def unavailable(self):        # None, or why it cannot run here
        ...
    def versions(self):           # soft validators
        return {"mylib": mylib.__version__}
    def supports(self, op, M, N, K, dtype):   # None, or why it refuses this problem
        ...
    def bind(self, op, choice):   # once per shape; host work only; output first
        return lambda out, A, B, C=None, split_k=1: ...
    def backend_entries(self):    # candidates for `oasr tune build` and oasr.autotune()
        ...

register_runner(MyGemm())
```

**Resolution.** A region naming a runner's config is served only if a runner of
that family owns the kind, the runner is available, it serves the op, and
`supports()` accepts the problem. Otherwise the lookup falls through to the next
tier and says so once. One config of a runner this process lacks used to make
the whole file unreadable; now it costs only its own regions. A refused shape is
not a declared gap: the fall-through lands on an OASR kernel.

**Dispatch.** The GEMM dispatcher binds a runner choice once per shape, when it
first plans it, into its memoised plan (`_PLAN_RUNNER`); the call path invokes
the bound callable, which takes the `gemm_torch` calling convention. `bind` may
plan on the host but must not launch device work: a shape can first be seen
inside a capture. cuBLAS takes
this path.

**Accounting.** The dispatcher counts each new plan by the runner it runs on
(`database.runner_counts()`, and a `kernel dispatch by runner` block in
`tuning_report()`, where `*` marks an external runner). It counts the plan, not
the resolution, because the batch-invariant selector reads the tuned choice and
may drop it. A plan on an external runner is also a policy hit
(`tuned-cublas:gemm.gemm` in `oasr.layers._backend.policy_hits()` and
`format_gap_report()`). A library winning on a shape is a measured decision and
a standing argument for kernel work, not a silent bypass.

**Build.** A runner contributes autotuner candidates through
`backend_entries()`. `gemm_tune.choice_of` maps its tactics back to choices
named by `Runner.config_id`, and those flow through measurement, the numerics
gate, partition and the file unchanged. They are never pruned by the cost model
and never dropped by the compile budget. An arm whose issue time was not
measured borrows the median of its `issue_class` (all OASR launchers share one).

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
  128, ×√2 to 1024, ×2 above), so a new M does not re-profile every candidate.
- **A result serves only sizes at or below the one it was measured at.**
  - **Why the bucket doesn't do this alone:** a call is profiled at its own
    shape, which is usually below its bucket's upper edge.
  - **What's recorded:** every result keeps the exact shape it was measured at,
    and a lookup takes the closest one measured at or above the call.
  - **A larger size in the same bucket:** it's a miss. Under `autotune(True)`
    it's measured; otherwise it takes the fallback.
  - **The cache file:** version 3 records each result's `measured` shape.
    A version-2 entry has none and still serves its whole bucket. A version-1
    file's exact shapes become measured results.
- A cache file tuned on another architecture is **not loaded** (hard validator);
  other environment differences warn.
- GEMM-family winners are also published to the **user tier** (`publish=True`),
  at the M they were measured at, so the production path serves them after the
  context exits.
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
