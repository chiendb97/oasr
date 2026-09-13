# OASR benchmarks

One CLI, `benchmarks/run.py`, for every benchmark in the tree: kernels,
decoders, the engine, the server in front of it, and accuracy.

```bash
python benchmarks/run.py --list                     # families and subroutines
python benchmarks/run.py --print-schema             # the two output schemas
python benchmarks/run.py --family gemm --subroutine bmm \
    --batch-count 256 --M 200 --N 200 --K 64 --dtype float16 --refcheck
python benchmarks/run.py --testlist benchmarks/testlists/all_kernels.txt \
    --output results.csv --refcheck
```

> Engine, service and accuracy runs take their checkpoint, corpus **and sizing**
> from `.env`. Source it first — `set -a; source .env; set +a` — per AGENTS.md
> rule 8. A missing `$CKPT_DIR` stops the run; a missing `$NUM_UTTERANCES` does
> not, it just measures a different working point without saying so.

## Layout

One directory per measurement family, mirroring the `tests/` split that
`ci/gpu_suites.py` enforces. A benchmark belongs to the directory that owns the
module it exercises.

```
benchmarks/
  run.py                  the CLI
  core/                   shared: schema, driver, timing, metrics, report, env
  kernels/                one file per oasr/functionals/ module
  features/               the feature frontend
  decoders/               CTC beam search, WFST
  engine/                 in-process ASREngine, plus the glue-op profiler
  service/                end-to-end oasr-server
  accuracy/               WER/CER sweep
  testlists/  manifests/
```

`bench_engine.py`, `bench_service.py`, `bench_accuracy.py` and
`oasr_benchmark.py` remain as one-line shims into `run.py`, so the commands the
docs name keep working.

## Families

| Family | Subroutines | Backends | Metric |
|---|---|---|---|
| `gemm` | gemm, bmm, bmm_strided, group_gemm, gemm_activation, gemm_log_softmax | `cutlass`, `torch` | TFLOPS |
| `norm` | layer_norm, add_layer_norm(+_residual), layer_norm_activation, rms_norm, add_rms_norm(+_residual), batch_norm, batch_norm_swish, batch_norm_activation, group_norm, cmvn | `cuda`, `torch` | TB/s |
| `conv` | depthwise_conv1d(+_causal), dense_conv1d, pointwise_conv1d(+_activation), conv2d(+_activation), grouped_conv2d, pointwise_conv2d, fsmn_chain | `cutlass`/`cuda`/`cudnn`, `torch` | TFLOPS + TB/s |
| `activation` | gelu, glu, relu, sigmoid, swish, tanh | `cuda`, `torch` | TB/s |
| `attention` | fmha_offline, fmha_bias, fmha_seqlens, fmha_bias_seqlens, fmha_paged, fmha_paged_bias | `cutlass`, `torch` | TFLOPS |
| `softmax` | softmax, masked_softmax | `cuda`, `cuda_unfused`, `torch` | TB/s |
| `topk` · `fft` · `pooling` | topk · rfft, rfft_power · avg_pool1d, max_pool1d | `cuda`, `torch` (+`torch_previous`) | TB/s |
| `recurrent` | lstm, rnn_tanh, rnn_relu, lstm_slot_step, lstm_step_cute | `oasr`, `native`, `cutlass16/32/64`, `streamk`, `splitk`, `cudnn` | TFLOPS |
| `mlp` | gated_mlp | `cute`, `oasr`, `torch` | TFLOPS + TB/s |
| `composite` | conv_block | `cuda`, `torch` | TFLOPS |
| `feature` | fbank_preprocess, mel_log, dct_lifter, fbank_pipeline, mfcc_pipeline | `cuda`, `cuda_unfold`, `torch`, `torchaudio` | TB/s |
| `ctc_decoder` | beam_search, streaming | `fused`, `legacy`, `torchaudio` | frames/s |
| `wfst_decoder` | decode | `in-tree`, `external` | RTFx |
| `engine` | offline, streaming, `*_wfst`, offline_packing, offline_length_batch, per-family | — | RTFx, utts/s, tokens/s |
| `service` | offline, streaming, grpc_offline, grpc_streaming, whisper | — | RTFx, req/s, latency |
| `accuracy` | wer | — | WER/CER + RTFx |

## Output

Two schemas, declared once in `core/schema.py` and printable with
`run.py --print-schema`. Both open with the same envelope — `schema_version`,
`category`, `run_id`, `timestamp`, `case_tag`, `git_commit`, `device`, `sm` —
so rows from different families join on `run_id`.

**Kernel** (32 columns) adds identity (`routine`, `subroutine`, `backend`,
`ref_backend`), the problem (`shape` as a human label, `params` as a parseable
`k=v;k=v`, `dtype`), the measurement (`median_ms`, `mean_ms`, `std_ms`,
`min_ms`, `p99_ms`, `iters`, `warmup_iters`, `timer`), the derived metrics
(`flops`, `bytes`, `tflops`, `bandwidth_tb_s`, `arith_intensity`,
`speedup_vs_ref`) and `refcheck` / `max_abs_diff` / `repro_command`.

Every column the previous 12-column schema had survives with the same name and
meaning, so the CSVs recorded under `.artifacts/` stay comparable.

**Workload** (69 columns) covers the engine, the service, the accuracy sweep and
the decoders, discriminated by `harness`. Cells a harness has no opinion about
stay empty. There is exactly one real-time figure:

```
rtfx = audio_s / wall_s        # higher is faster
```

`flops` and `bytes` are inputs, not outputs — a family declares how much work
its op does and how many bytes it touches, and the rates derive from them. A
family with no meaningful FLOP model leaves the cell empty rather than writing
`0.0`, which reads as "measured zero".

Two sidecars are written next to `--output <path>.csv`:

- `<path>.meta.json` — argv, git commit, device, library versions, every
  `OASR_*` switch, and the JIT cache hashes. That last one makes "confirm a
  fresh JIT hash directory" checkable after the fact instead of remembered.
- `<path>.raw.json` — per-sample arrays that do not belong in a cell: request
  latencies, first-partial times, the rejection-cause histogram.

## Measurement

`--iters` is a floor, not a cap. A 30 µs kernel measured for exactly 30
iterations spans 1 ms of wall time, which is not long enough for the part to
leave its idle clocks — the same kernel measures ~1.4× slower that way.
`--min-measure-ms` (default 20) raises the count until the measurement spans
enough wall time; the row records how many iterations actually ran.

Every statistic in a row comes from **one** timed loop. L2 is evicted between
iterations by default (`--no-flush` to measure a warm cache), and the flush is
outside the timed interval.

Families whose arms differ by a layout or a fusion interleave by default,
rotating the order between rounds — a single-order A/B lets the second arm run
on a warm allocator. `--interleave` / `--no-interleave` override.

## Adding a family

Create `benchmarks/<area>/<name>.py` exposing:

```python
SUBROUTINES: list[str]
DEFAULT_CONFIGS: dict[str, list[dict]]

def parse_args(parser) -> None: ...
def resolve_configs(args, subroutine) -> list[dict]: ...
def build_fns(subroutine, cfg, dtype, args) -> dict[str, Callable]: ...
def describe(subroutine, cfg, dtype) -> Work: ...
```

then register it in `core/registry.py`. The shared driver owns the rest: config
resolution, refcheck, the timed loop, speedup, and the row.

Optional declarations: `FORCE_DTYPE`, `DEFAULT_DTYPE`, `REF_BACKEND`,
`INTERLEAVE` / `INTERLEAVE_SUBROUTINES`, `TOLERANCES`, `NON_GATING_BACKENDS`.

A whole-pipeline harness defines `run(args, reporter, repro_command)` instead of
the four contract functions, and emits `WorkloadRow`s.

## Profiling and autotuning

```bash
ncu --set full -o gemm python benchmarks/run.py \
    --family gemm --subroutine gemm --M 16000 --N 512 --K 2048 \
    --backends cutlass --profile --warmup-iters 0

python benchmarks/run.py --family gemm --subroutine gemm --M 16000 --N 512 --K 512 \
    --autotune --cache gemm_tune.json
```

`--profile` runs one NVTX-wrapped iteration per backend and works for every
family. `--autotune` wraps the sweep in OASR's autotuner and works for every
kernel family, caching the winning config per shape.

See [`docs/benchmarks.md`](../docs/benchmarks.md) for the engine / service /
accuracy recipes and the measurement protocol, and the `/benchmark-kernel`
skill for the profiling workflow. Point-in-time results belong in `.artifacts/`.
