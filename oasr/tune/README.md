# `oasr.tune`

Kernel selection from measured data, per GPU architecture. User documentation:
[`docs/autotuning.md`](../../docs/autotuning.md).

| Module | Role |
|---|---|
| `database.py` | The tuning DB: file format, system/user tiers, validators, snapshot epochs, tier counts |
| `db/` | The shipped (system-tier) tuning files, one per `(arch family, kernel family)` |
| `shapes.py` | Dynamic-dimension buckets shared by tuning and lookup; which autotuner shape dims are dynamic |
| `census.py` | Which shapes to tune: engine keys × traffic × eager probes → `ShapeSet` |
| `bench.py` | The one benchmark protocol (graph loops, L2 rotation, interleaving, halving, gates) |
| `gemm_tune.py` | GEMM-family candidates measured with the protocol, numerics-gated |
| `cost_model.py` | Analytic, calibratable GEMM cost model (pruning, runtime fallback) |
| `arch.py` | `ArchProfile`: static ISA facts, device queries, cached micro-benchmarks |
| `partition.py` | Set cover + DP region segmentation + compile budget |
| `build.py` | `oasr tune build`: census in, tuning file out |
| `prebuild.py` | `oasr tune prebuild`: compile a build's modules without a GPU (`OASR_CUDA_ARCH_LIST`) |
| `evaluate.py` | Selection regret against the measured oracle |
| `telemetry.py` | Runtime misses as the next census |
| `cli.py` | `oasr tune census|build|prebuild|evaluate|diff|status|export-misses` |
| `autotuner.py`, `backends/` | The runtime `oasr.autotune()` (FlashInfer-style registry + cache), on the same protocol |
| `capture.py` | The GEMM shape recorder (`OASR_CAPTURE_GEMM`) the census probes use |
