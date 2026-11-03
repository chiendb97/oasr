# Contributing to OASR

Thanks for your interest in OASR. Bug reports, model requests, documentation
fixes, kernels and new architectures are all welcome. Everyone taking part is
expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

This page is the short version. The complete rules (the nine extension axes,
the kernel layering, and the traps that have already cost the project a
release) are in [`AGENTS.md`](AGENTS.md). It is written for human contributors
and coding agents alike, and it is worth reading before a non-trivial change.

## Ways to contribute

- **Report a bug.** Use the
  [bug report form](https://github.com/chiendb97/oasr/issues/new?template=bug_report.yml).
  The GPU, the checkpoint and the service mode matter most, so the form asks
  for them.
- **Request a model.** OASR serves checkpoints from WeNet, icefall/k2, FunASR
  and Hugging Face. If one you need does not load, open a
  [model request](https://github.com/chiendb97/oasr/issues/new?template=model_request.yml).
- **Pick up an issue.** Issues labelled
  [`good first issue`](https://github.com/chiendb97/oasr/labels/good%20first%20issue)
  and [`help wanted`](https://github.com/chiendb97/oasr/labels/help%20wanted)
  are scoped for outside contributors.
- **Improve the docs.** [`docs/`](docs/README.md) holds stable technical
  documentation, and a fix there needs no GPU to validate.

For a substantial change, such as a new architecture, decode family or kernel
family, please open an issue first so the design can be agreed before you
write the code.

## Development setup

```bash
git clone https://github.com/chiendb97/oasr.git && cd oasr
git submodule update --init          # provides 3rdparty/cutlass
pip install -e ".[all]"              # CUDA kernels JIT-compile on first call
pip install -r requirements-dev.txt  # the pinned lint/test versions CI uses
pre-commit install                   # optional: the fast half of the lint workflow

cp .env.example .env                 # then edit the checkpoint and audio paths
set -a; source .env; set +a          # every test and benchmark command assumes this
```

Requirements: CUDA ≥ 11.8, an NVIDIA GPU of compute capability 7.5 or newer,
Python ≥ 3.10, CMake ≥ 3.18, a Rust toolchain and `protobuf-compiler`.

## Making a change

1. Branch from `main` as `<type>/<topic>`, where the type is `feat/`, `fix/`,
   `perf/`, `refactor/`, `chore/` or `docs/`.
2. **Extend through a registry, not the engine core.** Model architectures,
   checkpoint formats, decode families, streaming runtimes, batching policies,
   tokenizers, feature frontends, speech detectors and quantization schemes are
   each added by subclassing a base class and registering it. Each one has a
   cookbook entry in [`docs/architecture.md`](docs/architecture.md).
3. **Build models from `oasr.layers`**, never from bare `nn.Linear`,
   `nn.LayerNorm`, `nn.Embedding` or `nn.Conv*`.
   `tests/models/test_layer_waist.py` enforces this.
4. **Add tests** next to the module they cover, under
   `tests/{kernels,models,decoders,features,engine,accuracy}/`. A new
   architecture also needs a tiny config in `tests/models/test_layer_waist.py`,
   and its checkpoint declared in `tests/assets.py` and `.env.example`.

## Before you open a pull request

```bash
set -a; source .env; set +a
black oasr/ tests/ benchmarks/ scripts/ ci/ && isort oasr/ tests/ benchmarks/ scripts/ ci/
ruff check oasr/ tests/ benchmarks/ scripts/ ci/
python scripts/mypy_ratchet.py
python ci/gpu_suites.py --check
pytest tests/ -m "not slow"
cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
```

- **Read the `external assets:` table** that pytest prints at the end. Without
  the checkpoints, the real-checkpoint tests *skip* and the run still exits 0.
  If the asset covering your change says `MISSING`, your change has not been
  tested. `--strict-assets` turns a missing asset into a failure.
- **Run cargo from `rust/`, and never with `--workspace`.** Two crates enable
  mutually exclusive pyo3 features. See
  [AGENTS.md § Rust workspace](AGENTS.md#rust-workspace).
- **Test a kernel change in a served dtype** (fp16/bf16 on CUDA). fp32 and CPU
  route to the torch reference, so they never reach the kernel.
- **Keep benchmark numbers out of `README.md`, `AGENTS.md` and `docs/`.**
  Those pages describe what stays true across releases. Put measurements in the
  pull request description.

Pull requests run lint and CPU tests on GitHub-hosted runners. The GPU suites
run on self-hosted and Modal runners that a maintainer triggers, so a
maintainer will run them on your branch before merging.

## Pull request guidelines

- Keep one concern per pull request.
- Describe *what* changed and *why*, and say how you validated it (which GPU,
  which checkpoints, and what the asset table said).
- Update the relevant page in `docs/` when you change behaviour that a user or
  an extender relies on.

## Reporting a security issue

Please do not open a public issue for a vulnerability. Follow
[`SECURITY.md`](SECURITY.md).

## License

OASR is licensed under the [Apache License 2.0](LICENSE). By contributing, you
agree that your contributions are licensed under the same terms.
