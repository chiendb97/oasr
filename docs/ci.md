# CI, linting, and what a green run means

Three workflows under `.github/workflows/`, plus a pre-commit config that
mirrors the fast half of them locally.

| Workflow | Runner | Trigger | Gates |
|---|---|---|---|
| `lint.yml` | GitHub-hosted | push to `main`, every PR | black, isort, ruff, the mypy ratchet, `cargo fmt`, `cargo clippy -D warnings`, `cargo test` |
| `test-cpu.yml` | GitHub-hosted | push to `main`, every PR | `pytest tests/` on Python 3.10 + 3.12 with no GPU, behind a `--min-passed` coverage floor |
| `test-gpu.yml` | self-hosted `oasr-gpu` | nightly 02:30 UTC + manual | the full suite **with `--strict-assets`**, split per family, plus the `slow` and `concurrent` markers |

---

## The thing this is actually for

`pytest tests/` used to report a fully green suite while silently skipping
every real-checkpoint test. Five env vars gated those suites, each read ad hoc
in its own module with its own default path, and pytest's summary does not
distinguish "passed" from "skipped because the checkpoint was not on this box".
Exporting the five by hand once turned a reported *1280 passed* into three more
failures, one of them a real bug (`audio_scale`, which costs the leading token
of every transcript). The bug was fixed; the mechanism that hid it was not.

Two changes close it.

**Every gated input is declared once**, in `tests/assets.py`, with the marker
file that proves it is really there — an HF snapshot present as dangling LFS
symlinks with no payload is not a usable checkpoint, and the `ZIPFORMER_CKPT`
probe says so. `assets.require(...)` is the single skip site, so every skip is
counted and every reason string is uniform.

**Every run prints what it did not check.** Not a flag — always:

```
external assets:
  CKPT_DIR                 ok       /.../20210610_u2pp_conformer_exp_librispeech
  WHISPER_CKPT             MISSING  /.../whisper-tiny  [18 test(s) skipped]
  ...
  -> 18 test(s) skipped for missing assets. A green run does not cover them;
     use --strict-assets to make this fatal.
```

**`--strict-assets` makes it fatal**, which is what the GPU workflow runs. An
asset the runner genuinely cannot have is named one by one:

```bash
pytest tests/ --strict-assets --allow-missing-asset WENET_REF_DIR
```

so the gap sits in the workflow file where it can be read, instead of inside a
skip nobody reads. An unknown name is a usage error, not a silent no-op.

### Adding a gated input

Add an `Asset` to `tests/assets.py`; nothing else changes. In the test, either

```python
@pytest.mark.requires_assets("MY_CKPT")      # class or function
```

or, inside a fixture or body:

```python
path = assets.require("MY_CKPT")
wavs = assets.require_wavs(4)                # also gates on the file count
```

Do not read the env var directly — that is exactly the pattern that produced
the trap.

---

## Running the gates locally

```bash
pip install -r requirements-dev.txt

black --check oasr/ tests/ benchmarks/ scripts/
isort --check-only oasr/ tests/ benchmarks/ scripts/
ruff check oasr/ tests/ benchmarks/ scripts/
python scripts/mypy_ratchet.py

cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
cd rust && cargo clippy -p oasr-core --lib -- -D warnings
```

Or install the hooks and let them run on commit:

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

Tool versions are **pinned** in `requirements-dev.txt` and `.pre-commit-config.yaml`,
and the two files must agree: black and ruff reformat differently across
releases, so an unpinned toolchain turns a green PR red on an upstream release
with no change to this repo.

### A full local GPU run

```bash
set -a; source .env; set +a                      # CKPT_DIR, AUDIO_DIR
export WAV_DIR="$AUDIO_DIR"
export AUDIO_PATH=$(ls "$AUDIO_DIR"/*.wav | head -1)
export ZIPFORMER_CKPT=/path/to/icefall-asr-librispeech-zipformer-large-cr-ctc-20241018

pytest tests/ -m "not slow and not concurrent" -q -rs
```

Read the asset table at the end before believing the count.

---

## Why mypy is a ratchet, not a gate

`mypy oasr/` reports ~495 errors across 81 files. Nearly all are untyped-torch
noise (`no-any-return`, `attr-defined` on tensors) rather than defects, and the
CuteDSL kernels under `oasr/kernels/cute/` are excluded outright because the
`cutlass.cute` decorators rewrite function bodies and mypy sees the DSL's
injected names as undefined — that was ~90% of the `name-defined` errors and
none of them were real.

Gating on zero would mean the job is red forever and nobody reads it. Gating on
*no file got worse* is a check that passes today and still fails on a real
regression:

```bash
python scripts/mypy_ratchet.py            # check against ci/mypy-baseline.json
python scripts/mypy_ratchet.py --update   # after a cleanup, lower the numbers
```

A file whose count drops is reported as slack to reclaim, not an error, so a
cleanup commit does not *have* to touch the baseline to stay green. A file whose
count rises fails the job and names the file. Fix the new error or narrow it
with a targeted `# type: ignore[code]` — do not raise the baseline.

The linters are worth running for what they *do* catch: the first pass over this
repo found `ClassVar` used without importing it in
`oasr/models/transducer/convert.py`, and a test calling a helper that a refactor
had deleted (`tests/test_conformer.py`, which had been raising `NameError` on
every run it was not skipped for).

---

## Why `test-cpu.yml` builds nothing

`oasr/_C*.so` (CMake + CUDA) and `oasr/_core*.so` (setuptools-rust) are both
absent on a GitHub-hosted runner, and `pip install -e .` would need the CUDA
toolkit for a job that cannot use a GPU anyway. It does not need to: `import
oasr` works without either, because the CUDA kernels JIT-compile on first
*call* and `_C` is loaded lazily. The job puts the repo on `PYTHONPATH` and runs
pytest directly.

Files whose every test allocates on `device="cuda"` carry a module-level
`pytestmark` skip, and `tests/test_decoder.py` carries an
`importorskip("oasr._C")`, so the CPU run is green and meaningful rather than a
wall of `RuntimeError: No CUDA GPUs are available`. Today that is **672 passed,
603 skipped**.

`--min-passed N` is the guard on that number. A CUDA guard added at the wrong
scope, or an import that quietly turns a module into skips, otherwise shrinks
coverage without turning anything red. Raise the floor to just under the
observed count after a green run; do not lower it to make a red run pass.

Python 3.10 is in the matrix because `pyproject.toml` declares
`requires-python = ">=3.10"`, and a floor nobody tests is a guess. That floor is
real, not conservative: `oasr/decoder/wfst/graph_image.py` annotates defaults
with PEP 604 unions (`np.ndarray | None`) and has no
`from __future__ import annotations`, so importing it on 3.9 raises `TypeError`
at def time. The repo previously claimed `>=3.8`.

---

## Rust: never `cargo --workspace`

`oasr-core` enables `pyo3/extension-module` while `oasr-server` enables
`pyo3/auto-initialize`; the two are mutually exclusive and Cargo unifies
features per build, so one invocation covering both fails to compile.
`oasr-core` is excluded from `default-members` and linted in its own step. Run
cargo from `rust/`, not the repo root, so `rust/.cargo/config.toml`'s target-dir
redirect applies.

One clippy lint is allowed at crate scope: `oasr-core`'s
`clippy::useless_conversion` fires inside `#[pyfunction]`'s own expansion, where
a function-level `#[allow]` does not reach. `oasr-server-grpc` allows
`clippy::result_large_err` for the same kind of reason — `tonic::Status` is 176
bytes and is tonic's type, not ours.

`B905` (`zip(..., strict=)`) is in ruff's ignore list. It only became reachable
when the target version moved to py310, it is a real bug class, and enabling it
is a worthwhile follow-up — but deciding `strict=True` per call site requires
knowing the lengths always match, and there are 38 of them across the engine and
the kernels. Turning it on blind would be a guess, not a fix.
