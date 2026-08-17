# CI and Testing

Four workflows under `.github/workflows/`, plus a pre-commit config that mirrors
the fast half of them locally.

| Workflow | Runner | Trigger | Gates |
|---|---|---|---|
| `lint.yml` | GitHub-hosted | push to `main`, every PR | black, isort, ruff, the mypy ratchet, `cargo fmt`, `cargo clippy -D warnings`, `cargo test` |
| `test-cpu.yml` | GitHub-hosted | push to `main`, every PR | `pytest tests/` on Python 3.10 + 3.12 with no GPU, behind a `--min-passed` coverage floor |
| `test-gpu.yml` | self-hosted `oasr-gpu` | nightly + manual | the full suite **with `--strict-assets`**, split per family, plus the `slow` and `concurrent` markers |
| `test-gpu-modal.yml` | GitHub-hosted → Modal GPU | weekly + manual | the same per-family split, on rented sm_120 |

## Running the gates locally

```bash
pip install -r requirements-dev.txt

black --check oasr/ tests/ benchmarks/ scripts/ ci/
isort --check-only oasr/ tests/ benchmarks/ scripts/ ci/
ruff check oasr/ tests/ benchmarks/ scripts/ ci/
python scripts/mypy_ratchet.py

cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
cd rust && cargo clippy -p oasr-core --lib -- -D warnings
```

Or install the hooks and let them run on commit:

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

Tool versions are **pinned** in `requirements-dev.txt` and
`.pre-commit-config.yaml`, and the two files must agree: black and ruff reformat
differently across releases, so an unpinned toolchain turns a green PR red on an
upstream release with no change to this repo.

A full local GPU run additionally needs the external assets exported:

```bash
set -a; source .env; set +a                      # CKPT_DIR, AUDIO_DIR
export WAV_DIR="$AUDIO_DIR"
export AUDIO_PATH=$(ls "$AUDIO_DIR"/*.wav | head -1)
export ZIPFORMER_CKPT=/path/to/icefall-asr-librispeech-zipformer-large-cr-ctc-...

pytest tests/ -m "not slow and not concurrent" -q -rs
```

**Read the asset table at the end before believing the count.**

## The asset gate

`pytest tests/` used to report a fully green suite while silently skipping every
real-checkpoint test. Several environment variables gated those suites, each read
ad hoc in its own module with its own default path, and pytest's summary does not
distinguish "passed" from "skipped because the checkpoint was not on this box".
That is how the `audio_scale` defect — which drops the leading token of every
transcript — shipped and survived a green suite.

Two mechanisms close it.

**Every gated input is declared once**, in `tests/assets.py`, together with the
marker file that proves it is really there. An HF snapshot present as dangling
LFS symlinks with no payload is not a usable checkpoint, and the probe says so.
`assets.require(...)` is the single skip site, so every skip is counted and every
reason string is uniform.

**Every run prints what it did not check.** Not behind a flag — always:

```
external assets:
  CKPT_DIR                 ok       /.../20210610_u2pp_conformer_exp_librispeech
  WHISPER_CKPT             MISSING  /.../whisper-tiny  [18 test(s) skipped]
  ...
  -> 18 test(s) skipped for missing assets. A green run does not cover them;
     use --strict-assets to make this fatal.
```

`--strict-assets` makes it fatal, which is what the GPU workflows run. An asset a
runner genuinely cannot have is named one by one, so the gap sits in the workflow
file where it can be read rather than inside a skip nobody reads:

```bash
pytest tests/ --strict-assets --allow-missing-asset WENET_REF_DIR
```

An unknown name is a usage error, not a silent no-op.

`--min-passed N` guards the passed *count*. A CUDA guard added at the wrong
scope, or an import that quietly turns a module into skips, otherwise shrinks
coverage without turning anything red. Raise the floor to just under the observed
count after a green run; never lower it to make a red run pass.

### Adding a gated input

Add an `Asset` to `tests/assets.py`; nothing else changes. Then, in the test:

```python
@pytest.mark.requires_assets("MY_CKPT")      # class or function
```

or, inside a fixture or body:

```python
path = assets.require("MY_CKPT")
wavs = assets.require_wavs(4)                # also gates on the file count
```

**Never read the environment variable directly** — that is exactly the pattern
that produced the trap.

## The accuracy gate

Everything else in this repo compares tensors. That is necessary and it is not
sufficient: a parity oracle feeds identical features to both sides, so a bug in
*how audio becomes features* cancels on both and the suite stays green.

`tests/test_accuracy.py` measures WER on a fixed 200-utterance LJSpeech subset and
fails when a rate exceeds its recorded value in `ci/wer-reference.json` plus a
tolerance of 0.3 absolute.

**Those rates are for regression detection, not publication.** LJSpeech is
out-of-domain for every architecture and the subset is 200 utterances. Do not
quote them as OASR's WER.

The tolerance is slack for a different GPU or torch build, not for noise: the
offline rates are bit-identical across consecutive runs *and* across
`max_batch_size` 1 / 8 / 16 / 32, which also makes "batching does not change the
answer" a measured property rather than an assumption. Where that does **not**
hold the entry says so and records the spread instead of widening the tolerance —
see the `transducer` and `nemotron_streaming` comments.

An entry may pin `decode_options`, which is the decode configuration the rate was
measured under. The `llm` row does: for a speech-LLM the prompt is part of what
is being asked, and a different prompt moves the WER with no defect behind it.
That row is also the family's only end-to-end check — its decode path is the one
with per-row KV offsets, group merging, paged KV and step CUDA graphs behind it,
and a graph that captured the wrong thing returns a *plausible transcript*, which
is exactly what a tensor oracle cannot see.

### Manifests ship; audio does not

A manifest is JSON Lines — `{"id", "audio", "text"}` — with `audio` resolved
against `--audio-root`, so 200 utterances is 31 KB of checked-in text. Build one
for a corpus you have:

```bash
python benchmarks/bench_accuracy.py --build-manifest out.jsonl \
    --audio-root $WAV_DIR --transcripts .../metadata.csv --limit 200
```

**One trap.** Whisper's `EnglishTextNormalizer` *deletes* `(...)` and `[...]`
spans, because in many corpora they annotate non-speech events. In LJSpeech the
parenthetical is read aloud, so leaving brackets in the manifest drops those
words from the reference while the hypothesis keeps them — scoring a correct
transcription as a run of insertions. `--build-manifest` unwraps brackets by
default (`--keep-brackets` to opt out) and
`TestManifest::test_no_bracketed_spans_survive` keeps it that way.

`oasr/testing/wer.py` reports the **corpus** rate — total edits over total
reference words — not the mean of per-utterance rates. Averaging weights a
three-word utterance like a thirty-word one and is not comparable to any
published figure.

Sweeping rather than gating is `benchmarks/bench_accuracy.py` — see
[benchmarks.md](benchmarks.md).

## Two GPU backends

The GPU suite runs in two places from **one** family split, `ci/gpu_suites.py`. A
split maintained twice drifts, and the failure mode is a test file that runs
nowhere — so `--check` (every `tests/test_*.py` in exactly one family) is both a
pre-commit hook and a `lint.yml` step.

|  | self-hosted (`test-gpu.yml`) | Modal (`test-gpu-modal.yml`) |
|---|---|---|
| GPU | the reference RTX 5090 — sm_120 | `RTX-PRO-6000` — GB202, also sm_120 |
| checkpoints | already on disk | a Volume, seeded once |
| cost | none | GPU-minutes per run |
| when | nightly sweep | weekly, on demand, and when the box is busy or wedged |

**Use the self-hosted box as the primary.** It is the SM this project ships on
and the assets are local. Modal exists because that box is also the benchmarking
machine (CI competing for clocks is exactly the noise the perf work fights),
because its GPU has fallen off the bus in a way only a *host* reset recovers, and
because a second SM is worth having — `test_gemm_heuristic.py` skips its whole
rule-table suite unless `_SM == 120`, so running once on another architecture is
the cheapest way to find code that assumed one.

```bash
modal run ci/modal_app.py                                  # all families
modal run ci/modal_app.py --suites kernels,engine          # a subset
OASR_MODAL_GPU=H100 modal run ci/modal_app.py --no-strict   # second SM
```

### Seeding the assets Volume

Once, from a machine that already has the checkpoints. Source paths come from the
same environment variables the suite reads, so a box set up for a local GPU run
needs no extra configuration:

```bash
set -a; source .env; set +a
export WAV_DIR="$AUDIO_DIR"
export ZIPFORMER_CKPT=/path/to/icefall-...-zipformer-large-cr-ctc-...
modal run ci/modal_app.py::seed_assets --dry-run   # print the plan first
modal run ci/modal_app.py::seed_assets
```

A second Volume caches `~/.cache/oasr/jit`. Without it every run recompiles the
kernels from cold, because they JIT on first *call*.

Two image details are load-bearing. The base is `nvidia/cuda:*-devel`, not
`-runtime`, because the JIT shells out to `nvcc` at run time. And
`CUDA_ARCHITECTURES=120` is passed explicitly: the CMake extension is built at
image-build time where there is no GPU, so `setup.py`'s torch-based arch
detection cannot see one and falls back to 80–90.

### Security — this repo is public

`test-gpu.yml` triggers on `schedule` and `workflow_dispatch` only. **Never add
`pull_request`** — a fork PR would execute arbitrary code on the maintainer's
machine. The Modal workflow does not have that problem: the GitHub-hosted job
only holds a token and the GPU work happens in Modal's sandbox. Fork PRs are not
given secrets either, so `modal run` from a fork fails closed rather than running
unauthenticated.

## Why `test-cpu.yml` builds nothing

`oasr/_C*.so` (CMake + CUDA) and `oasr/_core*.so` (setuptools-rust) are both
absent on a GitHub-hosted runner, and `pip install -e .` would need the CUDA
toolkit for a job that cannot use a GPU anyway. It does not need to: `import
oasr` works without either, because the CUDA kernels JIT-compile on first *call*
and `_C` is loaded lazily. The job puts the repo on `PYTHONPATH` and runs pytest
directly.

Files whose every test allocates on `device="cuda"` carry a module-level
`pytestmark` skip, `tests/test_decoder.py` carries an `importorskip("oasr._C")`,
and `@pytest.mark.cuda` actually gates.

Three rules follow, each of which cost a red run to learn:

- **"No GPU" and "no CUDA toolkit" are different conditions**, and a developer box
  has both. Reproducing a GitHub runner locally therefore needs *four* things
  removed, not one — the GPU, the compiled extensions, the checkpoints, **and**
  nvcc plus a warm `~/.cache/oasr/jit`. Anything that JIT-compiles will otherwise
  pass locally off the cache and fail in CI.
- **No pipes in a step that is supposed to gate.** GitHub runs steps under
  `bash -e`, so `pytest ... --collect-only | tail -5` reports *tail's* exit
  status. A collection error then looks like a passing step, and the real failure
  surfaces later as a bare `exit code 2` — pytest's `INTERRUPTED`, i.e. "a module
  failed to import", not "a test failed". A test module that imports a
  third-party package at module scope without `importorskip` can take down the
  entire session.
- **No source file may be gitignored.** A bare `checkpoints/` pattern matches a
  directory of that name at *any* depth, which once kept the whole
  `oasr/checkpoints/` package out of every clone while the local worktree had it
  — and silently excluded those files from black, isort and ruff, all three of
  which respect `.gitignore`. Patterns are anchored now (`/checkpoints/`) and
  `scripts/check_no_ignored_sources.py` runs as a pre-commit hook. It only works
  locally: on CI the files are already absent, so there is nothing left to find.

Python 3.10 is in the matrix because `pyproject.toml` declares
`requires-python = ">=3.10"`, and a floor nobody tests is a guess. That floor is
real: `oasr/decoder/wfst/graph_image.py` annotates defaults with PEP 604 unions
(`np.ndarray | None`) and has no `from __future__ import annotations`, so
importing it on 3.9 raises `TypeError` at def time.

## Why mypy is a ratchet, not a gate

`mypy oasr/` reports several hundred errors. Nearly all are untyped-torch noise
(`no-any-return`, `attr-defined` on tensors) rather than defects, and the CuteDSL
kernels under `oasr/kernels/cute/` are excluded outright because the
`cutlass.cute` decorators rewrite function bodies and mypy sees the DSL's
injected names as undefined.

Gating on zero would mean the job is red forever and nobody reads it. Gating on
*no file got worse* is a check that passes today and still fails on a real
regression:

```bash
python scripts/mypy_ratchet.py            # check against ci/mypy-baseline.json
python scripts/mypy_ratchet.py --update   # after a cleanup, lower the numbers
```

A file whose count *drops* is reported as slack to reclaim, not an error, so a
cleanup commit does not have to touch the baseline to stay green. A file whose
count *rises* fails the job and names the file. Fix the new error or narrow it
with a targeted `# type: ignore[code]` — do not raise the baseline.

Because the baseline counts errors coming out of *third-party stubs*, the
environment is part of it and is pinned in `ci/mypy-requirements.txt`:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r ci/mypy-requirements.txt -r requirements-dev.txt
```

Two settings there are load-bearing:

- **`[tool.mypy]` sets no `python_version`.** It applies to third-party stubs too,
  so pinning it below the interpreter aborts the run before a line of this repo
  is checked (`"3.8"` dies on torch's `match` statements; `"3.10"` dies on
  numpy's PEP 695 `type` statements). mypy runs at the interpreter's version; CI
  pins that to 3.12. Real 3.10 compatibility is covered by `test-cpu.yml`
  actually running the suite on 3.10.
- **`oasr/decoder/wfst/lattice.py` is excluded.** mypy hits an INTERNAL ERROR on
  its `np.lexsort` against recent numpy stubs, which kills the whole run; a
  per-module `ignore_errors` does not help, because the crash happens during
  checking rather than reporting.

## Known lint deviations

- **`B905` (`zip(..., strict=)`) is in ruff's ignore list.** It is a real bug
  class and enabling it is a worthwhile follow-up, but deciding `strict=True` per
  call site requires knowing the lengths always match, and there are dozens of
  them across the engine and the kernels. Turning it on blind would be a guess,
  not a fix.
- **`oasr-core` allows `clippy::useless_conversion` at crate scope** — it fires
  inside `#[pyfunction]`'s own expansion, where a function-level `#[allow]` does
  not reach.
- **`oasr-server-grpc` allows `clippy::result_large_err`** — `tonic::Status` is
  176 bytes and is tonic's type, not ours.

## Rust: never `cargo --workspace`

`oasr-core` enables `pyo3/extension-module` while `oasr-server` enables
`pyo3/auto-initialize`; the two are mutually exclusive and Cargo unifies features
per build, so one invocation covering both fails to compile. `oasr-core` is
excluded from `default-members` and linted in its own step. Run cargo from
`rust/`, not the repo root, so the target-dir redirect applies — see
[CLAUDE.md](../CLAUDE.md#rust-workspace).
