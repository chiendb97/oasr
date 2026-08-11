#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Run the OASR GPU test suite on Modal.

A second GPU backend alongside the self-hosted runner, for three reasons: the
self-hosted box is also the benchmarking machine (CI competing for clocks is
exactly the noise the perf work fights), its GPU has fallen off the bus before
in a way only a *host* reset recovers, and a public repo plus a self-hosted
runner is a bad combination the day anyone adds a `pull_request` trigger.

    # every family, on the default GPU, needing no external input
    modal run ci/modal_app.py::main

    # one family, on a different SM
    OASR_MODAL_GPU=H100 modal run ci/modal_app.py::main --suites kernels

    # the checkpoint-backed half, once the Volume has been seeded (~29 GiB)
    modal run ci/modal_app.py::seed_assets       # one-time, from a box with them
    modal run ci/modal_app.py::main --assets

    # a shell on the same image, to debug a suite that only fails up there
    modal shell ci/modal_app.py::run_suite

``::main`` is not optional.  This file defines two local entrypoints, and
``modal run`` on a file with more than one refuses to pick: bare
``modal run ci/modal_app.py`` exits with "Specify a Modal Function or local
entrypoint to run" before it reaches a GPU.

`--assets` is off by default.  Without it no asset env var is set, every
checkpoint- and audio-gated test skips, and what runs is the coverage that needs
only a GPU: the kernels, the layer waist, the JIT, the schedulers and the decode
plumbing on synthetic weights.  That is the half of the suite a rented GPU can
run on day one; the other half is a seeded Volume away.

The GPU default is **RTX-PRO-6000**: it is the GB202 die at compute capability
12.0, the same sm_120 as the RTX 5090 this project is tuned for.  Every other
Modal accelerator is a different SM, which silently skips the SM120-gated
suites (`tests/test_gemm_heuristic.py`) and exercises different autotuner
paths — useful as a *second* target, wrong as the only one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent
sys.path.insert(0, str(CI_DIR))
sys.path.insert(0, str(REPO_ROOT / "tests"))
import assets as test_assets  # noqa: E402  — both need the sys.path lines above
from gpu_suites import DEFAULT_MARKER_EXPR, SUITES, paths_for  # noqa: E402

APP_NAME = "oasr-gpu-ci"
REPO_REMOTE = "/repo"
ASSETS_MOUNT = "/assets"
JIT_MOUNT = "/root/.cache/oasr/jit"

# sm_120 (GB202) — same compute capability as the RTX 5090 the kernels target.
# Override for a second-SM run: OASR_MODAL_GPU=H100 modal run ...
# `or` rather than a get() default: the workflow passes this through from a
# dispatch input, and an unfilled input arrives as the empty string, which
# get() treats as a value and Modal then rejects as an accelerator.
GPU = os.environ.get("OASR_MODAL_GPU") or "RTX-PRO-6000"

# The CMake extension is built at *image build time*, where there is no GPU, so
# setup.py's torch-based arch detection cannot see one and falls back to 80-90.
# sm_120 has to be stated explicitly or `_C.so` is built for the wrong target.
CUDA_ARCHITECTURES = os.environ.get("OASR_MODAL_CUDA_ARCH", "120")

# Must be a torch build with sm_120 support (CUDA 12.8 or newer).
TORCH_INDEX = os.environ.get("OASR_MODAL_TORCH_INDEX", "https://download.pytorch.org/whl/cu128")

#: Where each declared asset lands inside the container, **derived** from
#: tests/assets.py rather than restated here.  ``Asset.relpath`` already *is*
#: the reference layout (see that module's "Where the paths come from"), so a
#: second copy is a copy that drifts: it did, the run after an architecture was
#: added — the new checkpoint had a declaration, a slot and an `.env.example`
#: line, and every suite that needed it still skipped, which `--strict-assets`
#: then turned into a red run naming an asset the Volume was never told about.
#: An asset with no ``relpath`` has no root-relative slot and is not seedable.
ASSET_LAYOUT: dict[str, str] = {
    asset.env: asset.relpath for asset in test_assets.ASSETS.values() if asset.relpath
}

#: Assets the Volume is not expected to hold *even when it is seeded*: upstream
#: source trees used only by parity oracles, and the k2 lang dir.  Named here so
#: --strict-assets stays on for everything else rather than being switched off
#: wholesale.
ALLOW_MISSING = ("WENET_REF_DIR", "ICEFALL_ZIPFORMER_DIR", "LANG_DIR")

app = modal.App(APP_NAME)

assets_vol = modal.Volume.from_name("oasr-ci-assets", create_if_missing=True)
# Kernels JIT-compile on first *call*; without a warm cache every run pays the
# full compile (~683 MiB of artifacts on the reference box).
jit_vol = modal.Volume.from_name("oasr-ci-jit-cache", create_if_missing=True)

image = (
    # -devel, not -runtime: the JIT shells out to nvcc at run time.
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "cmake", "ninja-build", "protobuf-compiler")
    # setuptools-rust builds the oasr._core PyO3 extension during pip install.
    .run_commands(
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        "echo 'source $HOME/.cargo/env' >> /root/.bashrc",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("torch", "torchaudio", "torchcodec", index_url=TORCH_INDEX)
    .pip_install(
        # pyproject [project.dependencies] + the optional extras whose absence
        # would turn real coverage into importorskips.
        "numpy",
        "apache-tvm-ffi>=0.1.0",
        "jinja2>=3.0",
        "PyYAML>=5.4",
        "safetensors",
        "huggingface_hub",
        "sentencepiece",
        "tokenizers",
        "transformers",
        "soundfile",
        # CuteDSL FMHA; without it oasr.jit.attention falls back to SDPA and the
        # cute-backend tests stop covering the kernel that ships.  Pinned, and
        # kept in step with the 3rdparty/cutlass submodule tag: the two halves
        # of CUTLASS are versioned together upstream, and a CI that compiles
        # kernels against one release while tracing them on another is testing a
        # combination nobody runs.  Floor lives in
        # oasr/jit/attention.py::MIN_CUTEDSL_VERSION.
        "nvidia-cutlass-dsl==4.6.1",
        "pytest==8.1.1",
    )
    # `pip install -e . --no-build-isolation` below needs pyproject's
    # [build-system] requires already present — setup.py must `import torch` to
    # find its cmake prefix, which an isolated build env cannot do.  Keep this
    # list in step with pyproject.toml.
    .pip_install(
        "setuptools>=61.0",
        "wheel",
        "cmake>=3.18",
        "pybind11>=2.10",
        "ninja",
        "setuptools-rust>=1.10",
    )
    # copy=True so the build below is baked into a cached layer.  A code change
    # re-runs only this final layer; the CUDA/Rust/torch layers stay cached.
    # Absolute, so `modal run` works from any cwd.
    .add_local_dir(
        REPO_ROOT,
        REPO_REMOTE,
        copy=True,
        ignore=["**/.git", "**/target", "**/build", "**/__pycache__", "**/*.so", "**/.venv"],
    )
    .run_commands(
        f"cd {REPO_REMOTE} && CUDA_ARCHITECTURES={CUDA_ARCHITECTURES} pip install -e . --no-build-isolation"
    )
    .workdir(REPO_REMOTE)
)


def _asset_env() -> dict[str, str]:
    """Asset env vars as the container sees them.  Evaluated *remotely*."""
    env = {name: f"{ASSETS_MOUNT}/{rel}" for name, rel in ASSET_LAYOUT.items()}
    # AUDIO_PATH is the one asset with no slot of its own: it is "some wav out
    # of WAV_DIR", a thing to pick rather than a thing to seed — which is what
    # tests/assets.py tells a human to do (`ls $WAV_DIR/*.wav | head -1`).  Left
    # unpicked it is simply unset in the container, and --strict-assets turns
    # that into a failed suite over a file the Volume already holds.  Sorted, so
    # every suite in a run agrees on which wav that is.
    wavs = sorted(Path(env["WAV_DIR"]).glob("*.wav")) if "WAV_DIR" in env else []
    if wavs:
        env["AUDIO_PATH"] = str(wavs[0])
    return env


@app.function(
    image=image,
    gpu=GPU,
    timeout=3600,
    volumes={ASSETS_MOUNT: assets_vol, JIT_MOUNT: jit_vol},
)
def run_suite(name: str, strict: bool = True, extra: str = "", assets: bool = True) -> str:
    """Run one family from ci/gpu_suites.py.  Raises on failure.

    ``assets=False`` runs the suite with the Volume unused: no asset env var is
    set, so every checkpoint- and audio-gated test skips and what is left is the
    kernel / layer / plumbing coverage that needs only a GPU.  ``strict`` stays
    meaningful there — the allow-list simply widens to every declared asset, so
    the run still *names* what it did not cover instead of turning the gate off.
    """
    env = {**os.environ, "OASR_JIT_DIR": JIT_MOUNT}
    if assets:
        env.update(_asset_env())

    subprocess.run(["nvidia-smi"], check=False)
    print(f"[modal] suite={name} gpu={GPU} strict={strict} assets={assets}", flush=True)

    cmd = [
        "python",
        "-m",
        "pytest",
        *paths_for(name),
        "-m",
        DEFAULT_MARKER_EXPR,
        "-q",
        "-rs",
        "-rE",
    ]
    if strict:
        cmd.append("--strict-assets")
        allowed = ALLOW_MISSING if assets else tuple(test_assets.ASSETS)
        for missing in allowed:
            cmd += ["--allow-missing-asset", missing]
    if extra:
        cmd += extra.split()

    try:
        subprocess.run(cmd, cwd=REPO_REMOTE, env=env, check=True)
    finally:
        # Persist whatever the JIT compiled so the next run starts warm.
        jit_vol.commit()
    return name


@app.local_entrypoint()
def main(suites: str = "", strict: bool = True, extra: str = "", assets: bool = False):
    """Run one, several, or all families concurrently.

    ``--suites`` is a comma-separated list of names from ci/gpu_suites.py;
    empty means all of them.

    ``--assets`` reads the checkpoints and audio out of the ``oasr-ci-assets``
    Volume and is **off by default**, because the Volume has to be seeded first
    (``seed_assets``, ~29 GiB) and a run against an empty one under
    ``--strict-assets`` fails on every gated test rather than skipping it.  Off,
    this is the GPU coverage that needs no external input; on, it is the whole
    suite.  Turn it on once ``seed_assets`` has run.
    """
    names = [s.strip() for s in suites.split(",") if s.strip()] or list(SUITES)
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s): {unknown}; known: {list(SUITES)}")

    scope = "with checkpoints" if assets else "no checkpoints (kernels/plumbing only)"
    print(f"[modal] launching {len(names)} suite(s) on {GPU}, {scope}: {', '.join(names)}")
    results = list(
        run_suite.starmap(
            [(n, strict, extra, assets) for n in names],
            return_exceptions=True,
        )
    )

    failed = [(n, r) for n, r in zip(names, results) if isinstance(r, Exception)]
    for n, r in zip(names, results):
        print(f"  {'FAIL' if isinstance(r, Exception) else 'ok  '}  {n}")
    if failed:
        # One line per failure, then a non-zero exit so `modal run` fails the job.
        for n, exc in failed:
            print(f"\n--- {n} ---\n{exc}", file=sys.stderr)
        raise SystemExit(f"{len(failed)}/{len(names)} suite(s) failed")


@app.local_entrypoint()
def seed_assets(dry_run: bool = False):
    """Upload the checkpoints and audio into the `oasr-ci-assets` Volume.

    Run once from a machine that has them.  Source paths come from the same
    env vars the test suite reads (see tests/assets.py), so a box already set
    up for a local GPU run needs no extra configuration beyond `.env`:

        set -a; source .env; set +a
        modal run ci/modal_app.py::seed_assets --dry-run   # check the plan
        modal run ci/modal_app.py::seed_assets

    Roughly 29 GiB in total, dominated by Qwen2-Audio-7B at 16 GiB.  Re-running
    overwrites in place, so it is safe to use for a single refreshed asset.
    """
    plan, missing = [], []
    for name, rel in ASSET_LAYOUT.items():
        src = test_assets.resolve(name)
        if src is None:
            missing.append(f"{name} (looked at {test_assets.declared(name) or '<unset>'})")
            continue
        plan.append((name, Path(src), rel))

    for name, src, rel in plan:
        kind = "dir " if src.is_dir() else "file"
        print(f"  {kind} {name:<22} {src}  ->  {ASSETS_MOUNT}/{rel}")
    for m in missing:
        print(f"  SKIP {m}")
    if not plan:
        raise SystemExit("nothing to upload; export the asset env vars first")
    if dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    with assets_vol.batch_upload(force=True) as batch:
        for _, src, rel in plan:
            if src.is_dir():
                batch.put_directory(str(src), f"/{rel}")
            else:
                batch.put_file(str(src), f"/{rel}")
    print(f"\nuploaded {len(plan)} asset(s) to the oasr-ci-assets Volume")
