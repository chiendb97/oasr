#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Run the OASR GPU test suite on Modal.

A second GPU backend alongside the self-hosted runner, for three reasons: the
self-hosted box is also the benchmarking machine (CI competing for clocks is
exactly the noise the perf work fights), its GPU has fallen off the bus before
in a way only a *host* reset recovers, and a public repo plus a self-hosted
runner is a bad combination the day anyone adds a `pull_request` trigger.

    # one-time, from a machine that has the checkpoints
    modal run ci/modal_app.py::seed_assets

    # everything, on the default GPU
    modal run ci/modal_app.py

    # one family, non-strict, on a different SM
    OASR_MODAL_GPU=H100 modal run ci/modal_app.py --suites kernels --no-strict

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
from gpu_suites import DEFAULT_MARKER_EXPR, SUITES, paths_for  # noqa: E402

APP_NAME = "oasr-gpu-ci"
REPO_REMOTE = "/repo"
ASSETS_MOUNT = "/assets"
JIT_MOUNT = "/root/.cache/oasr/jit"

# sm_120 (GB202) — same compute capability as the RTX 5090 the kernels target.
# Override for a second-SM run: OASR_MODAL_GPU=H100 modal run ...
GPU = os.environ.get("OASR_MODAL_GPU", "RTX-PRO-6000")

# The CMake extension is built at *image build time*, where there is no GPU, so
# setup.py's torch-based arch detection cannot see one and falls back to 80-90.
# sm_120 has to be stated explicitly or `_C.so` is built for the wrong target.
CUDA_ARCHITECTURES = os.environ.get("OASR_MODAL_CUDA_ARCH", "120")

# Must be a torch build with sm_120 support (CUDA 12.8 or newer).
TORCH_INDEX = os.environ.get("OASR_MODAL_TORCH_INDEX", "https://download.pytorch.org/whl/cu128")

#: Where each declared asset lands inside the container.  The *names* are the
#: env vars from tests/assets.py — that module stays the single source of truth
#: for what an asset is; this is only the layout of the Volume.
ASSET_LAYOUT: dict[str, str] = {
    "CKPT_DIR": "u2pp_conformer",
    "WAV_DIR": "wavs",
    "ZIPFORMER_CKPT": "zipformer_ctc",
    "WHISPER_CKPT": "whisper_tiny",
    "OASR_PARAFORMER_CKPT": "paraformer_zh",
    "SPEECH_LLM_TINY": "qwen2_audio_tiny",
    "SPEECH_LLM_CKPT": "qwen2_audio_7b",
    "OASR_TEST_FST": "lang_bpe/HLG.pt",
}

#: Assets the Volume is not expected to hold: upstream *source trees* used only
#: by parity oracles, and the k2 lang dir.  Named here so --strict-assets stays
#: on for everything else rather than being switched off wholesale.
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
    return {name: f"{ASSETS_MOUNT}/{rel}" for name, rel in ASSET_LAYOUT.items()}


@app.function(
    image=image,
    gpu=GPU,
    timeout=3600,
    volumes={ASSETS_MOUNT: assets_vol, JIT_MOUNT: jit_vol},
)
def run_suite(name: str, strict: bool = True, extra: str = "") -> str:
    """Run one family from ci/gpu_suites.py.  Raises on failure."""
    env = {**os.environ, **_asset_env(), "OASR_JIT_DIR": JIT_MOUNT}

    subprocess.run(["nvidia-smi"], check=False)
    print(f"[modal] suite={name} gpu={GPU} strict={strict}", flush=True)

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
        for missing in ALLOW_MISSING:
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
def main(suites: str = "", strict: bool = True, extra: str = ""):
    """Run one, several, or all families concurrently.

    ``--suites`` is a comma-separated list of names from ci/gpu_suites.py;
    empty means all of them.
    """
    names = [s.strip() for s in suites.split(",") if s.strip()] or list(SUITES)
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s): {unknown}; known: {list(SUITES)}")

    print(f"[modal] launching {len(names)} suite(s) on {GPU}: {', '.join(names)}")
    results = list(
        run_suite.starmap(
            [(n, strict, extra) for n in names],
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
    up for a local GPU run needs no extra configuration:

        set -a; source .env; set +a
        export WAV_DIR="$AUDIO_DIR"
        export ZIPFORMER_CKPT=/path/to/icefall-...-zipformer-large-cr-ctc-...
        modal run ci/modal_app.py::seed_assets

    Roughly 23 GiB in total, dominated by Qwen2-Audio-7B at 16 GiB.  Re-running
    overwrites in place, so it is safe to use for a single refreshed asset.
    """
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    import assets as test_assets  # noqa: PLC0415  — needs the sys.path line above

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
