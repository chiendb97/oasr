#!/usr/bin/env python3
# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""The GPU test suite split, defined once for every backend that runs it.

Two things execute the GPU suite — the self-hosted runner
(`.github/workflows/test-gpu.yml`) and Modal (`ci/modal_app.py`) — and a split
maintained twice drifts.  Both read the families from here.

The split exists so one failing area does not mask the rest, and so the box
dropping off the bus costs one suite instead of the sweep.

    python ci/gpu_suites.py --list                # names
    python ci/gpu_suites.py --paths engine        # that family's test paths
    python ci/gpu_suites.py --github-matrix       # JSON for `fromJSON()`
    python ci/gpu_suites.py --check               # every test file is covered

``--check`` is the part worth keeping honest: a new `tests/test_*.py` that
nobody adds to a family would never run on the split matrix, and the failure
mode is silence.  It runs in `lint.yml`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

#: family -> test paths.  Keep the groupings coarse; the point is isolation of
#: blast radius, not a taxonomy.
SUITES: dict[str, list[str]] = {
    "kernels": [
        "tests/test_activation.py",
        "tests/test_norm.py",
        "tests/test_conv.py",
        "tests/test_gemm.py",
        "tests/test_gemm_splitk.py",
        "tests/test_gemm_heuristic.py",
        "tests/test_gemm_log_softmax.py",
        "tests/test_softmax.py",
        "tests/test_topk.py",
        "tests/test_pooling.py",
        "tests/test_recurrent.py",
        "tests/test_recurrent_cute.py",
        "tests/test_gated_mlp.py",
        "tests/test_fft.py",
        "tests/test_cmvn.py",
        "tests/test_jit.py",
        "tests/test_attention.py",
        "tests/test_fmha.py",
        "tests/test_fmha_varlen.py",
        "tests/test_block_info.py",
        "tests/test_autotune.py",
        "tests/test_tune_asr_gemm.py",
    ],
    "decoders": [
        "tests/test_decoder.py",
        "tests/test_ctc_decoder_gpu.py",
        "tests/test_ctc_decoder_fused_parity.py",
        "tests/test_wfst_decoder.py",
        "tests/test_decoder_kv.py",
        # Word timings are per-decode-family (`oasr/engine/decode/{alignment,
        # ctc_align,attention_align}.py`), so they fail with the family.
        "tests/test_word_timings.py",
        "tests/test_alignment_cpp.py",
    ],
    "features": [
        "tests/test_features.py",
        "tests/test_features_registry.py",
        "tests/test_extract_features.py",
        "tests/test_sample_rate.py",
    ],
    "engine": [
        "tests/test_engine.py",
        "tests/test_vad.py",
        "tests/test_vad_segmenter.py",
        "tests/test_vad_streaming_segment.py",
        "tests/test_vad_silero.py",
        "tests/test_engine_seams.py",
        "tests/test_offline_graph.py",
        "tests/test_packing_device_layout.py",
        "tests/test_engine_isolation.py",
        "tests/test_pipeline.py",
        "tests/test_scheduler_length_batch.py",
        "tests/test_scheduler_preferred_batch.py",
        "tests/test_scheduler_split.py",
        "tests/test_streaming_backend.py",
        "tests/test_cache_manager.py",
        "tests/test_host_staging.py",
        "tests/test_incremental_executor.py",
        "tests/test_decode_options.py",
        "tests/test_decoding_options.py",
        "tests/test_packing_encoder.py",
        "tests/test_vram_sizing.py",
        "tests/test_engine_metrics.py",
        # The Python client / `oasr` CLI: the request-and-response surface the
        # engine is reached through, against a stub server rather than a GPU.
        "tests/test_client.py",
    ],
    # Its own family so a WER regression is attributable at a glance rather than
    # buried in a model-family failure — and because it is the one suite whose
    # failure means "the output got worse", not "a tensor moved".
    "accuracy": [
        "tests/test_accuracy.py",
    ],
    "models": [
        "tests/test_conformer.py",
        "tests/test_zipformer.py",
        "tests/test_whisper.py",
        "tests/test_paraformer.py",
        "tests/test_speech_llm.py",
        "tests/test_nemotron.py",
        "tests/test_transducer.py",
        "tests/test_transformer_decoder.py",
        "tests/test_rescoring.py",
        "tests/test_model_contract.py",
        "tests/test_model_registry.py",
        "tests/test_checkpoint_native.py",
        "tests/test_config_round_trip.py",
        "tests/test_from_pretrained.py",
        "tests/test_tokenizers.py",
        "tests/test_layer_waist.py",
    ],
}

#: Files deliberately outside the family split because they are reached through
#: an opt-in marker instead (`-m concurrent`), in its own job.
MARKER_ONLY: set[str] = {"tests/test_engine_concurrent.py"}

#: Markers the per-family jobs deselect; the opt-in job runs them separately.
DEFAULT_MARKER_EXPR = "not slow and not concurrent"


def paths_for(name: str) -> list[str]:
    try:
        return SUITES[name]
    except KeyError:
        raise SystemExit(f"unknown suite {name!r}; known: {', '.join(SUITES)}") from None


def github_matrix() -> str:
    """`include:` entries for a GitHub Actions matrix."""
    return json.dumps([{"name": n, "paths": " ".join(p)} for n, p in SUITES.items()])


def check() -> int:
    """Every tests/test_*.py must be in exactly one family (or marker-only)."""
    on_disk = {f"tests/{p.name}" for p in TESTS_DIR.glob("test_*.py")}
    listed: dict[str, list[str]] = {}
    for family, paths in SUITES.items():
        for p in paths:
            listed.setdefault(p, []).append(family)

    problems = []
    missing = sorted(on_disk - set(listed) - MARKER_ONLY)
    if missing:
        problems.append(
            "not in any suite (they would never run on the split matrix):\n"
            + "".join(f"    {p}\n" for p in missing)
        )
    dangling = sorted(set(listed) - on_disk)
    if dangling:
        problems.append("listed but not on disk:\n" + "".join(f"    {p}\n" for p in dangling))
    dupes = sorted(p for p, fams in listed.items() if len(fams) > 1)
    if dupes:
        problems.append(
            "in more than one suite:\n"
            + "".join(f"    {p}: {', '.join(listed[p])}\n" for p in dupes)
        )
    stale_marker = sorted(MARKER_ONLY - on_disk)
    if stale_marker:
        problems.append(
            "MARKER_ONLY entry not on disk:\n" + "".join(f"    {p}\n" for p in stale_marker)
        )

    if problems:
        print("ci/gpu_suites.py is out of sync with tests/:\n", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1
    print(
        f"OK: {len(on_disk)} test file(s) across {len(SUITES)} suites + {len(MARKER_ONLY)} marker-only"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="print suite names")
    g.add_argument("--paths", metavar="NAME", help="print one suite's test paths")
    g.add_argument("--github-matrix", action="store_true", help="print matrix JSON")
    g.add_argument("--check", action="store_true", help="verify every test file is covered")
    args = ap.parse_args()

    if args.list:
        print("\n".join(SUITES))
    elif args.paths:
        print(" ".join(paths_for(args.paths)))
    elif args.github_matrix:
        print(github_matrix())
    elif args.check:
        return check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
