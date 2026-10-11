# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""asrbench -- the ASR inference performance suite.

One workload definition, many backends.  A *dataset* is prepared once into a
manifest; a *workload* is a deterministic selection from it; a *benchmark*
drives a workload through a *backend* under a measurement protocol; a *run*
directory holds the raw per-request records and the summaries derived from
them, so every number can be recomputed without re-running inference.

    python -m benchmarks.asrbench datasets
    python -m benchmarks.asrbench prepare ljspeech --source-dir "$LJSPEECH_DIR"
    python -m benchmarks.asrbench run --profile smoke --backend oasr_engine
    python -m benchmarks.asrbench compare BASELINE_RUN CANDIDATE_RUN
    python -m benchmarks.asrbench report RUN [RUN ...]

See ``docs/asrbench.md``.  Performance is the subject; WER is only a guardrail
and lives in its own command (``validate``), never inside a timed run.
"""

#: Bumped when a result field changes meaning.  Readers key on names, so adding
#: a field does not require a bump.
RESULT_SCHEMA_VERSION = 1

#: The suite's own version, recorded in every run and every prepared dataset.
SUITE_VERSION = "1.0.0"
