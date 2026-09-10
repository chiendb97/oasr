# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared test helpers.

``tests/`` is on ``sys.path`` (see ``tests/conftest.py``), so these import by
name from any depth: ``from helpers import REPO_ROOT, tol``.

What lives here is what more than one *family* needs. A helper two files in the
same directory share belongs in that directory's ``conftest.py`` instead —
keeping it near its callers is what makes it get read before it gets copied.

:data:`REPO_ROOT` exists because three test modules used to derive it by
counting ``__file__.parents``, and the count is a function of where the file
sits. Moving such a file into a subdirectory turns it into a collection error
that names a missing path rather than the move.
"""

from __future__ import annotations

from pathlib import Path

#: The repository root, independent of how deep the calling test module sits.
REPO_ROOT = Path(__file__).resolve().parents[2]

from helpers.audio import (  # noqa: E402
    hiss,
    speech_corpus,
    tone,
    wav_path,
    wav_paths,
    waveform,
    waveforms,
)
from helpers.engine import engine_config, make_engine  # noqa: E402
from helpers.kernels import (  # noqa: E402
    ACTIVATION_IDS,
    ACTIVATIONS,
    assert_dest_passing,
    assert_graph_replay,
    device_sm,
    requires_cute,
    requires_sm,
)
from helpers.tolerances import tol  # noqa: E402

__all__ = [
    "ACTIVATIONS",
    "ACTIVATION_IDS",
    "REPO_ROOT",
    "assert_dest_passing",
    "assert_graph_replay",
    "device_sm",
    "engine_config",
    "hiss",
    "speech_corpus",
    "make_engine",
    "requires_cute",
    "requires_sm",
    "tol",
    "tone",
    "wav_path",
    "wav_paths",
    "waveform",
    "waveforms",
]
