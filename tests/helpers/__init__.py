# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared test helpers.

``tests/`` is on ``sys.path`` (see ``tests/conftest.py``), so these import by
name from any depth: ``from helpers import REPO_ROOT``.

:data:`REPO_ROOT` exists because three test modules used to derive it by
counting ``__file__.parents``, and the count is a function of where the file
sits.  Moving such a file into a subdirectory turns it into a collection error
that names a missing path rather than the move.
"""

from __future__ import annotations

from pathlib import Path

#: The repository root, independent of how deep the calling test module sits.
REPO_ROOT = Path(__file__).resolve().parents[2]

__all__ = ["REPO_ROOT"]
