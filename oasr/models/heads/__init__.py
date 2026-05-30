# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Decode-side output heads (CTC today; Transducer / AED are extension points)."""

from .ctc import CTCHead

__all__ = ["CTCHead"]
