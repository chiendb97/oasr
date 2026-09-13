# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""OASR benchmarks.

One CLI -- :mod:`benchmarks.run` -- over a family registry
(:mod:`benchmarks.core.registry`).  Families live in the directory that owns the
module they exercise: ``kernels/``, ``features/``, ``decoders/``, ``engine/``,
``service/``, ``accuracy/``.  See ``benchmarks/README.md``.

This file deliberately carries no inventory: the previous one listed fourteen
``bench_*.py`` scripts by hand and had fallen out of date with the tree.
"""
