# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Named recipes: a scheme plus the algorithms that choose its numbers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Recipe:
    """A quantization recipe: which scheme, produced by which algorithms, in order."""

    name: str
    scheme: str
    algorithms: Tuple[str, ...]
    doc: str = ""


RECIPES: Dict[str, Recipe] = {
    r.name: r
    for r in (
        Recipe(
            "int8_w8a8_rtn",
            "int8_w8a8",
            ("rtn",),
            "int8 per-channel W x per-token dynamic A, round-to-nearest; no calibration",
        ),
        Recipe(
            "smoothquant_w8a8",
            "int8_w8a8",
            ("smoothquant",),
            "SmoothQuant smoothing, then int8 per-channel W x per-token dynamic A",
        ),
        Recipe(
            "smoothquant_w8a8_fused",
            "int8_w8a8",
            ("smoothquant", "static_inputs"),
            "smoothquant_w8a8, except that GEMM-chain consumers (the FFN's w_2) take a "
            "calibrated static per-tensor input scale, so the producer's epilogue quantizes "
            "for them: no quantize launch and no fp round trip between the two",
        ),
    )
}


def get_recipe(name: str) -> Recipe:
    try:
        return RECIPES[name]
    except KeyError:
        raise ValueError(
            f"unknown quantization recipe {name!r}; known: {sorted(RECIPES)}"
        ) from None


__all__ = ["RECIPES", "Recipe", "get_recipe"]
