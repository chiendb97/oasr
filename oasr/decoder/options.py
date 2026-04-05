# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""Options dataclasses and SearchType enum for CTC decoders."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class SearchType(enum.Enum):
    """Decoder search algorithm type."""

    kGreedySearch = 0x02
    kPrefixBeamSearch = 0x00
    kWfstBeamSearch = 0x01


@dataclass
class CtcGreedySearchOptions:
    """Options for :class:`~oasr.decoder.CtcGreedySearch`.

    Attributes:
        blank: CTC blank token id (default 0).
    """

    blank: int = 0


@dataclass
class CtcPrefixBeamSearchOptions:
    """Options for :class:`~oasr.decoder.CtcPrefixBeamSearch`.

    Attributes:
        blank: CTC blank token id (default 0).
        first_beam_size: Top-k vocabulary pruning per frame.
        second_beam_size: Hypothesis beam width.
    """

    blank: int = 0
    first_beam_size: int = 10
    second_beam_size: int = 10


@dataclass
class CtcWfstBeamSearchOptions:
    """Options for :class:`~oasr.decoder.CtcWfstBeamSearch`.

    Attributes:
        blank: CTC blank token id (default 0).
        search_beam: Decoding beam width (larger = more accurate, slower).
        output_beam: Lattice pruning beam.
        min_active_states: Minimum active FSA states per frame.
        max_active_states: Maximum active FSA states per frame.
        subsampling_factor: Encoder subsampling factor (used for timestamps).
        nbest: Number of N-best paths to extract.
        blank_skip_thresh: Skip frames where P(blank) exceeds this threshold.
    """

    blank: int = 0
    search_beam: float = 20.0
    output_beam: float = 8.0
    min_active_states: int = 30
    max_active_states: int = 10000
    subsampling_factor: int = 1
    nbest: int = 10
    blank_skip_thresh: float = 0.98
