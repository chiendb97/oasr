# Copyright 2024 OASR Authors
# SPDX-License-Identifier: Apache-2.0
"""
High-level Python decoder API.

Provides a unified :class:`Decoder` interface that wraps the C++ search
implementations (greedy, prefix beam, K2 WFST) behind a clean API for both
offline and chunk-based streaming decoding.

Example — greedy offline::

    from oasr.decode import Decoder, DecoderConfig
    config = DecoderConfig(search_type="greedy")
    decoder = Decoder(config)
    result = decoder.decode(logp)           # logp: torch.Tensor [T, V]
    print(result.tokens[0])                 # best token sequence

Example — beam search streaming with phrase boosting::

    config = DecoderConfig(
        search_type="prefix_beam",
        context_phrases=[[45, 67, 89], [12, 34]],
        context_score=3.0,
    )
    decoder = Decoder(config)
    decoder.init_stream()
    for chunk_logp in chunks:
        partial = decoder.decode_chunk(chunk_logp)
    final = decoder.finalize_stream()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

import oasr.decoder as _dec


@dataclass
class DecoderConfig:
    """Configuration for the unified decoder.

    Args:
        search_type: One of ``"greedy"``, ``"prefix_beam"``, or ``"wfst"``.
        blank: CTC blank token id (default 0).
        beam_size: First-pass token beam size for prefix beam search.
        second_beam_size: Second-pass hypothesis beam size for prefix beam search.
        wfst_beam: Beam width for WFST decoding (only used when search_type="wfst").
        wfst_lattice_beam: Lattice beam for WFST decoding.
        wfst_max_active: Maximum active states per frame in WFST decoding.
        wfst_min_active: Minimum active states per frame in WFST decoding.
        wfst_acoustic_scale: Scale applied to log-probs in WFST decoding.
        wfst_nbest: Number of N-best paths to extract in WFST decoding.
        wfst_blank_skip_thresh: Skip frames where P(blank) > threshold in WFST decoding.
        context_phrases: Optional list of phrase token-ID sequences for boosting.
            Each phrase is a list of integer token IDs.
        context_score: Per-token bonus applied for each matched phrase token.
    """

    search_type: str = "prefix_beam"
    blank: int = 0

    # Prefix beam search options
    beam_size: int = 10
    second_beam_size: int = 10

    # WFST options (only used when search_type="wfst")
    wfst_search_beam: float = 20.0
    wfst_output_beam: float = 8.0
    wfst_max_active_states: int = 10000
    wfst_min_active_states: int = 30
    wfst_subsampling_factor: int = 1
    wfst_nbest: int = 10
    wfst_blank_skip_thresh: float = 0.98

    # Context biasing (phrase boosting)
    context_phrases: Optional[List[List[int]]] = None
    context_score: float = 3.0


@dataclass
class DecoderResult:
    """Result returned by decoding methods.

    Attributes:
        tokens: N-best list of decoded token ID sequences.
        scores: Log-probability (or total score) for each hypothesis.
        times: Per-token frame timestamps for each hypothesis. May be empty
            for search types that do not track timestamps.
    """

    tokens: List[List[int]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    times: List[List[int]] = field(default_factory=list)


def _to_result(searcher) -> DecoderResult:
    """Convert a C++ SearchInterface object to a :class:`DecoderResult`."""
    outputs = list(searcher.outputs)
    scores = list(searcher.likelihood)
    times = list(searcher.times)
    return DecoderResult(
        tokens=[list(seq) for seq in outputs],
        scores=list(scores),
        times=[list(t) for t in times],
    )


def _make_searcher(config: DecoderConfig, fst=None):
    """Construct the appropriate C++ searcher from *config*."""
    stype = config.search_type.lower()

    if stype == "greedy":
        opts = _dec.CtcGreedySearchOptions()
        opts.blank = config.blank
        searcher = _dec.CtcGreedySearch(opts)

    elif stype == "prefix_beam":
        opts = _dec.CtcPrefixBeamSearchOptions()
        opts.blank = config.blank
        opts.first_beam_size = config.beam_size
        opts.second_beam_size = config.second_beam_size
        searcher = _dec.CtcPrefixBeamSearch(opts)

    elif stype == "wfst":
        if not _dec.k2_available:
            raise RuntimeError(
                "K2 WFST decoder is not available. "
                "Rebuild with OASR_USE_K2=1 to enable it."
            )
        if fst is None:
            raise ValueError(
                "A decoding FST path must be provided for search_type='wfst'. "
                "Pass fst='/path/to/graph.pt' (file saved with torch.save(fsa.as_dict(), ...))."
            )
        opts = _dec.CtcWfstBeamSearchOptions()
        opts.blank = config.blank
        opts.search_beam = config.wfst_search_beam
        opts.output_beam = config.wfst_output_beam
        opts.max_active_states = config.wfst_max_active_states
        opts.min_active_states = config.wfst_min_active_states
        opts.subsampling_factor = config.wfst_subsampling_factor
        opts.nbest = config.wfst_nbest
        opts.blank_skip_thresh = config.wfst_blank_skip_thresh
        searcher = _dec.CtcWfstBeamSearch.from_file(fst, opts)

    else:
        raise ValueError(
            f"Unknown search_type {config.search_type!r}. "
            "Choose one of: 'greedy', 'prefix_beam', 'wfst'."
        )

    # Attach context graph if phrases are provided (greedy and beam search only)
    if config.context_phrases and stype in ("greedy", "prefix_beam"):
        ctx = _dec.ContextGraph(
            phrases=config.context_phrases,
            context_score=config.context_score,
        )
        searcher.set_context_graph(ctx)

    return searcher


def _logp_to_f32(logp: torch.Tensor) -> torch.Tensor:
    """Ensure *logp* is a 2-D float32 CPU tensor."""
    if logp.dim() != 2:
        raise ValueError(f"logp must be a 2-D tensor [T, V], got {logp.dim()}-D.")
    return logp.detach().cpu().to(torch.float32)


class Decoder:
    """Unified ASR decoder wrapping C++ search implementations.

    Supports offline decoding via :meth:`decode` and chunk-based streaming
    via :meth:`init_stream` / :meth:`decode_chunk` / :meth:`finalize_stream`.

    Args:
        config: Decoder configuration. Defaults to prefix beam search.
        fst: Path to a decoding graph file (saved with
            ``torch.save(fsa.as_dict(), path)``) required when
            ``config.search_type == "wfst"``.  Ignored otherwise.
    """

    def __init__(self, config: Optional[DecoderConfig] = None, fst=None):
        if config is None:
            config = DecoderConfig()
        self._config = config
        self._fst = fst
        self._searcher = _make_searcher(config, fst)

    # ------------------------------------------------------------------
    # Offline decoding
    # ------------------------------------------------------------------

    def decode(self, logp: torch.Tensor) -> DecoderResult:
        """Offline decode a full-length log-probability tensor.

        Args:
            logp: Log-probability tensor of shape ``[T, V]``.

        Returns:
            :class:`DecoderResult` with N-best token sequences and scores.
        """
        logp = _logp_to_f32(logp)
        self._searcher.reset()
        self._searcher.search(logp)
        self._searcher.finalize_search()
        return _to_result(self._searcher)

    # ------------------------------------------------------------------
    # Streaming decoding
    # ------------------------------------------------------------------

    def init_stream(self) -> None:
        """Initialize (or re-initialize) streaming decoding state."""
        self._searcher.reset()

    def decode_chunk(self, logp: torch.Tensor) -> DecoderResult:
        """Process one chunk of log-probability frames.

        Args:
            logp: Log-probability tensor of shape ``[T_chunk, V]``.

        Returns:
            :class:`DecoderResult` reflecting the current best hypotheses
            *without* finalizing.  The result may change as more chunks arrive.
        """
        logp = _logp_to_f32(logp)
        self._searcher.search(logp)
        return _to_result(self._searcher)

    def finalize_stream(self) -> DecoderResult:
        """Finalize streaming decoding and return the final result.

        Returns:
            :class:`DecoderResult` with the final N-best hypotheses.
        """
        self._searcher.finalize_search()
        return _to_result(self._searcher)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> DecoderConfig:
        """The configuration used to build this decoder."""
        return self._config
