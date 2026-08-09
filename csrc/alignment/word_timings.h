// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-token acoustic alignments -> word timings.  The C++ half of
// ``oasr/engine/decode/alignment.py``, and the reason it exists: this pass runs
// once per decoded token and once per rendered character of every transcript
// that asks for word timestamps, on the engine's step-loop thread, holding the
// GIL the whole time.  In Python it profiled larger than the CTC decode it was
// decorating.
//
// This is the only implementation: ``oasr/engine/decode/alignment.py`` is
// marshalling, with no Python twin to fall back to — a fallback costing more
// than the decode is one a deployment lands on silently.
// The contract is exact equivalence with a Python statement of the same rule
// kept in ``tests/test_alignment_cpp.py`` — an oracle, not a code path, so a
// change here lands in that file in the same commit.  Two places make the
// equivalence non-trivial and are pinned here rather than approximated:
//
//   * **Whitespace is Python's ``str.isspace()``**, not ``std::isspace`` and not
//     ASCII.  That set is 29 code points (Zs/Zl/Zp plus the bidi-class
//     WS/B/S controls), enumerated in ``kSpaceRanges``.
//   * **The text is UTF-8 and the word rule is per code point.**  Byte-oriented
//     scanning would split a CJK character mid-sequence and hand back a "word"
//     that is not a substring of the transcript in any useful sense.
//
// No torch dependency: this is plain data shuffling, so it compiles fast and is
// testable on its own.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace oasr {
namespace alignment {

/// One decoded token's acoustic span, in **encoder frames**.
struct TokenSpan {
    double start_frame = 0.0;
    double end_frame = 0.0;
    double confidence = 1.0;
};

/// One word of the transcript, in seconds.  ``word`` is a byte-for-byte
/// substring of the rendered transcript — see the module docstring on the
/// Python side for why that is a contract and not an implementation detail.
struct WordTiming {
    std::string word;
    double start = 0.0;
    double end = 0.0;
    double confidence = 1.0;
};

/// What one alignment pass produces: everything ``RequestOutput`` needs.
struct AlignmentResult {
    std::vector<WordTiming> words;
    /// Per-token ``(start, end)`` in seconds, offset applied.
    std::vector<std::pair<double, double>> timestamps;
    /// Mean per-token posterior; ``false`` when there were no tokens.
    double confidence = 0.0;
    bool has_confidence = false;
};

/// Emission frames -> per-token spans, for the frame-synchronous families.
///
/// Token *k* was emitted having consumed frames up to ``t_k``, with the
/// previous decision at ``t_{k-1}``, so it owns ``(t_{k-1} + 1, t_k + 1)`` and
/// the spans tile.  The first token starts at its own frame instead, so leading
/// silence is not attributed to the first word.  ``confidences`` may be shorter
/// than ``frames``; the tail defaults to 1.0.
std::vector<TokenSpan> emission_spans(const std::vector<int64_t>& frames,
                                      const std::vector<double>& confidences, int64_t frame_offset);

/// Per-token spans + the text each token contributed -> words, timestamps and
/// the utterance confidence.
///
/// ``pieces`` must be parallel to ``spans`` and concatenate to exactly the
/// transcript (the ``Tokenizer.token_pieces`` contract).  ``want_words``
/// false computes the timestamps and the confidence only — the Paraformer case,
/// where the alignment is free and always computed but the word pass is opt-in.
AlignmentResult align(const std::vector<TokenSpan>& spans, const std::vector<std::string>& pieces,
                      double seconds_per_frame, double offset, bool want_words);

/// ``emission_spans`` + ``align`` in one pass, so a frame-synchronous family
/// never materialises a per-token object at all.
AlignmentResult align_emissions(const std::vector<int64_t>& frames,
                                const std::vector<double>& confidences,
                                const std::vector<std::string>& pieces, double seconds_per_frame,
                                int64_t frame_offset, double offset, bool want_words);

// -- exposed for the differential tests -------------------------------------

/// Python's ``str.isspace()``, exactly.
bool is_space(uint32_t codepoint);

/// A script written without spaces (CJK, kana, Hangul), where each character is
/// its own word — the convention FunASR and Whisper both use.
bool is_spaceless(uint32_t codepoint);

}  // namespace alignment
}  // namespace oasr
