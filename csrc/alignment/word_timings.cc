// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "alignment/word_timings.h"

#include <algorithm>

namespace oasr {
namespace alignment {
namespace {

struct Range {
    uint32_t lo;
    uint32_t hi;
};

// Python's ``str.isspace()``: general category Zs/Zl/Zp, plus the controls
// whose bidirectional class is WS/B/S.  Enumerated rather than derived so the
// two implementations cannot drift with a locale or a libstdc++ version; the
// test regenerates this list from CPython and compares.
constexpr Range kSpaceRanges[] = {
    {0x0009, 0x000D}, {0x001C, 0x0020}, {0x0085, 0x0085}, {0x00A0, 0x00A0}, {0x1680, 0x1680},
    {0x2000, 0x200A}, {0x2028, 0x2029}, {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000},
};

// Scripts written without spaces.  Note 0x3000 (ideographic space) is *not*
// here — it is whitespace, and the kana block starts above it.
constexpr Range kSpacelessRanges[] = {
    {0x3040, 0x30FF},    // Hiragana + Katakana
    {0x3400, 0x4DBF},    // CJK Unified Ideographs Extension A
    {0x4E00, 0x9FFF},    // CJK Unified Ideographs
    {0xAC00, 0xD7AF},    // Hangul syllables
    {0xF900, 0xFAFF},    // CJK Compatibility Ideographs
    {0x20000, 0x2FA1F},  // CJK Extensions B-F + Compatibility Supplement
};

/// Decode one UTF-8 sequence at ``i``.  Returns its length in bytes and writes
/// the code point.  Malformed input is consumed one byte at a time as its own
/// code point: this runs on tokenizer output, so it should never happen, and
/// looping forever or reading past the end would be a worse answer than a
/// character that groups oddly.
inline size_t decode_utf8(const std::string& s, size_t i, uint32_t* cp) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    const size_t n = s.size();
    if (b0 < 0x80) {
        *cp = b0;
        return 1;
    }
    auto cont = [&](size_t k) -> bool {
        return i + k < n && (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80;
    };
    auto tail = [&](size_t k) -> uint32_t { return static_cast<unsigned char>(s[i + k]) & 0x3F; };
    if ((b0 & 0xE0) == 0xC0 && cont(1)) {
        *cp = ((b0 & 0x1Fu) << 6) | tail(1);
        return 2;
    }
    if ((b0 & 0xF0) == 0xE0 && cont(1) && cont(2)) {
        *cp = ((b0 & 0x0Fu) << 12) | (tail(1) << 6) | tail(2);
        return 3;
    }
    if ((b0 & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
        *cp = ((b0 & 0x07u) << 18) | (tail(1) << 12) | (tail(2) << 6) | tail(3);
        return 4;
    }
    *cp = b0;
    return 1;
}

template <size_t N>
inline bool in_ranges(uint32_t cp, const Range (&ranges)[N]) {
    for (size_t i = 0; i < N; ++i) {
        if (cp < ranges[i].lo)
            return false;  // sorted: no later range can match
        if (cp <= ranges[i].hi)
            return true;
    }
    return false;
}

}  // namespace

bool is_space(uint32_t cp) {
    return in_ranges(cp, kSpaceRanges);
}

bool is_spaceless(uint32_t cp) {
    return in_ranges(cp, kSpacelessRanges);
}

std::vector<TokenSpan> emission_spans(const std::vector<int64_t>& frames,
                                      const std::vector<double>& confidences,
                                      int64_t frame_offset) {
    std::vector<TokenSpan> out;
    out.reserve(frames.size());
    int64_t prev = 0;
    bool first = true;
    for (size_t k = 0; k < frames.size(); ++k) {
        const int64_t frame = frames[k];
        int64_t start = frame;
        if (!first) {
            start = prev + 1;
            if (start > frame)
                start = frame;
        }
        out.push_back(TokenSpan{static_cast<double>(start + frame_offset),
                                static_cast<double>(frame + frame_offset + 1),
                                k < confidences.size() ? confidences[k] : 1.0});
        prev = frame;
        first = false;
    }
    return out;
}

AlignmentResult align(const std::vector<TokenSpan>& spans, const std::vector<std::string>& pieces,
                      double seconds_per_frame, double offset, bool want_words) {
    AlignmentResult result;
    if (spans.empty())
        return result;

    result.timestamps.reserve(spans.size());
    double conf_sum = 0.0;
    for (const auto& s : spans) {
        const double t0 = s.start_frame * seconds_per_frame + offset;
        const double t1 = s.end_frame * seconds_per_frame + offset;
        result.timestamps.emplace_back(t0, std::max(t1, t0));
        conf_sum += s.confidence;
    }
    result.confidence = conf_sum / static_cast<double>(spans.size());
    result.has_confidence = true;
    if (!want_words)
        return result;

    // Character ownership as two parallel arrays rather than one entry per
    // character: ``ends[j]`` is one past the last *byte* piece ``j``
    // contributed and ``owner[j]`` is the token that produced it.  Ownership is
    // monotone in the text, so a word's member tokens are two binary searches
    // instead of a set built over its characters.  Byte offsets rather than
    // code-point offsets because a piece is always a whole number of code
    // points, so the two partition identically and bytes are free.
    std::string text;
    std::vector<size_t> ends;
    std::vector<size_t> owner;
    size_t total = 0;
    for (const auto& p : pieces)
        total += p.size();
    text.reserve(total);
    ends.reserve(pieces.size());
    owner.reserve(pieces.size());
    const size_t n_spans = spans.size();
    for (size_t i = 0; i < pieces.size() && i < n_spans; ++i) {
        if (pieces[i].empty())
            continue;
        text += pieces[i];
        ends.push_back(text.size());
        owner.push_back(i);
    }
    if (ends.empty())
        return result;

    auto emit = [&](size_t a, size_t b) {
        // ``b - 1`` rather than ``b`` keeps a word ending exactly on a piece
        // boundary from claiming the next one.
        const size_t first =
            static_cast<size_t>(std::upper_bound(ends.begin(), ends.end(), a) - ends.begin());
        const size_t last =
            static_cast<size_t>(std::upper_bound(ends.begin(), ends.end(), b - 1) - ends.begin());
        const TokenSpan& head = spans[owner[first]];
        double start = head.start_frame;
        double end = head.end_frame;
        double conf = head.confidence;
        for (size_t j = first + 1; j <= last; ++j) {
            const TokenSpan& s = spans[owner[j]];
            if (s.start_frame < start)
                start = s.start_frame;
            if (s.end_frame > end)
                end = s.end_frame;
            conf += s.confidence;
        }
        const double t0 = start * seconds_per_frame + offset;
        const double t1 = end * seconds_per_frame + offset;
        result.words.push_back(WordTiming{text.substr(a, b - a), t0, std::max(t1, t0),
                                          conf / static_cast<double>(last - first + 1)});
    };

    // One word is either a single space-less character or a maximal run of
    // ordinary non-space characters; whitespace is neither, so it is skipped.
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        uint32_t cp = 0;
        const size_t len = decode_utf8(text, i, &cp);
        if (is_space(cp)) {
            i += len;
            continue;
        }
        if (is_spaceless(cp)) {
            emit(i, i + len);
            i += len;
            continue;
        }
        size_t j = i + len;
        while (j < n) {
            uint32_t next = 0;
            const size_t next_len = decode_utf8(text, j, &next);
            if (is_space(next) || is_spaceless(next))
                break;
            j += next_len;
        }
        emit(i, j);
        i = j;
    }
    return result;
}

AlignmentResult align_emissions(const std::vector<int64_t>& frames,
                                const std::vector<double>& confidences,
                                const std::vector<std::string>& pieces, double seconds_per_frame,
                                int64_t frame_offset, double offset, bool want_words) {
    return align(emission_spans(frames, confidences, frame_offset), pieces, seconds_per_frame,
                 offset, want_words);
}

}  // namespace alignment
}  // namespace oasr
