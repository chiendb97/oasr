// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0

#include "tokenizers/symbol_table.h"

#include <utility>

#include "alignment/word_timings.h"

namespace oasr {
namespace tokenizers {
namespace {

/// U+2581 LOWER ONE EIGHTH BLOCK — sentencepiece's word boundary.
constexpr char kWordBoundary[] = "\xE2\x96\x81";
constexpr size_t kWordBoundaryLen = 3;

/// ``piece.replace("▁", " ")``.
std::string substituteBoundary(const std::string& piece) {
    const size_t hit = piece.find(kWordBoundary);
    if (hit == std::string::npos)
        return piece;  // the common case: no copy
    std::string out;
    out.reserve(piece.size());
    size_t i = 0;
    size_t next = hit;
    while (next != std::string::npos) {
        out.append(piece, i, next - i);
        out.push_back(' ');
        i = next + kWordBoundaryLen;
        next = piece.find(kWordBoundary, i);
    }
    out.append(piece, i, piece.size() - i);
    return out;
}

/// Decode the UTF-8 sequence starting at ``i``; returns its byte length.
size_t codepointAt(const std::string& s, size_t i, uint32_t* cp) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    if (b0 < 0x80) {
        *cp = b0;
        return 1;
    }
    size_t len = 1;
    uint32_t value = b0;
    if ((b0 & 0xE0) == 0xC0) {
        len = 2;
        value = b0 & 0x1Fu;
    } else if ((b0 & 0xF0) == 0xE0) {
        len = 3;
        value = b0 & 0x0Fu;
    } else if ((b0 & 0xF8) == 0xF0) {
        len = 4;
        value = b0 & 0x07u;
    } else {
        *cp = b0;
        return 1;
    }
    if (i + len > s.size()) {
        *cp = b0;
        return 1;
    }
    for (size_t k = 1; k < len; ++k) {
        const auto b = static_cast<unsigned char>(s[i + k]);
        if ((b & 0xC0) != 0x80) {
            *cp = b0;
            return 1;
        }
        value = (value << 6) | (b & 0x3Fu);
    }
    *cp = value;
    return len;
}

/// ``str.lstrip()`` — Python's whitespace set, not ``isspace(3)``.
std::string lstripWhitespace(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = 0;
        const size_t len = codepointAt(s, i, &cp);
        if (!alignment::is_space(cp))
            break;
        i += len;
    }
    return i == 0 ? s : s.substr(i);
}

/// ``str.rstrip()``.  Walks forward rather than backward: the strings here are
/// single tokens, and scanning UTF-8 backwards to find a code-point start is
/// more code than re-walking a handful of bytes.
std::string rstripWhitespace(const std::string& s) {
    size_t i = 0;
    size_t keep = 0;
    while (i < s.size()) {
        uint32_t cp = 0;
        const size_t len = codepointAt(s, i, &cp);
        i += len;
        if (!alignment::is_space(cp))
            keep = i;
    }
    return keep == s.size() ? s : s.substr(0, keep);
}

}  // namespace

SymbolTablePieces::SymbolTablePieces(std::unordered_map<int64_t, std::string> table,
                                     std::unordered_set<int64_t> special_ids)
    : table_(std::move(table)), special_(std::move(special_ids)) {}

std::vector<std::string> SymbolTablePieces::pieces(const std::vector<int64_t>& ids) const {
    std::vector<std::string> out;
    out.reserve(ids.size());
    for (const int64_t id : ids) {
        if (special_.count(id)) {
            out.emplace_back();
            continue;
        }
        const auto hit = table_.find(id);
        out.push_back(hit == table_.end() ? std::string() : substituteBoundary(hit->second));
    }
    // What ``decode``'s outer ``strip`` would take off each end.  Both loops
    // stop at the first piece that still has content afterwards: a piece that
    // was *entirely* whitespace is emptied and the next one has to be stripped
    // too.
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i].empty())
            continue;
        out[i] = lstripWhitespace(out[i]);
        if (!out[i].empty())
            break;
    }
    for (size_t i = out.size(); i-- > 0;) {
        if (out[i].empty())
            continue;
        out[i] = rstripWhitespace(out[i]);
        if (!out[i].empty())
            break;
    }
    return out;
}

}  // namespace tokenizers
}  // namespace oasr
