// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The rendering half of the ``symbol_table`` tokenizer kind (WeNet
// ``units.txt`` / icefall ``tokens.txt``), in C++.
//
// Only ``token_pieces`` lives here, and only because of where it runs: the word
// grouping calls it once per finished hypothesis on the engine's step-loop
// thread, and in Python it is a dict lookup, a ``▁``→space substitution and a
// strip fixup *per token*.  With the rest of the alignment pass moved off the
// interpreter it became the largest Python item left on that path, so it moved
// here entirely — ``SymbolTableTokenizer.token_pieces`` has no Python twin and
// raises without ``_C``, matching the alignment pass that is its only caller.
//
// ``decode`` deliberately stays in Python.  It is the same three operations,
// but it runs on every streaming partial and its output is the transcript
// itself — a second implementation there would be a correctness surface for a
// gain the alignment path does not need.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace oasr {
namespace tokenizers {

/// Id → piece rendering for a flat symbol table.
class SymbolTablePieces {
public:
    SymbolTablePieces(std::unordered_map<int64_t, std::string> table,
                      std::unordered_set<int64_t> special_ids);

    /// Per-token text contributions, concatenating to exactly what the Python
    /// ``decode`` returns for the same ids.
    ///
    /// A special id, or an id absent from the table, contributes ``""`` and
    /// owns no characters.  ``decode`` strips the joined text, so the leading
    /// whitespace of the first non-empty piece and the trailing whitespace of
    /// the last are removed here too — otherwise the pieces would no longer
    /// concatenate to it, which is the one property the word grouping depends
    /// on.
    std::vector<std::string> pieces(const std::vector<int64_t>& ids) const;

    size_t size() const { return table_.size(); }

private:
    std::unordered_map<int64_t, std::string> table_;
    std::unordered_set<int64_t> special_;
};

}  // namespace tokenizers
}  // namespace oasr
