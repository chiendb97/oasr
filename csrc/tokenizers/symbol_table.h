// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// C++ piece rendering for alignment's word-grouping path. Full transcript decode
// stays in Python to keep one authoritative implementation.

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
    /// Missing and special ids contribute nothing. Boundary whitespace is
    /// stripped so concatenated pieces exactly equal full decode.
    std::vector<std::string> pieces(const std::vector<int64_t>& ids) const;

    size_t size() const { return table_.size(); }

private:
    std::unordered_map<int64_t, std::string> table_;
    std::unordered_set<int64_t> special_;
};

}  // namespace tokenizers
}  // namespace oasr
