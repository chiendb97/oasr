#include "decoder/wfst/graph.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace oasr::wfst {
namespace {

constexpr uint64_t kMagic = 0x31474D4954534657ULL;  // "WFSTIMG1"
constexpr uint32_t kFlagFinalsAtEnd = 1;
constexpr uint32_t kFlagHasEps = 2;
constexpr uint32_t kFlagEpsFirst = 4;
constexpr size_t kHeaderBytes = 256;

#pragma pack(push, 1)
struct Header {  // v2 layout; v1 files lack the 7th offset
  uint64_t magic;
  uint32_t version;
  uint32_t flags;
  int64_t num_states;
  int64_t num_arcs;
  int32_t vocab_size;
  int32_t start_state;
  int64_t aux_pool_size;
  int64_t offsets[7];
};
#pragma pack(pop)

class File {
 public:
  explicit File(const std::string& path) : f_(std::fopen(path.c_str(), "rb")) {
    if (f_ == nullptr) throw std::runtime_error("cannot open graph image: " + path);
  }
  ~File() { std::fclose(f_); }
  template <typename T>
  void ReadAt(int64_t off, T* dst, int64_t count) {
    if (std::fseek(f_, static_cast<long>(off), SEEK_SET) != 0)
      throw std::runtime_error("seek failed in graph image");
    if (std::fread(dst, sizeof(T), static_cast<size_t>(count), f_) !=
        static_cast<size_t>(count))
      throw std::runtime_error("short read in graph image");
  }

 private:
  std::FILE* f_;
};

}  // namespace

std::unique_ptr<GraphImage> LoadGraphImage(const std::string& path) {
  File f(path);
  char raw[kHeaderBytes];
  f.ReadAt(0, raw, kHeaderBytes);
  Header h{};
  std::memcpy(&h, raw, sizeof(Header));
  if (h.magic != kMagic) throw std::runtime_error("bad magic in graph image: " + path);
  if (h.version != 1 && h.version != 2)
    throw std::runtime_error("unsupported graph image version");
  if (h.version == 1) h.offsets[6] = 0;  // v1: no epsilon section
  if (h.num_states <= 0 || h.num_arcs < 0 || h.num_states >= (1LL << 31))
    throw std::runtime_error("invalid graph image counts");

  auto img = std::make_unique<GraphImage>();
  img->num_states = h.num_states;
  img->num_arcs = h.num_arcs;
  img->vocab_size = h.vocab_size;
  img->start_state = h.start_state;
  img->finals_at_end = (h.flags & kFlagFinalsAtEnd) != 0;
  img->has_eps = h.version == 2 && (h.flags & kFlagHasEps) != 0;
  img->eps_first = (h.flags & kFlagEpsFirst) != 0;

  img->row_splits.resize(h.num_states + 1);
  img->final_count.resize(h.num_states);
  img->arc_dest_ilabel.resize(2 * h.num_arcs);
  img->arc_weight.resize(h.num_arcs);
  img->aux_row_splits.resize(h.num_arcs + 1);
  img->aux_pool.resize(h.aux_pool_size);

  f.ReadAt(h.offsets[0], img->row_splits.data(), h.num_states + 1);
  f.ReadAt(h.offsets[1], img->final_count.data(), h.num_states);
  f.ReadAt(h.offsets[2], img->arc_dest_ilabel.data(), 2 * h.num_arcs);
  f.ReadAt(h.offsets[3], img->arc_weight.data(), h.num_arcs);
  f.ReadAt(h.offsets[4], img->aux_row_splits.data(), h.num_arcs + 1);
  if (h.aux_pool_size > 0) f.ReadAt(h.offsets[5], img->aux_pool.data(), h.aux_pool_size);
  if (img->has_eps) {
    img->eps_count.resize(h.num_states);
    f.ReadAt(h.offsets[6], img->eps_count.data(), h.num_states);
  }

  if (img->row_splits.front() != 0 || img->row_splits.back() != img->num_arcs)
    throw std::runtime_error("corrupt row_splits in graph image");
  return img;
}

}  // namespace oasr::wfst
