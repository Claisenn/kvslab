#include "kvslab/tier.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <new>

namespace kvslab {
namespace {

// Page-granular so the arena can later be pinned, registered with an RDMA NIC,
// or handed to cudaHostRegister without straddling page boundaries.
constexpr std::size_t kArenaAlignment = 4096;

std::size_t round_up(std::size_t value, std::size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

}  // namespace

HostTier::HostTier(std::size_t capacity_bytes) {
  // std::aligned_alloc requires the size to be a multiple of the alignment, so
  // the tier rounds up and reports the rounded capacity rather than silently
  // owning bytes it claims not to have.
  capacity_bytes_ = round_up(capacity_bytes, kArenaAlignment);
  if (capacity_bytes_ == 0) return;

  void* raw = std::aligned_alloc(kArenaAlignment, capacity_bytes_);
  if (raw == nullptr) throw std::bad_alloc();
  arena_ = static_cast<std::byte*>(raw);
}

HostTier::~HostTier() { std::free(arena_); }

void HostTier::read(std::size_t offset, void* dst, std::size_t bytes) const {
  assert(offset + bytes <= capacity_bytes_ && "read past the end of the tier");
  std::memcpy(dst, arena_ + offset, bytes);
}

void HostTier::write(std::size_t offset, const void* src, std::size_t bytes) {
  assert(offset + bytes <= capacity_bytes_ && "write past the end of the tier");
  std::memcpy(arena_ + offset, src, bytes);
}

}  // namespace kvslab
