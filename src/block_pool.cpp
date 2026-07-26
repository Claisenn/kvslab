#include "kvslab/block_pool.hpp"

#include <cassert>
#include <cstdlib>
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

BlockPool::BlockPool(const CacheConfig& cfg) : cfg_(cfg) {
  arena_bytes_ = round_up(cfg_.total_bytes(), kArenaAlignment);
  if (arena_bytes_ > 0) {
    void* raw = std::aligned_alloc(kArenaAlignment, arena_bytes_);
    if (raw == nullptr) throw std::bad_alloc();
    arena_ = static_cast<std::byte*>(raw);
  }

  refcount_.assign(cfg_.num_blocks, 0);
  free_list_.reserve(cfg_.num_blocks);
  // Descending, so the first allocations hand out low block ids. Purely a
  // debugging nicety: block tables read in allocation order.
  for (std::size_t i = cfg_.num_blocks; i > 0; --i) {
    free_list_.push_back(static_cast<BlockId>(i - 1));
  }
}

BlockPool::~BlockPool() { std::free(arena_); }

BlockId BlockPool::allocate() {
  if (free_list_.empty()) return kInvalidBlock;
  const BlockId id = free_list_.back();
  free_list_.pop_back();
  assert(refcount_[id] == 0);
  refcount_[id] = 1;
  return id;
}

void BlockPool::incref(BlockId id) {
  assert(id < cfg_.num_blocks);
  assert(refcount_[id] > 0 && "reviving a freed block");
  ++refcount_[id];
}

bool BlockPool::decref(BlockId id) {
  assert(id < cfg_.num_blocks);
  assert(refcount_[id] > 0 && "double free");
  if (--refcount_[id] == 0) {
    free_list_.push_back(id);
    return true;
  }
  return false;
}

std::uint32_t BlockPool::refcount(BlockId id) const {
  assert(id < cfg_.num_blocks);
  return refcount_[id];
}

void* BlockPool::block_data(BlockId id) {
  assert(id < cfg_.num_blocks);
  return arena_ + static_cast<std::size_t>(id) * cfg_.block_bytes();
}

const void* BlockPool::block_data(BlockId id) const {
  assert(id < cfg_.num_blocks);
  return arena_ + static_cast<std::size_t>(id) * cfg_.block_bytes();
}

}  // namespace kvslab
