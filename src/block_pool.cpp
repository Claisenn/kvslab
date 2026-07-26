#include "kvslab/block_pool.hpp"

#include <cassert>
#include <stdexcept>

namespace kvslab {

BlockPool::BlockPool(const CacheConfig& cfg)
    : cfg_(cfg),
      owned_tier_(std::make_unique<HostTier>(cfg.total_bytes())),
      tier_(*owned_tier_) {
  init();
}

BlockPool::BlockPool(Tier& tier, const CacheConfig& cfg) : cfg_(cfg), tier_(tier) {
  init();
}

void BlockPool::init() {
  if (tier_.capacity_bytes() < cfg_.total_bytes()) {
    throw std::invalid_argument("kvslab: tier is too small for the requested pool");
  }

  refcount_.assign(cfg_.num_blocks, 0);
  free_list_.reserve(cfg_.num_blocks);
  // Descending, so the first allocations hand out low block ids. Purely a
  // debugging nicety: block tables read in allocation order.
  for (std::size_t i = cfg_.num_blocks; i > 0; --i) {
    free_list_.push_back(static_cast<BlockId>(i - 1));
  }
}

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

std::size_t BlockPool::block_offset(BlockId id) const {
  assert(id < cfg_.num_blocks);
  return static_cast<std::size_t>(id) * cfg_.block_bytes();
}

void* BlockPool::block_data(BlockId id) {
  std::byte* base = tier_.host_data();
  if (base == nullptr) return nullptr;  // device tier: use block_offset() instead
  return base + block_offset(id);
}

const void* BlockPool::block_data(BlockId id) const {
  // host_data() is non-const because a device tier may have to map on demand;
  // reading through it here does not mutate the pool's own state.
  std::byte* base = const_cast<Tier&>(tier_).host_data();
  if (base == nullptr) return nullptr;
  return base + block_offset(id);
}

}  // namespace kvslab
