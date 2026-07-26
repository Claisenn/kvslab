#include "kvslab/block_pool.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace kvslab {
namespace {

// Refcount and range violations are not crashes, they are silent corruption: a
// block handed back twice lands on the free list while the index still points
// at it, and the next allocation gives live KV to a second sequence. Nothing
// downstream can detect that, so these checks do not ride on assert() -- they
// stay in every build, at the cost of a branch that always predicts.
[[noreturn]] void fatal(const char* what, BlockId id) {
  std::fprintf(stderr, "kvslab: fatal: %s (block %u)\n", what, id);
  std::abort();
}

}  // namespace

BlockPool::BlockPool(const CacheConfig& cfg)
    : cfg_(cfg), owned_tier_(std::make_unique<HostTier>(cfg.total_bytes())) {
  init({{owned_tier_.get(), cfg.num_blocks}});
}

BlockPool::BlockPool(Tier& tier, const CacheConfig& cfg) : cfg_(cfg) {
  init({{&tier, cfg.num_blocks}});
}

BlockPool::BlockPool(const std::vector<TierSpec>& specs, const CacheConfig& cfg)
    : cfg_(cfg) {
  init(specs);
}

void BlockPool::init(const std::vector<TierSpec>& specs) {
  if (specs.empty()) {
    throw std::invalid_argument("kvslab: a pool needs at least one tier");
  }

  std::size_t total_blocks = 0;
  tiers_.reserve(specs.size());
  for (const TierSpec& spec : specs) {
    if (spec.tier == nullptr) {
      throw std::invalid_argument("kvslab: null tier in pool spec");
    }
    if (spec.tier->capacity_bytes() < spec.num_blocks * cfg_.block_bytes()) {
      throw std::invalid_argument("kvslab: tier is too small for the requested pool");
    }

    TierState state;
    state.tier = spec.tier;
    state.num_blocks = spec.num_blocks;
    // Descending, so the first allocations hand out low slots. Purely a
    // debugging nicety: block tables read in allocation order.
    state.free_slots.reserve(spec.num_blocks);
    for (std::size_t i = spec.num_blocks; i > 0; --i) {
      state.free_slots.push_back(static_cast<std::uint32_t>(i - 1));
    }
    tiers_.push_back(std::move(state));
    total_blocks += spec.num_blocks;
  }

  // One id per slot in the whole pool: every block could in principle be
  // resident somewhere at once, and ids are cheap.
  location_.assign(total_blocks, Location{});
  refcount_.assign(total_blocks, 0);
  free_ids_.reserve(total_blocks);
  for (std::size_t i = total_blocks; i > 0; --i) {
    free_ids_.push_back(static_cast<BlockId>(i - 1));
  }

  bounce_.resize(cfg_.block_bytes());
}

const BlockPool::Location& BlockPool::locate(BlockId id) const {
  if (id >= location_.size()) fatal("access to an out-of-range block", id);
  if (refcount_[id] == 0) fatal("access to a block that is not allocated", id);
  return location_[id];
}

BlockId BlockPool::allocate() {
  if (free_ids_.empty() || tiers_[0].free_slots.empty()) return kInvalidBlock;

  const BlockId id = free_ids_.back();
  free_ids_.pop_back();
  assert(refcount_[id] == 0);

  location_[id].tier = 0;
  location_[id].slot = tiers_[0].free_slots.back();
  tiers_[0].free_slots.pop_back();

  refcount_[id] = 1;
  return id;
}

void BlockPool::incref(BlockId id) {
  if (id >= location_.size()) fatal("incref of an out-of-range block", id);
  if (refcount_[id] == 0) fatal("incref of a block that is already free", id);
  ++refcount_[id];
}

bool BlockPool::decref(BlockId id) {
  if (id >= location_.size()) fatal("decref of an out-of-range block", id);
  if (refcount_[id] == 0) fatal("decref of a block that is already free", id);
  if (--refcount_[id] == 0) {
    const Location& loc = location_[id];
    tiers_[loc.tier].free_slots.push_back(loc.slot);
    free_ids_.push_back(id);
    return true;
  }
  return false;
}

std::uint32_t BlockPool::refcount(BlockId id) const {
  if (id >= location_.size()) fatal("refcount of an out-of-range block", id);
  return refcount_[id];
}

BlockPool::TierIndex BlockPool::block_tier(BlockId id) const {
  return locate(id).tier;
}

bool BlockPool::migrate(BlockId id, TierIndex dst) {
  const Location& loc = locate(id);
  if (dst >= tiers_.size()) fatal("migrate to an out-of-range tier", id);
  if (loc.tier == dst) return true;  // already there
  if (tiers_[dst].free_slots.empty()) return false;

  Tier& src_tier = *tiers_[loc.tier].tier;
  Tier& dst_tier = *tiers_[dst].tier;
  const std::size_t bytes = cfg_.block_bytes();
  const std::size_t src_off = static_cast<std::size_t>(loc.slot) * bytes;

  const std::uint32_t dst_slot = tiers_[dst].free_slots.back();
  tiers_[dst].free_slots.pop_back();
  const std::size_t dst_off = static_cast<std::size_t>(dst_slot) * bytes;

  // Through the bounce buffer, which works for any tier pairing. A direct
  // path for host<->host (or device peer copies) is an optimization for when
  // a profile asks for it.
  src_tier.read(src_off, bounce_.data(), bytes);
  dst_tier.write(dst_off, bounce_.data(), bytes);

  tiers_[loc.tier].free_slots.push_back(loc.slot);
  location_[id].tier = dst;
  location_[id].slot = dst_slot;
  return true;
}

std::size_t BlockPool::block_offset(BlockId id) const {
  const Location& loc = locate(id);
  return static_cast<std::size_t>(loc.slot) * cfg_.block_bytes();
}

void* BlockPool::block_data(BlockId id) {
  const Location& loc = locate(id);
  std::byte* base = tiers_[loc.tier].tier->host_data();
  if (base == nullptr) return nullptr;  // device tier: use block_offset() instead
  return base + static_cast<std::size_t>(loc.slot) * cfg_.block_bytes();
}

const void* BlockPool::block_data(BlockId id) const {
  const Location& loc = locate(id);
  // host_data() is non-const because a device tier may have to map on demand;
  // reading through it here does not mutate the pool's own state.
  std::byte* base = const_cast<Tier*>(tiers_[loc.tier].tier)->host_data();
  if (base == nullptr) return nullptr;
  return base + static_cast<std::size_t>(loc.slot) * cfg_.block_bytes();
}

}  // namespace kvslab
