#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "kvslab/tier.hpp"
#include "kvslab/types.hpp"

namespace kvslab {

// Fixed-size block allocator carving a tier into equal blocks.
//
// The pool owns the *bookkeeping* -- which blocks are free, who holds a
// reference -- and nothing about the storage itself. Where the bytes live is
// the tier's problem, which is what lets the same allocator sit on host memory
// today and on device memory or NVMe later without changing a line here.
//
// Not thread safe. Serving engines drive block allocation from a single
// scheduler thread; making this lock-free is a later exercise, not free
// complexity to take on now.
class BlockPool {
 public:
  // Allocates and owns a HostTier sized for `cfg`. The single-tier case should
  // not have to assemble the tier by hand.
  explicit BlockPool(const CacheConfig& cfg);

  // Borrows an externally-owned tier, which must outlive the pool and must be
  // at least `cfg.total_bytes()` large.
  BlockPool(Tier& tier, const CacheConfig& cfg);

  BlockPool(const BlockPool&) = delete;
  BlockPool& operator=(const BlockPool&) = delete;

  // Pops a block off the free list with refcount 1, or kInvalidBlock if full.
  BlockId allocate();

  void incref(BlockId id);
  // True when the block dropped to zero refs and went back on the free list.
  bool decref(BlockId id);
  std::uint32_t refcount(BlockId id) const;

  // Byte offset of the block within the tier. Valid on every tier, and the
  // only addressing a device-backed tier can offer.
  std::size_t block_offset(BlockId id) const;

  // Direct pointer to the block's bytes, or nullptr when the tier is not
  // host-addressable. Prefer block_offset() + tier().read/write in code that
  // must work on any tier.
  void* block_data(BlockId id);
  const void* block_data(BlockId id) const;

  Tier& tier() { return tier_; }
  const Tier& tier() const { return tier_; }

  std::size_t num_free() const { return free_list_.size(); }
  std::size_t num_blocks() const { return cfg_.num_blocks; }
  std::size_t num_used() const { return cfg_.num_blocks - free_list_.size(); }
  const CacheConfig& config() const { return cfg_; }

 private:
  void init();

  CacheConfig cfg_;
  std::unique_ptr<HostTier> owned_tier_;  // non-null only for the owning ctor
  Tier& tier_;
  std::vector<BlockId> free_list_;
  std::vector<std::uint32_t> refcount_;
};

}  // namespace kvslab
