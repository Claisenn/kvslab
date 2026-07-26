#pragma once

#include <cstddef>
#include <vector>

#include "kvslab/types.hpp"

namespace kvslab {

// Fixed-size block allocator over one contiguous, page-aligned arena.
//
// This is the storage substrate everything else sits on. A production stack
// backs the arena with device memory (cudaMalloc, or the VMM APIs when it wants
// to grow the pool without reserving it up front); here it is plain host memory
// so the whole engine builds and runs on a laptop with no GPU. Replacing the
// arena with a device allocation is the `Tier` abstraction on the roadmap.
//
// Not thread safe. Serving engines drive block allocation from a single
// scheduler thread; making this lock-free is a later exercise, not free
// complexity to take on now.
class BlockPool {
 public:
  explicit BlockPool(const CacheConfig& cfg);
  ~BlockPool();

  BlockPool(const BlockPool&) = delete;
  BlockPool& operator=(const BlockPool&) = delete;

  // Pops a block off the free list with refcount 1, or kInvalidBlock if full.
  BlockId allocate();

  void incref(BlockId id);
  // True when the block dropped to zero refs and went back on the free list.
  bool decref(BlockId id);
  std::uint32_t refcount(BlockId id) const;

  void* block_data(BlockId id);
  const void* block_data(BlockId id) const;

  std::size_t num_free() const { return free_list_.size(); }
  std::size_t num_blocks() const { return cfg_.num_blocks; }
  std::size_t num_used() const { return cfg_.num_blocks - free_list_.size(); }
  const CacheConfig& config() const { return cfg_; }

 private:
  CacheConfig cfg_;
  std::byte* arena_ = nullptr;
  std::size_t arena_bytes_ = 0;
  std::vector<BlockId> free_list_;
  std::vector<std::uint32_t> refcount_;
};

}  // namespace kvslab
