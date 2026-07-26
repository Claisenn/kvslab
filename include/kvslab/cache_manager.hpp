#pragma once

#include <cstdint>
#include <vector>

#include "kvslab/block_pool.hpp"
#include "kvslab/radix_tree.hpp"
#include "kvslab/types.hpp"

namespace kvslab {

// Ties the allocator and the prefix index into the interface a scheduler wants:
// hand it a token sequence, get back a block table.
//
// Lifecycle of one request:
//   acquire(tokens) -> reuse the cached prefix, allocate the rest, evicting if
//                      the pool is short. The reused prefix is pinned.
//   ... run the model, writing KV into the fresh blocks ...
//   release(tokens, alloc) -> publish the new blocks to the prefix index so the
//                      next request can reuse them, then unpin.
class CacheManager {
 public:
  struct Allocation {
    bool ok = false;
    std::size_t cached_tokens = 0;  // prefix served from cache
    std::size_t cached_blocks = 0;  // leading entries of `blocks` already cached
    std::vector<BlockId> blocks;    // full block table for the sequence
    RadixTree::Node* pinned = nullptr;
  };

  struct Stats {
    std::uint64_t requests = 0;
    std::uint64_t total_tokens = 0;
    std::uint64_t hit_tokens = 0;
    std::uint64_t evicted_blocks = 0;
    std::uint64_t alloc_failures = 0;

    double hit_rate() const {
      return total_tokens == 0 ? 0.0
                               : static_cast<double>(hit_tokens) /
                                     static_cast<double>(total_tokens);
    }
  };

  explicit CacheManager(const CacheConfig& cfg);

  Allocation acquire(const std::vector<TokenId>& tokens);
  void release(const std::vector<TokenId>& tokens, Allocation& alloc);

  const Stats& stats() const { return stats_; }
  const CacheConfig& config() const { return cfg_; }
  BlockPool& pool() { return pool_; }
  RadixTree& tree() { return tree_; }

 private:
  CacheConfig cfg_;
  BlockPool pool_;
  RadixTree tree_;
  Stats stats_;
};

}  // namespace kvslab
