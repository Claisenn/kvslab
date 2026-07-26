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
  // A sequence's claim on the cache: the block table to run against, plus the
  // pin holding its reused prefix in place.
  //
  // Move-only, because it owns those two things and neither survives being
  // duplicated. Releasing a copy would drop the pin a second time -- underflowing
  // the lock count and exposing the prefix to eviction while the original is
  // still reading it -- and decref the fresh blocks twice, handing live blocks
  // back to the free list for another sequence to overwrite.
  //
  // Destroying a live handle gives the blocks and the pin back rather than
  // stranding them. It cannot publish to the index on the way out -- that needs
  // the token sequence, which the handle does not carry -- so the KV is lost,
  // but a dropped handle costs a cache entry instead of leaking pool capacity
  // and pinning a prefix that can then never be evicted.
  struct Allocation {
    Allocation() = default;
    ~Allocation();
    Allocation(Allocation&& other) noexcept;
    Allocation& operator=(Allocation&& other) noexcept;
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;

    bool ok = false;
    std::size_t cached_tokens = 0;  // prefix served from cache
    std::size_t cached_blocks = 0;  // leading entries of `blocks` already cached
    std::vector<BlockId> blocks;    // full block table for the sequence
    RadixTree::Node* pinned = nullptr;

    // True once every block in the table is resident in the compute tier.
    // With async promotion the table comes back before the copies land, so
    // the engine checks (or waits) here before launching a kernel against it.
    // Always true when promotions ran synchronously.
    bool ready();
    void wait_ready();

   private:
    friend class CacheManager;
    // Give up ownership without releasing anything -- for the paths that have
    // already done the releasing themselves.
    void disarm();

    CacheManager* owner_ = nullptr;
  };

  struct Stats {
    std::uint64_t requests = 0;        // every acquire(), served or not
    std::uint64_t served = 0;          // those that returned a block table
    std::uint64_t total_tokens = 0;    // tokens across served requests
    std::uint64_t hit_tokens = 0;      // of those, the ones cache supplied
    std::uint64_t evicted_blocks = 0;
    std::uint64_t demoted_blocks = 0;      // moved to spill on the request path
    std::uint64_t background_demotions = 0;  // copies handed to the worker
    std::uint64_t promoted_blocks = 0;     // brought back to compute on a hit
    std::uint64_t alloc_failures = 0;
    std::uint64_t abandoned = 0;  // handles destroyed without release()

    // Share of served tokens that came from cache.
    //
    // A failed acquisition contributes to neither side. It served nothing, so
    // counting its tokens in the denominator alone would report a hit-rate
    // collapse for what is really a capacity problem -- and capacity already
    // has its own number in alloc_failures.
    double hit_rate() const {
      return total_tokens == 0 ? 0.0
                               : static_cast<double>(hit_tokens) /
                                     static_cast<double>(total_tokens);
    }
  };

  // Single-tier: one owned HostTier, capacity from cfg.num_blocks. Under
  // pressure the only relief is eviction, as before.
  explicit CacheManager(const CacheConfig& cfg);

  struct Options {
    // Compute slots the manager tries to keep free by demoting cold entries in
    // the background, so requests find room waiting instead of paying for a
    // copy on the acquire path. Zero disables background demotion; pressure is
    // then relieved synchronously, on demand.
    std::size_t spill_watermark = 0;
    // Run promotions on the worker too: acquire() returns the block table
    // before the copies land, and the caller gates on Allocation::ready().
    // Off means a hit on a spilled prefix pays its copies before returning,
    // and every allocation is born ready.
    bool async_promotion = false;
  };

  // Tiered: specs[0] is the compute tier, specs[1] the spill tier cold
  // entries demote into instead of being dropped. Tiers are borrowed and must
  // outlive the manager.
  CacheManager(const std::vector<BlockPool::TierSpec>& specs, const CacheConfig& cfg,
               Options options);
  // Split from the above because a nested struct's default member initializers
  // cannot back a default argument inside the enclosing class definition.
  CacheManager(const std::vector<BlockPool::TierSpec>& specs, const CacheConfig& cfg);

  Allocation acquire(const std::vector<TokenId>& tokens);
  void release(const std::vector<TokenId>& tokens, Allocation& alloc);

  const Stats& stats() const { return stats_; }
  const CacheConfig& config() const { return cfg_; }
  BlockPool& pool() { return pool_; }
  RadixTree& tree() { return tree_; }

 private:
  // Hand back what a destroyed handle still holds, without publishing it.
  void abandon(Allocation& alloc);
  // Top the compute tier's free slots back up to the watermark by scheduling
  // background demotions. Runs after every acquire and release.
  void maintain();

  CacheConfig cfg_;
  BlockPool pool_;
  RadixTree tree_;
  Stats stats_;
  Options options_;
  std::vector<BlockId> victims_;  // scratch for demotion picks
};

}  // namespace kvslab
