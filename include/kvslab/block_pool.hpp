#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "kvslab/tier.hpp"
#include "kvslab/types.hpp"

namespace kvslab {

// Fixed-size block allocator over one or more tiers.
//
// A BlockId is a stable identity, not an address. The pool keeps a per-block
// (tier, slot) location and resolves it on access, which is what lets a block
// migrate between tiers -- compute memory to spill memory and back -- while
// everything above (the prefix index, block tables held by requests) keeps
// naming it by the same id. Without the indirection, every demotion would
// have to rewrite the index.
//
// Tier 0 is the compute tier: allocate() only hands out slots there, because
// new KV is always written where the kernel runs. Other tiers exist to hold
// demoted blocks. migrate() moves a block's bytes and updates its location;
// ids, refcounts and pins are untouched.
//
// Not thread safe. Serving engines drive block allocation from a single
// scheduler thread; making this lock-free is a later exercise, not free
// complexity to take on now.
class BlockPool {
 public:
  using TierIndex = std::uint32_t;

  // A tier and how many blocks of cfg geometry it holds.
  struct TierSpec {
    Tier* tier = nullptr;         // borrowed; must outlive the pool
    std::size_t num_blocks = 0;
  };

  // Single-tier pools, as before: one owned HostTier sized for `cfg`, or one
  // borrowed tier that must be at least cfg.total_bytes() large.
  explicit BlockPool(const CacheConfig& cfg);
  BlockPool(Tier& tier, const CacheConfig& cfg);

  // Multi-tier pool. `specs[0]` is the compute tier. `cfg.num_blocks` is
  // ignored; capacity comes from the specs.
  BlockPool(const std::vector<TierSpec>& specs, const CacheConfig& cfg);

  BlockPool(const BlockPool&) = delete;
  BlockPool& operator=(const BlockPool&) = delete;

  // Pops a free block id bound to a free slot in tier 0, refcount 1, or
  // kInvalidBlock when tier 0 is full. Other tiers being free does not help:
  // new KV must land where compute can reach it.
  BlockId allocate();

  void incref(BlockId id);
  // True when the block dropped to zero refs and went back on the free list.
  bool decref(BlockId id);
  std::uint32_t refcount(BlockId id) const;

  // Where the block currently lives.
  TierIndex block_tier(BlockId id) const;

  // Moves the block's bytes to a free slot in `dst` and rebinds its location.
  // The id, refcount, and every pointer-free reference to the block stay
  // valid. Returns false when `dst` has no free slot; the block is untouched.
  // Synchronous: the copy happens on the calling thread.
  bool migrate(BlockId id, TierIndex dst);

  // --- Asynchronous migration -------------------------------------------
  //
  // The copy is the expensive half of a migration; the bookkeeping is a few
  // list operations. These entry points split the two: the calling thread
  // reserves the destination and hands the copy to a background worker, and
  // the block's location changes only when a later drain_migrations() call
  // observes the finished copy. Until then the block still lives -- and is
  // still readable -- where it was.
  //
  // Every function here must be called from the same thread that drives the
  // rest of the pool. The worker touches nothing but the job queue and the
  // bytes; all bookkeeping stays externally synchronized, same as ever.

  // Reserves a slot in `dst` and queues the copy. False when `dst` is full or
  // the block already has a migration in flight.
  bool demote_async(BlockId id, TierIndex dst);

  // True while a queued or running migration exists for the block.
  bool migrating(BlockId id) const;

  // Abandons an in-flight migration. The block keeps its current location;
  // the reserved destination slot is reclaimed once the worker is done with
  // it. No-op if nothing is in flight.
  void cancel_migration(BlockId id);

  // Applies every migration whose copy has finished: rebinds locations and
  // returns reserved slots from cancelled jobs. Cheap when nothing finished.
  void drain_migrations();

  // Blocks until the queue is empty, then drains. For tests and shutdown.
  void wait_for_migrations();

  std::size_t pending_migrations() const { return migrating_.size(); }

  // Byte offset of the block within its current tier's arena. Valid on every
  // tier; the offset is only meaningful together with block_tier().
  std::size_t block_offset(BlockId id) const;

  // Direct pointer to the block's bytes, or nullptr when its current tier is
  // not host-addressable.
  void* block_data(BlockId id);
  const void* block_data(BlockId id) const;

  Tier& tier(TierIndex idx = 0) { return *tiers_[idx].tier; }
  const Tier& tier(TierIndex idx = 0) const { return *tiers_[idx].tier; }
  std::size_t num_tiers() const { return tiers_.size(); }

  // Free slots in the compute tier -- the number acquire() can still allocate.
  std::size_t num_free() const { return tiers_[0].free_slots.size(); }
  std::size_t tier_free(TierIndex idx) const { return tiers_[idx].free_slots.size(); }
  std::size_t tier_blocks(TierIndex idx) const { return tiers_[idx].num_blocks; }

  // Ids in use across all tiers.
  std::size_t num_blocks() const { return location_.size(); }
  std::size_t num_used() const { return location_.size() - free_ids_.size(); }
  const CacheConfig& config() const { return cfg_; }

 public:
  ~BlockPool();

 private:
  struct TierState {
    Tier* tier = nullptr;
    std::size_t num_blocks = 0;
    std::vector<std::uint32_t> free_slots;
  };

  struct Location {
    TierIndex tier = 0;
    std::uint32_t slot = 0;
  };

  // One queued copy. The scheduler thread owns everything except `done`,
  // which the worker sets, and `cancelled`, which the worker reads to skip
  // work; src/dst pointers are immutable once queued.
  struct MigrationJob {
    BlockId id = kInvalidBlock;
    TierIndex dst_tier = 0;
    std::uint32_t dst_slot = 0;
    const std::byte* src = nullptr;
    std::byte* dst = nullptr;
    std::size_t bytes = 0;
    std::atomic<bool> done{false};
    std::atomic<bool> cancelled{false};
  };

  void init(const std::vector<TierSpec>& specs);
  const Location& locate(BlockId id) const;
  void worker_loop();
  void ensure_worker();

  CacheConfig cfg_;
  std::unique_ptr<HostTier> owned_tier_;  // non-null only for the owning ctor
  std::vector<TierState> tiers_;
  std::vector<Location> location_;        // indexed by BlockId
  std::vector<std::uint32_t> refcount_;   // indexed by BlockId
  std::vector<BlockId> free_ids_;
  std::vector<std::byte> bounce_;         // scratch for tier-to-tier copies

  // Jobs in queue order; the worker completes them front to back. The map is
  // the "in flight" flag: a block is migrating iff its id is a key.
  std::deque<std::unique_ptr<MigrationJob>> jobs_;
  std::unordered_map<BlockId, MigrationJob*> migrating_;

  std::thread worker_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::size_t next_unclaimed_ = 0;  // guarded by queue_mutex_: worker's cursor
  bool stopping_ = false;           // guarded by queue_mutex_
};

}  // namespace kvslab
