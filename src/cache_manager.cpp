#include "kvslab/cache_manager.hpp"

#include <cassert>
#include <utility>

namespace kvslab {

void CacheManager::Allocation::disarm() {
  ok = false;
  cached_tokens = 0;
  cached_blocks = 0;
  blocks.clear();
  pinned = nullptr;
  owner_ = nullptr;
}

CacheManager::Allocation::~Allocation() {
  if (ok && owner_ != nullptr) owner_->abandon(*this);
}

CacheManager::Allocation::Allocation(Allocation&& other) noexcept
    : ok(other.ok),
      cached_tokens(other.cached_tokens),
      cached_blocks(other.cached_blocks),
      blocks(std::move(other.blocks)),
      pinned(other.pinned),
      owner_(other.owner_) {
  // The moved-from handle must not release what this one now owns.
  other.disarm();
}

CacheManager::Allocation& CacheManager::Allocation::operator=(
    Allocation&& other) noexcept {
  if (this == &other) return *this;
  if (ok && owner_ != nullptr) owner_->abandon(*this);

  ok = other.ok;
  cached_tokens = other.cached_tokens;
  cached_blocks = other.cached_blocks;
  blocks = std::move(other.blocks);
  pinned = other.pinned;
  owner_ = other.owner_;
  other.disarm();
  return *this;
}

void CacheManager::abandon(Allocation& alloc) {
  // Everything release() does except publishing to the index. Only the blocks
  // this sequence allocated carry a reference from it; the cached prefix was
  // held by the pin.
  for (std::size_t i = alloc.cached_blocks; i < alloc.blocks.size(); ++i) {
    pool_.decref(alloc.blocks[i]);
  }
  if (alloc.pinned != nullptr) tree_.unlock(alloc.pinned);
  ++stats_.abandoned;
  alloc.disarm();
}

CacheManager::CacheManager(const CacheConfig& cfg)
    : cfg_(cfg), pool_(cfg), tree_(pool_, cfg.block_tokens) {}

CacheManager::CacheManager(const std::vector<BlockPool::TierSpec>& specs,
                           const CacheConfig& cfg, Options options)
    : cfg_(cfg),
      pool_(specs, cfg),
      tree_(pool_, cfg.block_tokens),
      options_(options) {}

CacheManager::CacheManager(const std::vector<BlockPool::TierSpec>& specs,
                           const CacheConfig& cfg)
    : CacheManager(specs, cfg, Options{}) {}

bool CacheManager::Allocation::ready() {
  if (!ok || owner_ == nullptr) return true;  // nothing outstanding to wait on
  owner_->pool_.drain_migrations();
  for (BlockId id : blocks) {
    if (owner_->pool_.block_tier(id) != 0) return false;
  }
  return true;
}

void CacheManager::Allocation::wait_ready() {
  // wait_for_migrations() waits on every queued job, not just this table's
  // promotions, so this can overshoot by a demotion or two. Acceptable until a
  // profile says targeted waiting is worth the bookkeeping.
  while (!ready()) owner_->pool_.wait_for_migrations();
}

void CacheManager::maintain() {
  if (pool_.num_tiers() < 2 || options_.spill_watermark == 0) return;

  // Count in-flight demotions as free-to-be: each will hand back a compute
  // slot at a future drain, and scheduling more for the same shortage would
  // demote deeper into the working set than the watermark asks for. In-flight
  // promotions already claimed their slot and contribute nothing.
  const std::size_t projected = pool_.num_free() + pool_.pending_departures(0);
  if (projected >= options_.spill_watermark) return;

  victims_.clear();
  tree_.pick_demotion_victims(options_.spill_watermark - projected, &victims_);
  for (BlockId id : victims_) {
    if (!pool_.migrate_async(id, 1)) break;  // spill full; no point continuing
    ++stats_.background_demotions;
  }
}

CacheManager::Allocation CacheManager::acquire(const std::vector<TokenId>& tokens) {
  ++stats_.requests;

  Allocation alloc;
  if (tokens.empty()) {
    ++stats_.served;
    alloc.ok = true;
    return alloc;
  }

  const std::size_t bt = cfg_.block_tokens;
  const std::size_t need_blocks = (tokens.size() + bt - 1) / bt;

  // Fold in any background copies that finished since the last call; their
  // compute slots are this request's first source of room.
  pool_.drain_migrations();

  RadixTree::MatchResult match = tree_.match_prefix(tokens);
  if (match.node != nullptr) {
    // Pin before doing anything that can evict, or the eviction below is free
    // to reclaim the very blocks we are about to hand out.
    tree_.lock(match.node);
    alloc.pinned = match.node;
  }
  alloc.cached_tokens = match.num_tokens;
  alloc.cached_blocks = match.blocks.size();
  alloc.blocks = std::move(match.blocks);

  // A matched block with a demotion in flight is about to be read; the move
  // must not land after the block table is handed out. Cancelling keeps the
  // block where it is -- it never left the compute tier. A promotion in
  // flight is the opposite case: it is exactly what this request needs, so it
  // is left to finish (possibly scheduled by an earlier request on the same
  // prefix, which is also why it must not be cancelled out from under them).
  for (BlockId id : alloc.blocks) {
    if (pool_.migrating(id) && pool_.migration_target(id) != 0) {
      pool_.cancel_migration(id);
    }
  }

  // Compute-tier demand: slots for the new blocks, plus one per matched block
  // currently in a spill tier -- the kernel reads from compute memory, so a
  // hit on a demoted prefix has to bring it back before it counts. A block
  // already promoting has its slot reserved by the in-flight job and costs
  // nothing more.
  const std::size_t fresh = need_blocks - alloc.cached_blocks;
  std::size_t promotions = 0;
  for (BlockId id : alloc.blocks) {
    if (pool_.block_tier(id) != 0 && !pool_.migrating(id)) ++promotions;
  }
  const std::size_t demand = fresh + promotions;

  if (pool_.num_free() < demand && pool_.num_tiers() > 1) {
    // Prefer demotion: it frees the same compute slots but keeps the entries,
    // so the cost is a future promotion instead of a recompute. The matched
    // path is pinned and the picker skips pinned nodes, so it cannot choose
    // the very blocks this request is about to promote.
    victims_.clear();
    tree_.pick_demotion_victims(demand - pool_.num_free(), &victims_);
    for (BlockId id : victims_) {
      if (!pool_.migrate(id, 1)) break;  // spill full; eviction is next
      ++stats_.demoted_blocks;
    }
  }
  while (pool_.num_free() < demand) {
    const std::size_t evicted = tree_.evict(demand - pool_.num_free());
    if (evicted == 0) break;
    stats_.evicted_blocks += evicted;
  }
  if (pool_.num_free() < demand) {
    // Oversubscribed by in-flight sequences. Back out cleanly and let the
    // scheduler retry once something finishes; a partial allocation would leak.
    if (alloc.pinned != nullptr) tree_.unlock(alloc.pinned);
    ++stats_.alloc_failures;
    return Allocation{};
  }

  // Promote what the hit needs. Slots were counted into `demand`, so these
  // cannot fail; the spill slots they vacate become room for future demotions.
  // Blocks already promoting are skipped -- their job carries them.
  for (BlockId id : alloc.blocks) {
    if (pool_.block_tier(id) == 0 || pool_.migrating(id)) continue;
    bool moved;
    if (options_.async_promotion) {
      // The copy overlaps whatever the engine does next; the caller gates on
      // ready() before reading. Falls back to a synchronous copy for a tier
      // the worker cannot address.
      moved = pool_.migrate_async(id, 0);
      if (!moved) moved = pool_.migrate(id, 0);
    } else {
      moved = pool_.migrate(id, 0);
    }
    assert(moved && "promotion failed despite reserved compute slots");
    (void)moved;
    ++stats_.promoted_blocks;
  }

  alloc.blocks.reserve(need_blocks);
  for (std::size_t i = 0; i < fresh; ++i) {
    const BlockId id = pool_.allocate();
    assert(id != kInvalidBlock && "free list disagrees with num_free()");
    alloc.blocks.push_back(id);
  }

  // Counted here rather than at entry so the hit rate's numerator and
  // denominator describe the same set of requests.
  ++stats_.served;
  stats_.total_tokens += tokens.size();
  stats_.hit_tokens += alloc.cached_tokens;
  alloc.ok = true;
  alloc.owner_ = this;  // arms the handle: dropping it now gives the blocks back

  // Start refilling the free headroom now, so the copies run while the engine
  // computes against the table we just returned.
  maintain();
  return alloc;
}

void CacheManager::release(const std::vector<TokenId>& tokens, Allocation& alloc) {
  if (!alloc.ok) return;

  // Publish the freshly computed blocks. Anything already in the tree -- because
  // a concurrent sequence got there first -- is left as is, and our duplicate
  // copies fall to zero refs below.
  tree_.insert(tokens, alloc.blocks);

  // Only the blocks this sequence allocated carry a reference from it. The
  // cached prefix was protected by the pin, not by a refcount, so decref'ing it
  // here would be a double free.
  for (std::size_t i = alloc.cached_blocks; i < alloc.blocks.size(); ++i) {
    pool_.decref(alloc.blocks[i]);
  }

  if (alloc.pinned != nullptr) tree_.unlock(alloc.pinned);
  // disarm() rather than assigning a fresh handle: move-assignment would see
  // this one still armed and abandon it, releasing everything a second time.
  alloc.disarm();

  pool_.drain_migrations();
  maintain();
}

}  // namespace kvslab
