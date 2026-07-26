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
                           const CacheConfig& cfg)
    : cfg_(cfg), pool_(specs, cfg), tree_(pool_, cfg.block_tokens) {}

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

  // Compute-tier demand: slots for the new blocks, plus one per matched block
  // currently in a spill tier -- the kernel reads from compute memory, so a
  // hit on a demoted prefix has to bring it back before it counts.
  const std::size_t fresh = need_blocks - alloc.cached_blocks;
  std::size_t promotions = 0;
  for (BlockId id : alloc.blocks) {
    if (pool_.block_tier(id) != 0) ++promotions;
  }
  const std::size_t demand = fresh + promotions;

  if (pool_.num_free() < demand && pool_.num_tiers() > 1) {
    // Prefer demotion: it frees the same compute slots but keeps the entries,
    // so the cost is a future promotion instead of a recompute. The matched
    // path is pinned and demote() skips pinned nodes, so it cannot move the
    // very blocks this request is about to promote.
    stats_.demoted_blocks += tree_.demote(demand - pool_.num_free(), 1);
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
  for (BlockId id : alloc.blocks) {
    if (pool_.block_tier(id) == 0) continue;
    const bool moved = pool_.migrate(id, 0);
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
}

}  // namespace kvslab
