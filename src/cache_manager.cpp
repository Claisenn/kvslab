#include "kvslab/cache_manager.hpp"

#include <cassert>

namespace kvslab {

CacheManager::CacheManager(const CacheConfig& cfg)
    : cfg_(cfg), pool_(cfg), tree_(pool_, cfg.block_tokens) {}

CacheManager::Allocation CacheManager::acquire(const std::vector<TokenId>& tokens) {
  ++stats_.requests;
  stats_.total_tokens += tokens.size();

  Allocation alloc;
  if (tokens.empty()) {
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

  const std::size_t fresh = need_blocks - alloc.cached_blocks;
  if (pool_.num_free() < fresh) {
    stats_.evicted_blocks += tree_.evict(fresh - pool_.num_free());
  }
  if (pool_.num_free() < fresh) {
    // Oversubscribed by in-flight sequences. Back out cleanly and let the
    // scheduler retry once something finishes; a partial allocation would leak.
    if (alloc.pinned != nullptr) tree_.unlock(alloc.pinned);
    ++stats_.alloc_failures;
    return Allocation{};
  }

  alloc.blocks.reserve(need_blocks);
  for (std::size_t i = 0; i < fresh; ++i) {
    const BlockId id = pool_.allocate();
    assert(id != kInvalidBlock && "free list disagrees with num_free()");
    alloc.blocks.push_back(id);
  }

  stats_.hit_tokens += alloc.cached_tokens;
  alloc.ok = true;
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
  alloc = Allocation{};
}

}  // namespace kvslab
