#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "kvslab/block_pool.hpp"
#include "kvslab/cache_manager.hpp"
#include "kvslab/radix_tree.hpp"
#include "kvslab/tier.hpp"

using namespace kvslab;

namespace {

// Tiny geometry: these tests care about block bookkeeping, not about moving
// realistic volumes of KV around.
CacheConfig tiny_config(std::size_t num_blocks, std::size_t block_tokens = 4) {
  CacheConfig cfg;
  cfg.num_blocks = num_blocks;
  cfg.block_tokens = block_tokens;
  cfg.num_layers = 1;
  cfg.num_kv_heads = 1;
  cfg.head_dim = 8;
  cfg.dtype_bytes = 2;
  return cfg;
}

std::vector<TokenId> iota_tokens(TokenId first, std::size_t count) {
  std::vector<TokenId> tokens(count);
  std::iota(tokens.begin(), tokens.end(), first);
  return tokens;
}

std::vector<TokenId> concat(std::vector<TokenId> head, const std::vector<TokenId>& tail) {
  head.insert(head.end(), tail.begin(), tail.end());
  return head;
}

// Stands in for what a request does: allocate blocks for the whole sequence,
// publish them to the index, then drop the request's own references.
std::vector<BlockId> publish(RadixTree& tree, BlockPool& pool,
                             const std::vector<TokenId>& tokens) {
  const std::size_t block_tokens = pool.config().block_tokens;
  const std::size_t count = tokens.size() / block_tokens;
  std::vector<BlockId> ids;
  ids.reserve(count);
  for (std::size_t i = 0; i < count; ++i) ids.push_back(pool.allocate());
  tree.insert(tokens, ids);
  for (BlockId id : ids) pool.decref(id);
  return ids;
}

// Blocks actually reachable from the root, counted by walking the tree.
//
// Deliberately not stored_blocks(): that counter is maintained by the same code
// paths a bookkeeping bug would corrupt, so comparing it against the pool tells
// you the two agree, not that either is right. Walking finds blocks the index
// dropped on the floor and blocks it stored twice.
std::size_t reachable_blocks(const RadixTree& tree) {
  std::vector<BlockId> seen;
  std::vector<const RadixTree::Node*> stack{tree.root()};
  while (!stack.empty()) {
    const RadixTree::Node* node = stack.back();
    stack.pop_back();
    seen.insert(seen.end(), node->blocks.begin(), node->blocks.end());
    for (const auto& [key, child] : node->children) stack.push_back(child.get());
  }
  std::sort(seen.begin(), seen.end());
  CHECK(std::adjacent_find(seen.begin(), seen.end()) == seen.end());  // no block stored twice
  return seen.size();
}

// One full request lifecycle through the manager.
void run_once(CacheManager& cm, const std::vector<TokenId>& tokens) {
  auto alloc = cm.acquire(tokens);
  CHECK(alloc.ok);
  cm.release(tokens, alloc);
}

}  // namespace

TEST(host_tier_round_trips_bytes) {
  HostTier tier(4000);
  // Capacity is rounded up to a page so the arena can be pinned or registered
  // whole; the tier reports what it actually owns rather than what was asked for.
  CHECK_EQ(tier.capacity_bytes(), std::size_t{4096});
  CHECK(tier.host_data() != nullptr);

  const std::uint64_t written = 0xfeedfacecafebeefull;
  tier.write(128, &written, sizeof(written));
  std::uint64_t read = 0;
  tier.read(128, &read, sizeof(read));
  CHECK_EQ(read, written);
}

TEST(block_pool_lays_blocks_out_on_a_borrowed_tier) {
  const CacheConfig cfg = tiny_config(4);
  HostTier tier(cfg.total_bytes());
  BlockPool pool(tier, cfg);

  const BlockId a = pool.allocate();
  const BlockId b = pool.allocate();
  CHECK_EQ(pool.block_offset(a), static_cast<std::size_t>(a) * cfg.block_bytes());
  CHECK(pool.block_data(a) == static_cast<void*>(tier.host_data() + pool.block_offset(a)));

  // Blocks must not overlap: filling one leaves its neighbour untouched.
  std::memset(pool.block_data(a), 0xAB, cfg.block_bytes());
  std::memset(pool.block_data(b), 0x00, cfg.block_bytes());
  const auto* bytes_a = static_cast<const unsigned char*>(pool.block_data(a));
  CHECK_EQ(bytes_a[cfg.block_bytes() - 1], static_cast<unsigned char>(0xAB));
}

TEST(block_pool_rejects_a_tier_that_cannot_hold_it) {
  const CacheConfig cfg = tiny_config(64);
  HostTier tier(cfg.total_bytes() / 2);  // half the room the pool needs

  bool threw = false;
  try {
    BlockPool pool(tier, cfg);
    (void)pool;
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

TEST(block_pool_allocates_and_recycles) {
  BlockPool pool(tiny_config(4));
  CHECK_EQ(pool.num_free(), std::size_t{4});

  std::vector<BlockId> ids;
  for (int i = 0; i < 4; ++i) {
    const BlockId id = pool.allocate();
    CHECK(id != kInvalidBlock);
    CHECK_EQ(pool.refcount(id), std::uint32_t{1});
    ids.push_back(id);
  }
  CHECK_EQ(pool.num_free(), std::size_t{0});
  CHECK(pool.allocate() == kInvalidBlock);

  // Distinct blocks must map to distinct storage.
  CHECK(pool.block_data(ids[0]) != pool.block_data(ids[1]));

  pool.incref(ids[0]);
  CHECK_EQ(pool.refcount(ids[0]), std::uint32_t{2});
  CHECK(!pool.decref(ids[0]));  // still held
  CHECK(pool.decref(ids[0]));   // now back on the free list
  CHECK_EQ(pool.num_free(), std::size_t{1});
}

TEST(radix_tree_matches_an_exact_sequence) {
  BlockPool pool(tiny_config(64));
  RadixTree tree(pool, 4);

  const std::vector<TokenId> seq = iota_tokens(1, 16);  // exactly 4 blocks
  const std::vector<BlockId> published = publish(tree, pool, seq);
  CHECK_EQ(tree.stored_blocks(), std::size_t{4});

  auto match = tree.match_prefix(seq);
  CHECK_EQ(match.num_tokens, std::size_t{16});
  // Reuse means the *same* physical blocks, not a fresh copy of the KV.
  CHECK(match.blocks == published);
}

TEST(radix_tree_reuses_a_shared_prefix) {
  BlockPool pool(tiny_config(64));
  RadixTree tree(pool, 4);
  publish(tree, pool, iota_tokens(1, 16));

  // Shares the first 8 tokens, then diverges on a block boundary.
  const std::vector<TokenId> forked = concat(iota_tokens(1, 8), {900, 901, 902, 903});
  auto match = tree.match_prefix(forked);
  CHECK_EQ(match.num_tokens, std::size_t{8});
  CHECK_EQ(match.blocks.size(), std::size_t{2});
}

TEST(radix_tree_match_stops_at_the_block_boundary) {
  BlockPool pool(tiny_config(64));
  RadixTree tree(pool, 4);
  publish(tree, pool, iota_tokens(1, 8));

  // Agrees on tokens 1..4 and then 5,6 -- but the second block differs, and
  // half a block is not reusable, so the hit has to stop at 4.
  const std::vector<TokenId> forked = {1, 2, 3, 4, 5, 6, 99, 100};
  auto match = tree.match_prefix(forked);
  CHECK_EQ(match.num_tokens, std::size_t{4});
}

TEST(radix_tree_split_preserves_both_branches) {
  BlockPool pool(tiny_config(64));
  RadixTree tree(pool, 4);

  const std::vector<TokenId> long_seq = iota_tokens(1, 16);  // one 4-block node
  publish(tree, pool, long_seq);

  // Forces a split of that node after 2 blocks.
  const std::vector<TokenId> forked = concat(iota_tokens(1, 8), {700, 701, 702, 703});
  publish(tree, pool, forked);

  // 2 shared blocks + 2 in the original tail + 1 in the fork.
  CHECK_EQ(tree.stored_blocks(), std::size_t{5});

  // Both branches must still match end to end across the split.
  CHECK_EQ(tree.match_prefix(long_seq).num_tokens, std::size_t{16});
  CHECK_EQ(tree.match_prefix(forked).num_tokens, std::size_t{12});

  // Blocks the fork allocated but did not get to keep must be back on the free
  // list, not stranded.
  CHECK_EQ(pool.num_used(), reachable_blocks(tree));
}

TEST(radix_tree_split_under_a_live_pin_keeps_the_prefix_protected) {
  BlockPool pool(tiny_config(8));
  RadixTree tree(pool, 4);

  const std::vector<TokenId> long_seq = iota_tokens(1, 16);  // one 4-block node
  publish(tree, pool, long_seq);

  // A request pins the whole sequence...
  auto held = tree.match_prefix(long_seq);
  CHECK_EQ(held.num_tokens, std::size_t{16});
  tree.lock(held.node);

  // ...and another diverges inside the pinned node, forcing a split. held.node
  // is now only the tail of what it covered a moment ago.
  publish(tree, pool, concat(iota_tokens(1, 8), {700, 701, 702, 703}));

  // Eviction may take the fork, but nothing the pin covers. This is the
  // invariant the split direction exists to preserve.
  tree.evict(8);
  CHECK_EQ(tree.match_prefix(long_seq).num_tokens, std::size_t{16});

  // Unlocking has to balance across the interposed parent. If the split had not
  // handed it the count, this would decrement a zero and trip the assert; if it
  // had over-counted, the nodes below would stay pinned and survive the sweep.
  tree.unlock(held.node);
  CHECK_EQ(tree.evict(4), std::size_t{4});
  CHECK_EQ(tree.stored_blocks(), std::size_t{0});
  CHECK_EQ(pool.num_used(), std::size_t{0});
}

TEST(radix_tree_does_not_cache_a_partial_trailing_block) {
  BlockPool pool(tiny_config(64));
  RadixTree tree(pool, 4);

  // 10 tokens = 2 full blocks + a half-filled one, whose KV still depends on
  // tokens the request has not committed to.
  const std::vector<TokenId> seq = iota_tokens(1, 10);
  std::vector<BlockId> ids;
  for (int i = 0; i < 3; ++i) ids.push_back(pool.allocate());
  CHECK_EQ(tree.insert(seq, ids), std::size_t{2});
  for (BlockId id : ids) pool.decref(id);

  CHECK_EQ(tree.stored_blocks(), std::size_t{2});
  CHECK_EQ(pool.num_used(), std::size_t{2});
}

TEST(radix_tree_evicts_the_least_recently_used_prefix) {
  BlockPool pool(tiny_config(8));
  RadixTree tree(pool, 4);

  // Four distinct 2-block sequences exactly fill the pool.
  for (TokenId base = 0; base < 4; ++base) {
    publish(tree, pool, iota_tokens(base * 1000 + 1, 8));
  }
  CHECK_EQ(pool.num_free(), std::size_t{0});

  const std::vector<TokenId> hot = iota_tokens(1, 8);
  tree.match_prefix(hot);  // moves `hot` to the front of the LRU order

  CHECK_EQ(tree.evict(2), std::size_t{2});
  CHECK_EQ(pool.num_free(), std::size_t{2});
  // The recently used sequence must not be what got dropped.
  CHECK_EQ(tree.match_prefix(hot).num_tokens, std::size_t{8});
}

TEST(radix_tree_eviction_leaves_blocks_that_are_still_held) {
  BlockPool pool(tiny_config(8));
  RadixTree tree(pool, 4);

  // Publish without dropping our own references, exactly like a sequence still
  // in flight: the tree and this test each hold every block.
  const std::vector<TokenId> held = iota_tokens(1, 8);
  std::vector<BlockId> ids;
  for (int i = 0; i < 2; ++i) ids.push_back(pool.allocate());
  tree.insert(held, ids);

  // A second sequence the tree owns outright.
  publish(tree, pool, iota_tokens(1000, 8));
  CHECK_EQ(tree.stored_blocks(), std::size_t{4});

  // Only the fully-owned sequence is reclaimable. Dropping the held one would
  // free nothing and strand its blocks, so it has to survive the sweep.
  CHECK_EQ(tree.evict(4), std::size_t{2});
  CHECK_EQ(tree.stored_blocks(), std::size_t{2});
  CHECK_EQ(tree.match_prefix(held).num_tokens, std::size_t{8});

  for (BlockId id : ids) pool.decref(id);
}

TEST(radix_tree_pin_protects_a_prefix_from_eviction) {
  BlockPool pool(tiny_config(4));
  RadixTree tree(pool, 4);
  publish(tree, pool, iota_tokens(1, 8));     // 2 blocks
  publish(tree, pool, iota_tokens(1000, 8));  // 2 blocks -> pool full
  CHECK_EQ(pool.num_free(), std::size_t{0});

  // Hold the first prefix the way an in-flight request would.
  auto held = tree.match_prefix(iota_tokens(1, 8));
  tree.lock(held.node);

  // Asked for 4 blocks, but only the unpinned sequence may be dropped.
  CHECK_EQ(tree.evict(4), std::size_t{2});
  CHECK_EQ(pool.num_free(), std::size_t{2});
  CHECK_EQ(tree.match_prefix(iota_tokens(1, 8)).num_tokens, std::size_t{8});

  // Once the holder is done, the same prefix becomes reclaimable.
  tree.unlock(held.node);
  CHECK_EQ(tree.evict(2), std::size_t{2});
  CHECK_EQ(pool.num_free(), std::size_t{4});
  CHECK_EQ(tree.stored_blocks(), std::size_t{0});
}

TEST(cache_manager_reuses_an_identical_sequence) {
  CacheManager cm(tiny_config(64));
  const std::vector<TokenId> seq = iota_tokens(1, 16);

  auto first = cm.acquire(seq);
  CHECK(first.ok);
  CHECK_EQ(first.cached_tokens, std::size_t{0});
  CHECK_EQ(first.blocks.size(), std::size_t{4});
  const std::vector<BlockId> original = first.blocks;
  cm.release(seq, first);

  auto second = cm.acquire(seq);
  CHECK(second.ok);
  CHECK_EQ(second.cached_tokens, std::size_t{16});
  CHECK(second.blocks == original);
  cm.release(seq, second);

  // Only the cached copy should still be resident.
  CHECK_EQ(cm.pool().num_used(), std::size_t{4});
  CHECK_EQ(cm.tree().stored_blocks(), std::size_t{4});
}

TEST(cache_manager_reuses_a_shared_prefix) {
  CacheManager cm(tiny_config(64));
  run_once(cm, iota_tokens(1, 16));

  const std::vector<TokenId> forked = concat(iota_tokens(1, 8), {900, 901, 902, 903});
  auto alloc = cm.acquire(forked);
  CHECK(alloc.ok);
  CHECK_EQ(alloc.cached_tokens, std::size_t{8});
  CHECK_EQ(alloc.cached_blocks, std::size_t{2});
  CHECK_EQ(alloc.blocks.size(), std::size_t{3});
  cm.release(forked, alloc);
}

TEST(cache_manager_does_not_publish_a_partial_trailing_block) {
  CacheManager cm(tiny_config(64));
  const std::vector<TokenId> seq = iota_tokens(1, 10);
  auto alloc = cm.acquire(seq);
  CHECK(alloc.ok);
  CHECK_EQ(alloc.blocks.size(), std::size_t{3});
  cm.release(seq, alloc);

  // The partial block is handed back rather than published to the index.
  CHECK_EQ(cm.tree().stored_blocks(), std::size_t{2});
  CHECK_EQ(cm.pool().num_used(), std::size_t{2});
}

TEST(cache_manager_evicts_under_pressure) {
  CacheManager cm(tiny_config(8));
  for (TokenId base = 0; base < 4; ++base) {
    run_once(cm, iota_tokens(base * 1000 + 1, 8));  // 2 blocks each -> pool full
  }
  CHECK_EQ(cm.pool().num_free(), std::size_t{0});

  const std::vector<TokenId> fresh = iota_tokens(50'000, 8);
  auto alloc = cm.acquire(fresh);
  CHECK(alloc.ok);
  CHECK_EQ(alloc.cached_tokens, std::size_t{0});
  CHECK(cm.stats().evicted_blocks >= 2);
  cm.release(fresh, alloc);
}

TEST(cache_manager_refuses_when_every_candidate_is_pinned) {
  CacheManager cm(tiny_config(4));
  const std::vector<TokenId> hot = iota_tokens(1, 8);
  run_once(cm, hot);

  auto held = cm.acquire(hot);
  CHECK(held.ok);
  CHECK_EQ(held.cached_tokens, std::size_t{8});

  // Needs 3 blocks with 2 free, and the only eviction candidate is pinned.
  const std::vector<TokenId> big = iota_tokens(5000, 12);
  auto blocked = cm.acquire(big);
  CHECK(!blocked.ok);
  CHECK_EQ(cm.stats().alloc_failures, std::uint64_t{1});
  // Failing must not have handed out the pinned blocks or leaked new ones.
  CHECK_EQ(cm.pool().refcount(held.blocks[0]), std::uint32_t{1});

  // Once the holder finishes, the same request goes through.
  cm.release(hot, held);
  auto retried = cm.acquire(big);
  CHECK(retried.ok);
  cm.release(big, retried);
}

TEST(cache_manager_release_spends_the_handle) {
  CacheManager cm(tiny_config(64));
  const std::vector<TokenId> seq = iota_tokens(1, 16);

  auto alloc = cm.acquire(seq);
  CHECK(alloc.ok);
  cm.release(seq, alloc);
  CHECK(!alloc.ok);

  // Releasing a spent handle must not decref the blocks or drop the pin a
  // second time. The Allocation being move-only stops a *copy* from doing that;
  // this covers the same handle being released twice.
  const std::size_t used = cm.pool().num_used();
  cm.release(seq, alloc);
  CHECK_EQ(cm.pool().num_used(), used);
  CHECK_EQ(cm.pool().num_used(), cm.tree().stored_blocks());

  // An underflowed lock count would leave the prefix unprotected or stuck
  // pinned; either way the cached entry must still be intact and reusable.
  auto again = cm.acquire(seq);
  CHECK(again.ok);
  CHECK_EQ(again.cached_tokens, std::size_t{16});
  cm.release(seq, again);
}

TEST(cache_manager_leaks_no_blocks_across_a_mixed_workload) {
  CacheManager cm(tiny_config(32));
  const std::vector<TokenId> prompt = iota_tokens(1, 12);
  for (TokenId turn = 0; turn < 20; ++turn) {
    run_once(cm, concat(prompt, iota_tokens(turn * 100 + 500, 9)));
  }
  // Every block still allocated must be one the index can actually reach;
  // anything else is a block nothing owns and nothing will ever free.
  CHECK_EQ(cm.pool().num_used(), reachable_blocks(cm.tree()));
  // And the counter must agree with the walk, which is a separate claim: one
  // catches leaks, the other catches the bookkeeping drifting away from them.
  CHECK_EQ(cm.tree().stored_blocks(), reachable_blocks(cm.tree()));
  // No entry should have been refused: a collision here would mean the 64-bit
  // block key is doing far worse than chance on a 20-sequence workload.
  CHECK_EQ(cm.tree().hash_collisions(), std::size_t{0});
}

int main() { return kvcheck::run_all(); }
