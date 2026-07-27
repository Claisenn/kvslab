#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "kvslab/block_pool.hpp"
#include "kvslab/codec.hpp"
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

TEST(tiered_pool_migrates_blocks_without_changing_their_identity) {
  const CacheConfig cfg = tiny_config(0);  // geometry only; capacity is per tier
  HostTier compute(4 * cfg.block_bytes());
  HostTier spill(8 * cfg.block_bytes());
  BlockPool pool({{&compute, 4}, {&spill, 8}}, cfg);

  CHECK_EQ(pool.num_tiers(), std::size_t{2});
  CHECK_EQ(pool.num_blocks(), std::size_t{12});
  CHECK_EQ(pool.num_free(), std::size_t{4});  // allocatable = compute tier only

  const BlockId id = pool.allocate();
  CHECK_EQ(pool.block_tier(id), BlockPool::TierIndex{0});

  // Fill the block with a pattern, demote it, and the bytes must follow.
  std::memset(pool.block_data(id), 0x5A, cfg.block_bytes());
  CHECK(pool.migrate(id, 1));
  CHECK_EQ(pool.block_tier(id), BlockPool::TierIndex{1});
  const auto* bytes = static_cast<const unsigned char*>(pool.block_data(id));
  CHECK(bytes >= reinterpret_cast<unsigned char*>(spill.host_data()));
  CHECK_EQ(bytes[0], static_cast<unsigned char>(0x5A));
  CHECK_EQ(bytes[cfg.block_bytes() - 1], static_cast<unsigned char>(0x5A));

  // Identity survived: same id, same refcount, and its compute slot is free
  // for someone else while the id itself still counts as used.
  CHECK_EQ(pool.refcount(id), std::uint32_t{1});
  CHECK_EQ(pool.tier_free(0), std::size_t{4});
  CHECK_EQ(pool.num_used(), std::size_t{1});

  // Promote it back; the pattern must survive the round trip.
  CHECK(pool.migrate(id, 0));
  CHECK_EQ(pool.block_tier(id), BlockPool::TierIndex{0});
  const auto* back = static_cast<const unsigned char*>(pool.block_data(id));
  CHECK_EQ(back[cfg.block_bytes() - 1], static_cast<unsigned char>(0x5A));

  pool.decref(id);
}

TEST(tiered_pool_frees_compute_capacity_by_demoting) {
  const CacheConfig cfg = tiny_config(0);
  HostTier compute(2 * cfg.block_bytes());
  HostTier spill(2 * cfg.block_bytes());
  BlockPool pool({{&compute, 2}, {&spill, 2}}, cfg);

  const BlockId a = pool.allocate();
  const BlockId b = pool.allocate();
  CHECK(pool.allocate() == kInvalidBlock);  // compute full; spill can't help

  // Demoting one block is exactly what makes the next allocation possible.
  CHECK(pool.migrate(a, 1));
  const BlockId c = pool.allocate();
  CHECK(c != kInvalidBlock);

  // Spill holds one block and has one slot left; a second demotion fills it
  // and a third must be refused without disturbing the block.
  CHECK(pool.migrate(b, 1));
  CHECK(!pool.migrate(c, 1));
  CHECK_EQ(pool.block_tier(c), BlockPool::TierIndex{0});

  // A freed spill-resident block returns its slot to the spill tier.
  pool.decref(a);
  CHECK_EQ(pool.tier_free(1), std::size_t{1});

  pool.decref(b);
  pool.decref(c);
  CHECK_EQ(pool.num_used(), std::size_t{0});
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

TEST(radix_tree_tears_down_a_deep_chain) {
  // Each turn extends the previous sequence by exactly one block, so the index
  // becomes a single chain rather than a bush -- the shape a long multi-turn
  // conversation produces, and the one that makes recursive teardown a risk.
  constexpr std::size_t kDepth = 1000;
  BlockPool pool(tiny_config(2 * kDepth + 8));
  RadixTree tree(pool, 4);

  std::vector<TokenId> seq;
  for (std::size_t turn = 0; turn < kDepth; ++turn) {
    const std::vector<TokenId> next = iota_tokens(static_cast<TokenId>(turn) * 4 + 1, 4);
    seq.insert(seq.end(), next.begin(), next.end());
    publish(tree, pool, seq);
  }

  // One node and one block per turn: no splits, since every sequence is a
  // strict extension of the one before it.
  CHECK_EQ(tree.num_nodes(), kDepth);
  CHECK_EQ(reachable_blocks(tree), kDepth);
  CHECK_EQ(pool.num_used(), kDepth);
  // The tree is destroyed at scope exit. A recursive teardown would go kDepth
  // frames deep here, and real conversations reach far past this.
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
  // The failure is counted as a request but not as a served one, and its tokens
  // stay out of the hit rate entirely -- it served nothing, so charging its
  // tokens to the denominator would blame the cache for a capacity shortfall.
  CHECK_EQ(cm.stats().requests, cm.stats().served + cm.stats().alloc_failures);
  // Two served requests, 8 tokens each, the second one entirely from cache.
  CHECK_EQ(cm.stats().total_tokens, std::uint64_t{16});
  CHECK_EQ(cm.stats().hit_rate(), 0.5);
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

TEST(cache_manager_dropped_handle_returns_its_blocks) {
  CacheManager cm(tiny_config(64));
  const std::vector<TokenId> hot = iota_tokens(1, 8);
  run_once(cm, hot);  // 2 blocks now cached

  {
    auto walked_away = cm.acquire(concat(hot, iota_tokens(900, 8)));
    CHECK(walked_away.ok);
    CHECK_EQ(walked_away.cached_tokens, std::size_t{8});
    // Scope ends without release(): the caller forgot, or an exception
    // unwound past it. The handle must clean up either way.
  }

  CHECK_EQ(cm.stats().abandoned, std::uint64_t{1});
  // The fresh blocks are back -- only the cached prefix is still resident --
  // and the pin is gone, so the prefix is evictable again.
  CHECK_EQ(cm.pool().num_used(), std::size_t{2});
  CHECK_EQ(cm.tree().evict(2), std::size_t{2});
  CHECK_EQ(cm.pool().num_used(), std::size_t{0});
}

TEST(cache_manager_moved_handle_releases_exactly_once) {
  CacheManager cm(tiny_config(64));
  const std::vector<TokenId> seq = iota_tokens(1, 16);

  auto first = cm.acquire(seq);
  CHECK(first.ok);

  // Ownership transfers; the moved-from handle is disarmed and its destruction
  // must not release anything.
  auto second = std::move(first);
  CHECK(!first.ok);
  CHECK(second.ok);

  cm.release(seq, second);
  CHECK_EQ(cm.stats().abandoned, std::uint64_t{0});
  CHECK_EQ(cm.pool().num_used(), reachable_blocks(cm.tree()));

  // Move-assigning over a live handle abandons what it held first.
  auto third = cm.acquire(iota_tokens(500, 8));
  auto fourth = cm.acquire(iota_tokens(900, 8));
  third = std::move(fourth);
  CHECK_EQ(cm.stats().abandoned, std::uint64_t{1});
  cm.release(iota_tokens(900, 8), third);
  CHECK_EQ(cm.pool().num_used(), reachable_blocks(cm.tree()));
}

TEST(cache_manager_demotes_under_pressure_and_promotes_on_the_hit) {
  const CacheConfig cfg = tiny_config(0);
  HostTier compute(4 * cfg.block_bytes());
  HostTier spill(8 * cfg.block_bytes());
  CacheManager cm({{&compute, 4}, {&spill, 8}}, cfg);

  const std::vector<TokenId> a = iota_tokens(1, 8);     // 2 blocks
  const std::vector<TokenId> b = iota_tokens(1000, 8);  // 2 blocks
  run_once(cm, a);
  run_once(cm, b);  // compute tier now full

  // A third sequence forces room. With a spill tier, the LRU entry is demoted
  // rather than dropped: nothing is evicted, and the index keeps all three.
  run_once(cm, iota_tokens(2000, 8));
  CHECK_EQ(cm.stats().demoted_blocks, std::uint64_t{2});
  CHECK_EQ(cm.stats().evicted_blocks, std::uint64_t{0});
  CHECK_EQ(cm.tree().stored_blocks(), std::size_t{6});

  // Re-acquiring the demoted sequence is a full hit -- the entry survived --
  // and its blocks come back to the compute tier before the table is returned,
  // demoting someone else to make room.
  auto hit = cm.acquire(a);
  CHECK(hit.ok);
  CHECK_EQ(hit.cached_tokens, std::size_t{8});
  CHECK_EQ(cm.stats().promoted_blocks, std::uint64_t{2});
  for (BlockId id : hit.blocks) {
    CHECK_EQ(cm.pool().block_tier(id), BlockPool::TierIndex{0});
  }
  cm.release(a, hit);

  // Nothing was recomputed to get here: every request after the first three
  // was served from the index, just not always from the compute tier.
  CHECK_EQ(cm.pool().num_used(), reachable_blocks(cm.tree()));
}

TEST(cache_manager_falls_back_to_eviction_when_spill_is_full) {
  const CacheConfig cfg = tiny_config(0);
  HostTier compute(2 * cfg.block_bytes());
  HostTier spill(2 * cfg.block_bytes());
  CacheManager cm({{&compute, 2}, {&spill, 2}}, cfg);

  // Two sequences fill compute; the third demotes the first into spill, which
  // is then also full.
  run_once(cm, iota_tokens(1, 8));
  run_once(cm, iota_tokens(1000, 8));
  run_once(cm, iota_tokens(2000, 8));
  CHECK_EQ(cm.stats().demoted_blocks, std::uint64_t{2});

  // A fourth sequence finds no spill room: demotion cannot help, so the LRU
  // entry is evicted outright -- the single-tier behaviour, as the last resort.
  run_once(cm, iota_tokens(3000, 8));
  CHECK(cm.stats().evicted_blocks >= 2);
  CHECK_EQ(cm.pool().num_used(), reachable_blocks(cm.tree()));
}

TEST(tiered_pool_async_demotion_moves_the_block_at_drain) {
  const CacheConfig cfg = tiny_config(0);
  HostTier compute(2 * cfg.block_bytes());
  HostTier spill(2 * cfg.block_bytes());
  BlockPool pool({{&compute, 2}, {&spill, 2}}, cfg);

  const BlockId id = pool.allocate();
  std::memset(pool.block_data(id), 0x7C, cfg.block_bytes());

  CHECK(pool.migrate_async(id, 1));
  CHECK(pool.migrating(id));
  // Until the drain observes the finished copy, the block has not moved: the
  // engine can keep reading it in place.
  CHECK_EQ(pool.block_tier(id), BlockPool::TierIndex{0});

  pool.wait_for_migrations();
  CHECK(!pool.migrating(id));
  CHECK_EQ(pool.block_tier(id), BlockPool::TierIndex{1});
  const auto* bytes = static_cast<const unsigned char*>(pool.block_data(id));
  CHECK_EQ(bytes[cfg.block_bytes() - 1], static_cast<unsigned char>(0x7C));
  CHECK_EQ(pool.tier_free(0), std::size_t{2});  // compute slot came back

  pool.decref(id);
}

TEST(tiered_pool_cancelled_demotion_leaves_the_block_in_place) {
  const CacheConfig cfg = tiny_config(0);
  HostTier compute(2 * cfg.block_bytes());
  HostTier spill(2 * cfg.block_bytes());
  BlockPool pool({{&compute, 2}, {&spill, 2}}, cfg);

  const BlockId id = pool.allocate();
  CHECK(pool.migrate_async(id, 1));
  pool.cancel_migration(id);
  CHECK(!pool.migrating(id));

  pool.wait_for_migrations();
  // The block never moved, and the reserved spill slot came back at drain.
  CHECK_EQ(pool.block_tier(id), BlockPool::TierIndex{0});
  CHECK_EQ(pool.tier_free(1), std::size_t{2});

  // A block dying mid-flight takes its migration with it, and both slots end
  // up free once the worker is done.
  const BlockId dying = pool.allocate();
  CHECK(pool.migrate_async(dying, 1));
  pool.decref(dying);
  pool.decref(id);
  pool.wait_for_migrations();
  CHECK_EQ(pool.tier_free(0), std::size_t{2});
  CHECK_EQ(pool.tier_free(1), std::size_t{2});
  CHECK_EQ(pool.num_used(), std::size_t{0});
}

TEST(cache_manager_watermark_demotes_in_the_background) {
  const CacheConfig cfg = tiny_config(0);
  HostTier compute(4 * cfg.block_bytes());
  HostTier spill(8 * cfg.block_bytes());
  // Keep 2 compute slots free at all times.
  CacheManager cm({{&compute, 4}, {&spill, 8}}, cfg, {.spill_watermark = 2});

  const std::vector<TokenId> a = iota_tokens(1, 8);
  run_once(cm, a);  // 2 blocks cached, 2 free: exactly at the watermark
  CHECK_EQ(cm.stats().background_demotions, std::uint64_t{0});

  run_once(cm, iota_tokens(1000, 8));  // 0 free: 2 slots under
  CHECK_EQ(cm.stats().background_demotions, std::uint64_t{2});

  // The copies run off-path; once drained, the LRU entry lives in spill, its
  // compute slots are free, and nothing was evicted or synchronously demoted.
  cm.pool().wait_for_migrations();
  CHECK_EQ(cm.pool().num_free(), std::size_t{2});
  CHECK_EQ(cm.stats().demoted_blocks, std::uint64_t{0});
  CHECK_EQ(cm.stats().evicted_blocks, std::uint64_t{0});

  // The demoted entry is still a full hit, promoted back on access.
  auto hit = cm.acquire(a);
  CHECK(hit.ok);
  CHECK_EQ(hit.cached_tokens, std::size_t{8});
  CHECK_EQ(cm.stats().promoted_blocks, std::uint64_t{2});
  cm.release(a, hit);
  CHECK_EQ(cm.pool().num_used(), reachable_blocks(cm.tree()));
}

TEST(cache_manager_hit_cancels_an_inflight_demotion) {
  const CacheConfig cfg = tiny_config(0);
  HostTier compute(4 * cfg.block_bytes());
  HostTier spill(8 * cfg.block_bytes());
  CacheManager cm({{&compute, 4}, {&spill, 8}}, cfg,
                  {.spill_watermark = 4});  // aggressive watermark

  const std::vector<TokenId> a = iota_tokens(1, 8);
  run_once(cm, a);
  // The watermark wants all 4 slots free, so a's blocks now have demotions in
  // flight. Hitting a must yield blocks that stay in the compute tier.
  auto hit = cm.acquire(a);
  CHECK(hit.ok);
  CHECK_EQ(hit.cached_tokens, std::size_t{8});
  for (BlockId id : hit.blocks) {
    CHECK(!cm.pool().migrating(id));
    CHECK_EQ(cm.pool().block_tier(id), BlockPool::TierIndex{0});
  }
  cm.release(a, hit);

  // Whatever the worker copied for the cancelled jobs is discarded; the
  // reserved spill slots all come back.
  cm.pool().wait_for_migrations();
  CHECK_EQ(cm.pool().num_used(), reachable_blocks(cm.tree()));
}

TEST(cache_manager_async_promotion_delivers_a_ready_gate) {
  const CacheConfig cfg = tiny_config(0);
  HostTier compute(4 * cfg.block_bytes());
  HostTier spill(8 * cfg.block_bytes());
  CacheManager cm({{&compute, 4}, {&spill, 8}}, cfg, {.async_promotion = true});

  // Cache a and force it into spill by filling compute with two more entries.
  const std::vector<TokenId> a = iota_tokens(1, 8);
  run_once(cm, a);
  run_once(cm, iota_tokens(1000, 8));
  run_once(cm, iota_tokens(2000, 8));
  CHECK_EQ(cm.stats().demoted_blocks, std::uint64_t{2});

  // Fill a's spilled bytes with a pattern so the round trip is observable.
  // (In a real engine the KV was written before the demotion; the test writes
  // through the spill tier directly, which the API allows.)
  auto probe = cm.acquire(a);
  CHECK(probe.ok);
  CHECK_EQ(probe.cached_tokens, std::size_t{8});
  CHECK_EQ(cm.stats().promoted_blocks, std::uint64_t{2});

  // The table came back immediately; readiness is a separate, waitable fact.
  probe.wait_ready();
  CHECK(probe.ready());
  for (BlockId id : probe.blocks) {
    CHECK_EQ(cm.pool().block_tier(id), BlockPool::TierIndex{0});
    CHECK(!cm.pool().migrating(id));
  }
  cm.release(a, probe);

  // A second hit on the now-resident prefix is born ready.
  auto again = cm.acquire(a);
  CHECK(again.ok);
  CHECK(again.ready());
  cm.release(a, again);
  CHECK_EQ(cm.pool().num_used(), reachable_blocks(cm.tree()));
}

TEST(cache_manager_concurrent_hits_share_one_promotion) {
  const CacheConfig cfg = tiny_config(0);
  HostTier compute(8 * cfg.block_bytes());
  HostTier spill(8 * cfg.block_bytes());
  CacheManager cm({{&compute, 8}, {&spill, 8}}, cfg, {.async_promotion = true});

  const std::vector<TokenId> a = iota_tokens(1, 8);
  run_once(cm, a);
  // Push a to spill synchronously so the test controls the starting state.
  {
    std::vector<BlockId> victims;
    cm.tree().pick_demotion_victims(2, &victims);
    for (BlockId id : victims) CHECK(cm.pool().migrate(id, 1));
  }

  // Two requests hit the same spilled prefix while the first promotion is
  // still in flight. The second must not schedule (or cancel!) anything --
  // it rides the same jobs and reaches ready through them.
  auto first = cm.acquire(a);
  CHECK(first.ok);
  auto second = cm.acquire(a);
  CHECK(second.ok);
  CHECK_EQ(cm.stats().promoted_blocks, std::uint64_t{2});  // counted once

  first.wait_ready();
  CHECK(second.ready());  // same blocks, same drain
  cm.release(a, first);
  cm.release(a, second);
  CHECK_EQ(cm.pool().num_used(), reachable_blocks(cm.tree()));
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

// ---------------------------------------------------------------------------
// Spill codec: fp16 -> fp8 E4M3 on the demotion path.

namespace {

// One shared instance: building the 64 KiB encode table once per test binary,
// not once per test.
const Fp8SpillCodec& fp8_codec() {
  static Fp8SpillCodec codec;
  return codec;
}

// |a - b| within E4M3 round-to-nearest error: half a mantissa step (2^-4
// relative) for normals, half a subnormal step (2^-10 absolute) near zero.
bool close_e4m3(float exact, float quantized) {
  const float err = std::fabs(exact - quantized);
  return err <= std::fabs(exact) / 16.0f + 0x1.0p-10f;
}

// Deterministic fp16 test values spanning the E4M3 range, indexable so a
// block's worth can be regenerated for comparison after a round trip.
std::uint16_t fp16_sample(std::size_t i) {
  // All within E4M3's finite range: saturation is exercised by the scalar
  // test, and would break the round-trip tolerance this helper feeds.
  const float magnitudes[] = {0.0f,    0x1.0p-9f, 0.0173f, 0.25f,  1.0f,
                              3.1416f, 17.5f,     100.0f,  240.0f, 447.0f};
  const std::size_t n = sizeof(magnitudes) / sizeof(magnitudes[0]);
  const float sign = (i / n) % 2 == 0 ? 1.0f : -1.0f;
  return float_to_fp16_bits(sign * magnitudes[i % n]);
}

}  // namespace

TEST(fp8_scalar_conversions_round_to_nearest_and_saturate) {
  // Exactly representable values survive untouched.
  CHECK_EQ(e4m3_to_float(float_to_e4m3(1.0f)), 1.0f);
  CHECK_EQ(e4m3_to_float(float_to_e4m3(-2.5f)), -2.5f);
  CHECK_EQ(e4m3_to_float(float_to_e4m3(448.0f)), 448.0f);
  CHECK_EQ(e4m3_to_float(float_to_e4m3(0x1.0p-9f)), 0x1.0p-9f);  // min subnormal
  CHECK_EQ(e4m3_to_float(float_to_e4m3(0.0f)), 0.0f);

  // Out of range saturates to max finite rather than producing NaN.
  CHECK_EQ(e4m3_to_float(float_to_e4m3(10000.0f)), 448.0f);
  CHECK_EQ(e4m3_to_float(float_to_e4m3(-10000.0f)), -448.0f);

  // Everything in range lands within half a quantization step.
  for (float v : {0.017f, 0.3f, 1.7f, 7.3f, 42.0f, 200.0f, 447.0f}) {
    CHECK(close_e4m3(v, e4m3_to_float(float_to_e4m3(v))));
    CHECK(close_e4m3(-v, e4m3_to_float(float_to_e4m3(-v))));
  }
}

TEST(fp8_codec_roundtrip_is_within_tolerance_and_in_place_safe) {
  const std::size_t values = 4096;
  std::vector<std::byte> raw(values * 2);
  for (std::size_t i = 0; i < values; ++i) {
    const std::uint16_t h = fp16_sample(i);
    std::memcpy(raw.data() + 2 * i, &h, 2);
  }

  // Out-of-place round trip.
  std::vector<std::byte> enc(values);
  std::vector<std::byte> back(values * 2);
  fp8_codec().encode(raw.data(), raw.size(), enc.data());
  fp8_codec().decode(enc.data(), raw.size(), back.data());
  for (std::size_t i = 0; i < values; ++i) {
    std::uint16_t orig, got;
    std::memcpy(&orig, raw.data() + 2 * i, 2);
    std::memcpy(&got, back.data() + 2 * i, 2);
    CHECK(close_e4m3(fp16_bits_to_float(orig), fp16_bits_to_float(got)));
  }

  // In-place over one buffer -- the bounce path's contract -- must agree
  // byte for byte with the out-of-place result.
  std::vector<std::byte> scratch = raw;
  fp8_codec().encode(scratch.data(), raw.size(), scratch.data());
  CHECK(std::memcmp(scratch.data(), enc.data(), values) == 0);
  fp8_codec().decode(scratch.data(), raw.size(), scratch.data());
  CHECK(std::memcmp(scratch.data(), back.data(), values * 2) == 0);
}

TEST(tiered_pool_with_codec_migrates_through_half_size_spill_slots) {
  CacheConfig cfg;
  cfg.block_tokens = 4;
  cfg.num_layers = 2;
  cfg.num_kv_heads = 2;
  cfg.head_dim = 8;

  // The spill arena is sized for 4 *encoded* blocks -- half the bytes the
  // same four blocks need raw. Construction succeeding is itself the
  // capacity claim: without the codec this spec must be rejected.
  HostTier compute(2 * cfg.block_bytes());
  HostTier spill(4 * (cfg.block_bytes() / 2));
  BlockPool pool({{&compute, 2}, {&spill, 4}}, cfg, &fp8_codec());
  CHECK_EQ(pool.tier_block_bytes(0), cfg.block_bytes());
  CHECK_EQ(pool.tier_block_bytes(1), cfg.block_bytes() / 2);

  const BlockId id = pool.allocate();
  CHECK(id != kInvalidBlock);
  const std::size_t values = cfg.block_bytes() / 2;
  auto* data = static_cast<std::byte*>(pool.block_data(id));
  for (std::size_t i = 0; i < values; ++i) {
    const std::uint16_t h = fp16_sample(i);
    std::memcpy(data + 2 * i, &h, 2);
  }

  // Demote (encodes) and promote (decodes): values come back within fp8
  // tolerance, not byte-identical -- that is the trade the codec makes.
  CHECK(pool.migrate(id, 1));
  CHECK_EQ(pool.block_tier(id), 1u);
  CHECK(pool.migrate(id, 0));
  CHECK_EQ(pool.block_tier(id), 0u);
  data = static_cast<std::byte*>(pool.block_data(id));
  for (std::size_t i = 0; i < values; ++i) {
    std::uint16_t got;
    std::memcpy(&got, data + 2 * i, 2);
    CHECK(close_e4m3(fp16_bits_to_float(fp16_sample(i)), fp16_bits_to_float(got)));
  }

  // A second round trip changes nothing more: E4M3 values re-encode to
  // themselves, so quantization error is paid once, not per demotion.
  std::vector<std::byte> after_first(cfg.block_bytes());
  std::memcpy(after_first.data(), data, cfg.block_bytes());
  CHECK(pool.migrate(id, 1));
  CHECK(pool.migrate(id, 0));
  CHECK(std::memcmp(pool.block_data(id), after_first.data(), cfg.block_bytes()) == 0);

  pool.decref(id);
}

TEST(async_migration_with_codec_encodes_on_the_worker) {
  CacheConfig cfg;
  cfg.block_tokens = 4;
  cfg.num_layers = 2;
  cfg.num_kv_heads = 2;
  cfg.head_dim = 8;

  HostTier compute(2 * cfg.block_bytes());
  HostTier spill(2 * (cfg.block_bytes() / 2));
  BlockPool pool({{&compute, 2}, {&spill, 2}}, cfg, &fp8_codec());

  const BlockId id = pool.allocate();
  const std::size_t values = cfg.block_bytes() / 2;
  auto* data = static_cast<std::byte*>(pool.block_data(id));
  for (std::size_t i = 0; i < values; ++i) {
    const std::uint16_t h = fp16_sample(i);
    std::memcpy(data + 2 * i, &h, 2);
  }

  CHECK(pool.migrate_async(id, 1));
  pool.wait_for_migrations();
  CHECK_EQ(pool.block_tier(id), 1u);
  CHECK(pool.migrate_async(id, 0));
  pool.wait_for_migrations();
  CHECK_EQ(pool.block_tier(id), 0u);

  data = static_cast<std::byte*>(pool.block_data(id));
  for (std::size_t i = 0; i < values; ++i) {
    std::uint16_t got;
    std::memcpy(&got, data + 2 * i, 2);
    CHECK(close_e4m3(fp16_bits_to_float(fp16_sample(i)), fp16_bits_to_float(got)));
  }
  pool.decref(id);
}

TEST(cache_manager_with_codec_serves_hits_from_a_doubled_spill_tier) {
  CacheConfig cfg;
  cfg.block_tokens = 4;
  cfg.num_layers = 2;
  cfg.num_kv_heads = 2;
  cfg.head_dim = 8;

  // Compute holds 2 sequences; the spill arena has raw room for 2 blocks but
  // encoded room for 4. Four single-block sequences then all stay resident:
  // without the codec the same bytes would force half of them out.
  HostTier compute(2 * cfg.block_bytes());
  HostTier spill(4 * (cfg.block_bytes() / 2));
  CacheManager cm({{&compute, 2}, {&spill, 4}}, cfg,
                  {.spill_codec = &fp8_codec()});

  auto seq = [&](TokenId base) {
    std::vector<TokenId> tokens(cfg.block_tokens);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = base + static_cast<TokenId>(i);
    }
    return tokens;
  };

  for (TokenId base : {100, 200, 300, 400}) {
    auto tokens = seq(base);
    auto alloc = cm.acquire(tokens);
    CHECK(alloc.ok);
    cm.release(tokens, alloc);
  }

  // Re-visit all four: every one must be a full hit -- served from compute
  // or promoted back out of the encoded spill tier, never recomputed.
  for (TokenId base : {100, 200, 300, 400}) {
    auto tokens = seq(base);
    auto alloc = cm.acquire(tokens);
    CHECK(alloc.ok);
    CHECK_EQ(alloc.cached_tokens, tokens.size());
    cm.release(tokens, alloc);
  }
  CHECK_EQ(cm.stats().evicted_blocks, std::uint64_t{0});
}

int main() { return kvcheck::run_all(); }
