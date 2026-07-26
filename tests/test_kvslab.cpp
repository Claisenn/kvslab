#include <numeric>
#include <vector>

#include "check.hpp"
#include "kvslab/block_pool.hpp"
#include "kvslab/radix_tree.hpp"

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

}  // namespace

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
  CHECK_EQ(pool.num_used(), tree.stored_blocks());
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

int main() { return kvcheck::run_all(); }
