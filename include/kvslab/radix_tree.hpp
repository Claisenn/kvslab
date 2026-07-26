#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "kvslab/types.hpp"

namespace kvslab {

class BlockPool;

// Block-aligned radix tree over token sequences: the prefix cache index.
//
// Each node owns a run of tokens whose length is a multiple of `block_tokens`,
// plus the blocks holding that run's KV. Splits therefore only ever land on
// block boundaries, which is exactly what lets a matched prefix be handed to an
// attention kernel as a block table with zero copying -- the whole point of
// prefix caching.
//
// Children are keyed by the hash of their first *block* of tokens, not by a
// single token: with block granularity, two sequences that diverge in the
// middle of a block must live in separate children, and a per-token key cannot
// represent that.
class RadixTree {
 public:
  struct Node {
    std::vector<TokenId> tokens;
    std::vector<BlockId> blocks;  // tokens.size() / block_tokens entries
    std::unordered_map<std::uint64_t, std::unique_ptr<Node>> children;
    Node* parent = nullptr;
    // Number of in-flight sequences whose matched path passes through this
    // node. Non-zero means "pinned": eviction must not touch it.
    std::uint32_t lock_count = 0;
    std::uint64_t last_access = 0;  // logical clock, for LRU
  };

  struct MatchResult {
    std::size_t num_tokens = 0;
    std::vector<BlockId> blocks;
    Node* node = nullptr;  // deepest matched node; exactly covers num_tokens
  };

  RadixTree(BlockPool& pool, std::size_t block_tokens);
  ~RadixTree();

  RadixTree(const RadixTree&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;

  // Longest block-aligned prefix of `tokens` already in the tree. Splits the
  // deepest node when the match ends inside it, so the returned node covers the
  // match and nothing more -- pinning it pins precisely the reused blocks.
  MatchResult match_prefix(const std::vector<TokenId>& tokens);

  // Stores whatever block-aligned suffix of `tokens` is not present yet, taking
  // a reference on each block it adopts. Blocks whose prefix is already cached
  // are left alone: the caller's copies are redundant and it should drop them.
  // Returns the number of blocks newly adopted.
  std::size_t insert(const std::vector<TokenId>& tokens,
                     const std::vector<BlockId>& blocks);

  // Frees up to `num_blocks` blocks by dropping least-recently-used unpinned
  // leaves. Returns how many blocks actually made it back to the free list.
  std::size_t evict(std::size_t num_blocks);

  // Pin/unpin `node` and every ancestor, so a sequence reading a cached prefix
  // cannot have it evicted out from under it mid-flight.
  void lock(Node* node);
  void unlock(Node* node);

  std::size_t stored_blocks() const { return stored_blocks_; }
  std::size_t num_nodes() const { return num_nodes_; }
  // Inserts refused because another block already occupied the hash slot.
  // Expected to stay at zero; a non-zero value means the index is silently
  // losing cache entries and the key width deserves a second look.
  std::size_t hash_collisions() const { return collisions_; }
  const Node* root() const { return root_.get(); }

 private:
  Node* split_node(Node* node, std::size_t offset);
  Node* find_lru_leaf(const std::vector<const Node*>& rejected);
  bool tree_is_sole_owner(const Node* node) const;
  std::size_t drop_leaf(Node* victim);
  void touch(Node* node) { node->last_access = ++clock_; }
  void release_blocks(Node* node);

  BlockPool& pool_;
  std::size_t block_tokens_;
  std::unique_ptr<Node> root_;
  std::uint64_t clock_ = 0;
  std::size_t stored_blocks_ = 0;
  std::size_t num_nodes_ = 0;
  std::size_t collisions_ = 0;
};

}  // namespace kvslab
