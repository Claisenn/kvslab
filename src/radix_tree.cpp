#include "kvslab/radix_tree.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

#include "kvslab/block_pool.hpp"

namespace kvslab {

RadixTree::RadixTree(BlockPool& pool, std::size_t block_tokens)
    : pool_(pool), block_tokens_(block_tokens), root_(std::make_unique<Node>()) {
  assert(block_tokens_ > 0);
}

RadixTree::~RadixTree() {
  // Hand every cached block back before the nodes go away, otherwise the pool
  // outlives the tree holding references to blocks nobody can reach.
  std::vector<Node*> stack{root_.get()};
  while (!stack.empty()) {
    Node* node = stack.back();
    stack.pop_back();
    release_blocks(node);
    for (auto& [key, child] : node->children) stack.push_back(child.get());
  }
}

void RadixTree::release_blocks(Node* node) {
  for (BlockId id : node->blocks) pool_.decref(id);
  node->blocks.clear();
}

// Splits `node` at `offset` tokens by interposing a *new parent* covering
// [0, offset) and leaving `node` itself covering the tail.
//
// The direction matters: callers hold Node* pins into the tree, and keeping the
// original object as the deeper half means an existing pin still covers every
// token it covered before the split. Interposing a parent, rather than a child,
// is what makes those pointers stable.
RadixTree::Node* RadixTree::split_node(Node* node, std::size_t offset) {
  assert(offset > 0 && offset < node->tokens.size());
  assert(offset % block_tokens_ == 0);

  Node* parent = node->parent;
  const std::uint64_t head_key = block_key(node->tokens.data(), block_tokens_);
  const std::size_t head_blocks = offset / block_tokens_;

  // Take ownership away from the parent map before mutating `node`.
  auto owned = std::move(parent->children[head_key]);

  auto head = std::make_unique<Node>();
  head->tokens.assign(node->tokens.begin(), node->tokens.begin() + offset);
  head->blocks.assign(node->blocks.begin(), node->blocks.begin() + head_blocks);
  head->parent = parent;
  head->last_access = node->last_access;
  // Every pin that reaches `node` necessarily passes through its new parent.
  head->lock_count = node->lock_count;

  node->tokens.erase(node->tokens.begin(), node->tokens.begin() + offset);
  node->blocks.erase(node->blocks.begin(), node->blocks.begin() + head_blocks);
  node->parent = head.get();

  const std::uint64_t tail_key = block_key(node->tokens.data(), block_tokens_);
  head->children[tail_key] = std::move(owned);

  Node* head_raw = head.get();
  parent->children[head_key] = std::move(head);
  ++num_nodes_;
  return head_raw;
}

RadixTree::MatchResult RadixTree::match_prefix(const std::vector<TokenId>& tokens) {
  MatchResult result;

  // Only whole blocks are reusable: a half-filled block's KV depends on tokens
  // the incoming request has not committed to yet.
  const std::size_t limit = (tokens.size() / block_tokens_) * block_tokens_;

  Node* cur = root_.get();
  std::size_t pos = 0;
  while (pos < limit) {
    auto it = cur->children.find(block_key(tokens.data() + pos, block_tokens_));
    if (it == cur->children.end()) break;

    Node* child = it->second.get();
    const std::size_t max_blocks =
        std::min(child->blocks.size(), (limit - pos) / block_tokens_);

    std::size_t common = 0;  // matching blocks
    while (common < max_blocks &&
           std::equal(child->tokens.begin() + common * block_tokens_,
                      child->tokens.begin() + (common + 1) * block_tokens_,
                      tokens.begin() + pos + common * block_tokens_)) {
      ++common;
    }
    if (common == 0) break;  // hash collision; treat as a miss

    result.blocks.insert(result.blocks.end(), child->blocks.begin(),
                         child->blocks.begin() + common);
    pos += common * block_tokens_;

    if (common < child->blocks.size()) {
      // Match ends inside this node. Split so the returned node covers the
      // match exactly -- pinning it must not pin the divergent tail.
      cur = split_node(child, common * block_tokens_);
      touch(cur);
      result.node = cur;
      break;
    }

    touch(child);
    cur = child;
    result.node = child;
  }

  result.num_tokens = pos;
  return result;
}

std::size_t RadixTree::insert(const std::vector<TokenId>& tokens,
                              const std::vector<BlockId>& blocks) {
  const std::size_t limit =
      std::min((tokens.size() / block_tokens_) * block_tokens_,
               blocks.size() * block_tokens_);

  Node* cur = root_.get();
  std::size_t pos = 0;
  while (pos < limit) {
    auto it = cur->children.find(block_key(tokens.data() + pos, block_tokens_));
    if (it == cur->children.end()) break;

    Node* child = it->second.get();
    const std::size_t max_blocks =
        std::min(child->blocks.size(), (limit - pos) / block_tokens_);

    std::size_t common = 0;
    while (common < max_blocks &&
           std::equal(child->tokens.begin() + common * block_tokens_,
                      child->tokens.begin() + (common + 1) * block_tokens_,
                      tokens.begin() + pos + common * block_tokens_)) {
      ++common;
    }
    if (common == 0) break;

    pos += common * block_tokens_;
    if (common < child->blocks.size()) {
      // Diverges inside `child`: split, then hang the remainder off the head.
      cur = split_node(child, common * block_tokens_);
      touch(cur);
      break;
    }
    touch(child);
    cur = child;
  }

  if (pos >= limit) {
    touch(cur);
    return 0;  // fully cached already; caller's blocks are redundant
  }

  auto leaf = std::make_unique<Node>();
  leaf->tokens.assign(tokens.begin() + pos, tokens.begin() + limit);
  leaf->blocks.assign(blocks.begin() + pos / block_tokens_,
                      blocks.begin() + limit / block_tokens_);
  leaf->parent = cur;
  touch(leaf.get());
  for (BlockId id : leaf->blocks) pool_.incref(id);

  const std::size_t adopted = leaf->blocks.size();
  const std::uint64_t key = block_key(leaf->tokens.data(), block_tokens_);
  cur->children[key] = std::move(leaf);
  stored_blocks_ += adopted;
  ++num_nodes_;
  return adopted;
}

// Least-recently-used unpinned leaf, or nullptr when everything is pinned.
//
// This is a full walk per eviction -- O(nodes) where an intrusive LRU list
// would be O(1). Correct first, and the profile says eviction is nowhere near
// the hot path at these pool sizes; see the roadmap.
RadixTree::Node* RadixTree::find_lru_leaf() {
  Node* best = nullptr;
  std::vector<Node*> stack{root_.get()};
  while (!stack.empty()) {
    Node* node = stack.back();
    stack.pop_back();
    for (auto& [key, child] : node->children) stack.push_back(child.get());

    if (node == root_.get()) continue;
    if (!node->children.empty()) continue;  // interior nodes hold live prefixes
    if (node->lock_count > 0) continue;
    if (best == nullptr || node->last_access < best->last_access) best = node;
  }
  return best;
}

std::size_t RadixTree::evict(std::size_t num_blocks) {
  std::size_t freed = 0;
  while (freed < num_blocks) {
    Node* victim = find_lru_leaf();
    if (victim == nullptr) break;  // everything left is pinned or interior

    for (BlockId id : victim->blocks) {
      if (pool_.decref(id)) ++freed;
    }
    stored_blocks_ -= victim->blocks.size();
    --num_nodes_;

    Node* parent = victim->parent;
    const std::uint64_t key = block_key(victim->tokens.data(), block_tokens_);
    parent->children.erase(key);  // victim dies here
  }
  return freed;
}

void RadixTree::lock(Node* node) {
  for (Node* n = node; n != nullptr && n != root_.get(); n = n->parent) {
    ++n->lock_count;
  }
}

void RadixTree::unlock(Node* node) {
  for (Node* n = node; n != nullptr && n != root_.get(); n = n->parent) {
    assert(n->lock_count > 0 && "unbalanced unlock");
    --n->lock_count;
  }
}

}  // namespace kvslab
