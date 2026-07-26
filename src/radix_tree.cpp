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
  // Two things have to happen here. Every cached block goes back to the pool,
  // or the pool outlives the tree still holding references to blocks nobody can
  // reach. And the nodes come apart iteratively: letting the unique_ptr chain
  // unwind on its own recurses once per level, and the tree is a chain, not a
  // bush, whenever a conversation keeps extending the same prefix -- exactly
  // the shape a long multi-turn session produces.
  std::vector<std::unique_ptr<Node>> pending;
  pending.push_back(std::move(root_));

  while (!pending.empty()) {
    std::unique_ptr<Node> node = std::move(pending.back());
    pending.pop_back();
    if (node == nullptr) continue;

    release_blocks(node.get());
    for (auto& [key, child] : node->children) {
      if (child != nullptr) pending.push_back(std::move(child));
    }
    node->children.clear();
    // `node` dies here with no children left to recurse into.
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

  const std::uint64_t key = block_key(tokens.data() + pos, block_tokens_);
  if (cur->children.find(key) != cur->children.end()) {
    // The walk above stopped without consuming this child, so the slot is held
    // by a block whose tokens differ from ours: a hash collision.
    //
    // Storing anyway would overwrite the existing child. That subtree's blocks
    // would lose their only owner and could never return to the free list, the
    // node counters would drift, and any pin reaching into it would dangle.
    // Refusing costs this sequence its cache entry, which is the right trade
    // against corrupting the index.
    ++collisions_;
    return 0;
  }

  auto leaf = std::make_unique<Node>();
  leaf->tokens.assign(tokens.begin() + pos, tokens.begin() + limit);
  leaf->blocks.assign(blocks.begin() + pos / block_tokens_,
                      blocks.begin() + limit / block_tokens_);
  leaf->parent = cur;
  touch(leaf.get());
  for (BlockId id : leaf->blocks) pool_.incref(id);

  const std::size_t adopted = leaf->blocks.size();
  cur->children.emplace(key, std::move(leaf));
  stored_blocks_ += adopted;
  ++num_nodes_;
  return adopted;
}

// True when the tree holds the only reference to every block in the node, so
// dropping it actually puts them back on the free list.
bool RadixTree::tree_is_sole_owner(const Node* node) const {
  for (BlockId id : node->blocks) {
    if (pool_.refcount(id) != 1) return false;
  }
  return true;
}

std::size_t RadixTree::drop_leaf(Node* victim) {
  std::size_t freed = 0;
  for (BlockId id : victim->blocks) {
    if (pool_.decref(id)) ++freed;
  }
  stored_blocks_ -= victim->blocks.size();
  --num_nodes_;

  Node* parent = victim->parent;
  const std::uint64_t key = block_key(victim->tokens.data(), block_tokens_);
  parent->children.erase(key);  // victim dies here
  return freed;
}

// Least-recently-used unpinned leaf, ignoring any node already rejected by the
// caller. Interior nodes are skipped because their blocks are the shared head
// of a longer cached sequence; pinned nodes because an in-flight request is
// reading them.
//
// This is a full walk per victim -- O(nodes) where an intrusive LRU list would
// be O(1). Correct first; see the roadmap. `rejected` is scanned linearly, but
// it is empty on every ordinary eviction.
RadixTree::Node* RadixTree::find_lru_leaf(const std::vector<const Node*>& rejected) {
  Node* best = nullptr;
  std::vector<Node*> stack{root_.get()};
  while (!stack.empty()) {
    Node* node = stack.back();
    stack.pop_back();
    for (auto& [key, child] : node->children) stack.push_back(child.get());

    if (node == root_.get()) continue;
    if (!node->children.empty()) continue;
    if (node->lock_count > 0) continue;
    if (std::find(rejected.begin(), rejected.end(), node) != rejected.end()) continue;
    if (best == nullptr || node->last_access < best->last_access) best = node;
  }
  return best;
}

std::size_t RadixTree::evict(std::size_t num_blocks) {
  std::size_t freed = 0;
  // Ownership is checked on the chosen victim rather than on every candidate
  // during the walk: the check costs one refcount lookup per block, so folding
  // it into the scan would turn an O(nodes) walk into an O(blocks) one for a
  // condition that virtually never holds.
  std::vector<const Node*> rejected;

  while (freed < num_blocks) {
    Node* victim = find_lru_leaf(rejected);
    if (victim == nullptr) break;

    if (!tree_is_sole_owner(victim)) {
      // Dropping this would free nothing and orphan its blocks -- the tree is
      // the only thing that could ever hand them back. Leave it and look
      // further down the LRU order.
      rejected.push_back(victim);
      continue;
    }

    const std::size_t reclaimed = drop_leaf(victim);
    // A node always holds at least one block, and the check above proved the
    // tree owns them all. Without this the loop could spin.
    assert(reclaimed > 0 && "a sole-owner leaf freed nothing");
    freed += reclaimed;
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
