#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace kvslab {

using TokenId = std::int32_t;
using BlockId = std::uint32_t;

inline constexpr BlockId kInvalidBlock = std::numeric_limits<BlockId>::max();

// Geometry of a KV cache pool, in the terms an attention kernel cares about.
// Every block holds `block_tokens` tokens' worth of K and V for every layer,
// which is what makes a block table handed to the kernel a pure index list.
struct CacheConfig {
  std::size_t num_blocks = 1024;
  std::size_t block_tokens = 16;
  std::size_t num_layers = 32;
  std::size_t num_kv_heads = 8;
  std::size_t head_dim = 128;
  std::size_t dtype_bytes = 2;  // fp16 / bf16

  // K and V, across all layers, for a single token.
  std::size_t bytes_per_token() const {
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes;
  }
  std::size_t block_bytes() const { return block_tokens * bytes_per_token(); }
  std::size_t total_bytes() const { return num_blocks * block_bytes(); }
};

// FNV-1a over one block's worth of token ids. The prefix tree keys its children
// by whole blocks rather than single tokens: the tree is block aligned, so a
// divergence *inside* a block has to produce two distinct children, and a
// per-token key cannot express that.
//
// Equality of the key is never trusted on its own -- a match is confirmed by
// comparing the tokens themselves, and an insert whose key is already taken by
// a different block is refused rather than overwriting it. A collision
// therefore costs a cache entry, never correctness.
inline std::uint64_t block_key(const TokenId* tokens, std::size_t n) {
  std::uint64_t h = 1469598103934665603ull;
  for (std::size_t i = 0; i < n; ++i) {
    auto v = static_cast<std::uint32_t>(tokens[i]);
    for (int byte = 0; byte < 4; ++byte) {
      h ^= (v >> (byte * 8)) & 0xffu;
      h *= 1099511628211ull;
    }
  }
  return h;
}

}  // namespace kvslab
