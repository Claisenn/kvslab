#include <vector>

#include "check.hpp"
#include "kvslab/block_pool.hpp"

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

int main() { return kvcheck::run_all(); }
