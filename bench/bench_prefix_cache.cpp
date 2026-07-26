// Prefix-cache microbenchmark.
//
// Measures the block-management path only: prefix match, block allocation,
// eviction, index insert. No KV is actually written, so the numbers are a
// ceiling on what the manager itself can sustain, not end-to-end serving
// throughput. That is the point -- this layer should never be the bottleneck,
// and the benchmark exists to notice when it becomes one.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "kvslab/cache_manager.hpp"

using namespace kvslab;

namespace {

// Deterministic, so two runs are comparable and a regression is a regression.
class Lcg {
 public:
  explicit Lcg(std::uint64_t seed) : state_(seed) {}
  std::uint32_t next() {
    state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(state_ >> 33);
  }
  TokenId token() { return static_cast<TokenId>(next() % 32000); }

 private:
  std::uint64_t state_;
};

std::vector<TokenId> random_tokens(Lcg& rng, std::size_t count) {
  std::vector<TokenId> tokens;
  tokens.reserve(count);
  for (std::size_t i = 0; i < count; ++i) tokens.push_back(rng.token());
  return tokens;
}

struct Result {
  double seconds;
  double hit_rate;
  std::uint64_t requests;
  std::uint64_t tokens;
  std::uint64_t evicted;
  std::uint64_t failures;
};

Result run(const CacheConfig& cfg, const std::vector<std::vector<TokenId>>& workload) {
  CacheManager cm(cfg);
  const auto start = std::chrono::steady_clock::now();
  for (const std::vector<TokenId>& tokens : workload) {
    auto alloc = cm.acquire(tokens);
    if (!alloc.ok) continue;  // pool oversubscribed; a scheduler would requeue
    cm.release(tokens, alloc);
  }
  const auto end = std::chrono::steady_clock::now();

  const CacheManager::Stats& s = cm.stats();
  return Result{std::chrono::duration<double>(end - start).count(),
                s.hit_rate(),
                s.requests,
                s.total_tokens,
                s.evicted_blocks,
                s.alloc_failures};
}

// Every request is unique: nothing to reuse. This is the floor.
std::vector<std::vector<TokenId>> workload_cold(std::size_t requests, std::size_t len) {
  Lcg rng(1);
  std::vector<std::vector<TokenId>> out;
  out.reserve(requests);
  for (std::size_t i = 0; i < requests; ++i) out.push_back(random_tokens(rng, len));
  return out;
}

// A long shared system prompt plus a short unique user turn -- the shape most
// production traffic actually has, and what prefix caching exists for.
std::vector<std::vector<TokenId>> workload_shared_prefix(std::size_t requests,
                                                         std::size_t prefix_len,
                                                         std::size_t suffix_len) {
  Lcg rng(2);
  const std::vector<TokenId> prefix = random_tokens(rng, prefix_len);
  std::vector<std::vector<TokenId>> out;
  out.reserve(requests);
  for (std::size_t i = 0; i < requests; ++i) {
    std::vector<TokenId> tokens = prefix;
    const std::vector<TokenId> suffix = random_tokens(rng, suffix_len);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    out.push_back(std::move(tokens));
  }
  return out;
}

// Multi-turn chat: each turn re-sends the whole history, so turn N should hit
// almost everything turn N-1 wrote.
std::vector<std::vector<TokenId>> workload_multi_turn(std::size_t conversations,
                                                      std::size_t turns,
                                                      std::size_t turn_len) {
  Lcg rng(3);
  std::vector<std::vector<TokenId>> out;
  out.reserve(conversations * turns);
  for (std::size_t c = 0; c < conversations; ++c) {
    std::vector<TokenId> history;
    for (std::size_t t = 0; t < turns; ++t) {
      const std::vector<TokenId> turn = random_tokens(rng, turn_len);
      history.insert(history.end(), turn.begin(), turn.end());
      out.push_back(history);
    }
  }
  return out;
}

void print_row(const std::string& name, const Result& r) {
  const double reqs_per_sec = r.seconds > 0 ? static_cast<double>(r.requests) / r.seconds : 0.0;
  const double us_per_req =
      r.requests > 0 ? r.seconds * 1e6 / static_cast<double>(r.requests) : 0.0;
  std::printf("%-22s %10.1f %12.1f %10.1f%% %11llu %9llu\n", name.c_str(), reqs_per_sec,
              us_per_req, r.hit_rate * 100.0,
              static_cast<unsigned long long>(r.evicted),
              static_cast<unsigned long long>(r.failures));
}

}  // namespace

int main() {
  CacheConfig cfg;
  cfg.block_tokens = 16;
  cfg.num_layers = 32;
  cfg.num_kv_heads = 8;
  cfg.head_dim = 128;
  cfg.dtype_bytes = 2;
  cfg.num_blocks = 8192;  // 16 GiB of KV at this geometry

  std::printf("kvslab prefix cache benchmark\n");
  std::printf("  block_tokens=%zu  blocks=%zu  block=%.1f KiB  pool=%.2f GiB\n\n",
              cfg.block_tokens, cfg.num_blocks,
              static_cast<double>(cfg.block_bytes()) / 1024.0,
              static_cast<double>(cfg.total_bytes()) / (1024.0 * 1024.0 * 1024.0));

  std::printf("%-22s %10s %12s %11s %11s %9s\n", "workload", "req/s", "us/req", "hit rate",
              "evicted", "failed");
  std::printf("%s\n", std::string(80, '-').c_str());

  print_row("cold (no sharing)", run(cfg, workload_cold(20000, 512)));
  print_row("shared prefix 1k+64", run(cfg, workload_shared_prefix(20000, 1024, 64)));
  print_row("multi-turn 8x256", run(cfg, workload_multi_turn(2000, 8, 256)));

  return 0;
}
