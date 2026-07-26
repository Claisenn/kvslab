#include "kvslab/block_pool.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace kvslab {
namespace {

// Refcount and range violations are not crashes, they are silent corruption: a
// block handed back twice lands on the free list while the index still points
// at it, and the next allocation gives live KV to a second sequence. Nothing
// downstream can detect that, so these checks do not ride on assert() -- they
// stay in every build, at the cost of a branch that always predicts.
[[noreturn]] void fatal(const char* what, BlockId id) {
  std::fprintf(stderr, "kvslab: fatal: %s (block %u)\n", what, id);
  std::abort();
}

}  // namespace

BlockPool::BlockPool(const CacheConfig& cfg)
    : cfg_(cfg), owned_tier_(std::make_unique<HostTier>(cfg.total_bytes())) {
  init({{owned_tier_.get(), cfg.num_blocks}});
}

BlockPool::BlockPool(Tier& tier, const CacheConfig& cfg) : cfg_(cfg) {
  init({{&tier, cfg.num_blocks}});
}

BlockPool::BlockPool(const std::vector<TierSpec>& specs, const CacheConfig& cfg)
    : cfg_(cfg) {
  init(specs);
}

void BlockPool::init(const std::vector<TierSpec>& specs) {
  if (specs.empty()) {
    throw std::invalid_argument("kvslab: a pool needs at least one tier");
  }

  std::size_t total_blocks = 0;
  tiers_.reserve(specs.size());
  for (const TierSpec& spec : specs) {
    if (spec.tier == nullptr) {
      throw std::invalid_argument("kvslab: null tier in pool spec");
    }
    if (spec.tier->capacity_bytes() < spec.num_blocks * cfg_.block_bytes()) {
      throw std::invalid_argument("kvslab: tier is too small for the requested pool");
    }

    TierState state;
    state.tier = spec.tier;
    state.num_blocks = spec.num_blocks;
    // Descending, so the first allocations hand out low slots. Purely a
    // debugging nicety: block tables read in allocation order.
    state.free_slots.reserve(spec.num_blocks);
    for (std::size_t i = spec.num_blocks; i > 0; --i) {
      state.free_slots.push_back(static_cast<std::uint32_t>(i - 1));
    }
    tiers_.push_back(std::move(state));
    total_blocks += spec.num_blocks;
  }

  // One id per slot in the whole pool: every block could in principle be
  // resident somewhere at once, and ids are cheap.
  location_.assign(total_blocks, Location{});
  refcount_.assign(total_blocks, 0);
  free_ids_.reserve(total_blocks);
  for (std::size_t i = total_blocks; i > 0; --i) {
    free_ids_.push_back(static_cast<BlockId>(i - 1));
  }

  bounce_.resize(cfg_.block_bytes());
}

BlockPool::~BlockPool() {
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    stopping_ = true;
  }
  queue_cv_.notify_all();
  for (std::thread& worker : workers_) {
    if (worker.joinable()) worker.join();
  }
  // Unfinished jobs die with the pool; the tiers outlive it by contract, and
  // nothing will ever observe the locations that were never rebound.
}

void BlockPool::ensure_worker() {
  if (!workers_.empty()) return;
  const unsigned hw = std::thread::hardware_concurrency();
  const unsigned count = hw >= 4 ? 3 : 1;
  workers_.reserve(count);
  for (unsigned i = 0; i < count; ++i) {
    workers_.emplace_back([this] { worker_loop(); });
  }
}

void BlockPool::worker_loop() {
  std::unique_lock<std::mutex> lk(queue_mutex_);
  while (true) {
    queue_cv_.wait(lk, [&] { return stopping_ || next_unclaimed_ < jobs_.size(); });
    if (stopping_) return;

    MigrationJob* job = jobs_[next_unclaimed_].get();
    ++next_unclaimed_;
    lk.unlock();

    // A cancelled job skips the copy but still completes: its reserved
    // destination slot is handed back at the next drain, not here, so the
    // scheduler never sees a slot freed by another thread.
    if (!job->cancelled.load(std::memory_order_acquire)) {
      std::memcpy(job->dst, job->src, job->bytes);
    }
    job->done.store(true, std::memory_order_release);
    queue_cv_.notify_all();

    lk.lock();
  }
}

bool BlockPool::migrate_async(BlockId id, TierIndex dst) {
  const Location& loc = locate(id);
  if (dst >= tiers_.size()) fatal("async migrate to an out-of-range tier", id);
  if (loc.tier == dst) return false;
  if (migrating_.find(id) != migrating_.end()) return false;
  if (tiers_[dst].free_slots.empty()) return false;

  // The worker copies through raw pointers, so both ends must be
  // host-addressable. A device tier will need the worker to speak the Tier
  // interface; until one exists, callers fall back to synchronous migrate().
  std::byte* src_base = tiers_[loc.tier].tier->host_data();
  std::byte* dst_base = tiers_[dst].tier->host_data();
  if (src_base == nullptr || dst_base == nullptr) return false;

  auto job = std::make_unique<MigrationJob>();
  job->id = id;
  job->dst_tier = dst;
  job->dst_slot = tiers_[dst].free_slots.back();
  tiers_[dst].free_slots.pop_back();
  job->bytes = cfg_.block_bytes();
  job->src = src_base + static_cast<std::size_t>(loc.slot) * job->bytes;
  job->dst = dst_base + static_cast<std::size_t>(job->dst_slot) * job->bytes;

  migrating_.emplace(id, job.get());
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    jobs_.push_back(std::move(job));
  }
  ensure_worker();
  queue_cv_.notify_one();
  return true;
}

bool BlockPool::migrating(BlockId id) const {
  return migrating_.find(id) != migrating_.end();
}

std::size_t BlockPool::pending_departures(TierIndex tier) const {
  std::size_t count = 0;
  for (const auto& [id, job] : migrating_) {
    // The location is not rebound until drain, so it still names the source.
    if (location_[id].tier == tier && job->dst_tier != tier) ++count;
  }
  return count;
}

BlockPool::TierIndex BlockPool::migration_target(BlockId id) const {
  auto it = migrating_.find(id);
  if (it == migrating_.end()) fatal("migration_target of a block not migrating", id);
  return it->second->dst_tier;
}

void BlockPool::cancel_migration(BlockId id) {
  auto it = migrating_.find(id);
  if (it == migrating_.end()) return;
  // The block never moves: its location was untouched while the job was in
  // flight. The job stays queued so the worker can finish with the reserved
  // destination slot, which drain_migrations() then reclaims.
  it->second->cancelled.store(true, std::memory_order_release);
  migrating_.erase(it);
}

void BlockPool::drain_migrations() {
  // Unlocked fast path: the deque is only ever mutated on this thread (the
  // worker reads entries but never adds or removes them), so observing it
  // empty needs no lock. Keeps the single-tier request path mutex-free.
  if (jobs_.empty()) return;

  std::vector<std::unique_ptr<MigrationJob>> finished;
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    while (!jobs_.empty() && jobs_.front()->done.load(std::memory_order_acquire)) {
      finished.push_back(std::move(jobs_.front()));
      jobs_.pop_front();
      --next_unclaimed_;  // done implies claimed, so the cursor covers the front
    }
  }

  for (const auto& job : finished) {
    if (job->cancelled.load(std::memory_order_acquire)) {
      tiers_[job->dst_tier].free_slots.push_back(job->dst_slot);
      continue;
    }
    // An uncancelled finished job implies the block is still allocated and
    // still where it was: decref-to-zero and sync migrate both cancel first.
    Location& loc = location_[job->id];
    tiers_[loc.tier].free_slots.push_back(loc.slot);
    loc.tier = job->dst_tier;
    loc.slot = job->dst_slot;
    migrating_.erase(job->id);
  }
}

void BlockPool::wait_for_migrations() {
  {
    std::unique_lock<std::mutex> lk(queue_mutex_);
    queue_cv_.wait(lk, [&] {
      for (const auto& job : jobs_) {
        if (!job->done.load(std::memory_order_acquire)) return false;
      }
      return true;
    });
  }
  drain_migrations();
}

const BlockPool::Location& BlockPool::locate(BlockId id) const {
  if (id >= location_.size()) fatal("access to an out-of-range block", id);
  if (refcount_[id] == 0) fatal("access to a block that is not allocated", id);
  return location_[id];
}

BlockId BlockPool::allocate() {
  if (free_ids_.empty() || tiers_[0].free_slots.empty()) return kInvalidBlock;

  const BlockId id = free_ids_.back();
  free_ids_.pop_back();
  assert(refcount_[id] == 0);

  location_[id].tier = 0;
  location_[id].slot = tiers_[0].free_slots.back();
  tiers_[0].free_slots.pop_back();

  refcount_[id] = 1;
  return id;
}

void BlockPool::incref(BlockId id) {
  if (id >= location_.size()) fatal("incref of an out-of-range block", id);
  if (refcount_[id] == 0) fatal("incref of a block that is already free", id);
  ++refcount_[id];
}

bool BlockPool::decref(BlockId id) {
  if (id >= location_.size()) fatal("decref of an out-of-range block", id);
  if (refcount_[id] == 0) fatal("decref of a block that is already free", id);
  if (--refcount_[id] == 0) {
    // A dying block takes any in-flight migration with it. The freed source
    // slot may be reallocated while the worker still reads it -- a torn copy
    // into a destination that is discarded at drain, which is harmless.
    cancel_migration(id);
    const Location& loc = location_[id];
    tiers_[loc.tier].free_slots.push_back(loc.slot);
    free_ids_.push_back(id);
    return true;
  }
  return false;
}

std::uint32_t BlockPool::refcount(BlockId id) const {
  if (id >= location_.size()) fatal("refcount of an out-of-range block", id);
  return refcount_[id];
}

BlockPool::TierIndex BlockPool::block_tier(BlockId id) const {
  return locate(id).tier;
}

bool BlockPool::migrate(BlockId id, TierIndex dst) {
  // A synchronous move supersedes a queued one; the stale job's copy targets
  // a still-reserved slot and is discarded at drain.
  cancel_migration(id);

  const Location& loc = locate(id);
  if (dst >= tiers_.size()) fatal("migrate to an out-of-range tier", id);
  if (loc.tier == dst) return true;  // already there
  if (tiers_[dst].free_slots.empty()) return false;

  Tier& src_tier = *tiers_[loc.tier].tier;
  Tier& dst_tier = *tiers_[dst].tier;
  const std::size_t bytes = cfg_.block_bytes();
  const std::size_t src_off = static_cast<std::size_t>(loc.slot) * bytes;

  const std::uint32_t dst_slot = tiers_[dst].free_slots.back();
  tiers_[dst].free_slots.pop_back();
  const std::size_t dst_off = static_cast<std::size_t>(dst_slot) * bytes;

  // One copy when both sides are host-addressable, which every tier today is;
  // the bounce buffer stays as the fallback that works for any pairing. The
  // tiering benchmark moves 64 MiB per request at the default geometry, so the
  // halved traffic is worth the branch.
  std::byte* src_base = src_tier.host_data();
  std::byte* dst_base = dst_tier.host_data();
  if (src_base != nullptr && dst_base != nullptr) {
    std::memcpy(dst_base + dst_off, src_base + src_off, bytes);
  } else {
    src_tier.read(src_off, bounce_.data(), bytes);
    dst_tier.write(dst_off, bounce_.data(), bytes);
  }

  tiers_[loc.tier].free_slots.push_back(loc.slot);
  location_[id].tier = dst;
  location_[id].slot = dst_slot;
  return true;
}

std::size_t BlockPool::block_offset(BlockId id) const {
  const Location& loc = locate(id);
  return static_cast<std::size_t>(loc.slot) * cfg_.block_bytes();
}

void* BlockPool::block_data(BlockId id) {
  const Location& loc = locate(id);
  std::byte* base = tiers_[loc.tier].tier->host_data();
  if (base == nullptr) return nullptr;  // device tier: use block_offset() instead
  return base + static_cast<std::size_t>(loc.slot) * cfg_.block_bytes();
}

const void* BlockPool::block_data(BlockId id) const {
  const Location& loc = locate(id);
  // host_data() is non-const because a device tier may have to map on demand;
  // reading through it here does not mutate the pool's own state.
  std::byte* base = const_cast<Tier*>(tiers_[loc.tier].tier)->host_data();
  if (base == nullptr) return nullptr;
  return base + static_cast<std::size_t>(loc.slot) * cfg_.block_bytes();
}

}  // namespace kvslab
