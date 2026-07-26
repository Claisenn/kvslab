#pragma once

#include <cstddef>
#include <string_view>

#include "kvslab/types.hpp"

namespace kvslab {

// A flat byte arena that KV blocks live in.
//
// Tiers exist so the allocator stops caring where the bytes physically are.
// The interface is deliberately expressed as bulk copies rather than pointers:
// host memory can be dereferenced by the CPU and device memory cannot, so a
// pointer-only interface would either lie about device tiers or force every
// caller to special-case them. `host_data()` is the fast path for tiers the CPU
// *can* address and returns nullptr for the ones it can't -- callers that need
// to work on any tier go through read/write.
class Tier {
 public:
  virtual ~Tier() = default;

  Tier(const Tier&) = delete;
  Tier& operator=(const Tier&) = delete;

  virtual std::string_view name() const = 0;
  virtual std::size_t capacity_bytes() const = 0;

  // Non-null only when the CPU can dereference this tier directly.
  virtual std::byte* host_data() = 0;

  // Bulk moves against a host buffer. `offset` is a byte offset into the arena.
  virtual void read(std::size_t offset, void* dst, std::size_t bytes) const = 0;
  virtual void write(std::size_t offset, const void* src, std::size_t bytes) = 0;

 protected:
  Tier() = default;
};

// Page-aligned host memory: the tier every other one is measured against, and
// the only one that needs no special build configuration.
//
// Page alignment is not cosmetic -- it is what leaves the door open to pinning
// the arena or registering it with an RDMA NIC without the registration
// straddling a page boundary.
class HostTier final : public Tier {
 public:
  explicit HostTier(std::size_t capacity_bytes);
  ~HostTier() override;

  std::string_view name() const override { return "host"; }
  std::size_t capacity_bytes() const override { return capacity_bytes_; }
  std::byte* host_data() override { return arena_; }

  void read(std::size_t offset, void* dst, std::size_t bytes) const override;
  void write(std::size_t offset, const void* src, std::size_t bytes) override;

 private:
  std::byte* arena_ = nullptr;
  std::size_t capacity_bytes_ = 0;
};

}  // namespace kvslab
