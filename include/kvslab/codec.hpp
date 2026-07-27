#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace kvslab {

// Transforms a block's bytes on the way into and out of a spill tier.
//
// The pool treats KV as opaque bytes everywhere else; a codec is the one
// place that knows the dtype. It exists so a spill tier can hold blocks in a
// cheaper representation than the compute tier -- the classic move being
// fp16 KV quantized to fp8 on demotion, which doubles what the same spill
// arena holds at an accuracy cost attention tolerates far better than the
// alternative, which is not caching the prefix at all.
//
// encode() and decode() must tolerate src == dst (in-place over one buffer):
// the bounce path through a non-host-addressable tier has one scratch buffer,
// not two. `raw_bytes` is always the uncompressed size; encoded_bytes() maps
// it to the compressed one.
class SpillCodec {
 public:
  virtual ~SpillCodec() = default;

  SpillCodec(const SpillCodec&) = delete;
  SpillCodec& operator=(const SpillCodec&) = delete;

  virtual std::string_view name() const = 0;
  virtual std::size_t encoded_bytes(std::size_t raw_bytes) const = 0;
  virtual void encode(const std::byte* src, std::size_t raw_bytes,
                      std::byte* dst) const = 0;
  virtual void decode(const std::byte* src, std::size_t raw_bytes,
                      std::byte* dst) const = 0;

 protected:
  SpillCodec() = default;
};

// --- Scalar conversions, exposed for tests --------------------------------
//
// fp16 here is IEEE binary16. fp8 is OCP E4M3: 1 sign, 4 exponent (bias 7),
// 3 mantissa, no infinities, 0x7f/0xff as NaN, max finite 448. Conversion
// rounds to nearest even and saturates past 448 -- saturating rather than
// producing NaN is what inference stacks do, because one outlier poisoning a
// whole block is strictly worse than clipping it.

inline float fp16_bits_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1fu;
  std::uint32_t man = h & 0x3ffu;
  std::uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        ++shift;
      }
      man &= 0x3ffu;
      bits = sign | ((127u - 15u - static_cast<std::uint32_t>(shift)) << 23) |
             (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
  }
  return std::bit_cast<float>(bits);
}

inline std::uint16_t float_to_fp16_bits(float f) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
  const std::uint32_t abs = bits & 0x7fffffffu;

  if (abs >= 0x7f800000u) {  // inf or nan
    const std::uint16_t man = abs > 0x7f800000u ? 0x200u : 0u;
    return static_cast<std::uint16_t>(sign | 0x7c00u | man);
  }
  if (abs >= 0x477ff000u) {  // rounds to >= 65536: overflow to inf
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (abs < 0x38800000u) {  // subnormal half, or zero
    if (abs < 0x33000000u) return sign;  // rounds to zero
    const int shift = 126 - static_cast<int>(abs >> 23);
    std::uint32_t man = (abs & 0x7fffffu) | 0x800000u;
    // Round to nearest even at bit (shift + 13).
    const std::uint32_t lsb = 1u << (shift + 13);
    const std::uint32_t round = (lsb >> 1) - 1u + ((man >> (shift + 13)) & 1u);
    man += round;
    return static_cast<std::uint16_t>(sign | (man >> (shift + 13)));
  }
  std::uint32_t man = abs + 0xfffu + ((abs >> 13) & 1u);  // RNE at bit 13
  return static_cast<std::uint16_t>(sign | ((man - 0x38000000u) >> 13));
}

inline std::uint8_t float_to_e4m3(float f) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
  const std::uint8_t sign = static_cast<std::uint8_t>((bits >> 24) & 0x80u);
  const std::uint32_t abs = bits & 0x7fffffffu;

  if (abs > 0x7f800000u) return sign | 0x7fu;  // nan stays nan
  if (abs >= 0x43e10000u) return sign | 0x7eu;  // >= 450: saturate to 448
  if (abs < 0x3a800000u) {                      // < 2^-10: rounds to zero
    return sign;
  }

  // Round the float's 23-bit mantissa to E4M3's step at this magnitude.
  // Normal E4M3 keeps 3 mantissa bits; subnormals (exp < -6) keep fewer.
  const int exp = static_cast<int>(abs >> 23) - 127;
  const int drop = exp >= -6 ? 20 : 20 + (-6 - exp);  // mantissa bits to shed
  std::uint32_t man = (abs & 0x7fffffu) | 0x800000u;
  const std::uint32_t lsb = 1u << drop;
  man += (lsb >> 1) - 1u + ((man >> drop) & 1u);  // RNE
  std::uint32_t q = man >> drop;                  // implicit bit still present

  int e = exp;
  if (exp >= -6) {
    if (q == 16) {  // mantissa rounded up into the next binade
      q = 8;
      ++e;
    }
    if (e > 8 || (e == 8 && (q - 8) > 6)) return sign | 0x7eu;  // saturate
    return static_cast<std::uint8_t>(sign | ((e + 7) << 3) | (q - 8));
  }
  // Subnormal result: q is already in units of 2^-9.
  if (q >= 8) return static_cast<std::uint8_t>(sign | 0x08u);  // min normal
  return static_cast<std::uint8_t>(sign | q);
}

inline float e4m3_to_float(std::uint8_t v) {
  const std::uint32_t sign = static_cast<std::uint32_t>(v & 0x80u) << 24;
  const int exp = (v >> 3) & 0xf;
  const int man = v & 0x7;
  std::uint32_t bits;
  if (exp == 0xf && man == 0x7) {
    bits = sign | 0x7fc00000u;  // nan
  } else if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      // Subnormal: man * 2^-9. Normalize man's leading bit into the float's
      // implicit one.
      const int p = man >= 4 ? 2 : (man >= 2 ? 1 : 0);
      bits = sign | (static_cast<std::uint32_t>(127 + p - 9) << 23) |
             ((static_cast<std::uint32_t>(man) << (23 - p)) & 0x7fffffu);
    }
  } else {
    bits = sign | (static_cast<std::uint32_t>(exp - 7 + 127) << 23) |
           (static_cast<std::uint32_t>(man) << 20);
  }
  return std::bit_cast<float>(bits);
}

// FP16 -> FP8 E4M3 over a buffer of binary16 values, halving its size.
//
// Both directions are table lookups: 64 KiB maps every fp16 pattern to its
// nearest E4M3, 512 B maps every E4M3 back to exact fp16 bits (every E4M3
// value is exactly representable in fp16, so decode loses nothing on top of
// what encode already gave up). The tables are built once in the constructor
// from the scalar conversions above, which keeps the hot path free of
// floating-point edge cases.
class Fp8SpillCodec final : public SpillCodec {
 public:
  Fp8SpillCodec();

  std::string_view name() const override { return "fp8-e4m3"; }
  std::size_t encoded_bytes(std::size_t raw_bytes) const override {
    return raw_bytes / 2;
  }
  void encode(const std::byte* src, std::size_t raw_bytes,
              std::byte* dst) const override;
  void decode(const std::byte* src, std::size_t raw_bytes,
              std::byte* dst) const override;

 private:
  std::vector<std::uint8_t> enc_;   // fp16 bits -> e4m3
  std::vector<std::uint16_t> dec_;  // e4m3 -> fp16 bits
};

}  // namespace kvslab
