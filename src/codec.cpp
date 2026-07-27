#include "kvslab/codec.hpp"

#include <cassert>
#include <cstring>

namespace kvslab {

Fp8SpillCodec::Fp8SpillCodec() : enc_(1u << 16), dec_(1u << 8) {
  for (std::uint32_t h = 0; h < (1u << 16); ++h) {
    enc_[h] = float_to_e4m3(fp16_bits_to_float(static_cast<std::uint16_t>(h)));
  }
  for (std::uint32_t v = 0; v < (1u << 8); ++v) {
    dec_[v] = float_to_fp16_bits(e4m3_to_float(static_cast<std::uint8_t>(v)));
  }
}

void Fp8SpillCodec::encode(const std::byte* src, std::size_t raw_bytes,
                           std::byte* dst) const {
  assert(raw_bytes % 2 == 0 && "fp16 payloads are an even number of bytes");
  const std::size_t n = raw_bytes / 2;
  // Forward walk is in-place safe: the write at i trails both reads at 2i.
  for (std::size_t i = 0; i < n; ++i) {
    std::uint16_t h;
    std::memcpy(&h, src + 2 * i, 2);
    dst[i] = static_cast<std::byte>(enc_[h]);
  }
}

void Fp8SpillCodec::decode(const std::byte* src, std::size_t raw_bytes,
                           std::byte* dst) const {
  assert(raw_bytes % 2 == 0 && "fp16 payloads are an even number of bytes");
  const std::size_t n = raw_bytes / 2;
  // Backward walk is in-place safe: the writes at 2i land at or beyond the
  // read at i, and everything still unread sits below i.
  for (std::size_t i = n; i > 0; --i) {
    const std::uint16_t h = dec_[static_cast<std::uint8_t>(src[i - 1])];
    std::memcpy(dst + 2 * (i - 1), &h, 2);
  }
}

}  // namespace kvslab
