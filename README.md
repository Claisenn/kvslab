# kvslab

A KV cache management engine for LLM inference, written from scratch in C++20.

KV cache is the dominant memory pressure in LLM serving: a long-context request
can hold more KV than the model weights themselves, while requests share large
common prefixes — system prompts, few-shot examples, conversation history. How
that memory is partitioned, reused, evicted and moved between devices decides
both throughput and time-to-first-token.

kvslab implements that layer directly rather than integrating an existing one.
The allocator, the prefix index, the tiering and the transfer path are all
written here.

## Status

Phase 2 — tiered storage. Phase 1 (block management and prefix reuse) is
complete; the cache now runs over a compute tier plus a spill tier, demoting
cold entries instead of dropping them and migrating blocks on background
workers behind a readiness gate. Single node, host memory, no GPU required.

## Design

Three layers, bottom up.

**`BlockPool`** — a fixed-size block allocator over one contiguous, page-aligned
arena, with a free list and refcounts. Blocks are fixed size, so the block table
handed to an attention kernel is a plain index list with no copying involved.
Page alignment leaves the door open to pinning the arena or registering it with
an RDMA NIC later.

**`RadixTree`** — the prefix index. Token sequences are organised into a radix
tree so requests sharing a prefix share the physical blocks holding its KV.

**`CacheManager`** — ties the two into the interface a scheduler wants: hand it a
token sequence, get back a block table.

### Two decisions worth calling out

*Children are keyed by the hash of a whole block of tokens, not by a single
token.* The tree is block aligned, so two sequences diverging in the middle of a
block must land in separate children, and a per-token key cannot express that.
Equality of the key is never trusted on its own: a match is confirmed by
comparing the tokens themselves, and an insert whose key is already taken by a
different block is refused rather than overwriting it — overwriting would strand
that subtree's blocks with no owner to return them to the free list. A collision
therefore costs a cache entry, never correctness, and `hash_collisions()`
reports how many have been seen.

*Splitting a node interposes a new **parent**, not a new child.* Callers hold
node pointers as pins across a request's lifetime. Keeping the original object
as the deeper half means an existing pin still protects every token it protected
before the split: pinning walks to the root, and the interposed parent inherits
the count. The node object on its own covers less afterwards — it is the pin,
not the pointer's own range, that callers rely on.

Eviction is LRU over unpinned leaves by default, driven by a logical clock. A
prefix an in-flight request is reading is pinned along with all its ancestors
and cannot be reclaimed underneath it.

An optional scan-resistant policy (`RadixTree::EvictionPolicy::kScanResistant`)
splits candidates into a probation segment (stored once, never reused) and a
protected one (reused at least once), and spends probation first — two-segment
SLRU. It exists for the workload that kills plain LRU: a hot set re-visited on
a period longer than the flood of one-shot sequences arriving in between. On
the scan-pollution benchmark it takes the hit rate from 3.8% to 21.2% — the
workload's ceiling is 23% — while evicting 20% fewer blocks, each of which is
a full prefill recompute in a real engine.

## Build

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Benchmarks want an optimized build:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
./build-release/bench_prefix_cache
```

Needs a C++20 compiler and CMake 3.16+, and nothing else. Built and tested with
Apple clang on macOS/arm64. The MSVC path is wired up — `_aligned_malloc` for
the arena, `/W4` and `/UNDEBUG` for the flags — but has never been compiled, so
treat it as a starting point rather than a supported target.

Block refcount and range checks are unconditional. Violating one is not a crash
but silent corruption — a block handed back twice lands on the free list while
the index still points at it — and nothing downstream can detect that, so they
are not worth trading away for a branch. The remaining invariants, chiefly pin
balance and block alignment, ride on `assert()`, which stays live in optimized
builds by default; `-DKVSLAB_ENABLE_ASSERTS=OFF` turns those off.

## Roadmap

- **Phase 1** ✅ — block allocator, block-aligned radix prefix index, pinning
  and LRU eviction, tests and benchmarks.
- **Phase 2** — tiered storage. Done: the `Tier` abstraction, stable block
  identities that survive migration, demotion to spill instead of eviction with
  promotion on hit, and asynchronous migration in both directions — background
  demotion to a watermark, promotion behind an `Allocation::ready()` gate, a
  small worker pool. On the oversubscription benchmark this serves 80% of the
  requests a single tier misses entirely, at 93% of its request rate.
  Also done: an optional spill codec (`Fp8SpillCodec`, fp16 → fp8 E4M3 by
  table lookup) that stores demoted blocks at half size, so the same spill
  arena holds twice the entries. Quantization error is bounded by E4M3
  round-to-nearest and paid once — a re-demotion re-encodes to the same
  bits. On the capped-spill benchmark the codec turns a working set that
  misses entirely (fp16 spill, 0% hit) into one that fits (80% hit, zero
  evictions) in the same bytes.
  Remaining: an NVMe-backed tier via `io_uring`.
- **Phase 3** — transfer engine. Zero-copy KV movement between nodes for
  prefill/decode disaggregation: TCP baseline first for correctness, then RDMA
  over `ibverbs`.
- **Phase 4** — plug into a real stack. Implement vLLM's `KVConnector` and
  report end-to-end TTFT and throughput.

## Prior art

[Mooncake](https://github.com/kvcache-ai/Mooncake),
[InfiniStore](https://github.com/bytedance/InfiniStore),
[tair-kvcache](https://github.com/alibaba/tair-kvcache) and
[LMCache](https://github.com/LMCache/LMCache) all solve pieces of this problem in
production. kvslab is not trying to replace them — it is a from-scratch
implementation of the same layer, built to understand it end to end.

## License

Apache-2.0
