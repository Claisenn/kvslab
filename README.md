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

Phase 1 — block management and prefix reuse. Single node, host memory, no GPU
required.

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
as the deeper half means an existing pin still covers every token it covered
before the split, which is what makes those pointers safe to hold.

Eviction is LRU over unpinned leaves, driven by a logical clock. A prefix an
in-flight request is reading is pinned along with all its ancestors and cannot
be reclaimed underneath it.

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

Requires only a C++20 compiler and CMake 3.16+. `assert()` stays live in
optimized builds by default (`-DKVSLAB_ENABLE_ASSERTS=OFF` to disable) — the
invariants it guards are refcount and pin balance, which otherwise fail
silently.

## Roadmap

- **Phase 1** — block allocator, block-aligned radix prefix index, pinning and
  LRU eviction, tests and benchmarks.
- **Phase 2** — tiered storage. Abstract the arena into a `Tier`, stack
  HBM / DRAM / NVMe, move blocks asynchronously on a background thread so
  transfer overlaps compute. NVMe via `io_uring`.
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
