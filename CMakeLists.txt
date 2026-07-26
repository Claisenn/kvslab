cmake_minimum_required(VERSION 3.16)
project(kvslab LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# The invariants in this engine -- refcount balance, block alignment, pin
# balance -- are exactly the ones that go wrong silently. Keeping assert() live
# in optimized builds costs a branch on cold paths and buys real safety, so it
# is on by default rather than only in Debug.
option(KVSLAB_ENABLE_ASSERTS "Keep assert() live even in optimized builds" ON)

# Compiler-specific spellings, kept in one place so the targets below stay
# readable. MSVC is wired up but not exercised -- clang and GCC are what this
# is built and tested with.
if(MSVC)
  set(KVSLAB_WARNINGS /W4)
  set(KVSLAB_KEEP_ASSERTS /UNDEBUG)
else()
  set(KVSLAB_WARNINGS -Wall -Wextra -Wpedantic)
  set(KVSLAB_KEEP_ASSERTS -UNDEBUG)
endif()

add_library(kvslab
  src/tier.cpp
  src/block_pool.cpp
  src/radix_tree.cpp
  src/cache_manager.cpp
)
target_include_directories(kvslab PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(kvslab PRIVATE ${KVSLAB_WARNINGS})
if(KVSLAB_ENABLE_ASSERTS)
  target_compile_options(kvslab PUBLIC ${KVSLAB_KEEP_ASSERTS})
endif()

enable_testing()

add_executable(kvslab_tests tests/test_kvslab.cpp)
target_link_libraries(kvslab_tests PRIVATE kvslab)
target_include_directories(kvslab_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(kvslab_tests PRIVATE ${KVSLAB_WARNINGS})
add_test(NAME kvslab_tests COMMAND kvslab_tests)

add_executable(bench_prefix_cache bench/bench_prefix_cache.cpp)
target_link_libraries(bench_prefix_cache PRIVATE kvslab)
target_compile_options(bench_prefix_cache PRIVATE ${KVSLAB_WARNINGS})
