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

add_library(kvslab
  src/block_pool.cpp
  src/radix_tree.cpp
)
target_include_directories(kvslab PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(kvslab PRIVATE -Wall -Wextra -Wpedantic)
if(KVSLAB_ENABLE_ASSERTS)
  target_compile_options(kvslab PUBLIC -UNDEBUG)
endif()

enable_testing()

add_executable(kvslab_tests tests/test_kvslab.cpp)
target_link_libraries(kvslab_tests PRIVATE kvslab)
target_include_directories(kvslab_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(kvslab_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME kvslab_tests COMMAND kvslab_tests)
