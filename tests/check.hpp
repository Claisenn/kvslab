#pragma once

// A ~50 line test harness, so the project stays buildable with nothing but a
// compiler and CMake. GoogleTest would pull in a network fetch for no capability
// this needs.

#include <cstdio>
#include <vector>

namespace kvcheck {

inline int g_checks = 0;
inline int g_failures = 0;

inline void report(bool ok, const char* expr, const char* file, int line) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "    FAIL %s:%d: %s\n", file, line, expr);
  }
}

struct Case {
  const char* name;
  void (*fn)();
};

inline std::vector<Case>& registry() {
  static std::vector<Case> cases;
  return cases;
}

struct Register {
  Register(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline int run_all() {
  for (const Case& c : registry()) {
    const int before = g_failures;
    c.fn();
    std::fprintf(stderr, "[%s] %s\n", g_failures == before ? " ok " : "FAIL", c.name);
  }
  std::fprintf(stderr, "\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

}  // namespace kvcheck

#define CHECK(expr) ::kvcheck::report(static_cast<bool>(expr), #expr, __FILE__, __LINE__)
#define CHECK_EQ(a, b) ::kvcheck::report((a) == (b), #a " == " #b, __FILE__, __LINE__)

#define TEST(name)                                      \
  static void name();                                   \
  static ::kvcheck::Register kvcheck_reg_##name(#name, &name); \
  static void name()
