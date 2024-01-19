#ifndef BRIDGE_LEARNING_BRIDGE_LIB_UTILS_H
#define BRIDGE_LEARNING_BRIDGE_LIB_UTILS_H
#include <cassert>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace bridge_learning_env {
using GameParameters = std::unordered_map<std::string, std::string>;

// Returns string associated with key in params, parsed as template type.
// If key is not in params, returns the provided default value.
template<class T>
T ParameterValue(const GameParameters& params, const std::string& key, T default_value);

template<>
int ParameterValue(const GameParameters& params, const std::string& key, int default_value);

template<>
double ParameterValue(const GameParameters& params, const std::string& key, double default_value);

template<>
std::string ParameterValue(const GameParameters& params, const std::string& key, std::string default_value);

template<>
bool ParameterValue(const GameParameters& params, const std::string& key, bool default_value);

template<typename... Args>
std::string StrFormat(const std::string& format, Args... args) {
  const int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  const auto size = static_cast<size_t>(size_s);
  const std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return {buf.get(), buf.get() + size - 1}; // We don't want the '\0' inside
}

std::vector<int> Arange(int start, int end);

std::vector<int> Permutation(int num);

std::vector<int> Permutation(int num, std::mt19937& rng);

template<typename T>
void PrintVector(const std::vector<T>& vec) {
  for (const T item : vec) {
    std::cout << item << ",";
  }
  std::cout << std::endl;
}

template<class... Args>
std::string StrCat(const Args&... args) {
  using Expander = int[];
  std::stringstream ss;
  (void)Expander{0, (void(ss << args), 0)...};
  return ss.str();
}

std::vector<std::string> StrSplit(const std::string& str, char delimiter);

template<class... Args>
void AssertWithMessage(bool condition, Args&&... args) {
  if (!condition) {
    const std::string msg = StrCat(std::forward<Args>(args)...);
    std::cerr << msg << std::endl;
    abort();
  }
}

// fprintf(stderr, "Input requirements failed at %s:%d in %s: %s\n", __FILE__, __LINE__, __func__, #expr)
#if defined(NDEBUG)
#define REQUIRE(expr) \
  bridge_learning_env::AssertWithMessage(expr, #expr, ", check failed at ", __FILE__, ":", __LINE__, ". ");
#define REQUIRE_EQ(x, y)\
  bridge_learning_env::AssertWithMessage((x) == (y), #x " == " #y, " check failed at ", __FILE__, ":", __LINE__, \
": ", (x), " vs ", (y), ". ");
#define REQUIRE_VECTOR_EQ(vec1, vec2) \
do { \
if ((vec1) != (vec2)) { \
std::cerr << "Assertion failed: Vectors not equal at " << __FILE__ << ":" << __LINE__ << std::endl; \
std::cerr << "Vector 1: "; \
PrintVector((vec1)); \
std::cerr << "Vector 2: "; \
PrintVector((vec2)); \
std::abort(); \
} \
} while (0)
#else
#define REQUIRE(expr) assert(expr)
#define REQUIRE_EQ(x, y) assert((x)==(y))
#define REQUIRE_VECTOR_EQ(vec1, vec2) assert((vec1)==(vec2))
#endif

// #if defined(NDEBUG)
// #define REQUIRE_EQ(expr1, expr2)                                                     \
//   (expr1 == expr2 ? (void)0                                                          \
//                   : (fprintf(stderr,                                                 \
//                              "Input requirements failed at %s:%d in %s: %s vs %s\n", \
//                              __FILE__,                                               \
//                              __LINE__,                                               \
//                              __func__,                                               \
//                              #expr1,                                                 \
//                              #expr2),                                                \
//                      std::abort()))
// #else
// #define REQUIRE_EQ(expr1, expr2) assert(expr1 == expr2)
// #endif
} // namespace bridge_learning_env

#endif // BRIDGE_LEARNING_BRIDGE_LIB_UTILS_H
