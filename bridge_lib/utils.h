#ifndef BRIDGE_LEARNING_BRIDGE_LIB_UTILS_H
#define BRIDGE_LEARNING_BRIDGE_LIB_UTILS_H
#include <memory>
#include <cassert>
#include <random>
#include <stdexcept>
#include <string>
#include <iostream>
#include <unordered_map>
#include <sstream>

namespace bridge {
using GameParameters = std::unordered_map<std::string, std::string>;

// Returns string associated with key in params, parsed as template type.
// If key is not in params, returns the provided default value.
template <class T>
T ParameterValue(const GameParameters& params, const std::string& key,
                 T default_value);

template <>
int ParameterValue(const GameParameters& params, const std::string& key,
                   int default_value);
template <>
double ParameterValue(const GameParameters& params, const std::string& key,
                      double default_value);
template <>
std::string ParameterValue(const GameParameters& params, const std::string& key,
                           std::string default_value);
template <>
bool ParameterValue(const GameParameters& params, const std::string& key,
                    bool default_value);

template <typename... Args>
std::string StrFormat(const std::string& format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1;  // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return {buf.get(), buf.get() + size - 1};  // We don't want the '\0' inside
}


std::vector<int> Arange(int start, int end);

std::vector<int> Permutation(int num);

std::vector<int> Permutation(int num, std::mt19937& rng);

template <typename T>
void PrintVector(const std::vector<T>& vec) {
  for (const T item : vec) {
    std::cout << item << ",";
  }
  std::cout << std::endl;
}

template <class... Args>
std::string StrCat(const Args&... args) {
  using Expander = int[];
  std::stringstream ss;
  (void)Expander{0, (void(ss << args), 0)...};
  return ss.str();
}

#if defined(NDEBUG)
#define REQUIRE(expr)                                                        \
  (expr ? (void)0                                                            \
        : (fprintf(stderr, "Input requirements failed at %s:%d in %s: %s\n", \
                   __FILE__, __LINE__, __func__, #expr),                     \
           std::abort()))
#else
#define REQUIRE(expr) assert(expr)
#endif

}  // namespace bridge

#endif // BRIDGE_LEARNING_BRIDGE_LIB_UTILS_H
