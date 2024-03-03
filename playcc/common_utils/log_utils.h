//
// Created by qzz on 2023/12/3.
//

#ifndef BRIDGE_LEARNING_PLAYCC_LOG_UTILS_H_
#define BRIDGE_LEARNING_PLAYCC_LOG_UTILS_H_
#include <cmath>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include "absl/types/optional.h"
#include "absl/types/span.h"



// Make sure that arbitrary structures can be printed out.
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::unique_ptr<T>& v);
template <typename T, typename U>
std::ostream& operator<<(std::ostream& stream, const std::pair<T, U>& v);
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& v);
template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& stream, const std::array<T, N>& v);
template <typename T>
std::ostream& operator<<(std::ostream& stream, const absl::optional<T>& v);
std::ostream& operator<<(std::ostream& stream, const absl::nullopt_t& v);
template <typename T>
std::ostream& operator<<(std::ostream& stream, absl::Span<T> v);

// Actual template implementations.
template <typename T>
std::ostream& operator<<(std::ostream& stream, absl::Span<T> v) {
  stream << "[";
  for (const auto& element : v) {
    stream << element << " ";
  }
  stream << "]";
  return stream;
}
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& v) {
  stream << "[";
  for (const auto& element : v) {
    stream << element << " ";
  }
  stream << "]";
  return stream;
}
template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& stream, const std::array<T, N>& v) {
  stream << "[";
  for (const auto& element : v) {
    stream << element << " ";
  }
  stream << "]";
  return stream;
}
template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::unique_ptr<T>& v) {
  return stream << *v;
}
template <typename T>
std::ostream& operator<<(std::ostream& stream, const absl::optional<T>& v) {
  return stream << *v;
}
template <typename T, typename U>
std::ostream& operator<<(std::ostream& stream, const std::pair<T, U>& v) {
  stream << "(" << v.first << "," << v.second << ")";
  return stream;
}

namespace internal {
// SpielStrOut(out, a, b, c) is equivalent to:
//    out << a << b << c;
// It is used to enable SpielStrCat, below.
template<typename Out, typename T>
void SpielStrOut(Out& out, const T& arg) {
  out << arg;
}

template<typename Out, typename T, typename... Args>
void SpielStrOut(Out& out, const T& arg1, Args&&... args) {
  out << arg1;
  SpielStrOut(out, std::forward<Args>(args)...);
}

// Builds a string from pieces:
//
//  SpielStrCat(1, " + ", 1, " = ", 2) --> "1 + 1 = 2"
//
// Converting the parameters to strings is done using the stream operator<<.
// This is only kept around to be used in the SPIEL_CHECK_* macros and should
// not be called by any code outside of this file. Prefer absl::StrCat instead.
// It is kept here due to support for more types, including char.
template<typename... Args>
std::string SpielStrCat(Args&&... args) {
  std::ostringstream out;
  SpielStrOut(out, std::forward<Args>(args)...);
  return out.str();
}

} // namespace internal

// Macros to check for error conditions.
// These trigger SpielFatalError if the condition is violated.
// These macros are always executed. If you want to use checks
// only for debugging, use SPIEL_DCHECK_*

#define SPIEL_CHECK_OP(x_exp, op, y_exp)                                                                          \
  do {                                                                                                            \
    auto x = x_exp;                                                                                               \
    auto y = y_exp;                                                                                               \
    if (!((x)op(y)))                                                                                              \
      SpielFatalError(internal::SpielStrCat(                                                                      \
          __FILE__, ":", __LINE__, " ", #x_exp " " #op " " #y_exp, "\n" #x_exp, " = ", x, ", " #y_exp " = ", y)); \
  }                                                                                                               \
  while (false)

#define SPIEL_CHECK_FN2(x_exp, y_exp, fn)                                                                          \
  do {                                                                                                             \
    auto x = x_exp;                                                                                                \
    auto y = y_exp;                                                                                                \
    if (!fn(x, y))                                                                                                 \
      SpielFatalError(internal::SpielStrCat(                                                                       \
          __FILE__, ":", __LINE__, " ", #fn "(" #x_exp ", " #y_exp ")\n", #x_exp " = ", x, ", " #y_exp " = ", y)); \
  }                                                                                                                \
  while (false)

#define SPIEL_CHECK_FN3(x_exp, y_exp, z_exp, fn)                                          \
  do {                                                                                    \
    auto x = x_exp;                                                                       \
    auto y = y_exp;                                                                       \
    auto z = z_exp;                                                                       \
    if (!fn(x, y, z))                                                                     \
      SpielFatalError(internal::SpielStrCat(__FILE__,                                     \
                                            ":",                                          \
                                            __LINE__,                                     \
                                            " ",                                          \
                                            #fn "(" #x_exp ", " #y_exp ", " #z_exp ")\n", \
                                            #x_exp " = ",                                 \
                                            x,                                            \
                                            ", " #y_exp " = ",                            \
                                            y,                                            \
                                            ", " #z_exp " = ",                            \
                                            z));                                          \
  }                                                                                       \
  while (false)

#define SPIEL_CHECK_GE(x, y) SPIEL_CHECK_OP(x, >=, y)
#define SPIEL_CHECK_GT(x, y) SPIEL_CHECK_OP(x, >, y)
#define SPIEL_CHECK_LE(x, y) SPIEL_CHECK_OP(x, <=, y)
#define SPIEL_CHECK_LT(x, y) SPIEL_CHECK_OP(x, <, y)
#define SPIEL_CHECK_EQ(x, y) SPIEL_CHECK_OP(x, ==, y)
#define SPIEL_CHECK_NE(x, y) SPIEL_CHECK_OP(x, !=, y)
#define SPIEL_CHECK_PROB(x) \
  SPIEL_CHECK_GE(x, 0);     \
  SPIEL_CHECK_LE(x, 1);     \
  SPIEL_CHECK_FALSE(std::isnan(x) || std::isinf(x))
#define SPIEL_CHECK_PROB_TOLERANCE(x, tol) \
  SPIEL_CHECK_GE(x, -(tol));               \
  SPIEL_CHECK_LE(x, 1.0 + (tol));          \
  SPIEL_CHECK_FALSE(std::isnan(x) || std::isinf(x))

// Checks that x and y are equal to the default dynamic threshold proportional
// to max(|x|, |y|).
#define SPIEL_CHECK_FLOAT_EQ(x, y) SPIEL_CHECK_FN2(static_cast<float>(x), static_cast<float>(y), Near)

// Checks that x and y are epsilon apart or closer.
#define SPIEL_CHECK_FLOAT_NEAR(x, y, epsilon) \
  SPIEL_CHECK_FN3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(epsilon), Near)

#define SPIEL_CHECK_TRUE(x) \
  while (!(x)) SpielFatalError(internal::SpielStrCat(__FILE__, ":", __LINE__, " CHECK_TRUE(", #x, ")"))

// A verbose checker that will print state info:
// Use as SPIEL_CHECK_TRUE_WSI(bool cond, const std::string& error_message,
//                             const Game& game_ref, const State& state_ref)
#define SPIEL_CHECK_TRUE_WSI(x, e, g, s) \
  while (!(x))                           \
  SpielFatalErrorWithStateInfo(internal::SpielStrCat(__FILE__, ":", __LINE__, " CHECK_TRUE(", #x, "): ", e), (g), (s))

#define SPIEL_CHECK_FALSE(x) \
  while (x) SpielFatalError(internal::SpielStrCat(__FILE__, ":", __LINE__, " CHECK_FALSE(", #x, ")"))

#if !defined(NDEBUG)

// Checks that are executed in Debug / Testing build type,
// and turned off for Release build type.
#define SPIEL_DCHECK_OP(x_exp, op, y_exp) SPIEL_CHECK_OP(x_exp, op, y_exp)
#define SPIEL_DCHECK_FN2(x_exp, y_exp, fn) SPIEL_CHECK_FN2(x_exp, y_exp, fn)
#define SPIEL_DCHECK_FN3(x_exp, y_exp, z_exp, fn) SPIEL_CHECK_FN3(x_exp, y_exp, z_exp, fn)
#define SPIEL_DCHECK_GE(x, y) SPIEL_CHECK_GE(x, y)
#define SPIEL_DCHECK_GT(x, y) SPIEL_CHECK_GT(x, y)
#define SPIEL_DCHECK_LE(x, y) SPIEL_CHECK_LE(x, y)
#define SPIEL_DCHECK_LT(x, y) SPIEL_CHECK_LT(x, y)
#define SPIEL_DCHECK_EQ(x, y) SPIEL_CHECK_EQ(x, y)
#define SPIEL_DCHECK_NE(x, y) SPIEL_CHECK_NE(x, y)
#define SPIEL_DCHECK_PROB(x) SPIEL_DCHECK_PROB(x)
#define SPIEL_DCHECK_FLOAT_EQ(x, y) SPIEL_CHECK_FLOAT_EQ(x, y)
#define SPIEL_DCHECK_FLOAT_NEAR(x, y, epsilon) SPIEL_CHECK_FLOAT_NEAR(x, y, epsilon)
#define SPIEL_DCHECK_TRUE(x) SPIEL_CHECK_TRUE(x)
#define SPIEL_DCHECK_FALSE(x) SPIEL_CHECK_FALSE(x)

#else // defined(NDEBUG)

// Turn off checks for the (optimized) Release build type.
#define SPIEL_DCHECK_OP(x_exp, op, y_exp)
#define SPIEL_DCHECK_FN2(x_exp, y_exp, fn)
#define SPIEL_DCHECK_FN3(x_exp, y_exp, z_exp, fn)
#define SPIEL_DCHECK_GE(x, y)
#define SPIEL_DCHECK_GT(x, y)
#define SPIEL_DCHECK_LE(x, y)
#define SPIEL_DCHECK_LT(x, y)
#define SPIEL_DCHECK_EQ(x, y)
#define SPIEL_DCHECK_NE(x, y)
#define SPIEL_DCHECK_PROB(x)
#define SPIEL_DCHECK_FLOAT_EQ(x, y)
#define SPIEL_DCHECK_FLOAT_NEAR(x, y, epsilon)
#define SPIEL_DCHECK_TRUE(x)
#define SPIEL_DCHECK_FALSE(x)

#endif // !defined(NDEBUG)

// When an error is encountered, OpenSpiel code should call SpielFatalError()
// which will forward the message to the current error handler.
// The default error handler outputs the error message to stderr, and exits
// the process with exit code 1.

// When called from Python, a different error handled is used, which returns
// RuntimeException to the caller, containing the error message.

// Report a runtime error.
[[noreturn]] void SpielFatalError(const std::string& error_msg);

// Specify a new error handler.
using ErrorHandler = void (*)(const std::string&);
void SetErrorHandler(ErrorHandler error_handler);

// Floating point comparisons use this as a multiplier on the larger of the two
// numbers as the threshold.
inline constexpr float FloatingPointDefaultThresholdRatio() { return 1e-5f; }

// Returns whether the absolute difference between floating point values a and
// b is less than or equal to FloatingPointThresholdRatio() * max(|a|, |b|).
template<typename T>
bool Near(T a, T b) {
  static_assert(std::is_floating_point<T>::value, "Near() is only for floating point args.");
  return fabs(a - b) <= (std::max(fabs(a), fabs(b)) * FloatingPointDefaultThresholdRatio());
}

// Returns whether |a - b| <= epsilon.
template<typename T>
bool Near(T a, T b, T epsilon) {
  static_assert(std::is_floating_point<T>::value, "Near() is only for floating point args.");
  return fabs(a - b) <= epsilon;
}
#endif // BRIDGE_LEARNING_PLAYCC_LOG_UTILS_H_
