//
// Created by qzz on 2023/9/19.
//

#ifndef BRIDGE_LEARNING_RELA_UTILS_H_
#define BRIDGE_LEARNING_RELA_UTILS_H_
#include <iostream>
#include <unordered_map>
#include <vector>
#include <random>

namespace rela::utils{
// inline int getProduct(const std::vector<int64_t>& nums) {
//   int prod = 1;
//   for (auto v : nums) {
//     prod *= v;
//   }
//   return prod;
// }

template <typename T>
inline std::vector<T> pushLeft(T left, const std::vector<T>& vals) {
  std::vector<T> vec;
  vec.reserve(1 + vals.size());
  vec.push_back(left);
  for (auto v : vals) {
    vec.push_back(v);
  }
  return vec;
}

template <typename T>
inline void printVector(const std::vector<T>& vec) {
  for (const auto& v : vec) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
inline void printMapKey(const T& map) {
  for (const auto& name2sth : map) {
    std::cout << name2sth.first << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
inline void printMap(const T& map) {
  for (const auto& name2sth : map) {
    std::cout << name2sth.first << ": " << name2sth.second << std::endl;
  }
}

inline void RelaFatalError(const std::string& error_msg){
  std::cerr << "Rela Fatal Error: " << error_msg << std::endl
            << std::endl
            << std::flush;
  std::exit(1);
}

template<typename T>
inline T UniformSample(const std::vector<T> &vec, std::mt19937 &rng){
  std::uniform_int_distribution<std::size_t> dist(0, vec.size() - 1);
  return vec[dist(rng)];
}
}
#endif //BRIDGE_LEARNING_RELA_UTILS_H_
