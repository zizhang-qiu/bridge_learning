//
// Created by qzz on 2023/11/2.
//

#ifndef BRIDGE_LEARNING_PLAYCC_VECTOR_UTILS_H_
#define BRIDGE_LEARNING_PLAYCC_VECTOR_UTILS_H_
#include <vector>
#include <iostream>
#include <sstream>
template<typename T>
void CheckVectorSize(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  if (lhs.size() != rhs.size()) {
    std::cerr << "Vector size not equal, " << lhs.size() << " vs " << rhs.size() << "." << std::endl;
  }
}

template<typename T>
std::vector<T> VectorMax(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  CheckVectorSize(lhs, rhs);
  size_t size = lhs.size();
  std::vector<T> res(size);
  for (size_t i = 0; i < size; ++i) {
    res[i] = std::max(lhs[i], rhs[i]);
  }
  return res;
}

template<typename T>
std::vector<T> VectorMin(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  CheckVectorSize(lhs, rhs);
  size_t size = lhs.size();
  std::vector<T> res(size);
  for (size_t i = 0; i < size; ++i) {
    res[i] = std::min(lhs[i], rhs[i]);
  }
  return res;
}

template<typename T>
bool VectorGreaterEqual(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  CheckVectorSize(lhs, rhs);
  size_t size = lhs.size();
  for (size_t i = 0; i < size; ++i) {
    if (lhs[i] < rhs[i]) {
      return false;
    }
  }
  return true;
}

template<typename T>
bool VectorDominate(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  CheckVectorSize(lhs, rhs);
  size_t size = lhs.size();
  for (size_t i = 0; i < size; ++i) {
    if (lhs[i] < rhs[i]) {
      return false;
    }
  }
  return true;
}

template<typename T>
std::string VectorToString(const std::vector<T> &vec) {
  std::ostringstream oss;

  // Use an iterator to traverse the vector and append elements to the string stream
  for (auto it = vec.begin(); it != vec.end(); ++it) {
    oss << *it;

    // Add a comma and space if it's not the last element
    if (std::next(it) != vec.end()) {
      oss << ", ";
    }
  }

  return oss.str();
}

// The dot product of 2 vectors.
template<typename T>
std::vector<T> VectorProduct(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  CheckVectorSize(lhs, rhs);
  std::vector<T> res(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    res[i] = lhs[i] * rhs[i];
  }
  return res;
}
#endif //BRIDGE_LEARNING_PLAYCC_VECTOR_UTILS_H_
