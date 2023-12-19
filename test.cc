//
// Created by qzz on 2023/12/16.
//
#include <iostream>
#include <vector>
#include <algorithm>

template <typename T, typename Predicate>
void MoveItemToFirst(std::vector<T>& vec, Predicate constraint_function) {
  auto it = std::find_if(vec.begin(), vec.end(), constraint_function);

  if (it != vec.end()) {
    // Rotate the vector so that the item satisfying the constraint is at the beginning
    std::rotate(vec.begin(), it, it + 1);
  }
}

// Example usage:
int main() {
  std::vector<int> yourVector = {2, 8, 4, 6, 1, 5};

  // Using a lambda function as the constraint
  MoveItemToFirst(yourVector, [](int x) { return x ==6; });  // Move even numbers to the first position

  // Print the vector to verify the result
  for (const auto& item : yourVector) {
    std::cout << item << " ";
  }

  return 0;
}