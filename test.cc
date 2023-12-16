//
// Created by qzz on 2023/12/16.
//
#include "bridge_lib/bridge_state.h"
#include "playcc/utils.h"

namespace ble = bridge_learning_env;

int main() {
//  std::vector<int> input = {1, 2, 3, 5, 6, 7, 9};
//
//  std::vector<int> result = KeepLargestConsecutive(input);
//
//  // Display the result
//  for (int value : result) {
//    std::cout << value << " ";
//  }
  auto cards = GenerateAllCardsBySuits({ble::kClubsSuit, ble::kDiamondsSuit});
//  for(const auto& card:cards){
//    std::cout << card.ToString() << std::endl;
//  }
  std::cout << cards << std::endl;

  return 0;
}