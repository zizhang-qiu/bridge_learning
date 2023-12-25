//
// Created by qzz on 2023/12/22.
//
#include <iostream>
#include <algorithm>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_cat.h"

#include "playcc/alpha_mu_search.h"
#include "playcc/bridge_state_without_hidden_info.h"
#include "bridge_lib/bridge_state.h"
#include "playcc/file.h"
#include "playcc/pimc.h"
#include "playcc/transposition_table.h"
#include "playcc/deal_analyzer.h"

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);

int main() {
  std::vector<int> myVector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // Define the constraint (for example, remove even numbers)
  auto isEven = [](int n) { return n % 2 == 0; };

  // Use the erase-remove idiom
  myVector.erase(std::remove_if(myVector.begin(), myVector.end(), isEven), myVector.end());

  // Display the modified vector
  for (int num : myVector) {
    std::cout << num << " ";
  }

  return 0;
}
