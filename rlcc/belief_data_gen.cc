//
// Created by qzz on 2024/1/18.
//

#include "belief_data_gen.h"

std::vector<int> NoPlayTrajectory(const std::vector<int>& trajectory) {
  if (const size_t size = trajectory.size(); size == ble::kNumCards + ble::kNumPlayers) {
    return trajectory;
  }
  return std::vector<int>(trajectory.begin(), trajectory.end() - ble::kNumCards);
}
