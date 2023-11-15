//
// Created by qzz on 2023/10/19.
//

#ifndef BRIDGE_LEARNING_PLAYCC_UTILS_H_
#define BRIDGE_LEARNING_PLAYCC_UTILS_H_
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include "bridge_lib/bridge_state.h"
namespace ble = bridge_learning_env;

std::vector<ble::BridgeHistoryItem> GetPlayHistory(const std::vector<ble::BridgeHistoryItem> &history);

template<typename T>
std::vector<T> FlattenVector(const std::vector<std::vector<T>> &nested_vector) {
  std::vector<T> flattened_vector;

  for (const auto &inner_vector : nested_vector) {
    flattened_vector.insert(flattened_vector.end(), inner_vector.begin(), inner_vector.end());
  }
  flattened_vector.shrink_to_fit();
  return flattened_vector;
}

std::array<int, ble::kNumCards> HandsToCardIndices(const std::vector<ble::BridgeHand> &hands);

std::unique_ptr<ble::BridgeState> ConstructStateFromDeal(std::array<int, ble::kNumCards> deal,
                                                         const std::shared_ptr<ble::BridgeGame> &game);

std::unique_ptr<ble::BridgeState> ConstructStateFromDeal(std::array<int, ble::kNumCards> deal,
                                                         const std::shared_ptr<ble::BridgeGame> &game,
                                                         const ble::BridgeState &original_state);

deal StateToDeal(const ble::BridgeState &state);

std::vector<int> MovesToUids(const std::vector<ble::BridgeMove>& moves, const ble::BridgeGame& game);

#endif //BRIDGE_LEARNING_PLAYCC_UTILS_H_
