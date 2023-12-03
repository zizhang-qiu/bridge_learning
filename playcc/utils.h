//
// Created by qzz on 2023/10/19.
//

#ifndef BRIDGE_LEARNING_PLAYCC_UTILS_H_
#define BRIDGE_LEARNING_PLAYCC_UTILS_H_
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include "log_utils.h"
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

ble::BridgeState ConstructStateFromDeal(std::array<int, ble::kNumCards> deal,
                                        const std::shared_ptr<ble::BridgeGame> &game);

ble::BridgeState ConstructStateFromDeal(const std::array<int, ble::kNumCards> &deal,
                                        const std::shared_ptr<ble::BridgeGame> &game,
                                        const ble::BridgeState &original_state);

// Convert a BridgeState to DDS deal
deal StateToDDSDeal(const ble::BridgeState &state);

std::vector<int> MovesToUids(const std::vector<ble::BridgeMove> &moves, const ble::BridgeGame &game);

bool IsActingPlayerDeclarerSide(const ble::BridgeState &state);
#endif // BRIDGE_LEARNING_PLAYCC_UTILS_H_
