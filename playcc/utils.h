//
// Created by qzz on 2023/10/19.
//

#ifndef BRIDGE_LEARNING_PLAYCC_UTILS_H_
#define BRIDGE_LEARNING_PLAYCC_UTILS_H_
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "bridge_lib/bridge_state.h"

#include "log_utils.h"

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

template<typename T, size_t N>
void PrintArray(std::array<T, N> arr) {
  for (const auto item : arr) {
    std::cout << item << ",";
  }
  std::cout << std::endl;
}

std::array<std::vector<ble::BridgeCard>,
           ble::kNumSuits> SplitCardsVecBySuits(const std::vector<ble::BridgeCard> &cards);

std::set<ble::Suit> GetSuitsFromCardsVec(const std::vector<ble::BridgeCard> &cards);

std::set<ble::Suit> GetSuitsFromMovesVec(const std::vector<ble::BridgeMove> &moves);

std::vector<ble::BridgeCard> ExtractCardsBySuitsFromCardsVec(const std::vector<ble::BridgeCard> &cards,
                                                             const std::set<ble::Suit> &suits);

std::vector<ble::BridgeCard> GenerateAllCardsBySuits(const std::set<ble::Suit> &suits);

std::vector<ble::BridgeMove> GetLegalMovesWithoutEquivalentCards(const ble::BridgeState &state);

std::vector<int> KeepLargestConsecutive(const std::vector<int> &input);

std::vector<int> FindSetBitPositions(int decimalNumber);

std::vector<ble::BridgeMove> GetMovesFromFutureTricks(const futureTricks& fut);

template<typename T, typename RNG>
T UniformSample(const std::vector<T> &vec, RNG &rng) {
  // Check if the vector is not empty
  if (vec.empty()) {
    throw std::out_of_range("Vector is empty");
  }

  // Use the provided RNG to generate an index uniformly
  std::uniform_int_distribution<std::size_t> dist(0, vec.size() - 1);

  // Return the sampled element
  return vec[dist(rng)];
}

template <typename T>
void MoveItemToFirst(std::vector<T>& vec, bool (*constraint_function)(T)) {
  auto it = std::find_if(vec.begin(), vec.end(), constraint_function);

  if (it != vec.end()) {
    // Rotate the vector so that the item satisfying the constraint is at the beginning
    std::rotate(vec.begin(), it, it + 1);
  }
}
#endif // BRIDGE_LEARNING_PLAYCC_UTILS_H_
