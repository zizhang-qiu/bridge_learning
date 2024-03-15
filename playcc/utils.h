//
// Created by qzz on 2023/10/19.
//

#ifndef PLAYCC_UTILS_H
#define PLAYCC_UTILS_H
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>


#include "bridge_lib/bridge_game.h"
#include "bridge_lib/bridge_utils.h"
#include "bridge_lib/utils.h"

#include "bridge_lib/bridge_state.h"

#include "bridge_lib/example_cards_ddts.h"
#include "common_utils/log_utils.h"

namespace ble = bridge_learning_env;

std::vector<ble::BridgeHistoryItem> GetPlayHistory(
    const std::vector<ble::BridgeHistoryItem>& history);

template <typename T>
std::vector<T> FlattenVector(const std::vector<std::vector<T>>& nested_vector) {
  std::vector<T> flattened_vector;

  for (const auto& inner_vector : nested_vector) {
    flattened_vector.insert(flattened_vector.end(), inner_vector.begin(),
                            inner_vector.end());
  }
  flattened_vector.shrink_to_fit();
  return flattened_vector;
}

std::array<int, ble::kNumCards> HandsToCardIndices(
    const std::vector<ble::BridgeHand>& hands);

template <typename Container>
ble::BridgeState ConstructStateFromDeal(
    const Container& deal, const std::shared_ptr<ble::BridgeGame>& game) {
  SPIEL_CHECK_EQ(deal.size(), ble::kNumCards);
  auto state = ble::BridgeState(game);
  for (int i = 0; i < ble::kNumCards; ++i) {
    ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
    state.ApplyMove(move);
  }
  return state;
}

template <typename Container>
ble::BridgeState ConstructStateFromDealAndOriginalState(
    const Container& deal, const std::shared_ptr<ble::BridgeGame>& game,
    const ble::BridgeState& original_state) {
  auto state = ble::BridgeState(game);
  for (int i = 0; i < ble::kNumCards; ++i) {
    ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
    state.ApplyMove(move);
  }
  const auto& history = original_state.History();
  for (int i = ble::kNumCards; i < history.size(); ++i) {
    ble::BridgeMove move = history[i].move;
    state.ApplyMove(move);
  }
  return state;
}

ble::BridgeState ConstructStateFromTrajectory(
    const std::vector<int>& trajectory,
    const std::shared_ptr<ble::BridgeGame>& game);

// Convert a BridgeState to DDS deal
deal StateToDDSDeal(const ble::BridgeState& state);

std::vector<int> MovesToUids(const std::vector<ble::BridgeMove>& moves,
                             const ble::BridgeGame& game);

bool IsActingPlayerDeclarerSide(const ble::BridgeState& state);

template <typename T, size_t N>
void PrintArray(std::array<T, N> arr) {
  for (const auto item : arr) {
    std::cout << item << ",";
  }
  std::cout << std::endl;
}

std::array<std::vector<ble::BridgeCard>, ble::kNumSuits> SplitCardsVecBySuits(
    const std::vector<ble::BridgeCard>& cards);

std::set<ble::Suit> GetSuitsFromCardsVec(
    const std::vector<ble::BridgeCard>& cards);

std::set<ble::Suit> GetSuitsFromMovesVec(
    const std::vector<ble::BridgeMove>& moves);

std::vector<ble::BridgeCard> ExtractCardsBySuitsFromCardsVec(
    const std::vector<ble::BridgeCard>& cards,
    const std::set<ble::Suit>& suits);

std::vector<ble::BridgeCard> GenerateAllCardsBySuits(
    const std::set<ble::Suit>& suits);

std::vector<ble::BridgeMove> GetLegalMovesWithoutEquivalentCards(
    const ble::BridgeState& state);

std::vector<int> KeepLargestConsecutive(const std::vector<int>& input);

std::vector<int> FindSetBitPositions(int decimalNumber);

std::vector<ble::BridgeMove> GetMovesFromFutureTricks(const futureTricks& fut);

void ApplyRandomMove(ble::BridgeState& state, std::mt19937& rng);

template <typename T>
T UniformSample(const std::vector<T>& vec, std::mt19937& rng) {
  // Check if the vector is not empty
  if (vec.empty()) {
    throw std::out_of_range("Vector is empty");
  }

  if (vec.size() == 1) {
    return vec[0];
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

ble::BridgeCard CardFromString(const std::string& card_string);

ble::BridgeHand ConstructHandFromCardStrings(
    const std::vector<std::string>& card_strings);

// Construct a Brdige state from strings of cards in four players' hand
// e.g. {{"H5"},{"H10","H9","H6"},{"HK","HJ","H2"},{}}
// Usually used for debugging algorithms.
ble::BridgeState ConstructStateFromCardStrings(
    const std::array<std::vector<std::string>, ble::kNumPlayers>& cards,
    const std::shared_ptr<ble::BridgeGame>& game, std::mt19937& rng);

ble::BridgeState SampleRandomStateFromGivenCards(
    std::array<std::vector<ble::BridgeCard>, ble::kNumPlayers> known_cards);

inline void Test(int n = 100) {
  SetMaxThreads(0);
  boards bo;
  bo.noOfBoards = n;
  solvedBoards solved;
  std::mt19937 rng;

  for (int i = 0; i < n; ++i) {
    // auto cur_deal = ble::example_deals[i];
    auto cur_deal = ble::Permutation(ble::kNumCards);
    auto state = ConstructStateFromDeal(cur_deal, ble::default_game);

    for(const auto uid:{55, 52, 52, 52}){
      auto move = ble::default_game->GetMove(uid);
      state.ApplyMove(move);
    }
    SPIEL_CHECK_FALSE(state.IsTerminal());
    ::deal dl = StateToDDSDeal(state);

    bo.deals[i] = dl;
    bo.mode[i] = 2;
    bo.solutions[i] = 1;
    bo.target[i] = -1;
  }
  std::chrono::high_resolution_clock::time_point st, ed;
  st = std::chrono::high_resolution_clock::now();
  ::futureTricks fut;
  for (int i = 0; i < n; ++i) {
    auto dl = bo.deals[i];
    const int res = SolveBoard(dl, -1, 1, 2, &fut, 0);
    char line[80];
    if (res != RETURN_NO_FAULT) {
      ErrorMessage(res, line);
      printf("DDS error: %s\n", line);
    }
    // std::cout << "Solve board done." << std::endl;
  }
  ed = std::chrono::high_resolution_clock::now();
  std::cout << "Duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>((ed - st))
                   .count()
            << std::endl;

  // SolveAllBoardsBin(&bo, &solved);
  st = std::chrono::high_resolution_clock::now();
  SolveAllBoardsBin(&bo, &solved);
  ed = std::chrono::high_resolution_clock::now();

  std::cout << "Solve all boards done!" << std::endl;
  std::cout << "Duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>((ed - st))
                   .count()
            << std::endl;
}

// Custom hash function for std::vector<int>
struct VectorHash {
  std::size_t operator()(const std::vector<int>& vec) const {
    std::hash<int> hasher;
    std::size_t hash_value = 0;

    for (const int& element : vec) {
      hash_value ^=
          hasher(element) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
    }

    return hash_value;
  }
};
#endif /* PLAYCC_UTILS_H */
