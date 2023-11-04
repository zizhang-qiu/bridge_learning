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

std::vector<ble::BridgeHistoryItem> GetPlayHistory(const std::vector<ble::BridgeHistoryItem> &history) {

  std::vector<ble::BridgeHistoryItem> play_history;
  for (const auto item : history) {
    if (item.move.MoveType() == ble::BridgeMove::Type::kPlay) {
      play_history.push_back(item);
    }
  }
  return play_history;
}

template<typename T>
std::vector<T> FlattenVector(const std::vector<std::vector<T>> &nested_vector) {
  std::vector<T> flattened_vector;

  for (const auto &inner_vector : nested_vector) {
    flattened_vector.insert(flattened_vector.end(), inner_vector.begin(), inner_vector.end());
  }
  flattened_vector.shrink_to_fit();
  return flattened_vector;
}

std::array<int, ble::kNumCards> HandsToCardIndices(const std::vector<ble::BridgeHand> &hands) {
  std::array<int, ble::kNumCards> res{};
  for (int i = 0; i < ble::kNumCardsPerHand; ++i) {
    for (int pl = 0; pl < ble::kNumPlayers; ++pl) {
      res[i * ble::kNumPlayers + pl] = hands[pl].Cards()[i].Index();
    }
  }
  return res;
}

std::unique_ptr<ble::BridgeState> ConstructStateFromDeal(const std::array<int, ble::kNumCards> deal,
                                                         const std::shared_ptr<ble::BridgeGame> &game) {
  auto state = std::make_unique<ble::BridgeState>(game);
  for (int i = 0; i < ble::kNumCards; ++i) {
    ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
    state->ApplyMove(move);
  }
  return state;
}

std::unique_ptr<ble::BridgeState> ConstructStateFromDeal(const std::array<int, ble::kNumCards> deal,
                                                         const std::shared_ptr<ble::BridgeGame> &game,
                                                         const ble::BridgeState &original_state) {
  auto state = std::make_unique<ble::BridgeState>(game);
  for (int i = 0; i < ble::kNumCards; ++i) {
    ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
    state->ApplyMove(move);
  }
  const auto &history = original_state.History();
  for (int i = ble::kNumCards; i < history.size(); ++i) {
    ble::BridgeMove move = history[i].move;
    state->ApplyMove(move);
  }
  return state;
}

deal StateToDeal(const ble::BridgeState &state) {
  if (state.CurrentPhase() != ble::Phase::kPlay) {
    std::cerr << "Should be play phase." << std::endl;
    std::abort();
  }
  deal dl{};
  const ble::Contract contract = state.GetContract();
  dl.trump = ble::DenominationToDDSStrain(contract.denomination);
//  std::cout << "dl.trump: " << dl.trump << std::endl;
  const ble::Trick current_trick = state.CurrentTrick();
  dl.first = current_trick.Leader() != ble::kInvalidPlayer ? current_trick.Leader() :
             state.IsDummyActing() ? state.GetDummy() : state.CurrentPlayer();
//  std::cout << "dl.first: " << dl.first << std::endl;

  const auto &history = state.History();
  std::vector<ble::BridgeHistoryItem> play_history;
  for (const auto move : history) {
    if (move.move.MoveType() == ble::BridgeMove::Type::kPlay) {
      play_history.push_back(move);
    }
  }

  int num_tricks_played = static_cast<int>(play_history.size()) / ble::kNumPlayers;
  int num_card_played_current_trick = static_cast<int>(play_history.size()) - num_tricks_played * ble::kNumPlayers;
  memset(dl.currentTrickSuit, 0, 3 * sizeof(dl.currentTrickSuit));
  memset(dl.currentTrickRank, 0, 3 * sizeof(dl.currentTrickSuit));
  for (int i = 0; i < num_card_played_current_trick; ++i) {
    ble::BridgeHistoryItem item = play_history[num_tricks_played * ble::kNumPlayers + i];
    dl.currentTrickSuit[i] = ble::SuitToDDSSuit(item.suit);
    dl.currentTrickRank[i] = ble::RankToDDSRank(item.rank);
  }

//  std::cout << "currentTrickSuit: ";
//  for(int i : dl.currentTrickSuit){
//    std::cout << i << std::endl;
//  }
//
//  std::cout << "currentTrickRank: ";
//  for(int i : dl.currentTrickRank){
//    std::cout << i << std::endl;
//  }

  const auto &hands = state.Hands();
  for (ble::Player pl : ble::kAllSeats) {
    for (const auto card : hands[pl].Cards()) {
      dl.remainCards[pl][SuitToDDSSuit(card.CardSuit())] += 1
          << (2 + card.Rank());
    }
  }

//  futureTricks fut{};
//  const int res = SolveBoard(
//      dl,
//      /*target=*/-1,
//      /*solutions=*/1,
//      /*mode=*/2,
//      &fut,
//      /*threadIndex=*/0);
//  if (res != RETURN_NO_FAULT){
//    char error_message[80];
//    ErrorMessage(res, error_message);
//    std::cerr << "double dummy solver: " << error_message << std::endl;
//    std::exit(1);
//  }
  return dl;
}

#endif //BRIDGE_LEARNING_PLAYCC_UTILS_H_
