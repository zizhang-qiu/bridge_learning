//
// Created by qzz on 2023/12/22.
//
#include <random>

#include "playcc/transposition_table.h"
#include "playcc/bridge_state_without_hidden_info.h"
#include "playcc/utils.h"
#include "bridge_lib/utils.h"

namespace ble = bridge_learning_env;

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);
ble::BridgeStateWithoutHiddenInfo GenerateRandomState(std::mt19937 &rng, int num_play_actions) {
  const std::vector<int> deal =
      {27, 3, 32, 38, 25, 44, 4, 17, 34, 46, 42, 15, 11, 31, 41, 0, 24, 16, 7, 39, 21, 30, 33, 47, 49, 29, 5, 1, 37, 22,
       14, 8, 23, 40, 12, 43, 20, 2, 45, 28, 13, 9, 26, 18, 6, 19, 51, 10, 48, 35, 50, 36};
  ble::BridgeState state{game};
  for (const int uid : deal) {
    const auto move = game->GetChanceOutcome(uid);
    state.ApplyMove(move);
  }
  std::vector<int> bid_uid;
  bid_uid.push_back(ble::BidIndex(1, ble::kSpadesTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(3, ble::kNoTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  // Bidding
  for (const int uid : bid_uid) {
    ble::BridgeMove move = game->GetMove(uid);
    state.ApplyMove(move);
  }

  for (int i = 0; i < num_play_actions; ++i) {
    const auto legal_moves = state.LegalMoves();
    const auto move = UniformSample(legal_moves, rng);
    state.ApplyMove(move);
  }
  return ble::BridgeStateWithoutHiddenInfo(state);
}

void TranspositionTableSerializationTest() {
  TranspositionTable tt{};
  std::string sr;
  std::mt19937 rng(12);
  ble::BridgeStateWithoutHiddenInfo state;
  // No play.
  state = GenerateRandomState(rng, 0);
  ParetoFront front{};
  tt[state] = front;
//  std::cout << tt << std::endl;
  sr = tt.Serialize();
//  std::cout << sr << std::endl;

  SPIEL_CHECK_EQ(TranspositionTable::Deserialize(sr, game), tt);


  // Several cards played.
  state = GenerateRandomState(rng, 30);
  tt[state] = front;
  sr = tt.Serialize();
  std::cout << sr << std::endl;
  SPIEL_CHECK_EQ(TranspositionTable::Deserialize(sr, game), tt);

  const OutcomeVector vec1{{0, 1, 1}, {true, true, true}, {ble::BridgeMove::Type::kPlay, ble::kClubsSuit, 2, ble::kInvalidDenomination, -1,
                                                           ble::kNotOtherCall}};
  const OutcomeVector vec2{{1, 1, 0}, {true, true, true}};
  front = ParetoFront{{vec1, vec2}};
  tt[state] = front;
  sr = tt.Serialize();
  std::cout << sr << std::endl;
  SPIEL_CHECK_EQ(TranspositionTable::Deserialize(sr, game), tt);

  // All cards played.
  state = GenerateRandomState(rng, ble::kNumCards);
  std::cout << state << std::endl;
  std::cout << state.Serialize() << std::endl;
}

int main() {
  TranspositionTableSerializationTest();
}