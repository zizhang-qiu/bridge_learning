//
// Created by qzz on 2023/12/16.
//
#include <iostream>

// #include "torch/torch.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_cat.h"
// #include "rela/batcher.h"

#include "playcc/alpha_mu_search.h"
#include "playcc/bridge_state_without_hidden_info.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/example_cards_ddts.h"
#include "bridge_lib/canonical_encoder.h"
#include "playcc/common_utils/file.h"
#include "playcc/pimc.h"
#include "playcc/transposition_table.h"
#include "playcc/deal_analyzer.h"
#include "playcc/rule_based_defender.h"
#include "playcc/wbridge5_trajectory_bot.h"
#include "playcc/sayc/constraints.h"
#include "playcc\sayc\hand_analyzer.h"
// #include "rlcc/belief_data_gen.h"

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);

std::vector<size_t> FindNonZeroIndices(const std::vector<int>& vec) {
  std::vector<size_t> indices;

  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] != 0) {
      indices.push_back(i);
    }
  }

  return indices;
}

enum A { a };

enum B { b };

int main(int argc, char** argv) {
  // rela::FutureReply fut(0);
  // rela::TensorDict d1 = {
  //   {"a", torch::tensor({{1, 2, 3}, {11, 22, 33}})},
  //   {"b", torch::tensor({{4, 5, 6}, {44, 55, 66}})}
  // };
  // rela::TensorDict d2 = {
  //   {"a", torch::tensor({{11, 22, 33}, {111, 222, 333}})},
  //   {"b", torch::tensor({{44, 55, 66}, {444, 555, 666}})}
  // };

  // //
  // torch::Tensor basic_indices = torch::arange(0, ble::kNumCardsPerHand) * ble::kNumPlayers;
  // std::cout << basic_indices << std::endl;
  // basic_indices.scatter_(0, torch::tensor({0, 1}), 3);
  // std::cout << basic_indices << std::endl;
  // std::cout << basic_indices << std::endl;
  // auto deal = ble::example_deals[0];
  // auto state = ConstructStateFromDeal(deal, game);
  // std::mt19937 rng(1);
  // while (state.CurrentPhase() == ble::Phase::kAuction) {
  //   const auto legal_moves = state.LegalMoves();
  //   const auto random_move = UniformSample(legal_moves, rng);
  //   state.ApplyMove(random_move);
  // }
  //
  // std::cout << torch::cuda::is_available() << std::endl;
  // torch::Device d{"cuda"};
  //
  // auto state2 = std::make_unique<ble::BridgeState>(game);
  //
  // std::cout << state2->ToString() << std::endl;

  // for (int i = 0; i < 4; ++i)
  // ApplyRandomMove(state, rng);
  // std::cout << state << std::endl;
  // const auto encoder = ble::CanonicalEncoder(game, ble::kNumTricks);
  // // const auto encoding = encoder.Encode({state});
  // // rela::utils::printVector(encoding);
  std::vector trajectory = {
      27, 46, 36, 43, 18, 22, 0, 20, 24, 2, 40, 41, 28, 16, 21, 10, 42, 32, 48,
      47, 13, 17, 3, 5, 12, 25, 34, 8, 29, 38,
      23, 4, 30, 26, 35, 19, 9, 44, 7, 51, 14, 1, 45, 15, 49, 33, 11, 50, 39,
      31, 6, 37, 52, 52, 58, 59,
      52, 61, 52, 62, 52, 64, 52, 52, 52, 39, 31, 3, 47, 41, 13, 1, 45, 21, 5,
      49, 17, 27, 16, 7, 51, 50,
      18, 2, 6, 15, 12, 22, 23, 48, 4, 24, 32, 0, 20, 28, 44, 25, 34, 37, 9, 43,
      14, 26, 11, 10, 30, 46,
      36, 33, 40, 8, 29, 38, 35, 19, 42
  };
  std::vector<std::vector<int>> trajectories = {trajectory};

  // BeliefDataGen gen{trajectories, 1, game};
  // auto data = gen.AllData("cuda");
  //
  // rela::utils::printMap(data);

  // WBridge5TrajectoryBot bot{trajectories, game};

  // auto non_zero_indices = FindNonZeroIndices(encoding);
  // rela::utils::printVector(non_zero_indices);
  //
  // std::cout << encoder.Shape()[0] << std::endl;

  // auto observation = ble::BridgeObservation(state);
  // auto encoding = encoder.Encode(observation);
  // encoding = std::vector<int>(encoding.begin(), encoding.begin() + encoder.GetAuctionTensorSize());
  // rela::utils::printVector(encoding);
  // encoding = encoder.EncodeOtherHands({state});
  // rela::utils::printVector(encoding);
  // for(const auto suit:ble::kSuitChar) {
  //   for(const auto rank:ble::kRankChar) {
  //     std::string card_string = {suit, rank};
  //     std::cout << card_string << std::endl;
  //     auto card = CardFromString(card_string);
  //     std::cout << card << std::endl;
  //   }
  // }

  // const std::array<std::vector<std::string>, ble::kNumPlayers> cards = {{
  //         {"HA", "HQ", "H8", "H7", "H5"}, {"HT", "H9", "H6"},
  //         {"HK", "HJ", "H2"}, {"H4", "H3"}}
  // };
  //
  // std::mt19937 rng(1);
  // auto state = ConstructStateFromCardStrings(cards,
  //                                            ble::default_game, rng);
  // std::cout << state << std::endl;
  //
  // std::vector<int> bids = {52, 52, 52,
  //                          ble::BidIndex(1, ble::Denomination::kClubsTrump) +
  //                          52,
  //                          52, 52, 52};
  // for (const int bid : bids) {
  //   state.ApplyMove(state.ParentGame()->GetMove(bid));
  // }
  // std::cout << state << std::endl;
  //
  // // Opening lead.
  // const ble::BridgeMove opening_lead{ble::BridgeMove::Type::kPlay,
  //                                    ble::Suit::kHeartsSuit, 3,
  //                                    ble::Denomination::kInvalidDenomination,
  //                                    -1, ble::kNotOtherCall};
  // state.ApplyMove(opening_lead);
  // std::cout << state << std::endl;
  //
  // auto rule_result = RuleOf10And12(state);
  // std::cout << rule_result.higher_cards_declarer_hold << std::endl;
  std::vector<std::string> card_strings = {"SK", "SQ", "S5", "S4", "HA", "H8",
                                           "H7", "H3", "D6", "CK", "CT", "C6",
                                           "C4"};
  ble::BridgeHand hand{};
  for (const auto& card_str : card_strings) {
    auto card = CardFromString(card_str);
    hand.AddCard(card);
  }

  std::cout << hand << std::endl;

  auto state = ble::BridgeState(game);

  while (state.IsChanceNode()) {
    state.ApplyRandomChance();
  }

  sayc::HandAnalyzer hand_analyzer{hand};

  bool is_balanced = hand_analyzer.IsBalanced();
  std::cout << std::boolalpha << is_balanced << std::endl;

  std::cout << "hcp: " << hand_analyzer.HighCardPoints() << std::endl;

  // const auto fit = sayc::OneNoTrumpOpeningConstraint.Fits(hand_analyzer,
  //   ble::BridgeObservation{state});

  // std::cout << fit << std::endl;

  auto c = sayc::LoadConstraint("balanced_hand", {});
  const auto fit3 = c->Fits(hand_analyzer, ble::BridgeObservation{state}, {});
  std::cout << fit3 << std::endl;

  auto c_20 = sayc::LoadConstraint("rule_of_20", {});
  const auto fit4 = c_20->
      Fits(hand_analyzer, ble::BridgeObservation{state}, {});
  std::cout << fit4 << std::endl;

  auto sorted_length_with_suits = hand_analyzer.GetSortedSuitLengthWithSuits();
  for (const auto& pair : sorted_length_with_suits) {
    std::cout << pair.first << ": ";
    for (const auto suit : pair.second) {
      std::cout << suit << ", ";
    }
    std::cout << std::endl;
  }
}