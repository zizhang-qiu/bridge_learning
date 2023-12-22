//
// Created by qzz on 2023/12/16.
//
#include <iostream>

#include "absl/strings/str_format.h"

#include "playcc/alpha_mu_search.h"
#include "playcc/bridge_state_without_hidden_info.h"
#include "bridge_lib/bridge_state.h"
#include "playcc/file.h"
#include "playcc/pimc.h"
#include "playcc/transposition_table.h"
#include "playcc/deal_analyzer.h"


const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);
int main(int argc, char **argv) {
//  std::vector<int> trajectory = {
//      27, 3, 32, 38, 25, 44, 4, 17, 34, 46, 42, 15, 11, 31, 41, 0, 24, 16, 7, 39, 21, 30, 33, 47, 49, 29, 5, 1, 37, 22,
//      14, 8, 23, 40, 12, 43, 20, 2, 45, 28, 13, 9, 26, 18, 6, 19, 51, 10, 48, 35, 50, 36, 52, 52, 69, 52, 52, 52, 0, 48,
//      16, 4, 13, 9, 33, 1, 5, 17, 37, 29, 34, 2, 14, 38, 47, 11, 3, 51, 45, 8, 25, 44, 41, 18, 49, 22, 21, 19, 7, 10, 6,
//      46, 50, 28, 42, 36, 24, 30
//  };
  std::mt19937 rng;
  ble::BridgeState state{game};
  const ble::Contract contract{3, ble::kNoTrump, ble::kUndoubled, ble::kSouth};

  while (true) {
    state = ble::BridgeState(game);
    const auto deal = ble::Permutation(ble::kNumCards, rng);
    for (int i = 0; i < ble::kNumCards; ++i) {
      ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
      state.ApplyMove(move);
    }
    std::vector<int> bid_uid;
    bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
    bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
    bid_uid.push_back(ble::BidIndex(contract.level, contract.denomination) + ble::kBiddingActionBase);
    bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
    bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
    bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
    // Bidding
    for (const int uid : bid_uid) {
      ble::BridgeMove move = game->GetMove(uid);
      state.ApplyMove(move);
    }
    auto ddt = state.DoubleDummyResults();
    if (ddt[state.GetContract().denomination][state.GetContract().declarer] > (6 + contract.level)){
      break;
    }
  }
//
//  for (int i = 0; i < ble::kNumCards; ++i) {
//    state.ApplyMove(game->GetChanceOutcome(trajectory[i]));
//  }
//  for (int i = 52; i < trajectory.size(); ++i) {
//    state.ApplyMove(game->GetMove(trajectory[i]));
//  }

  std::cout << state << std::endl;

  int num_worlds = 20;
  const AlphaMuConfig alpha_mu_cfg{3,
                                   num_worlds,
                                   false,
                                   true,
                                   true,
                                   true};
  auto resampler = std::make_shared<UniformResampler>(1);
  auto alpha_mu_bot = AlphaMuBot(resampler, alpha_mu_cfg);
  const PIMCConfig pimc_cfg{num_worlds, false};
  auto pimc_bot = PIMCBot(resampler, pimc_cfg);
  auto analyzer = DealAnalyzer("D:/Projects/bridge/analysis");
  analyzer.Analyze(state, alpha_mu_bot, pimc_bot);
//  auto tt = TranspositionTableFromFile("D:/Projects/bridge/", game);
//  alpha_mu_bot.SetTT(tt);
//  const auto move = alpha_mu_bot.Act(state);
//  std::cout << move << std::endl;
//  auto dds_moves = DealAnalyzer::DDSMoves(state);
//  std::cout << absl::StrFormat("The dds give moves: %s",VectorToString(dds_moves)) << std::endl;

}