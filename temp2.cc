//
// Created by qzz on 2023/12/22.
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
int main() {
  std::vector<int> trajectory = {
      32, 36, 21, 5, 40, 3, 19, 9, 18, 13, 39, 38, 0, 4, 16, 12, 23, 10, 28, 51, 7, 27, 46, 45, 50, 31, 15, 6, 8, 1, 37,
      2, 42, 14, 20, 43, 48, 29, 44, 11, 17, 33, 41, 25, 49, 35, 26, 24, 30, 22, 47, 34, 52, 52, 69, 52, 52, 52, 12, 0,
      36, 44
  };
  ble::BridgeState state{game};
    for (int i = 0; i < ble::kNumCards; ++i) {
    state.ApplyMove(game->GetChanceOutcome(trajectory[i]));
  }
  for (int i = 52; i < trajectory.size(); ++i) {
    state.ApplyMove(game->GetMove(trajectory[i]));
  }

  std::cout << state << std::endl;

  int num_worlds = 40;
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

//  auto tt = TranspositionTableFromFile("D:/Projects/bridge/analysis/tt.txt", game);
//  alpha_mu_bot.SetTT(tt);

  const auto move = alpha_mu_bot.Act(state);
  std::cout << move << std::endl;
  auto dds_moves = DealAnalyzer::DDSMoves(state);
  std::cout << absl::StrFormat("The dds give moves: %s",VectorToString(dds_moves)) << std::endl;
}