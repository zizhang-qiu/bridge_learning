//
// Created by qzz on 2023/12/22.
//
#include <iostream>
#include <algorithm>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_cat.h"

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
//  std::vector<int> trajectory = {
//      32, 36, 21, 5, 40, 3, 19, 9, 18, 13, 39, 38, 0, 4, 16, 12, 23, 10, 28, 51, 7, 27, 46, 45, 50, 31, 15, 6, 8, 1, 37,
//      2, 42, 14, 20, 43, 48, 29, 44, 11, 17, 33, 41, 25, 49, 35, 26, 24, 30, 22, 47, 34, 52, 52, 69, 52, 52, 52, 12, 0,
//      36, 44
//  };
//  ble::BridgeState state{game};
//    for (int i = 0; i < ble::kNumCards; ++i) {
//    state.ApplyMove(game->GetChanceOutcome(trajectory[i]));
//  }
//  for (int i = 52; i < trajectory.size(); ++i) {
//    state.ApplyMove(game->GetMove(trajectory[i]));
//  }
//
//  std::cout << state << std::endl;
//
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
//
////  auto tt = TranspositionTableFromFile("D:/Projects/bridge/analysis/tt.txt", game);
////  alpha_mu_bot.SetTT(tt);
//
//  const auto move = alpha_mu_bot.Act(state);
//  std::cout << move << std::endl;
//  auto dds_moves = DealAnalyzer::DDSMoves(state);
//  std::cout << absl::StrFormat("The dds give moves: %s",VectorToString(dds_moves)) << std::endl;

  std::vector<int> traj1 =
      {36, 41, 46, 3, 47, 6, 44, 1, 0, 20, 49, 16, 30, 18, 33, 29, 23, 32, 27, 37, 9, 28, 14, 48, 21, 24, 2, 7, 8, 51,
       45, 40, 22, 11, 19, 35, 31, 10, 50, 26, 42, 38, 34, 5, 43, 15, 17, 39, 12, 13, 25, 4, 52, 52, 69, 52, 52, 52, 48,
       0, 32, 44, 40, 12, 28, 17, 4, 8, 24, 19, 13, 49, 5, 21, 27, 7, 47, 15, 9, 41, 45, 1, 2, 26, 42, 10, 36, 20, 25,
       16, 30, 6, 50, 29, 14, 37, 22, 38, 51, 33, 3, 31, 18, 46, 39, 23, 34, 35, 43, 11};
  std::vector<int> traj2 =
      {36, 41, 46, 3, 47, 6, 44, 1, 0, 20, 49, 16, 30, 18, 33, 29, 23, 32, 27, 37, 9, 28, 14, 48, 21, 24, 2, 7, 8, 51,
       45, 40, 22, 11, 19, 35, 31, 10, 50, 26, 42, 38, 34, 5, 43, 15, 17, 39, 12, 13, 25, 4, 52, 52, 69, 52, 52, 52, 48,
       0, 32, 44, 40, 12, 28, 17, 4, 8, 24, 25, 20, 33, 16, 36, 9, 13, 49, 5, 2, 26, 42, 10, 30, 6, 50, 1, 19, 39, 47,
       51, 41, 45, 37, 21, 27, 7, 43, 15, 22, 18, 34, 29, 46, 3, 31, 38, 14, 35, 23, 11};
  auto it = std::mismatch(traj1.begin(), traj1.end(), traj2.begin(), traj2.end());
  int index = static_cast<int>(std::distance(traj1.begin(), it.first));
  std::cout << index << std::endl;

  auto state1 = ConstructStateFromTrajectory(std::vector<int>(traj1.begin(), traj1.begin() + ble::kNumCards + 6), game);
//  auto state2 = ConstructStateFromTrajectory(std::vector<int>(traj2.begin(), it.second), game);
//
//  std::cout << state1 << "\n" << state2 << std::endl;
//
//  std::cout << "original move: " << game->GetMove(traj1[index]) << std::endl;
//
//  auto move = alpha_mu_bot.Act(state1);
//  std::cout << move << std::endl;
//
//  auto dds_moves = DealAnalyzer::DDSMoves(state1);
//  std::cout << "dds moves:\n" << VectorToString(dds_moves) << std::endl;
  auto analyzer = DealAnalyzer(absl::StrCat("D:/Projects/bridge/analysis", "/", 0));
  analyzer.Analyze(state1, alpha_mu_bot, pimc_bot);



}