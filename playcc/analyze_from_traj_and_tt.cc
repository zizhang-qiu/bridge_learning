//
// Created by qzz on 2023/12/23.
//
#include "absl/strings/str_format.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include "playcc/alpha_mu_bot.h"
#include "playcc/alpha_mu_search.h"
#include "playcc/bridge_state_without_hidden_info.h"
#include "bridge_lib/bridge_state.h"
#include "playcc/file.h"
#include "playcc/pimc.h"
#include "playcc/transposition_table.h"
#include "playcc/deal_analyzer.h"

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);

std::vector<int> TrajectoryFromFile(const std::string &path) {
  file::File traj_file{path, "r"};
  std::vector<std::string> lines = absl::StrSplit(traj_file.ReadContents(), ",");
  std::vector<int> trajectory;
  for (auto &line : lines) {
    if (line.empty())continue;
    int uid = std::stoi(line);
    trajectory.push_back(uid);
  }
  return trajectory;
}

int main() {
//  const std::string file_dir = R"(D:\Projects\bridge\analysis\0)";
//  const std::string tt_path = absl::StrCat(file_dir, "/", "tt.txt");
//  const std::string traj_path = absl::StrCat(file_dir, "/", "traj.txt");
//
//  auto tt = TranspositionTableFromFile(tt_path, game);
//  auto traj = TrajectoryFromFile(traj_path);
  std::vector<int> traj = {
      20, 12, 24, 0, 43, 33, 48, 32, 6, 23, 21, 39, 26, 16, 4, 29, 15, 50, 18, 30, 11, 51, 2, 17, 8, 35, 45, 38, 31, 25,
      44, 22, 37, 47, 34, 19, 42, 10, 46, 13, 28, 1, 7, 14, 36, 9, 49, 5, 3, 40, 41, 27, 52, 52, 69, 52, 52, 52,
      19, 31,
      51, 7, 16, 48, 0, 8, 21, 5, 37, 9,
//      20, 12, 44, 32, 49, 17, 15, 1, 45, 13, 11, 25, 41, 29, 36, 33, 2, 30, 42, 50,
//      40, 24, 14, 28, 47, 18, 27, 3, 10, 34, 38, 6, 22, 26, 35, 46, 4, 39, 43, 23
  };

  int num_worlds = 20;
  const AlphaMuConfig alpha_mu_cfg{2,
                                   num_worlds,
                                   false,
                                   true,
                                   true,
                                   true};
  auto resampler = std::make_shared<UniformResampler>(1);
  auto alpha_mu_bot = VanillaAlphaMuBot(resampler, alpha_mu_cfg);
  const PIMCConfig pimc_cfg{num_worlds, false};
  auto pimc_bot = PIMCBot(resampler, pimc_cfg);

  auto state = ConstructStateFromTrajectory(traj, game);
  int seed = 6989;
//  int seed = 42;
  resampler->ResetWithParams({{"seed", std::to_string(seed)}});
  while (!state.IsTerminal()) {
    if (IsActingPlayerDeclarerSide(state)) {
      std::cout << state << std::endl;
      auto dds_moves = DealAnalyzer::DDSMoves(state);
      std::cout << "dds moves:\n" << VectorToString(dds_moves) << std::endl;

//  alpha_mu_bot.SetTT(tt);

      auto move = alpha_mu_bot.Act(state);
      std::cout << "alphamu move: " << move << std::endl;
      resampler->ResetWithParams({{"seed", std::to_string(seed)}});
      auto pimc_move = pimc_bot.Act(state);
      std::cout << "pimc move: " << pimc_move << std::endl;
      state.ApplyMove(move);
    } else {
//      resampler->ResetWithParams({{"seed", std::to_string(seed)}});
      auto pimc_move = pimc_bot.Act(state);
      state.ApplyMove(pimc_move);
    }
  }
  std::cout << state << std::endl;

}