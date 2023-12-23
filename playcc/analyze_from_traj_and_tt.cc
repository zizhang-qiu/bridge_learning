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
  const std::string file_dir = R"(D:\Projects\bridge\analysis\0)";
  const std::string tt_path = absl::StrCat(file_dir, "/", "tt.txt");
  const std::string traj_path = absl::StrCat(file_dir, "/", "traj.txt");

  auto tt = TranspositionTableFromFile(tt_path, game);
  auto traj = TrajectoryFromFile(traj_path);

  int num_worlds = 80;
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

  auto state = ConstructStateFromTrajectory(traj, game);

  auto dds_moves = DealAnalyzer::DDSMoves(state);
  std::cout << "dds moves:\n" << VectorToString(dds_moves) << std::endl;

//  alpha_mu_bot.SetTT(tt);
  resampler->ResetWithParams({{"seed", std::to_string(3)}});
  auto move = alpha_mu_bot.Act(state);
  std::cout << move << std::endl;
  resampler->ResetWithParams({{"seed", std::to_string(3)}});
  auto pimc_move = pimc_bot.Act(state);
  std::cout << pimc_move << std::endl;

}