//
// Created by qzz on 2023/12/12.
//
#include <chrono>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "bridge_lib/bridge_scoring.h"
#include "third_party/cxxopts/include/cxxopts.hpp"

#include "alpha_mu_bot.h"
#include "dds_bot.h"
#include "logger.h"
#include "pimc.h"
#include "stat_manager.h"

namespace ble = bridge_learning_env;

struct RunningAverage {
  double sum = 0.0;
  int count = 0;

  void AddNumber(double number) {
    sum += number;
    count++;
  }

  double GetAverage() const {
    if (count == 0) {
      return 0.0; // Avoid division by zero
    }
    return sum / count;
  }
};

ble::Contract ParseContractFromString(std::string contract_string) {
  int contract_level = std::stoi(contract_string.substr(0, 1));
  absl::AsciiStrToUpper(&contract_string);
  ble::Denomination denomination;
  SPIEL_CHECK_GE(contract_level, 1);
  SPIEL_CHECK_LE(contract_level, 7);
  if (contract_string.size() == 2) {
    switch (contract_string[1]) {
      case 'C':
        denomination = ble::Denomination::kClubsTrump;
        return {contract_level, denomination, ble::kUndoubled, ble::kSouth};
      case 'D':
        denomination = ble::Denomination::kDiamondsTrump;
        return {contract_level, denomination, ble::kUndoubled, ble::kSouth};
      case 'H':
        denomination = ble::Denomination::kHeartsTrump;
        return {contract_level, denomination, ble::kUndoubled, ble::kSouth};
      case 'S':
        denomination = ble::Denomination::kSpadesTrump;
        return {contract_level, denomination, ble::kUndoubled, ble::kSouth};
      default:
        SpielFatalError(absl::StrCat("Invalid contract string: ", contract_string));
    }
  }
  if (contract_string.size() != 3 || contract_string.substr(1) != "NT") {
    SpielFatalError(absl::StrCat("Invalid contract string: ", contract_string));
  }
  denomination = ble::Denomination::kNoTrump;
  return {contract_level, denomination, ble::kUndoubled, ble::kSouth};
}

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);

ble::BridgeState ConstructRandomState(std::mt19937 &rng, const ble::Contract &contract) {
  auto state = ble::BridgeState(game);
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
  return state;
}

std::shared_ptr<PlayBot> CreatePlayer(const std::string &str,
                                      const PIMCConfig &pimc_cfg,
                                      const AlphaMuConfig &alpha_mu_cfg,
                                      std::shared_ptr<Resampler> &resampler) {
  if (str == "alpha_mu") {
    return std::make_shared<AlphaMuBot>(resampler, alpha_mu_cfg);
  }
  if (str == "pimc") {
    return std::make_shared<PIMCBot>(resampler, pimc_cfg);
  }
  if (str == "dds") {
    return std::make_shared<DDSBot>();
  }
  std::cerr << absl::StrFormat("The algorithm %s is not supported, supported algorithms are alpha_mu, pimc and dds.")
            << std::endl;
  std::exit(1);
}

std::tuple<std::shared_ptr<PlayBot>, std::shared_ptr<PlayBot>, std::shared_ptr<PlayBot>> CreatePlayers(
    const cxxopts::ParseResult &result, std::shared_ptr<Resampler> resampler) {
  const std::string defender_str = result["defender"].as<std::string>();
  const std::string player1_str = result["p1"].as<std::string>();
  const std::string player2_str = result["p2"].as<std::string>();
  const bool early_cut = result["early_cut"].as<bool>();
  const bool root_cut = result["root_cut"].as<bool>();
  const int num_worlds = result["num_worlds"].as<int>();
  const int num_max_moves = result["num_max_moves"].as<int>();
  const AlphaMuConfig alpha_mu_cfg{num_max_moves, num_worlds, false, true, root_cut, early_cut};
  const PIMCConfig pimc_cfg{num_worlds, false};
  std::shared_ptr<PlayBot> defenfer = CreatePlayer(defender_str, pimc_cfg, alpha_mu_cfg, resampler);
  std::shared_ptr<PlayBot> player1 = CreatePlayer(player1_str, pimc_cfg, alpha_mu_cfg, resampler);
  std::shared_ptr<PlayBot> player2 = CreatePlayer(player2_str, pimc_cfg, alpha_mu_cfg, resampler);
  return {player1, player2, defenfer};
}

int main(int argc, char **argv) {
  cxxopts::Options options("Bridge Play Match", "A program plays bridge between alphamu and pimc.");
  options.add_options()(
      "m, num_max_moves", "Number of max moves in alphamu search", cxxopts::value<int>()->default_value("1"))(
      "w, num_worlds", "Number of possible worlds", cxxopts::value<int>()->default_value("20"))(
      "num_deals", "Number of deals with different results", cxxopts::value<int>()->default_value("200"))(
      "early_cut", "Whether use early cut", cxxopts::value<bool>()->default_value("false"))(
      "root_cut", "Whether use root cut", cxxopts::value<bool>()->default_value("false"))(
      "contract", "The contract of the deals", cxxopts::value<std::string>()->default_value("3NT"))(
      "seed", "Random seed for generating deals", cxxopts::value<int>()->default_value("77"))(
      "show_play", "Whether to show the played games", cxxopts::value<bool>()->default_value("false"))(
      "file_dir", "The directory to save log", cxxopts::value<std::string>()->default_value("D:/Projects/bridge"))(
      "p1", "The algorithm of player1", cxxopts::value<std::string>()->default_value("alpha_mu"))(
      "p2", "The algorithm of player2", cxxopts::value<std::string>()->default_value("dds"))(
      "defender", "The algorithm of defenders", cxxopts::value<std::string>()->default_value("pimc"));

  auto result = options.parse(argc, argv);
  int played_deals = 0;
  int total_deals = result["num_deals"].as<int>();
  int num_different_score_deals = 0;
  int num_deals_win_by_p1 = 0;
  constexpr int double_dummy_tolerance = 1;
  // const int num_worlds = result["num_worlds"].as<int>();

  int seed = result["seed"].as<int>();
  while (seed == -1) {
    seed = static_cast<int>(std::random_device()());
  }
  std::mt19937 rng(seed);
  auto resampler = std::make_shared<UniformResampler>(1);
  // const int num_max_moves = result["num_max_moves"].as<int>();
  // const AlphaMuConfig alpha_mu_cfg{num_max_moves, num_worlds, false};
  // const PIMCConfig pimc_cfg{num_worlds, false};
  //  auto alpha_mu_bot = VanillaAlphaMuBot(resampler, alpha_mu_cfg);
  std::shared_ptr<PlayBot> player1;
  std::shared_ptr<PlayBot> player2;
  std::shared_ptr<PlayBot> defender;

  std::tie(player1, player2, defender) = CreatePlayers(result, resampler);

  StatManager stat_manager{};
  const std::string defender_str = result["defender"].as<std::string>();
  const std::string player1_str = result["p1"].as<std::string>();
  const std::string player2_str = result["p2"].as<std::string>();

  // auto alpha_mu_bot = AlphaMuBot(resampler, alpha_mu_cfg);
  // auto pimc_bot = PIMCBot(resampler, pimc_cfg);

  const ble::Contract contract = ParseContractFromString(result["contract"].as<std::string>());
  //  std::cout << absl::StrFormat("The contract is set to %s", contract.ToString()) << std::endl;
  //
  //  std::cout << absl::StrFormat("The config of alpha mu: num_max_moves: %d, num_worlds: %d",
  //                               result["num_max_moves"].as<int>(),
  //                               num_worlds) << std::endl;
  //  std::cout << absl::StrFormat("The config of pimc: num_worlds: %d",
  //                               num_worlds) << std::endl;

  RunningAverage avg{};
  //  spdlog::set_level(spdlog::level::info);
  std::string time = absl::FormatTime("%Y%m%d%H%M%E3S", absl::Now(), absl::LocalTimeZone());
  std::string file_name =
      absl::StrFormat("Match_%s_and_%s_vs_%s__%s", player1->Name(), player2->Name(), defender->Name(), time);
  std::cout << file_name << std::endl;
  auto logger = FileLogger(result["file_dir"].as<std::string>(), file_name, "w");
  logger.Print("Match Start.");

  while (num_different_score_deals < total_deals) {
    std::cout << absl::StrCat("Running deal No. ", played_deals) << std::endl;

    auto state1 = ConstructRandomState(rng, contract);
    auto ddt = state1.DoubleDummyResults();
    std::cout << absl::StrCat("Double dummy result: ",
                              ddt[state1.GetContract().denomination][state1.GetContract().declarer])
              << "\n";
    if (std::abs(ddt[state1.GetContract().denomination][state1.GetContract().declarer] - (6 + contract.level)) >
        double_dummy_tolerance) {
      continue;
    }
    rela::utils::printVector(state1.UidHistory());
    auto state2 = state1.Clone();
    resampler->ResetWithParams({{"seed", std::to_string(played_deals)}});
    while (!state1.IsTerminal()) {
      //      std::cout << state1 << std::endl;
      ble::BridgeMove move;
      if (IsActingPlayerDeclarerSide(state1)) {
        const auto st = std::chrono::high_resolution_clock::now();
        move = player1->Step(state1);
        const auto ed = std::chrono::high_resolution_clock::now();
        const auto elapsed = ed - st;
        //        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
        // avg.AddNumber(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
        stat_manager.AddValue(player1_str, std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
      }
      else {
        const auto st = std::chrono::high_resolution_clock::now();
        move = defender->Step(state1);
        const auto ed = std::chrono::high_resolution_clock::now();
        const auto elapsed = ed - st;
        stat_manager.AddValue(defender_str, std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
      }
      state1.ApplyMove(move);
    }
    //    std::cout << state.ToString() << std::endl;
    //      //
    //    std::cout << "State 1:\n" << state.ToString() << std::endl;
    //    std::cout << "Playing state2" << std::endl;
    resampler->ResetWithParams({{"seed", std::to_string(played_deals)}});
    while (!state2.IsTerminal()) {
      ble::BridgeMove move;
      if (IsActingPlayerDeclarerSide(state2)) {
        const auto st = std::chrono::high_resolution_clock::now();
        move = player2->Step(state2);
        const auto ed = std::chrono::high_resolution_clock::now();
        const auto elapsed = ed - st;
        //        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
        // avg.AddNumber(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
        stat_manager.AddValue(player2_str, std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
      }
      else {
        const auto st = std::chrono::high_resolution_clock::now();
        move = defender->Step(state2);
        const auto ed = std::chrono::high_resolution_clock::now();
        const auto elapsed = ed - st;
        stat_manager.AddValue(defender_str, std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
      }
      state2.ApplyMove(move);
    }
    ++played_deals;
    const bool table_open_win = state1.Scores()[contract.declarer] > 0;
    const bool table_close_win = state2.Scores()[contract.declarer] > 0;

    if (table_open_win != table_close_win) {
      ++num_different_score_deals;
      num_deals_win_by_p1 += table_open_win;
      logger.Print("Deal No.%d, state1:\n%s\ntrajectory:\n%s\nstate2:\n%s\ntrajectory:\n%s",
                   played_deals,
                   state1.ToString(),
                   VectorToString(state1.UidHistory()),
                   state2.ToString(),
                   VectorToString(state2.UidHistory()));
    }
    if (result["show_play"].as<bool>()) {
      std::cout << absl::StrFormat(
                       "Deal No.%d, state1:\n%s\nstate2:\n%s", played_deals, state1.ToString(), state2.ToString())
                << std::endl;
    }
    const std::string res_str = absl::StrFormat("%d deals have been played, num != : %d, num win by %s: %d",
                                                played_deals,
                                                num_different_score_deals,
                                                player1->Name(),
                                                num_deals_win_by_p1);
    std::cout << res_str << std::endl;
    logger.Print(res_str);

    std::cout << absl::StrFormat("Average execution time of %s: ", player1->Name())
              << stat_manager.GetAverage(player1_str) << std::endl;
    std::cout << absl::StrFormat("Average execution time of %s: ", player2->Name())
              << stat_manager.GetAverage(player2_str) << std::endl;
    std::cout << absl::StrFormat("Average execution time of %s: ", defender->Name())
              << stat_manager.GetAverage(defender_str) << std::endl;
    // std::cout << "Average execution time of alphamu: " << avg.GetAverage() << std::endl;
    // logger.Print("Average execution time of alphamu: %f", avg.GetAverage());
  }
  const std::string final_res_str =
      absl::StrFormat("Match is over, the result for %s is %d/%d = %f",
                      player1->Name(),
                      num_deals_win_by_p1,
                      total_deals,
                      static_cast<double>(num_deals_win_by_p1) / static_cast<double>(total_deals));
  std::cout << final_res_str << std::endl;
  logger.Print(final_res_str);
  return 0;
}
