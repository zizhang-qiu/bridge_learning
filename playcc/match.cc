//
// Created by qzz on 2023/12/12.
//
#include <chrono>

#include "third_party/cxxopts/include/cxxopts.hpp"
#include "absl/strings/str_cat.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "bridge_lib/bridge_scoring.h"

#include "pimc.h"
#include "alpha_mu_bot.h"
#include "logger.h"


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
      case 'C':denomination = ble::Denomination::kClubsTrump;
        return {contract_level, denomination, ble::kUndoubled, ble::kSouth};
      case 'D':denomination = ble::Denomination::kDiamondsTrump;
        return {contract_level, denomination, ble::kUndoubled, ble::kSouth};
      case 'H':denomination = ble::Denomination::kHeartsTrump;
        return {contract_level, denomination, ble::kUndoubled, ble::kSouth};
      case 'S':denomination = ble::Denomination::kSpadesTrump;
        return {contract_level, denomination, ble::kUndoubled, ble::kSouth};
      default:SpielFatalError(absl::StrCat("Invalid contract string: ", contract_string));
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

int main(int argc, char **argv) {
  cxxopts::Options options("Bridge Play Match", "A program plays bridge between alphamu and pimc.");
  options.add_options()
      ("m, num_max_moves", "Number of max moves in alphamu search", cxxopts::value<int>()->default_value("1"))
      ("w, num_worlds", "Number of possible worlds", cxxopts::value<int>()->default_value("20"))
      ("num_deals", "Number of deals with different results", cxxopts::value<int>()->default_value("200"))
      ("contract", "The contract of the deals", cxxopts::value<std::string>()->default_value("3NT"))
      ("seed", "Random seed for generating deals", cxxopts::value<int>()->default_value("66"))
      ("show_play", "Whether to show the played games", cxxopts::value<bool>()->default_value("false"));

  auto result = options.parse(argc, argv);
  int played_deals = 0;
  int total_deals = result["num_deals"].as<int>();
  int num_different_score_deals = 0;
  int num_deals_win_by_alpha_mu = 0;
  int double_dummy_tolerance = 3;
  const int num_worlds = result["num_worlds"].as<int>();

  int seed = result["seed"].as<int>();
  while (seed == -1) {
    seed = static_cast<int>(std::random_device()());
  }
  std::mt19937 rng(seed);
  auto resampler = std::make_shared<UniformResampler>(1);
  const AlphaMuConfig alpha_mu_cfg{result["num_max_moves"].as<int>(),
                                   num_worlds,
                                   false};
  const PIMCConfig pimc_cfg{num_worlds, false};
//  auto alpha_mu_bot = VanillaAlphaMuBot(resampler, alpha_mu_cfg);
  auto alpha_mu_bot = AlphaMuBot(resampler, alpha_mu_cfg);
  auto pimc_bot = PIMCBot(resampler, pimc_cfg);

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
  auto logger = FileLogger("D:/Projects/bridge", "log.txt");
  logger.Print("Match Start.");

  while (num_different_score_deals < total_deals) {
    std::cout << absl::StrCat("Running deal No. ", played_deals) << std::endl;

    auto state1 = ConstructRandomState(rng, contract);
    auto ddt = state1.DoubleDummyResults();
    std::cout
        << absl::StrCat("Double dummy result: ", ddt[state1.GetContract().denomination][state1.GetContract().declarer])
        << "\n";
    if (std::abs(ddt[state1.GetContract().denomination][state1.GetContract().declarer] - (6 + contract.level))
        > double_dummy_tolerance) {
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
        move = alpha_mu_bot.Act(state1);
        const auto ed = std::chrono::high_resolution_clock::now();
        const auto elapsed = ed - st;
//        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
        avg.AddNumber(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
      } else {
        move = pimc_bot.Act(state1);
      }
      state1.ApplyMove(move);
    }
    //    std::cout << state.ToString() << std::endl;
    //      //
    //    std::cout << "State 1:\n" << state.ToString() << std::endl;
    //    std::cout << "Playing state2" << std::endl;
    resampler->ResetWithParams({{"seed", std::to_string(played_deals)}});
    while (!state2.IsTerminal()) {
      const auto move = pimc_bot.Act(state2);
      //    std::cout << "pimc move: " << move.ToString() << std::endl;
      state2.ApplyMove(move);
    }
    ++played_deals;
    const bool table_open_win = state1.Scores()[contract.declarer] > 0;
    const bool table_close_win = state2.Scores()[contract.declarer] > 0;

    if (table_open_win != table_close_win) {
      ++num_different_score_deals;
      num_deals_win_by_alpha_mu += table_open_win;
      spdlog::info("Deal No.{}, state1:\n{}\nstate2:\n{}",
                   played_deals,
                   state1.ToString(),
                   state2.ToString());
    }
    if (result["show_play"].as<bool>()) {
      std::cout
          << absl::StrFormat("Deal No.%d, state1:\n%s\nstate2:\n%s", played_deals, state1.ToString(), state2.ToString())
          << std::endl;
    }
    std::cout << absl::StrCat(played_deals,
                              " Deals have been played, num != : ",
                              num_different_score_deals,
                              ", num win by alphamu: ",
                              num_deals_win_by_alpha_mu)
              << std::endl;

    std::cout << "Average execution time of alphamu: " << avg.GetAverage() << std::endl;
  }
  std::cout
      << absl::StrFormat("Match is over, the result for alphamu is %d/%d = %f", num_deals_win_by_alpha_mu, total_deals,
                         static_cast<double>(num_deals_win_by_alpha_mu) / static_cast<double>(total_deals))
      << std::endl;
  return 0;
}