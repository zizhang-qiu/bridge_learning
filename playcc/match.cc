//
// Created by qzz on 2023/12/12.
//
#include "pimc.h"
#include "alpha_mu_bot.h"
#include "third_party/cxxopts/include/cxxopts.hpp"
#include "absl/strings/str_cat.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "bridge_lib/bridge_scoring.h"

namespace ble = bridge_learning_env;
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
  bid_uid.push_back(ble::BidIndex(3, ble::kNoTrump) + ble::kBiddingActionBase);
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
      ("seed", "Random seed for generating deals", cxxopts::value<int>()->default_value("42"));

  auto result = options.parse(argc, argv);
  int played_deals = 0;
  int total_deals = 100;
  int num_different_score_deals = 0;
  int num_deals_win_by_alpha_mu = 0;
  const int num_worlds = result["num_worlds"].as<int>();

  std::mt19937 rng(result["seed"].as<int>());
  auto resampler = std::make_shared<UniformResampler>(1);
  const AlphaMuConfig alpha_mu_cfg{result["num_max_moves"].as<int>(), num_worlds, true};
  const PIMCConfig pimc_cfg{num_worlds, true};
  auto alpha_mu_bot = VanillaAlphaMuBot(resampler, alpha_mu_cfg);
  auto pimc_bot = PIMCBot(resampler, pimc_cfg);

  const ble::Contract contract = ParseContractFromString(result["contract"].as<std::string>());
  std::cout << absl::StrFormat("The contract is set to %s", contract.ToString()) << std::endl;

  std::cout << absl::StrFormat("The config of alpha mu: num_max_moves: %d, num_worlds: %d",
                               result["num_max_moves"].as<int>(),
                               num_worlds) << std::endl;
  std::cout << absl::StrFormat("The config of pimc: num_worlds: %d",
                               num_worlds) << std::endl;
  while (num_different_score_deals < total_deals) {
    std::cout << absl::StrCat("Running deal No. ", played_deals) << std::endl;
    auto state1 = ConstructRandomState(rng, contract);
    auto state2 = state1.Clone();
    resampler->ResetWithParams({{"seed", std::to_string(played_deals)}});
    while (!state1.IsTerminal()) {
      ble::BridgeMove move;
      if (IsActingPlayerDeclarerSide(state1)) {
        move = alpha_mu_bot.Act(state1);
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
    }
    std::cout << absl::StrCat(played_deals,
                              " Deals have been played, num != : ",
                              num_different_score_deals,
                              ", num win by alphamu: ",
                              num_deals_win_by_alpha_mu)
              << std::endl;
  }
  std::cout
      << absl::StrFormat("Match is over, the result for alphamu is %d/%d = %f", num_deals_win_by_alpha_mu, total_deals,
                         static_cast<double>(num_deals_win_by_alpha_mu) / static_cast<double>(total_deals))
      << std::endl;
  return 0;
}