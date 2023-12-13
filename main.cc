// #define NOMINMAX
#include "bridge_lib/bridge_observation.h"
#include "bridge_lib/canonical_encoder.h"
#include "bridge_lib/example_cards_ddts.h"
// #include "rela/batch_runner.h"
// #include "rlcc/bridge_env.h"
#include <iostream>
#include <sstream>
#include "playcc/pimc.h"
#include "playcc/resampler.h"
#include "rela/utils.h"
// #include "playcc/outcome_vector.h"
#include <array>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "bridge_lib/utils.h"
#include "playcc/alpha_mu_bot.h"
#include "playcc/alpha_mu_search.h"
#include "playcc/bridge_state_without_hidden_info.h"
#include "playcc/log_utils.h"
#include "playcc/pareto_front.h"
#include "playcc/worlds.h"
namespace ble = bridge_learning_env;

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);
ble::BridgeState ConstructRandomState(std::mt19937 &rng) {
  auto state = ble::BridgeState(game);
  const auto deal = ble::Permutation(ble::kNumCards, rng);
  for (int i = 0; i < ble::kNumCards; ++i) {
    ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
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
  return state;
}
int main() {
  // Deal
  //  const auto &deal = ble::example_deals[0];
  std::mt19937 rng(222);
  //  //  const int num_max_moves = 3;
  const int num_worlds = 20;
  const AlphaMuConfig cfg{1, num_worlds, true};

  //
  int played_deals = 0;
  int total_deals = 10;
  int num_different_score_deals = 0;
  int num_deals_win_by_alpha_mu = 0;
  int num_discrepancy = 0;
  //
  //  const ble::Player declarer = ble::kEast;
  //  const ble::Player dummy = ble::kWest;
  auto resampler = std::make_shared<UniformResampler>(1);
  auto alpha_mu_bot = VanillaAlphaMuBot(resampler, cfg);
  auto pimc_bot = PIMCBot(resampler, {num_worlds, true});
  while (num_different_score_deals < total_deals) {
    //      std::vector<std::shared_ptr<Resampler>> resamplers_open;
    //      std::vector<std::shared_ptr<Resampler>> resamplers_close;
    //      for (int j = 0; j < ble::kNumPlayers; ++j) {
    //        resamplers_open.push_back(std::make_shared<UniformResampler>(num_different_score_deals));
    //        resamplers_close.push_back(std::make_shared<UniformResampler>(num_different_score_deals));
    //      }
    //      auto pimc_bot = PIMCBot(resamplers_open[ble::kSouth], num_worlds);
    //      auto pimc_bot2 = PIMCBot(resamplers_open[ble::kNorth], num_worlds);
    //      auto alpha_mu_declarer = VanillaAlphaMuBot(resamplers_open[ble::kEast], cfg);
    //      auto alpha_mu_dummy = VanillaAlphaMuBot(resamplers_open[ble::kWest], cfg);

    std::cout << absl::StrCat("Running deal No. ", played_deals) << std::endl;

    //    std::vector<PIMCBot> close_bots;
    //      close_bots.reserve(ble::kNumPlayers);
    //      for (int pl = 0; pl < ble::kNumPlayers; ++pl) {
    //        close_bots.emplace_back(resamplers_close[pl], num_worlds);
    //      }

    auto state = ConstructRandomState(rng);
    auto state2 = state.Clone();
    resampler->ResetWithParams({{"seed", std::to_string(played_deals)}});
//    std::cout << "Playing state1" << std::endl;

    while (!state.IsTerminal()) {
      ble::BridgeMove move;
      if (IsActingPlayerDeclarerSide(state)) {
//        const auto res = alpha_mu_bot.Search(state);
////        std::cout << state.ToString() << "\n" << res << std::endl;

        move = alpha_mu_bot.Act(state);
//        if (move.MoveType() == bridge_learning_env::BridgeMove::kInvalid) {
//          move = state.LegalMoves()[0];
//        }
      } else {
//        const auto res = pimc_bot.Search(state);
//        //        std::cout << state.ToString() << std::endl;
//        //        for (int i = 0; i < res.moves.size(); ++i) {
//        //          std::cout << absl::StrFormat("move: %s, score: %d", res.moves[i].ToString(), res.scores[i]) <<
//        //          std::endl;
//        //        }
//        move = std::get<0>(GetBestAction(res));
        move = pimc_bot.Act(state);
      }
      state.ApplyMove(move);
    }
    //    std::cout << state.ToString() << std::endl;
    //      //
    //    std::cout << "State 1:\n" << state.ToString() << std::endl;
    //    std::cout << "Playing state2" << std::endl;
    resampler->ResetWithParams({{"seed", std::to_string(played_deals)}});
    while (!state2.IsTerminal()) {
      ble::Player current_player;
      if (state2.IsDummyActing()) {
        current_player = state2.GetDummy();
      } else {
        current_player = state2.CurrentPlayer();
      }
      SPIEL_CHECK_GE(current_player, 0);
//      const auto res = pimc_bot.Search(state2);
//      //      std::cout << current_player << "\n";
//      //      const auto &move = pimc_bot.Act(state2);
//      //      std::cout << state2.ToString() << std::endl;
//      //      for (int i = 0; i < res.moves.size(); ++i) {
//      //        std::cout << absl::StrFormat("move: %s, score: %d", res.moves[i].ToString(), res.scores[i]) <<
//      //        std::endl;
//      //      }
//      const auto move = std::get<0>(GetBestAction(res));
      const auto move = pimc_bot.Act(state2);
      //    std::cout << "pimc move: " << move.ToString() << std::endl;
      state2.ApplyMove(move);
    }
//    std::cout << "State 2:\n" << state2.ToString() << std::endl;
    ++played_deals;
    //      std::cout << state2.ToString() << std::endl;
    if (state.UidHistory() != state2.UidHistory()) {
//      std::exit(1);
      ++num_discrepancy;
    }
    const bool table_open_win = state.Scores()[ble::kWest] > 0;
    const bool table_close_win = state2.Scores()[ble::kWest] > 0;

    if (table_open_win != table_close_win) {
      ++num_different_score_deals;
      num_deals_win_by_alpha_mu += table_open_win;
      //        std::exit(1);
    }
    std::cout << absl::StrCat(played_deals,
                              " Deals have been played, num != : ",
                              num_different_score_deals,
                              ", num win by alphamu: ",
                              num_deals_win_by_alpha_mu,
                              ", num discrepancy: ",
                              num_discrepancy)
              << std::endl;
  }
  //  auto state = ConstructRandomState(rng);
  //  auto resampler = std::make_shared<UniformResampler>(2);
  //  //
  //  for (int i = 0; i < 5; ++i) {
  //    std::cout << absl::StrFormat("Trial No.%d", i) << std::endl;
  //    resampler->ResetWithParams({{"seed", std::to_string(42)}});
  //    const auto sampled_deals = ResampleMultipleDeals(resampler, state, 3);
  //    for (const auto d : sampled_deals) {
  //      PrintArray(d);
  //    }
  //    auto worlds = Worlds(sampled_deals, state);
  //    //  auto front = VanillaAlphaMu(ble::BridgeStateWithoutHiddenInfo(state), 3, worlds);
  //    //  std::cout << front << std::endl;
  //    std::cout << worlds << std::endl;
  //  }
  //  const auto possible_moves = worlds.GetAllPossibleMoves();
  //  for(const auto move:possible_moves){
  //    std::cout << move.ToString() << std::endl;
  //  }
  //  const std::vector<std::array<int, ble::kNumCards>> deals = {
  //      {2,  8,  6,  46, 13, 44, 23, 31, 27, 25, 33, 12, 51, 49, 9, 19, 30, 47, 38, 1, 37, 45, 39, 41, 50, 20,
  //       16, 42, 11, 17, 35, 0,  29, 48, 40, 7,  26, 24, 34, 15, 4, 28, 43, 10, 36, 5, 18, 3,  14, 21, 32, 22},
  //      {2,  8,  6,  46, 38, 44, 16, 31, 37, 25, 34, 12, 50, 49, 32, 19, 43, 47, 9,  1, 4,  45, 14, 41, 13, 20,
  //       18, 42, 26, 17, 35, 0,  36, 48, 29, 7,  40, 24, 33, 15, 23, 28, 11, 10, 27, 5, 51, 3,  30, 21, 39, 22},
  //      {2,  8,  6,  46, 27, 44, 23, 31, 16, 25, 14, 12, 32, 49, 30, 19, 33, 47, 29, 1, 35, 45, 39, 41, 51, 20,
  //       34, 42, 38, 17, 9,  0,  43, 48, 4,  7,  36, 24, 11, 15, 18, 28, 37, 10, 26, 5, 40, 3,  50, 21, 13, 22},
  //      {2,  8,  6,  46, 35, 44, 9,  31, 27, 25, 16, 12, 11, 49, 36, 19, 34, 47, 14, 1, 32, 45, 26, 41, 37, 20,
  //       51, 42, 38, 17, 4,  0,  30, 48, 43, 7,  39, 24, 40, 15, 29, 28, 13, 10, 18, 5, 50, 3,  23, 21, 33, 22},
  //      {2,  8,  6,  46, 27, 44, 23, 31, 39, 25, 37, 12, 4,  49, 16, 19, 33, 47, 18, 1, 40, 45, 29, 41, 32, 20,
  //       13, 42, 50, 17, 51, 0,  11, 48, 9,  7,  43, 24, 35, 15, 14, 28, 30, 10, 38, 5, 36, 3,  34, 21, 26, 22}};
  //  std::vector<int> bid_uid;
  //  bid_uid.push_back(ble::BidIndex(1, ble::kSpadesTrump) + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::BidIndex(3, ble::kNoTrump) + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  //  std::vector<int> play_uid;
  //  play_uid.push_back(ble::CardIndex(bridge_learning_env::kHeartsSuit, 1));
  //  play_uid.push_back(ble::CardIndex(bridge_learning_env::kHeartsSuit, 10));
  //  play_uid.push_back(ble::CardIndex(bridge_learning_env::kHeartsSuit, 0));
  //  for (const auto& d : deals) {
  //    auto state = ConstructStateFromDeal(d, game);
  //    // Bidding
  //    for (const int uid : bid_uid) {
  //      ble::BridgeMove move = game->GetMove(uid);
  //      state.ApplyMove(move);
  //    }
  //    for (const int uid : play_uid) {
  //      ble::BridgeMove move = game->GetMove(uid);
  //      state.ApplyMove(move);
  //    }
  //    std::cout << state.ToString() << std::endl;
  //
  //    const auto legal_moves = state.LegalMoves();
  //    std::cout << "evaluation 1" << std::endl;
  //    for (const auto& move : legal_moves) {
  //      int evaluation = Rollout(state, move);
  //      std::cout << "move: " << move.ToString() << ", evaluation: " << evaluation << std::endl;
  //    }
  //    std::cout << "evaluation 2" << std::endl;
  //    for (const auto& move : legal_moves) {
  //      auto child = state.Child(move);
  //      bool evaluation = DoubleDummyEvaluation(child);
  //      std::cout << "move: " << move.ToString() << ", evaluation: " << evaluation << std::endl;
  //    }
  //  }

  return 0;
}
