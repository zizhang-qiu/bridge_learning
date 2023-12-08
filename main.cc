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
  //  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::BidIndex(3, ble::kDiamondsTrump) + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::BidIndex(3, ble::kHeartsTrump) + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::BidIndex(4, ble::kClubsTrump) + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::BidIndex(4, ble::kHeartsTrump) + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::BidIndex(5, ble::kClubsTrump) + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::BidIndex(5, ble::kHeartsTrump) + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  //  bid_uid.push_back(ble::kDouble + ble::kBiddingActionBase);
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
  std::mt19937 rng(3);
  const int num_max_moves = 3;
  const int num_worlds = 20;

  //  const ble::Contract contract{3, ble::Denomination::kClubsTrump, ble::DoubleStatus::kUndoubled, ble::kEast};

  //  state.ApplyMove(state.LegalMoves()[0]);
  //
  //  auto legal_moves = state2.LegalMoves();
  //  for(const auto move:legal_moves){
  //    std::cout << move.ToString() << std::endl;
  //  }
  //  state2.ApplyMove(legal_moves[0]);
  //  std::cout << state2.ToString() << std::endl;
  //  ble::BridgeHand hand{};
  //  hand.AddCard({ble::Suit::kClubsSuit, 2});
  //  std::cout << std::boolalpha << hand.IsCardInHand({ble::Suit::kClubsSuit, 2}) << std::endl;
  const AlphaMuConfig cfg{2, 20};

  int total_deals = 100;
  int num_different_score_deals = 0;
  int num_deals_win_by_alpha_mu = 0;

  const ble::Player declarer = ble::kEast;
  const ble::Player dummy = ble::kWest;
  for (int i = 0; i < total_deals; ++i) {
    std::vector<std::shared_ptr<Resampler>> resamplers_open;
    std::vector<std::shared_ptr<Resampler>> resamplers_close;
    for (int j = 0; j < ble::kNumPlayers; ++j) {
      resamplers_open.push_back(std::make_shared<UniformResampler>(i));
      resamplers_close.push_back(std::make_shared<UniformResampler>(i));
    }
    auto pimc_bot = PIMCBot(resamplers_open[ble::kSouth], num_worlds);
    auto pimc_bot2 = PIMCBot(resamplers_open[ble::kNorth], num_worlds);
    auto alpha_mu_declarer = VanillaAlphaMuBot(resamplers_open[ble::kEast], cfg);
    auto alpha_mu_dummy = VanillaAlphaMuBot(resamplers_open[ble::kWest], cfg);
    std::cout << absl::StrCat("Running deal No. ", i) << std::endl;

    std::vector<PIMCBot> close_bots;
    close_bots.reserve(ble::kNumPlayers);
    for (int pl = 0; pl < ble::kNumPlayers; ++pl) {
      close_bots.emplace_back(resamplers_close[pl], num_worlds);
    }

    auto state = ConstructRandomState(rng);
    auto state2 = state.Clone();

    while (!state.IsTerminal()) {
//      std::cout << state.ToString() << std::endl;
      if (IsActingPlayerDeclarerSide(state)) {
        if (state.IsDummyActing()) {
          const auto move = alpha_mu_dummy.Act(state);
          state.ApplyMove(move);
        }
        else {
          const auto move = alpha_mu_declarer.Act(state);
          state.ApplyMove(move);
        }
        //      std::exit(1);
      }
      else {
        if (state.CurrentPlayer() == ble::kSouth) {
          const auto &move = pimc_bot.Act(state);
          //        std::cout << "pimc move: " << move.ToString() << std::endl;
          state.ApplyMove(move);
        }
        else {
          const auto &move = pimc_bot2.Act(state);
          //        std::cout << "pimc move: " << move.ToString() << std::endl;
          state.ApplyMove(move);
        }
      }
    }
    std::cout << state.ToString() << std::endl;
    //
    while (!state2.IsTerminal()) {
      ble::Player current_player;
      if (state2.IsDummyActing()) {
        current_player = state2.GetDummy();
      }
      else {
        current_player = state2.CurrentPlayer();
      }
//      std::cout << current_player << "\n";
      const auto &move = close_bots[current_player].Act(state2);
      //    std::cout << "pimc move: " << move.ToString() << std::endl;
      state2.ApplyMove(move);
    }
    std::cout << state2.ToString() << std::endl;
    const bool table_open_win = state.Scores()[ble::kWest] > 0;
    const bool table_close_win = state2.Scores()[ble::kWest] > 0;

    if (table_open_win != table_close_win) {
      ++num_different_score_deals;
      num_deals_win_by_alpha_mu += table_open_win;
    }
    std::cout << absl::StrCat(i,
                              "Deals have been played, num != : ",
                              num_different_score_deals,
                              ", num win by alphamu: ",
                              num_deals_win_by_alpha_mu)
              << std::endl;
  }

  //  ParetoFront front{};
  //  ParetoFront pf1{{OutcomeVector{{0, 1, 1}, {true, true, true}}}};
  //  ParetoFront pf2{{OutcomeVector{{1, 1, 0}, {true, true, true}}}};
  //  front = ParetoFrontMax(front, pf1);
  //  front = ParetoFrontMax(front, pf2);
  //  std::cout << front.ToString() << std::endl;
  //
  //  ParetoFront front2{};
  //  ParetoFront pf3{{OutcomeVector{{1, 1, 0}, {true, true, true}}}};
  //  ParetoFront pf4{{OutcomeVector{{1, 0, 1}, {true, true, true}}}};
  //  ParetoFront pf5{{OutcomeVector{{1, 0, 0}, {true, true, true}}}};
  //  front2 = ParetoFrontMax(front2, pf3);
  //  front2 = ParetoFrontMax(front2, pf4);
  //  front2 = ParetoFrontMax(front2, pf5);
  //  std::cout << front2.ToString() << std::endl;
  //
  //  ParetoFront front3{};
  //  front3 = ParetoFrontMin(front3, front);
  //  front3 = ParetoFrontMin(front3, front2);
  //  std::cout << "front3:\n" << front3.ToString() << std::endl;
  //  std::string s = absl::StrCat("1", 2, "hello");
  //  std::cout << s << std::endl;
  //  absl::StrAppend(&s, "345");
  //  std::cout << s << std::endl;
  //  s = absl::StrJoin({1, 2, 3}, ",");
  //  std::cout << s << std::endl;
  //  const auto ss = absl::StrSplit(s, ",");
  //  for(const auto str:ss){
  //    std::cout << str << std::endl;
  //  }
  //  s = absl::StrFormat("Welcome to %s, Number %d!", "The Village", 6);
  //
  //  std::cout << s << std::endl;
  //
  //  s = absl::StrReplaceAll(
  //       "$who bought $count #Noun. Thanks $who!",
  //       {{"$count", absl::StrCat(5)},
  //        {"$who", "Bob"},
  //        {"#Noun", "Apples"}});
  //  std::cout << s << std::endl;
  return 0;
}
