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
#include <unordered_map>
#include <array>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "bridge_lib/utils.h"
#include "playcc/alpha_mu_bot.h"
#include "playcc/bridge_state_without_hidden_info.h"
#include "playcc/common_utils/log_utils.h"
#include "playcc/pareto_front.h"
#include "playcc/worlds.h"
#include "playcc/transposition_table.h"
#include "playcc/common_utils/logger.h"
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
  std::mt19937 rng;
  std::vector<int> trajectory = {
      27, 3, 32, 38, 25, 44, 4, 17, 34, 46, 42, 15, 11, 31, 41, 0, 24, 16, 7, 39, 21, 30, 33, 47, 49, 29, 5, 1, 37, 22,
      14, 8, 23, 40, 12, 43, 20, 2, 45, 28, 13, 9, 26, 18, 6, 19, 51, 10, 48, 35, 50, 36, 52, 52, 69, 52, 52, 52
  };
  ble::BridgeState state{game};
//  while (true) {

//    state = ConstructRandomState(rng);


  for (int i = 0; i < ble::kNumCards; ++i) {
    state.ApplyMove(game->GetChanceOutcome(trajectory[i]));
  }
  for (int i = 52; i < trajectory.size(); ++i) {
    state.ApplyMove(game->GetMove(trajectory[i]));
  }
  auto state2 = state.Clone();
  auto ddt = state.DoubleDummyResults();
  std::cout
      << absl::StrCat("Double dummy result: ", ddt[state.GetContract().denomination][state.GetContract().declarer])
      << "\n";
//    if (ddt[state.GetContract().denomination][state.GetContract().declarer] >= 9) {
//      break;
//    }
//  }

  int num_worlds = 20;
  auto resampler = std::make_shared<UniformResampler>(1);
  const AlphaMuConfig alpha_mu_cfg{3,
                                   num_worlds,
                                   false,
                                   true,
                                   true,
                                   true};
  const PIMCConfig pimc_cfg{num_worlds, false};
//  auto alpha_mu_bot = VanillaAlphaMuBot(resampler, alpha_mu_cfg);
  auto alpha_mu_bot = AlphaMuBot(resampler, alpha_mu_cfg);
  auto pimc_bot = PIMCBot(resampler, pimc_cfg);
  resampler->ResetWithParams({{"seed", std::to_string(3)}});
  while (!state.IsTerminal()) {
    std::cout << state << std::endl;
    if (IsActingPlayerDeclarerSide(state)) {
      SetMaxThreads(0);
      auto dl = StateToDDSDeal(state);
      futureTricks fut{};

      const int res = SolveBoard(
          dl,
          /*target=*/-1,
          /*solutions=*/3, // We only want one card.
          /*mode=*/2,
          &fut,
          /*threadIndex=*/0);
      if (res != RETURN_NO_FAULT) {
        char error_message[80];
        ErrorMessage(res, error_message);
        std::cerr << "double dummy solver: " << error_message << std::endl;
        std::exit(1);
      }
      for (int i = 0; i < ble::kNumCardsPerHand; ++i) {
        if (fut.rank[i] != 0)
          std::cout << absl::StrFormat("%d, %c%c, score: %d", i,
                                       ble::kSuitChar[ble::DDSSuitToSuit(fut.suit[i])],
                                       ble::kRankChar[ble::DDSRankToRank(fut.rank[i])],
                                       fut.score[i]) << std::endl;
      }
      auto search_move = alpha_mu_bot.Step(state);
      auto str = alpha_mu_bot.GetTT().Serialize();
      auto logger = FileLogger("D:/Projects/bridge", "tt", "a");
      logger.Print(str);
//      std::cout << "str:\n" << str << std::endl;
      auto tt = TranspositionTable::Deserialize(str, game);
      auto str2 = tt.Serialize();
      std::cout << (tt.Table() == alpha_mu_bot.GetTT().Table()) << std::endl;
      std::cout << "Search move: " << search_move << std::endl;
      state.ApplyMove(search_move);
    } else {
      auto move = pimc_bot.Step(state);
//      if (move.CardSuit() == ble::kHeartsSuit && move.CardRank() == 7) {
//
////        std::cout << alpha_mu_bot.GetTT().Serialize() << std::endl;
//      }
      state.ApplyMove(move);
    }
//    state.ApplyMove(move);

    std::cout << state << std::endl;
//  resampler->ResetWithParams({{"seed", std::to_string(23)}});
//  while (!state2.IsTerminal()) {
//    const auto move = pimc_bot.Act(state2);
//    //    std::cout << "pimc move: " << move.ToString() << std::endl;
//    state2.ApplyMove(move);
////    break;
//  }
//  std::cout << state2 << std::endl;
//  auto front = alpha_mu_bot.Search(state);
//  std::cout << front << std::endl;
//  auto search_res = pimc_bot.Search(state);
//  PrintSearchResult(search_res);
//  auto move = alpha_mu_bot.Act(state);
//  std::cout << move << std::endl;
//  SetMaxThreads(0);
//  auto dl = StateToDDSDeal(state);
//  futureTricks fut{};
//
//  const int res = SolveBoard(
//      dl,
//      /*target=*/-1,
//      /*solutions=*/2, // We only want one card.
//      /*mode=*/2,
//      &fut,
//      /*threadIndex=*/0);
//  if (res != RETURN_NO_FAULT) {
//    char error_message[80];
//    ErrorMessage(res, error_message);
//    std::cerr << "double dummy solver: " << error_message << std::endl;
//    std::exit(1);
//  }
//  for(int i=0; i<ble::kNumCardsPerHand; ++i){
//    std::cout << ble::DDSSuitToSuit(fut.suit[i]) << " " << ble::DDSRankToRank(fut.rank[i]) << std::endl;
//  }
//  ble::BridgeMove move{
//      /*move_type=*/ble::BridgeMove::Type::kPlay,
//      /*suit=*/ble::DDSSuitToSuit(fut.suit[0]),
//      /*rank=*/ble::DDSRankToRank(fut.rank[0]),
//      /*denomination=*/ble::kInvalidDenomination,
//      /*level=*/-1,
//      /*other_call=*/ble::kNotOtherCall};
//  std::cout << "dds move: " << move << std::endl;

  }
  return 0;
}

