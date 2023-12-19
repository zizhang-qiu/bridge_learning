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
#include "playcc/alpha_mu_search.h"
#include "playcc/bridge_state_without_hidden_info.h"
#include "playcc/log_utils.h"
#include "playcc/pareto_front.h"
#include "playcc/worlds.h"
#include "playcc/transposition_table.h"
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
  std::mt19937 rng(23);
  std::vector<int> trajectory = {
      40, 46, 35, 29, 31, 26, 32, 10, 47, 28, 19, 38, 12, 11, 1, 42, 2, 25, 0, 9, 50, 3, 8, 27, 4, 22, 18, 7, 6, 14, 30,
      5, 44, 37, 21, 23, 13, 20, 48, 33, 16, 45, 51, 36, 34, 43, 17, 49, 39, 15, 24, 41, 52, 52, 69, 52, 52, 52,
//      ble::CardIndex(ble::kDiamondsSuit, 1),
//      ble::CardIndex(ble::kDiamondsSuit, 3),
//      ble::CardIndex(ble::kDiamondsSuit, 11),
//      ble::CardIndex(ble::kDiamondsSuit, 0),
//      ble::CardIndex(ble::kHeartsSuit, 3),
//      ble::CardIndex(ble::kHeartsSuit, 4),
//      ble::CardIndex(ble::kHeartsSuit, 9),
//      ble::CardIndex(ble::kHeartsSuit, 12),
//      ble::CardIndex(ble::kClubsSuit, 1),
//      ble::CardIndex(ble::kClubsSuit, 5),
//      ble::CardIndex(ble::kClubsSuit, 12),
//      ble::CardIndex(ble::kClubsSuit, 9),
//      ble::CardIndex(ble::kClubsSuit, 0),
//      ble::CardIndex(ble::kSpadesSuit, 1),
//      ble::CardIndex(ble::kClubsSuit, 10),
//      ble::CardIndex(ble::kClubsSuit, 7),
//      ble::CardIndex(ble::kClubsSuit, 3),
//      ble::CardIndex(ble::kSpadesSuit, 0),
//      ble::CardIndex(ble::kClubsSuit, 2),
//      ble::CardIndex(ble::kDiamondsSuit, 2),
//      ble::CardIndex(ble::kClubsSuit, 4),
//      ble::CardIndex(ble::kHeartsSuit, 5),
//      ble::CardIndex(ble::kClubsSuit, 8),
//      ble::CardIndex(ble::kSpadesSuit, 5),
//      ble::CardIndex(ble::kSpadesSuit, 4),
//      ble::CardIndex(ble::kSpadesSuit, 6),
//      ble::CardIndex(ble::kSpadesSuit, 11),
//      ble::CardIndex(ble::kSpadesSuit, 2),
//      ble::CardIndex(ble::kClubsSuit, 11),
//      ble::CardIndex(ble::kDiamondsSuit, 6),
//      ble::CardIndex(ble::kClubsSuit, 6),
//      ble::CardIndex(ble::kDiamondsSuit, 7),
//      ble::CardIndex(ble::kSpadesSuit, 7),
//      ble::CardIndex(ble::kSpadesSuit, 3),
  };
//  auto state = ConstructRandomState(rng);
  auto state = ble::BridgeState(game);

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
  int num_worlds = 10;
  auto resampler = std::make_shared<UniformResampler>(1);
  const AlphaMuConfig alpha_mu_cfg{2, num_worlds, false};
  const PIMCConfig pimc_cfg{num_worlds, false};
  auto alpha_mu_bot = VanillaAlphaMuBot(resampler, alpha_mu_cfg);
//  auto alpha_mu_bot = AlphaMuBot(resampler, alpha_mu_cfg);
  auto pimc_bot = PIMCBot(resampler, pimc_cfg);
  resampler->ResetWithParams({{"seed", std::to_string(23)}});
  while (!state.IsTerminal()) {
    std::cout << state << std::endl;
    ble::BridgeMove move;
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
      move = alpha_mu_bot.Act(state);
//      break;
    } else {
      move = pimc_bot.Act(state);
    }
    state.ApplyMove(move);
  }

  std::cout << state << std::endl;
  resampler->ResetWithParams({{"seed", std::to_string(23)}});
  while (!state2.IsTerminal()) {
    const auto move = pimc_bot.Act(state2);
    //    std::cout << "pimc move: " << move.ToString() << std::endl;
    state2.ApplyMove(move);
//    break;
  }
  std::cout << state2 << std::endl;
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
  return 0;
}
