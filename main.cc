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
#include "playcc/alpha_mu_search.h"
#include "playcc/pareto_front.h"
namespace ble = bridge_learning_env;
int main() {
  const ble::GameParameters params = {};
  const auto game = std::make_shared<ble::BridgeGame>(params);
  auto state = std::make_unique<ble::BridgeState>(game);

  // Deal
  const auto &deal = ble::example_deals[0];
  for (int i = 0; i < ble::kNumCards; ++i) {
    ble::BridgeMove move = game->GetChanceOutcome(deal[i]);
    state->ApplyMove(move);
  }



  std::vector<int> bid_uid;
  bid_uid.push_back(ble::BidIndex(1, ble::kSpadesTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(3, ble::kClubsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(3, ble::kDiamondsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(3, ble::kHeartsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(4, ble::kClubsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(4, ble::kHeartsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(5, ble::kClubsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::BidIndex(5, ble::kHeartsTrump) + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kDouble + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  bid_uid.push_back(ble::kPass + ble::kBiddingActionBase);
  // Bidding
  for (const int uid : bid_uid) {
    ble::BridgeMove move = game->GetMove(uid);
    state->ApplyMove(move);
  }

  std::cout << state->ToString() << std::endl;

  const int num_max_moves = 1;
  const int num_worlds = 20;
  auto resampler = std::make_shared<UniformResampler>(1);
  auto pimc_bot = PIMCBot(resampler, 20);
  while (!state->IsTerminal()) {
    if (IsActingPlayerDeclarerSide(state)) {
      std::vector<bridge_learning_env::BridgeState> worlds;
      worlds.reserve(num_worlds);
      for (int i = 0; i < num_worlds; ++i) {
        auto d = resampler->Resample(*state);
        auto world = ConstructStateFromDeal(d, game, *state);
        worlds.push_back(world);
      }
      const std::vector<bool> possible_worlds(num_worlds, true);
      ParetoFront front = VanillaAlphaMu(*state, num_max_moves, worlds, possible_worlds);
      std::cout << "front:\n" << front.ToString() << std::endl;
      std::exit(1);
    }
    else {
      const auto& move = pimc_bot.Act(*state);
      std::cout << "pimc move: " << move.ToString() << std::endl;
      state->ApplyMove(move);
    }
  }
  std::cout << state->ToString() << std::endl;

  // std::vector<bridge_learning_env::BridgeState> worlds;
  // worlds.reserve(20);
  // for (int i = 0; i < 20; ++i) {
  //   auto d = resampler->Resample(state);
  //   auto world = ConstructStateFromDeal(d, game, state);
  //   worlds.push_back(*world);
  // }
  //
  // std::vector<bool> possible_worlds(20, true);
  // possible_worlds[3] = false;
  // ParetoFront result{};
  // StopSearch(state, 0, worlds, possible_worlds, result);
  // std::cout << result.ToString() << std::endl;

  // const OutcomeVector vec1{{0, 1, 1}, {true, true, true}};
  // const OutcomeVector vec2{{1, 1, 0}, {true, true, true}};
  // //  bool dominate = OutcomeVectorDominate(vec1, vec2);
  // //  std::cout << std::boolalpha << dominate << std::endl;
  // const ParetoFront pf{{vec1, vec2}};
  // std::cout << pf.ToString() << std::endl;
  //
  // const OutcomeVector vec3{{1, 1, 0}, {true, true, true}};
  // const OutcomeVector vec4{{1, 0, 1}, {true, true, true}};
  // //  bool dominate = OutcomeVectorDominate(vec1, vec2);
  // //  std::cout << std::boolalpha << dominate << std::endl;
  // const ParetoFront pf2{{vec3, vec4}};
  // std::cout << pf2.ToString() << std::endl;
  //
  // const ParetoFront product = pf * pf2;
  // std::cout << "product: " << product.ToString() << std::endl;
  // const ParetoFront join = ParetoFrontMin(pf, pf2);
  // std::cout << "join: " << join.ToString() << std::endl;
  // const ParetoFront m = ParetoFrontMax(pf, pf2);
  // std::cout << "max: " << m.ToString() << std::endl;
  // bool is_less_equal = pf <= pf2;
  // std::cout << std::boolalpha << is_less_equal << std::endl;
  return 0;
}
