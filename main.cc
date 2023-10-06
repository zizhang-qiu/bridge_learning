#include "bridge_lib/canonical_encoder.h"
#include "bridge_lib/bridge_observation.h"
#include "bridge_lib/example_cards_ddts.h"



using namespace bridge;
int main(int, char **) {
  // for(int i=0; i<bridge::kNumCards; ++i){
  //     std::cout << bridge::CardString(i) << "\n";
  // }
//   const bridge::GameParameters params = {
//      {"expect_true", "true"},
//       {"expect_1", "1"},
//       {"expect_5.66", "5.66"},
//       {"expect_yes", "yes"}
//   };
  // assert( bridge::ParameterValue(params, "expect_true", false)==true);
  // assert(bridge::ParameterValue(params, "expect_1", 1) == 1);
  // assert(bridge::ParameterValue(params, "expect_5.66", 0.01) == 5.66);
  // assert(bridge::ParameterValue(params, "expect_yes", "no") == "yes");
  // for(int i=0; i<bridge::kNumContracts; ++i){
  //     std::cout << bridge::kAllContracts[i].ToString() << "\n";
  // }

  //    unsigned int seed = 1;
  //    std::mt19937 rng(seed);
  ////    auto cards = bridge::Permutation(bridge::kNumCards, rng);
  ////    bridge::PrintVector(cards);
  auto cards = bridge::example_cards[0];
  //
  //
  //    // auto state = std::make_shared<bridge::BridgeState>(false, false);
  const bridge::GameParameters params = {{"is_dealer_vulnerable", "false"},
                                         {"is_non_dealer_vulnerable", "false"},
                                         {"seed", "42"},
                                         {"dealer", "1"}};
  auto game = std::make_shared<BridgeGame>(params);

  auto state = BridgeState2(game);

  for(const auto card:cards){
    auto move = game->GetChanceOutcome(card);
    state.ApplyMove(move);
  }

  BridgeMove move = game->GetMove(55);
  state.ApplyMove(move);
  move = game->GetMove(53);
  state.ApplyMove(move);
  move = game->GetMove(52);
  state.ApplyMove(move);

  auto obs = BridgeObservation(state, state.CurrentPlayer());
  std::cout << obs.ToString() << std::endl;

  CanonicalEncoder encoder{game};
  std::vector<int> encoding(480, 0);
  int offset = 0;
  int length = EncodeVulnerability(obs, game, 0, &encoding);
  std::cout << length << std::endl;
  offset += length;

  length = EncodeAuction(obs, offset, &encoding);
  std::cout << length << std::endl;
  offset += length;

  length = EncodePlayerHand(obs, offset, &encoding);
  std::cout << length << std::endl;
  PrintVector(encoding);

  auto encoding_ = encoder.Encode(obs);
  std::cout << "Equal? :" << (encoding == encoding_) << std::endl;

  while(!state.IsTerminal()){
    auto legal_moves = state.LegalMoves();
    move = legal_moves[0];
    std::cout << move.ToString() << std::endl;
    state.ApplyMove(move);
  }

  std::cout << state.GetContract().ToString() << std::endl;
  std::cout << state.ToString() << std::endl;

}
