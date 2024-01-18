//
// Created by qzz on 2023/12/16.
//
#include <iostream>

#include "torch/torch.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_cat.h"
#include "rela/batcher.h"
#include "playcc/alpha_mu_search.h"
#include "playcc/bridge_state_without_hidden_info.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/example_cards_ddts.h"
#include "bridge_lib/canonical_encoder.h"
#include "playcc/file.h"
#include "playcc/pimc.h"
#include "playcc/transposition_table.h"
#include "playcc/deal_analyzer.h"

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);

int main(int argc, char** argv) {
  // rela::FutureReply fut(0);
  rela::TensorDict d1 = {
    {"a", torch::tensor({{1, 2, 3}, {11, 22, 33}})},
    {"b", torch::tensor({{4, 5, 6}, {44, 55, 66}})}
  };
  rela::TensorDict d2 = {
    {"a", torch::tensor({{11, 22, 33}, {111, 222, 333}})},
    {"b", torch::tensor({{44, 55, 66}, {444, 555, 666}})}
  };

  // //
  // torch::Tensor basic_indices = torch::arange(0, ble::kNumCardsPerHand) * ble::kNumPlayers;
  // std::cout << basic_indices << std::endl;
  // basic_indices.scatter_(0, torch::tensor({0, 1}), 3);
  // std::cout << basic_indices << std::endl;
  // std::cout << basic_indices << std::endl;
  auto deal = ble::example_deals[0];
  auto state = ConstructStateFromDeal(deal, game);
  std::mt19937 rng(1);
  while (state.CurrentPhase() == ble::Phase::kAuction) {
    const auto legal_moves = state.LegalMoves();
    const auto random_move = UniformSample(legal_moves, rng);
    state.ApplyMove(random_move);
  }


  ApplyRandomMove(state, rng);
  std::cout << state << std::endl;
  const auto encoder = ble::CanonicalEncoder(game, ble::kNumTricks);
  const auto encoding = encoder.Encode({state});
  rela::utils::printVector(encoding);
}
