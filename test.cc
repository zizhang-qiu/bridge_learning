//
// Created by qzz on 2023/12/16.
//
#include <iostream>

#include "torch/torch.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_cat.h"
// #include "rela/batcher.h"
#include "playcc/alpha_mu_search.h"
#include "playcc/bridge_state_without_hidden_info.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/example_cards_ddts.h"
#include "bridge_lib/canonical_encoder.h"
#include "playcc/file.h"
#include "playcc/pimc.h"
#include "playcc/transposition_table.h"
#include "playcc/deal_analyzer.h"
#include "rlcc/belief_data_gen.h"

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);

std::vector<size_t> FindNonZeroIndices(const std::vector<int>& vec) {
  std::vector<size_t> indices;

  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] != 0) {
      indices.push_back(i);
    }
  }

  return indices;
}

int main(int argc, char** argv) {
  // rela::FutureReply fut(0);
  // rela::TensorDict d1 = {
  //   {"a", torch::tensor({{1, 2, 3}, {11, 22, 33}})},
  //   {"b", torch::tensor({{4, 5, 6}, {44, 55, 66}})}
  // };
  // rela::TensorDict d2 = {
  //   {"a", torch::tensor({{11, 22, 33}, {111, 222, 333}})},
  //   {"b", torch::tensor({{44, 55, 66}, {444, 555, 666}})}
  // };

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

  std::cout << torch::cuda::is_available() << std::endl;
  torch::Device d{"cuda"};

  auto state2 = std::make_unique<ble::BridgeState>(game);

  std::cout << state2->ToString() << std::endl;


  // for (int i = 0; i < 4; ++i)
    // ApplyRandomMove(state, rng);
  // std::cout << state << std::endl;
  // const auto encoder = ble::CanonicalEncoder(game, ble::kNumTricks);
  // // const auto encoding = encoder.Encode({state});
  // // rela::utils::printVector(encoding);
  std::vector trajectory = {
    41, 47, 7, 44, 50, 45, 5, 0, 48, 22, 32, 25, 11, 24, 10, 16, 3, 29, 12, 13, 30, 51, 31, 34, 15, 49, 23, 43, 42, 6,
    26, 46, 40, 28, 36, 9, 39, 27, 21, 37, 19, 14, 1, 2, 4, 33, 17, 38, 35, 20, 8, 18, 58, 59, 52, 61, 63, 67, 68, 72,
    52, 52, 52
  };
  std::vector<std::vector<int>> trajectories = {trajectory};

  BeliefDataGen gen{trajectories, 1, game};
  auto data = gen.AllData("cuda");

  rela::utils::printMap(data);

  // auto non_zero_indices = FindNonZeroIndices(encoding);
  // rela::utils::printVector(non_zero_indices);
  //
  // std::cout << encoder.Shape()[0] << std::endl;

  // auto observation = ble::BridgeObservation(state);
  // auto encoding = encoder.Encode(observation);
  // encoding = std::vector<int>(encoding.begin(), encoding.begin() + encoder.GetAuctionTensorSize());
  // rela::utils::printVector(encoding);
  // encoding = encoder.EncodeOtherHands({state});
  // rela::utils::printVector(encoding);
}
