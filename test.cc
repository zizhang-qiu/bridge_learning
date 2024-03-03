//
// Created by qzz on 2023/12/16.
//
#include <iostream>
#include <memory>
#include <random>

// #include "torch/torch.h"

// #include "rela/batcher.h"

#include "bridge_lib/bridge_game.h"
#include "bridge_lib/bridge_move.h"
#include "bridge_lib/bridge_state.h"
#include "bridge_lib/bridge_utils.h"
#include "bridge_lib/example_cards_ddts.h"
#include "bridge_lib/utils.h"
// #include "dll.h"
#include "dll.h"
#include "playcc/alpha_mu_bot.h"
#include "playcc/dds_evaluator.h"
#include "playcc/pimc.h"
#include "playcc/resampler.h"
#include "playcc/utils.h"

// #include "rlcc/belief_data_gen.h"

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

enum A { a };

enum B { b };

int main(int argc, char** argv) {
  std::vector trajectory = {
      27, 46, 36, 43, 18, 22, 0,  20, 24, 2,  40, 41, 28, 16, 21, 10, 42,
      32, 48, 47, 13, 17, 3,  5,  12, 25, 34, 8,  29, 38, 23, 4,  30, 26,
      35, 19, 9,  44, 7,  51, 14, 1,  45, 15, 49, 33, 11, 50, 39, 31, 6,
      37, 52, 52, 58, 59, 52, 61, 52, 62, 52, 64, 52, 52, 52, 39, 31, 3,
      47, 41, 13, 1,  45, 21, 5,  49, 17, 27, 16, 7,  51, 50, 18, 2,  6,
      15, 12, 22, 23, 48, 4,  24, 32, 0,  20, 28, 44, 25, 34, 37, 9,  43,
      14, 26, 11, 10, 30, 46, 36, 33, 40, 8,  29, 38, 35, 19, 42};
  std::vector<std::vector<int>> trajectories = {trajectory};

  std::mt19937 rng(13);
  const auto deal = ble::Permutation(ble::kNumCards, rng);

  auto state = ConstructStateFromDeal(deal, ble::default_game);

  while (state.IsInPhase(ble::Phase::kAuction)) {
    ApplyRandomMove(state, rng);
  }

  std::cout << state << std::endl;

  // Test(200);


  

  std::shared_ptr<Resampler> resampler = std::make_shared<UniformResampler>(22);
  const AlphaMuConfig cfg{2, 40, false, true, true, true, kNumTotalTricks};
  AlphaMuBot bot{resampler, cfg, state.GetContract().declarer};

  const PIMCConfig pimc_cfg{40, false};
  PIMCBot pimc_bot{resampler, pimc_cfg};

  while (!state.IsTerminal()) {
    ble::BridgeMove move;
    if (state.CurrentPlayer() == state.GetContract().declarer) {
      move = bot.Step(state);
      std::cout << "Alpha mu move: " << move << std::endl;
    }else{
      move = pimc_bot.Step(state);
      std::cout << "PIMC move: " << move << std::endl;
    }
    state.ApplyMove(move);
  }
  std::cout << state << std::endl;
}