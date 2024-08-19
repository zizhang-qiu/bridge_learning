//
// Created by qzz on 2023/12/16.
//
#include <ios>
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
#include "playcc/alpha_mu_bot.h"
#include "playcc/dds_evaluator.h"
#include "playcc/pimc.h"
#include "playcc/resampler.h"
#include "playcc/utils.h"
#include "rlcc/bridge_actor.h"
#include "rlcc/bridge_dataset.h"
#include "rlcc/bridge_env.h"
#include "rlcc/bridge_env_actor.h"
#include "rlcc/duplicate_env.h"
#include "rlcc/env_actor.h"
#include "rlcc/detailed_encoder.h"

// #include "rlcc/belief_data_gen.h"

const ble::GameParameters params = {};
const auto game = std::make_shared<ble::BridgeGame>(params);

std::vector<size_t> FindNonZeroIndices(const std::vector<int> &vec) {
  std::vector<size_t> indices;

  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] != 0) {
      indices.push_back(i);
    }
  }

  return indices;
}

int main(int argc, char **argv) {
  std::mt19937 rng(22);
  ble::BridgeState state{ble::default_game};
  while (state.IsChanceNode()) {
    state.ApplyRandomChance();
  }

  int num_moves = 0;
  while (state.IsInPhase(ble::Phase::kAuction) && num_moves < 7) {
    const auto legal_moves = state.LegalMoves();
    const auto random_move = rela::utils::UniformSample(legal_moves, rng);
    state.ApplyMove(random_move);
    ++num_moves;
  }

  std::cout << state << std::endl;

  auto encoder = DetailedEncoder(ble::default_game);
  auto encoding = encoder.Encode({state});
  std::cout << encoding << std::endl;
}