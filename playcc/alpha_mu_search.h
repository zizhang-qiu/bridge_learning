//
// Created by qzz on 2023/11/14.
//

#ifndef ALPHA_MU_SEARCHER_H
#define ALPHA_MU_SEARCHER_H

#include "worlds.h"
#include "bridge_lib/bridge_state.h"
#include "pareto_front.h"
#include "bridge_state_without_hidden_info.h"
#include "utils.h"
namespace ble = bridge_learning_env;

// The stop function should return a boolean value which
// indicates whether to stop the search and a pareto front
// represents the double dummy evaluation if the search is end.
struct StopResult{
  bool stop;
  ParetoFront result;
};

bool DoubleDummyEvaluation(const ble::BridgeState& state);

std::vector<int> DoubleDummyEvaluation(const Worlds& worlds);

ble::BridgeState ApplyMove(const ble::BridgeMove& move, ble::BridgeState state);


StopResult StopSearch(const ble::BridgeStateWithoutHiddenInfo& state,
                int num_max_moves,
                const Worlds& worlds);
//
//ParetoFront VanillaAlphaMu(const ble::BridgeStateWithoutHiddenInfo& state,
//                           int num_max_moves,
//                           const std::vector<ble::BridgeState>& worlds,
//                           const std::vector<bool>& possible_worlds);

ParetoFront VanillaAlphaMu(const ble::BridgeStateWithoutHiddenInfo &state,
                           int num_max_moves,
                           const Worlds &worlds);

#endif // ALPHA_MU_SEARCHER_H
