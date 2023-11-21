//
// Created by qzz on 2023/11/14.
//

#ifndef ALPHA_MU_SEARCHER_H
#define ALPHA_MU_SEARCHER_H

#include "bridge_lib/bridge_state.h"
#include "pareto_front.h"
#include "utils.h"
namespace ble = bridge_learning_env;

bool DoubleDummyEvaluation(const ble::BridgeState& state);

ble::BridgeState ApplyMove(const ble::BridgeMove& move, ble::BridgeState state);

bool StopSearch(const ble::BridgeState& state,
                int num_max_moves,
                const std::vector<ble::BridgeState>& worlds,
                const std::vector<bool>& possible_worlds,
                ParetoFront& result);

ParetoFront VanillaAlphaMu(const ble::BridgeState& state,
                           int num_max_moves,
                           const std::vector<ble::BridgeState>& worlds,
                           const std::vector<bool>& possible_worlds);

#endif // ALPHA_MU_SEARCHER_H
