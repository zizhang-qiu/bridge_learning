//
// Created by qzz on 2023/12/7.
//

#include "alpha_mu_bot.h"

ble::BridgeMove VanillaAlphaMuBot::Act(const ble::BridgeState& state) {
  SPIEL_CHECK_FALSE(state.IsTerminal());
  const auto& legal_moves = state.LegalMoves();
  const int num_legal_moves = static_cast<int>(legal_moves.size());
  // Only one legal move, return it.
  if (num_legal_moves == 1 && !cfg_.search_with_one_legal_move) {
    return legal_moves[0];
  }

  const ParetoFront front = Search(state);
  auto best = front.BestOutcome();
  if (best.move.MoveType() == bridge_learning_env::BridgeMove::kInvalid) {
    best.move = state.LegalMoves()[0];
  }
  return best.move;
}
ParetoFront VanillaAlphaMuBot::Search(const ble::BridgeState& state) const {
  //  std::vector<bridge_learning_env::BridgeState> worlds;
  //  worlds.reserve(cfg_.num_worlds);
  //  for (int i = 0; i < cfg_.num_worlds; ++i) {
  //    auto resample_result = resampler_->Resample(state);
  //    if (resample_result.success) {
  //      auto world = ConstructStateFromDeal(resample_result.result, state.ParentGame(), state);
  //      worlds.push_back(world);
  //    }
  //    else {
  //      --i;
  //    }
  //  }
  //  const std::vector<bool> possible_worlds(cfg_.num_worlds, true);
  const auto deals = ResampleMultipleDeals(resampler_, state, cfg_.num_worlds);
//  std::cout << "Deals sampled in alpha mu:\n" << std::endl;
//  for (const auto d : deals) {
//    PrintArray(d);
//  }
  ParetoFront front =
      VanillaAlphaMu(ble::BridgeStateWithoutHiddenInfo(state), cfg_.num_max_moves, Worlds(deals, state));
  return front;
}
ble::BridgeMove VanillaAlphaMuBot::Act(const ble::BridgeState& state, const vector<ble::BridgeState>& worlds) const {
  const auto& legal_moves = state.LegalMoves();
  const int num_legal_moves = static_cast<int>(legal_moves.size());
  // Only one legal move, return it.
  if (num_legal_moves == 1) {
    return legal_moves[0];
  }

  const std::vector<bool> possible_worlds(cfg_.num_worlds, true);
  ParetoFront front = VanillaAlphaMu(ble::BridgeStateWithoutHiddenInfo(state), cfg_.num_max_moves, Worlds(worlds));
  auto best = front.BestOutcome();
  if (best.move.MoveType() == bridge_learning_env::BridgeMove::kInvalid) {
    best.move = state.LegalMoves()[0];
  }
  return best.move;
}
