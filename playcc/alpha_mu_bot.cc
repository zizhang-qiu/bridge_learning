//
// Created by qzz on 2023/12/7.
//

#include "alpha_mu_bot.h"

bool IsFirstMaxNode(const ble::BridgeStateWithoutHiddenInfo &state) {
  return state.PlayHistory().size() == 1;
}

bool IsFirstMaxNode(const ble::BridgeState &state) {
  return state.PlayHistory().size() == 1;
}

ble::BridgeMove VanillaAlphaMuBot::Act(const ble::BridgeState &state) {
  SPIEL_CHECK_FALSE(state.IsTerminal());
//  const auto &legal_moves = state.LegalMoves();
  const auto legal_moves = GetLegalMovesWithoutEquivalentCards(state);
//  for (const auto move : legal_moves) {
//    std::cout << move << std::endl;
//  }
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
ParetoFront VanillaAlphaMuBot::Search(const ble::BridgeState &state) const {

  const auto deals = ResampleMultipleDeals(resampler_, state, cfg_.num_worlds);
//  std::cout << "Deals sampled in alpha mu:\n" << std::endl;
//  for (const auto d : deals) {
//    PrintArray(d);
//  }
  ParetoFront front =
      VanillaAlphaMu(ble::BridgeStateWithoutHiddenInfo(state), cfg_.num_max_moves, Worlds(deals, state));
  return front;
}
ble::BridgeMove VanillaAlphaMuBot::Act(const ble::BridgeState &state, const vector<ble::BridgeState> &worlds) const {
  const auto &legal_moves = state.LegalMoves();
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

ble::BridgeMove AlphaMuBot::Act(const ble::BridgeState &state) {
  SPIEL_CHECK_FALSE(state.IsTerminal());

//  const auto &legal_moves = state.LegalMoves();
  const auto legal_moves = GetLegalMovesWithoutEquivalentCards(state);
  const int num_legal_moves = static_cast<int>(legal_moves.size());
  // Only one legal move, return it.
  if (num_legal_moves == 1 && !cfg_.search_with_one_legal_move) {
    return legal_moves[0];
  }

  const auto deals = ResampleMultipleDeals(resampler_, state, cfg_.num_worlds);
  const ParetoFront
      front = Search(ble::BridgeStateWithoutHiddenInfo(state), cfg_.num_max_moves, Worlds(deals, state), {});
  auto best = front.BestOutcome();
  if (best.move.MoveType() == bridge_learning_env::BridgeMove::kInvalid) {
    best.move = state.LegalMoves()[0];
  }
  return best.move;
}

ParetoFront AlphaMuBot::Search(const ble::BridgeStateWithoutHiddenInfo &state, int num_max_moves,
                               const Worlds &worlds, ParetoFront alpha) {
  if (IsFirstMaxNode(state)) {
    tt_.Clear();
  }
  auto [stop, result] = StopSearch(state, num_max_moves, worlds);
  if (stop) {
    tt_[state] = result;
    return result;
  }

  bool is_state_in_tt = tt_.HasKey(state);

  if (ble::Partnership(state.CurrentPlayer()) != ble::Partnership(state.GetContract().declarer)) {
    // Min node.
    ParetoFront mini{};
    // Early cut.
    if (cfg_.early_cut) {
      if (is_state_in_tt && (tt_[state] <= alpha)) {
        return mini;
      }
    }

    std::vector<ble::BridgeMove> all_moves = worlds.GetAllPossibleMoves();
    if (is_state_in_tt) {
      auto it = std::find(all_moves.begin(), all_moves.end(), tt_[state].BestOutcome().move);
      if (it != all_moves.end()) {
        all_moves.erase(it);
      }
    }

    for (const auto &move : all_moves) {
      const auto s = state.Child(move);
      const auto next_worlds = worlds.Child(move);
      ParetoFront f = Search(s, num_max_moves, next_worlds, {});
      f.SetMove(move);
      mini = ParetoFrontMin(mini, f);
    }

    tt_[state] = mini;
    return mini;
  } else {
    // Max node.
    ParetoFront front{};
    const std::vector<ble::BridgeMove> all_moves = worlds.GetAllPossibleMoves();
    for (const auto &move : all_moves) {

      const auto s = state.Child(move);
      const auto next_worlds = worlds.Child(move);
      ParetoFront f = Search(s, num_max_moves - 1, next_worlds, front);
      f.SetMove(move);

      front = ParetoFrontMax(front, f);

      if (cfg_.root_cut) {

        if (num_max_moves == cfg_.num_max_moves) {
          // Root node.
          if (tt_.HasKey(state) && tt_[state].BestOutcome().Score() == front.BestOutcome().Score()) {
            break;
          }
        }
      }
    }
    tt_[state] = front;
    return front;
  }

}

