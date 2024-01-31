//
// Created by qzz on 2023/11/27.
//

#include "worlds.h"

#include "common_utils/log_utils.h"
std::vector<bool> Worlds::MoveIsLegal(const ble::BridgeMove &move) const {
  std::vector<bool> results;
  results.reserve(states_.size());
  for (const auto &s : states_) {
    results.push_back(s.MoveIsLegal(move));
  }
  return results;
}

void Worlds::ApplyMove(const ble::BridgeMove &move) {
  for (int i = 0; i < Size(); ++i) {
    if (possible_[i]) {
      if (states_[i].MoveIsLegal(move)) {
        states_[i].ApplyMove(move);
      }
      else {
        possible_[i] = false;
      }
    }
  }
}

std::string Worlds::ToString() const {
  std::string rv;
  for (int i = 0; i < Size(); ++i) {
    rv += states_[i].ToString() + "\n\n";
  }
  return rv;
}

std::vector<ble::BridgeMove> Worlds::GetAllPossibleMoves() const {
  SPIEL_CHECK_FALSE(states_.empty());
  const auto game = states_[0].ParentGame();
  auto compare_func = [&game](const ble::BridgeMove &lhs, const ble::BridgeMove &rhs) {
    return game->GetMoveUid(lhs) < game->GetMoveUid(rhs);
  };
  std::set<ble::BridgeMove, decltype(compare_func)> possible_moves(compare_func);
  for (int i = 0; i < Size(); ++i) {
    if (possible_[i]) {
      std::vector<ble::BridgeMove> legal_moves;
      if (IsMaxNode()) {
        legal_moves = GetLegalMovesWithoutEquivalentCards(states_[i]);
      } else {
        legal_moves = states_[i].LegalMoves();
      }
//      const auto legal_moves = GetLegalMovesWithoutEquivalentCards(states_[i]);
      for (const ble::BridgeMove &move : legal_moves) {
        possible_moves.emplace(move);
      }
    }
  }
  return {possible_moves.begin(), possible_moves.end()};
}

std::ostream &operator<<(ostream &stream, const Worlds &worlds) {
  stream << worlds.ToString();
  return stream;
}
