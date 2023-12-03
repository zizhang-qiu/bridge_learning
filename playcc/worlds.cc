//
// Created by qzz on 2023/11/27.
//

#include "worlds.h"
std::vector<bool> Worlds::MoveIsLegal(const ble::BridgeMove& move) const {
  std::vector<bool> results;
  results.reserve(states_.size());
  for (const auto& s : states_) {
    results.push_back(s.MoveIsLegal(move));
  }
  return results;
}
void Worlds::ApplyMove(const ble::BridgeMove& move) {
  for (int i = 0; i < Size(); ++i) {
    if (possible_[i]) {
      states_[i].ApplyMove(move);
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


