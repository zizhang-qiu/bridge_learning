//
// Created by qzz on 2023/11/27.
//

#ifndef BRIDGE_LEARNING_PLAYCC_WORLDS_H_
#define BRIDGE_LEARNING_PLAYCC_WORLDS_H_
#include "bridge_lib/bridge_state.h"

#include "utils.h"
namespace ble = bridge_learning_env;
class Worlds {
 public:
  Worlds() = default;

  Worlds(const Worlds &) = default;

  explicit Worlds(const std::vector<ble::BridgeState> &worlds) : states_(worlds), possible_(states_.size(), true) {}

  Worlds(const std::vector<ble::BridgeState> &worlds, const std::vector<int> &possible) {
    SPIEL_CHECK_EQ(worlds.size(), possible.size());
    states_ = worlds;
    possible_ = possible;
  }

  Worlds(const std::vector<std::array<int, ble::kNumCards>> &deals, const ble::BridgeState &state) :
      possible_(deals.size(), true) {
    for (const auto &deal : deals) {
      auto s = ConstructStateFromDealAndOriginalState(deal, state.ParentGame(), state);
      states_.push_back(s);
    }
  }

  void ApplyMove(const ble::BridgeMove &move);

  [[nodiscard]] std::vector<ble::BridgeMove> GetAllPossibleMoves() const;

  [[nodiscard]] std::vector<bool> MoveIsLegal(const ble::BridgeMove &move) const;

  [[nodiscard]] int Size() const { return static_cast<int>(states_.size()); }

  [[nodiscard]] std::string ToString() const;

  [[nodiscard]] std::vector<ble::BridgeState> States() const { return states_; }

  [[nodiscard]] std::vector<int> Possible() const { return possible_; }

  Worlds CloneWithPossibility(const std::vector<int> &possible) { return {states_, possible}; }

  [[nodiscard]] Worlds Child(const ble::BridgeMove &move) const {
    auto cloned = *this;
    cloned.ApplyMove(move);
    return cloned;
  }

  [[nodiscard]] bool IsMaxNode() const {
    for (int i = 0; i < Size(); ++i) {
      if (possible_[i]) {
        return ble::Partnership(states_[i].CurrentPlayer()) == ble::Partnership(states_[i].GetContract().declarer);
      }
    }
    return false;
  }

 private:
  std::vector<ble::BridgeState> states_;
  std::vector<int> possible_;
};

std::ostream &operator<<(std::ostream &stream, const Worlds &worlds);
#endif // BRIDGE_LEARNING_PLAYCC_WORLDS_H_
