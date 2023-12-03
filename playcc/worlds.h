//
// Created by qzz on 2023/11/27.
//

#ifndef BRIDGE_LEARNING_PLAYCC_WORLDS_H_
#define BRIDGE_LEARNING_PLAYCC_WORLDS_H_
#include "bridge_lib/bridge_state.h"
namespace ble = bridge_learning_env;
class Worlds {
  public:
  Worlds() = default;

  Worlds(const Worlds&) = default;

  explicit Worlds(const std::vector<ble::BridgeState>& worlds) : states_(worlds), possible_(states_.size(), true) {}

  Worlds(const std::vector<ble::BridgeState>& worlds, const std::vector<bool>& possible) {
    assert(worlds.size() == possible.size());
    states_ = worlds;
    possible_ = possible;
  }

  void ApplyMove(const ble::BridgeMove& move);

  [[nodiscard]] std::vector<bool> MoveIsLegal(const ble::BridgeMove& move) const;

  [[nodiscard]] int Size() const { return static_cast<int>(states_.size()); }

  [[nodiscard]] std::string ToString() const;

  [[nodiscard]] std::vector<ble::BridgeState> States() const { return states_; }

  [[nodiscard]] std::vector<bool> Possible() const { return possible_; }

  Worlds CloneWithPossibility(const std::vector<bool>& possible) { return {states_, possible}; }

  private:
  std::vector<ble::BridgeState> states_;
  std::vector<bool> possible_;
};
#endif // BRIDGE_LEARNING_PLAYCC_WORLDS_H_
