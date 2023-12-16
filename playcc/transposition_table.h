//
// Created by qzz on 2023/12/13.
//

#ifndef BRIDGE_LEARNING_PLAYCC_TRANSPOSITION_TABLE_H_
#define BRIDGE_LEARNING_PLAYCC_TRANSPOSITION_TABLE_H_
#include "bridge_state_without_hidden_info.h"
#include "pareto_front.h"
#include <unordered_map>
namespace ble = bridge_learning_env;
class TranspositionTable {
 public:
  void Clear() {
    table_.clear();
  }

  [[nodiscard]] bool HasKey(const ble::BridgeStateWithoutHiddenInfo &state) const {
    return (table_.find(state) != table_.end());
  }

  [[nodiscard]] auto Table() const {
    return table_;
  }

  void Insert(const ble::BridgeStateWithoutHiddenInfo &state,
              const ParetoFront &front) {
    table_[state] = front;
  }

  ParetoFront &operator[](const ble::BridgeStateWithoutHiddenInfo &state) {
    auto it = table_.find(state);

    if (it == table_.end()) {
      auto result = table_.emplace(state, ParetoFront{});
      it = result.first;
    }

    return it->second;
  }

  [[nodiscard]] std::string ToString() const;

 private:
  std::unordered_map<ble::BridgeStateWithoutHiddenInfo, ParetoFront, ble::BridgeStateWithoutHiddenInfo> table_{};
};

std::ostream &operator<<(std::ostream &stream, const TranspositionTable &tt);
#endif //BRIDGE_LEARNING_PLAYCC_TRANSPOSITION_TABLE_H_
