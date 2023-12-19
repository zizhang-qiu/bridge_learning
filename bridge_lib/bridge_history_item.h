//
// Created by qzz on 2023/9/24.
//

#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_HISTORY_ITEM_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_HISTORY_ITEM_H_
#include <iostream>

#include "bridge_move.h"

namespace bridge_learning_env {

struct BridgeHistoryItem {
  explicit BridgeHistoryItem(BridgeMove move_made) : move(move_made) {}
  BridgeHistoryItem(const BridgeHistoryItem &past_move) = default;
  std::string ToString() const;

  BridgeMove move;
  Player player = -1;
  Player deal_to_player = -1;

  Suit suit = kInvalidSuit;
  int rank = -1;
  Denomination denomination = kInvalidDenomination;
  int level = -1;
  OtherCalls other_call = kNotOtherCall;
};

std::ostream &operator<<(std::ostream &stream, const BridgeHistoryItem &item);

} // namespace bridge

#endif // BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_HISTORY_ITEM_H_
