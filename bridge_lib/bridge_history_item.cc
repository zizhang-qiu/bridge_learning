//
// Created by qzz on 2023/9/24.
//

#include "bridge_history_item.h"
#include "utils.h"

namespace bridge {
std::string BridgeHistoryItem::ToString() const {
  std::string str = StrCat("<", move.ToString());
  if (player >= 0) {
    str += StrCat(" by player ", player, ">");
  }
  return str;
}
} // bridge