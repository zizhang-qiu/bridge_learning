//
// Created by qzz on 2023/9/21.
//

#include "bridge_card.h"

namespace bridge_learning_env {

std::string BridgeCard::ToString() const {
  if (!IsValid()) {
    return {"Invalid"};
  }
  return {kSuitChar[suit_], kRankChar[rank_]};
}

bool BridgeCard::operator==(const BridgeCard &other_card) const {
  return other_card.CardSuit() == CardSuit() && other_card.Rank() == Rank();
}

} // bridge