//
// Created by qzz on 2023/9/21.
//

#ifndef BRIDGE_LIB_BRIDGE_CARD_H
#define BRIDGE_LIB_BRIDGE_CARD_H
#include <iostream>

#include "bridge_utils.h"

namespace bridge_learning_env {

class BridgeCard {
 public:
  BridgeCard() = default; // Create an invalid card.
  BridgeCard(const Suit suit, const int rank) : suit_(suit), rank_(rank) {}
  bool operator==(const BridgeCard &other_card) const;
  [[nodiscard]] bool IsValid() const {
    return suit_ >= 0 && rank_ >= 0;
  }
  [[nodiscard]] std::string ToString() const;
  [[nodiscard]] Suit CardSuit() const { return suit_; }
  [[nodiscard]] int Rank() const { return rank_; }
  [[nodiscard]] int Index() const { return CardIndex(suit_, rank_); }

 private:
  Suit suit_ = Suit::kInvalidSuit;
  int rank_ = -1;
};

std::ostream &operator<<(std::ostream &stream, const BridgeCard &card);

std::ostream &operator<<(std::ostream &stream, const std::vector<BridgeCard> &cards);

} // namespace bridge

#endif /* BRIDGE_LIB_BRIDGE_CARD_H */
