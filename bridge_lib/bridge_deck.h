//
// Created by qzz on 2023/10/23.
//

#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_STATE_2_CC_BRIDGE_DECK_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_STATE_2_CC_BRIDGE_DECK_H_
#include <random>

#include "utils.h"
#include "bridge_card.h"
#include "bridge_hand.h"
#include "bridge_utils.h"

namespace bridge_learning_env {
class BridgeDeck {
  public:
  // The deck of bridge always contains 52 cards, no other arguments.
  BridgeDeck() : card_in_deck_(kNumCards, true), total_count_(kNumCards) {}
  BridgeCard DealCard(Suit suit, int rank);
  BridgeCard DealCard(int card_index);
  BridgeCard DealCard(std::mt19937 &rng);
  [[nodiscard]] int Size() const { return total_count_; }
  [[nodiscard]] bool Empty() const { return total_count_ == 0; }
  [[nodiscard]] bool CardInDeck(const Suit suit, const int rank) const { return card_in_deck_[CardIndex(suit, rank)]; }

  [[nodiscard]] std::vector<BridgeCard> Cards() const;
  [[nodiscard]] std::array<std::vector<BridgeCard>, kNumSuits> CardsBySuits() const;

  private:
  std::vector<bool> card_in_deck_;
  int total_count_;
};
} // namespace bridge_learning_env
#endif // BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_STATE_2_CC_BRIDGE_DECK_H_
