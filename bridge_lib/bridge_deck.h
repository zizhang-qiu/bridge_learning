//
// Created by qzz on 2023/10/23.
//

#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_STATE_2_CC_BRIDGE_DECK_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_STATE_2_CC_BRIDGE_DECK_H_

#include "third_party/dds/src/TransTableL.h"
#include "third_party/dds/src/SolverIF.h"
#include "third_party/dds/src/Memory.h"
#include "auction_tracker.h"
#include "bridge_history_item.h"
#include "bridge_hand.h"
#include "bridge_card.h"
#include "bridge_utils.h"
#include "bridge_game.h"
#include <set>
#include <random>
namespace bridge_learning_env {
class BridgeDeck {
 public:
  // The deck of bridge always contains 52 cards, no other arguments.
  BridgeDeck() : total_count_(kNumCards), card_in_deck_(kNumCards, true) {}
  BridgeCard DealCard(Suit suit, int rank);
  BridgeCard DealCard(int card_index);
  BridgeCard DealCard(mt19937 &rng);
  [[nodiscard]] int Size() const { return total_count_; }
  [[nodiscard]] bool Empty() const { return total_count_ == 0; }
  [[nodiscard]] bool CardInDeck(Suit suit, int rank) const {
    return card_in_deck_[CardIndex(suit, rank)];
  }

  [[nodiscard]] std::vector<BridgeCard> Cards() const;
  [[nodiscard]] std::array<std::vector<BridgeCard>, kNumSuits> CardsBySuits() const;

 private:

  std::vector<bool> card_in_deck_;
  int total_count_;

};
}
#endif //BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_STATE_2_CC_BRIDGE_DECK_H_
