//
// Created by qzz on 2023/9/21.
//

#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_HAND_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_HAND_H_
#include <vector>
#include <array>

#include "bridge_card.h"

namespace bridge_learning_env {
struct HandEvaluation {
  int hcp;
  int control;
  int zar_hcp;
  int zar_dp;
  std::array<int, kNumSuits> suit_length;
};

class BridgeHand {
  public:
    BridgeHand() = default;

    BridgeHand(const BridgeHand& hand) = default;

    [[nodiscard]] const std::vector<BridgeCard>& Cards() const { return cards_; }

    void AddCard(const BridgeCard& card);

    void RemoveFromHand(Suit suit, int rank, std::vector<BridgeCard>* played_cards);

    [[nodiscard]] std::string ToString() const;

    bool IsFullHand() const { return cards_.size() == kNumCardsPerHand; }

    int HighCardPoints() const;

    int ControlValue() const;

    int ZarHighCardPoints() const;

    bool IsCardInHand(BridgeCard card) const;

    std::array<int, kNumSuits> SuitLength() const;

    int ZarDistributionPoints() const;

    std::array<std::vector<BridgeCard>, kNumSuits> CardsBySuits() const;

    HandEvaluation GetHandEvaluation() const;

  private:
    std::vector<BridgeCard> cards_;
};

std::ostream& operator<<(std::ostream& stream, const BridgeHand& hand);
} // bridge

#endif //BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_HAND_H_
