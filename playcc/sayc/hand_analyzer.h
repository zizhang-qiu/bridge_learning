//
// Created by qzz on 2024/1/26.
//

#ifndef BRIDGE_LEARNING_PLAYCC_SAYC_HAND_ANALYZER_H_
#define BRIDGE_LEARNING_PLAYCC_SAYC_HAND_ANALYZER_H_
#include <algorithm>

#include "bridge_lib/bridge_hand.h"
namespace ble = bridge_learning_env;

namespace sayc {
// https://en.wikipedia.org/wiki/Balanced_hand
inline constexpr int kNumBalancedHandTypes = 3; // 4-4-3-2, 4-3-3-3, 5-3-3-2.

constexpr std::array<std::array<int, ble::kNumSuits>, kNumBalancedHandTypes>
balanced_hand = {
    {
        {4, 4, 3, 2}, {4, 3, 3, 3}, {5, 3, 3, 2}
    }
};

class HandAnalyzer {
  public:
    explicit HandAnalyzer(const ble::BridgeHand& hand)
      : hand_(hand) {
      cards_by_suit_ = hand_.CardsBySuits();
    }

    [[nodiscard]] bool IsBalanced() const {
      const auto suit_length = GetSortedSuitLength();
      const auto it = std::find(balanced_hand.begin(), balanced_hand.end(),
                                suit_length);
      return it != balanced_hand.end();
    }

    [[nodiscard]] std::array<int, ble::kNumSuits> GetSuitLength() const {
      std::array<int, ble::kNumSuits> suit_length{};
      for (const ble::Suit suit : ble::kAllSuits) {
        suit_length[suit] = static_cast<int>(cards_by_suit_[suit].size());
      }
      return suit_length;
    }

    // Get sorted suit length, descending order.
    [[nodiscard]] std::array<int, ble::kNumSuits> GetSortedSuitLength() const {
      auto suit_length = GetSuitLength();
      std::sort(suit_length.begin(), suit_length.end(), std::greater<>());
      return suit_length;
    }

    int HighCardPoints() const {
      return hand_.HighCardPoints();
    }



  private:
    ble::BridgeHand hand_;
    std::array<std::vector<ble::BridgeCard>, ble::kNumSuits> cards_by_suit_;
};
}
#endif //BRIDGE_LEARNING_PLAYCC_SAYC_HAND_ANALYZER_H_
