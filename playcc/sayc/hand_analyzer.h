//
// Created by qzz on 2024/1/26.
//

#ifndef BRIDGE_LEARNING_PLAYCC_SAYC_HAND_ANALYZER_H_
#define BRIDGE_LEARNING_PLAYCC_SAYC_HAND_ANALYZER_H_
#include <algorithm>
#include <map>
#include "bridge_lib/bridge_hand.h"
#include "../common_utils/log_utils.h"
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
    HandAnalyzer() = default;

    explicit HandAnalyzer(const ble::BridgeHand& hand) {
      SetHand(hand);
    }

    const ble::BridgeHand Hand() const { return hand_; }

    void SetHand(const ble::BridgeHand& hand) {
      SPIEL_CHECK_TRUE(hand.IsFullHand());
      hand_ = hand;
      cards_by_suit_ = hand_.CardsBySuits();
    }

    [[nodiscard]] bool IsBalanced() const {
      const auto suit_length = GetSortedSuitLength();
      const auto iterator = std::find(balanced_hand.begin(),
                                      balanced_hand.end(),
                                      suit_length);
      return iterator != balanced_hand.end();
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

    // Get sorted suit length, descending order.
    [[nodiscard]] std::vector<std::pair<int, std::vector<ble::Suit>>>
    GetSortedSuitLengthWithSuits() const {
      auto suit_length = GetSuitLength();
      std::array<std::pair<ble::Suit, int>, ble::kNumSuits>
          suit_length_with_suit{};
      for (const ble::Suit suit : ble::kAllSuits) {
        suit_length_with_suit[suit] = {suit, suit_length[suit]};
      }
      std::sort(suit_length_with_suit.begin(), suit_length_with_suit.end(), [](
                const std::pair<ble::Suit, int>& lhs,
                const std::pair<ble::Suit, int>& rhs) {
                  return lhs.second > rhs.second;
                });
      std::map<int, std::vector<ble::Suit>, std::greater<int>> result_map;
      for (const auto& pair : suit_length_with_suit) {
        result_map[pair.second].push_back(pair.first);
      }

      std::vector<std::pair<int, std::vector<ble::Suit>>> result;
      result.reserve(result_map.size());
      for (const auto& entry : result_map) {
        auto suits = entry.second;
        std::sort(suits.begin(), suits.end(), std::greater<>());
        result.emplace_back(std::make_pair(entry.first, suits));
      }
      return result;
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