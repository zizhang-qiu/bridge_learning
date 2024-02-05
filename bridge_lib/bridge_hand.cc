//
// Created by qzz on 2023/9/21.
//
#include "bridge_hand.h"

#include <array>
#include <algorithm>

#include "utils.h"

namespace bridge_learning_env {
int BridgeHand::HighCardPoints() const {
  REQUIRE(IsFullHand());
  int point = 0;
  for (const BridgeCard& card : cards_) {
    if (card.Rank() > 8) {
      point += card.Rank() - 8;
    }
  }
  return point;
}

int BridgeHand::ControlValue() const {
  REQUIRE(IsFullHand());
  int point = 0;
  for (const BridgeCard& card : cards_) {
    if (card.Rank() >= 10) {
      point += card.Rank() - 10;
    }
  }
  return point;
}

int BridgeHand::ZarHighCardPoints() const {
  REQUIRE(IsFullHand());
  return HighCardPoints() + ControlValue();
}

std::string BridgeHand::ToString() const {
  std::string rv;
  std::array<std::string, kNumSuits> cards_string;
  for (const Suit suit : kAllSuits) {
    cards_string[suit] = kSuitChar[suit];
  }
  auto cards_by_suits = CardsBySuits();
  for (auto& cards_this_suit : cards_by_suits) {
    std::sort(cards_this_suit.begin(),
              cards_this_suit.end(),
              [](const BridgeCard& lhs, const BridgeCard& rhs) {
                return lhs.Index() > rhs.Index();
              });
  }

  for (const Suit suit : kAllSuits) {
    for (const auto& card : cards_by_suits[suit]) {
      cards_string[suit].push_back(kRankChar[card.Rank()]);
    }
  }
  for (const Suit suit : kAllSuits) {
    cards_string[suit] += " ";
    rv += cards_string[suit];
  }
  return rv;
}

void BridgeHand::AddCard(const BridgeCard& card) {
  REQUIRE(card.IsValid());

  cards_.push_back(card);
}

void BridgeHand::RemoveFromHand(Suit suit, int rank, std::vector<BridgeCard>* played_cards) {
  auto card_it = std::find(cards_.begin(), cards_.end(), BridgeCard(suit, rank));
  int card_index = static_cast<int>(card_it - cards_.begin());
  if (played_cards != nullptr && card_it != cards_.end()) {
    played_cards->push_back(cards_[card_index]);
  }
  cards_.erase(cards_.begin() + card_index);
}

bool BridgeHand::IsCardInHand(BridgeCard card) const {
  auto card_it = std::find(cards_.begin(), cards_.end(), card);
  return card_it != cards_.end();
}

std::array<int, kNumSuits> BridgeHand::SuitLength() const {
  std::array<int, kNumSuits> res{};
  res.fill(0);
  for (const auto card : cards_) {
    res[card.CardSuit()] += 1;
  }
  return res;
}

int BridgeHand::ZarDistributionPoints() const {
  REQUIRE(IsFullHand());
  auto suit_length = SuitLength();
  std::sort(suit_length.begin(), suit_length.end(), std::greater<>());
  // Zar distribution points are the sum of the lengths of the two longest suits plus
  // the difference between the longest suit and the shortest suit.
  const int zar_dp = suit_length[3] + suit_length[2] + (suit_length[3] - suit_length[0]);
  return zar_dp;
}

std::array<std::vector<BridgeCard>, kNumSuits> BridgeHand::CardsBySuits() const {
  std::array<std::vector<BridgeCard>, kNumSuits> cards_by_suits{};
  for (const auto& card : Cards()) {
    cards_by_suits[card.CardSuit()].push_back(card);
  }
  return cards_by_suits;
}

HandEvaluation BridgeHand::GetHandEvaluation() const {
  const int hcp = HighCardPoints();
  const int control = ControlValue();
  const int zar_hcp = hcp + control;
  const int zar_dp = ZarDistributionPoints();
  const auto suit_length = SuitLength();
  return {hcp, control, zar_hcp, zar_dp, suit_length};
}

std::ostream& operator<<(std::ostream& stream, const BridgeHand& hand) {
  stream << hand.ToString();
  return stream;
}
} // bridge
