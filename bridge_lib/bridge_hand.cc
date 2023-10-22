//
// Created by qzz on 2023/9/21.
//

#include "bridge_hand.h"
#include "utils.h"
#include <cassert>
#include <array>
#include <algorithm>

namespace bridge_learning_env {
int BridgeHand::HighCardPoints() const {
  REQUIRE(IsFullHand());
  int point = 0;
  for (const BridgeCard &card : cards_) {
    if (card.Rank() >= 9) {
      point += card.Rank() - 9;
    }
  }
  return point;
}

int BridgeHand::ControlValue() const {
  REQUIRE(IsFullHand());
  int point = 0;
  for (const BridgeCard &card : cards_) {
    if (card.Rank() >= 11) {
      point += card.Rank() - 11;
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
  for (const auto &card : cards_) {
    cards_string[card.CardSuit()].push_back(kRankChar[card.Rank()]);
  }
  for (const Suit suit : kAllSuits) {
    cards_string[suit] += " ";
    rv += cards_string[suit];
  }
  return rv;
}

void BridgeHand::AddCard(const BridgeCard &card) {
  REQUIRE(card.IsValid());

  cards_.push_back(card);
}

void BridgeHand::RemoveFromHand(Suit suit, int rank, std::vector<BridgeCard> *played_cards) {
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
int BridgeHand::CardIndex(BridgeCard card) const {
  auto card_it = std::find(cards_.begin(), cards_.end(), card);
  if (card_it == cards_.end()) {
    return -1;
  }
  return static_cast<int>(card_it - cards_.begin());
}

} // bridge