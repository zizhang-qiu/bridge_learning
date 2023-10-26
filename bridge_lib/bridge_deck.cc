//
// Created by qzz on 2023/10/23.
//
#include "bridge_deck.h"
namespace bridge_learning_env {
BridgeCard BridgeDeck::DealCard(Suit suit,
                                int rank) {
  int index = CardToIndex(suit, rank);
  REQUIRE(card_in_deck_[index] == true);
  card_in_deck_[index] = false;
  --total_count_;
  return {suit, rank};
}
BridgeCard BridgeDeck::DealCard(int card_index) {
  REQUIRE(card_in_deck_[card_index] == true);
  card_in_deck_[card_index] = false;
  --total_count_;
  return {CardSuit(card_index), CardRank(card_index)};
}
BridgeCard BridgeDeck::DealCard(std::mt19937 &rng) {
  if (Empty()) {
    return {};
  }
  std::discrete_distribution<std::mt19937::result_type> dist(
      card_in_deck_.begin(), card_in_deck_.end());
  int index = static_cast<int>(dist(rng));
  REQUIRE(card_in_deck_[index] == true);
  card_in_deck_[index] = false;
  --total_count_;
  return {IndexToSuit(index), IndexToRank(index)};
}
std::vector<BridgeCard> BridgeDeck::Cards() const {
  std::vector<BridgeCard> cards;
  cards.reserve(total_count_);
  for (const Suit suit : kAllSuits) {
    for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
      if (CardInDeck(suit, rank)) {
        cards.emplace_back(suit, rank);
      }
    }
  }
  return cards;
}
std::array<std::vector<BridgeCard>, kNumSuits> BridgeDeck::CardsBySuits() const {
  std::array<std::vector<BridgeCard>, kNumSuits> cards_by_suits{};
  for (const Suit suit : kAllSuits) {
    for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
      if (CardInDeck(suit, rank)) {
        cards_by_suits[suit].emplace_back(suit, rank);
      }
    }
  }
  return cards_by_suits;
}
}