//
// Created by qzz on 2023/10/23.
//

#ifndef BRIDGE_LEARNING_PLAYCC_DECK_SAMPLER_H_
#define BRIDGE_LEARNING_PLAYCC_DECK_SAMPLER_H_
#include "bridge_lib/bridge_deck.h"
namespace ble = bridge_learning_env;

class DeckSampler {
  public:
    DeckSampler() {
      Reset();
    }

    void Reset() { deck_ = ble::BridgeDeck(); }

    // Sample a card which is not a specified suit from deck.
    ble::BridgeCard SampleNotSuit(const ble::Suit suit, std::mt19937& rng) {
      const std::array<std::vector<ble::BridgeCard>, ble::kNumSuits>
          cards_by_suits = deck_.CardsBySuits();
      std::vector<ble::BridgeCard> legal_cards;
      for (const ble::Suit s : ble::kAllSuits) {
        if (s == suit) {
          continue;
        }
        for (const auto card : cards_by_suits[s]) {
          legal_cards.push_back(card);
        }
      }
      if (legal_cards.empty()) {
        return {};
      }
      std::uniform_int_distribution<int> dis(
          0, static_cast<int>(legal_cards.size()) - 1);
      const auto random_index = dis(rng);
      const ble::BridgeCard sampled_card = legal_cards[random_index];
      deck_.DealCard(sampled_card.Index());
      return sampled_card;
    }

    ble::BridgeCard SampleNotSuits(const std::vector<ble::Suit>& suits,
                                   std::mt19937& rng) {
      const std::array<std::vector<ble::BridgeCard>, ble::kNumSuits>
          cards_by_suits = deck_.CardsBySuits();
      std::vector<ble::BridgeCard> legal_cards;
      for (const ble::Suit s : ble::kAllSuits) {
        if (std::find(suits.begin(), suits.end(), s) != suits.end()) {
          continue;
        }
        for (const auto card : cards_by_suits[s]) {
          legal_cards.push_back(card);
        }
      }
      if (legal_cards.empty()) {
        return {};
      }
      std::uniform_int_distribution<int> dis(
          0, static_cast<int>(legal_cards.size()) - 1);
      const auto random_index = dis(rng);
      const ble::BridgeCard sampled_card = legal_cards[random_index];
      deck_.DealCard(sampled_card.Index());
      return sampled_card;
    }

    // Just sample a card, no constraints.
    ble::BridgeCard Sample(std::mt19937& rng) {
      const std::vector<ble::BridgeCard> legal_cards = deck_.Cards();

      std::uniform_int_distribution<int> dis(
          0, static_cast<int>(legal_cards.size()) - 1);
      const auto random_index = dis(rng);
      //  std::cout << "random index: " << random_index << std::endl;
      const ble::BridgeCard sampled_card = legal_cards[random_index];
      //  std::cout << "sampled card: " << sampled_card.ToString() << std::endl;
      deck_.DealCard(sampled_card.Index());
      return sampled_card;
    }

    void DealKnownCards(const std::vector<ble::BridgeHand>& known_cards) {
      for (const auto& hand : known_cards) {
        for (const auto& card : hand.Cards()) {
          deck_.DealCard(card.Index());
        }
      }
    }

    const ble::BridgeDeck& Deck() const { return deck_; }

  private:
    ble::BridgeDeck deck_;
};

#endif // BRIDGE_LEARNING_PLAYCC_DECK_SAMPLER_H_
