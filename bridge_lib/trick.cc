#include "trick.h"
namespace bridge_learning_env {
Trick::Trick(const Player leader, const Denomination trumps, const int card) :
    trumps_(trumps),
    led_suit_(CardSuit(card)),
    winning_suit_(CardSuit(card)),
    winning_rank_(CardRank(card)),
    leader_(leader),
    winning_player_(leader) {}

void Trick::Play(const Player player, const int card) {
  const Suit card_suit = CardSuit(card);
  const int card_rank = CardRank(card);
  if (card_suit == winning_suit_) {
    if (card_rank > winning_rank_) {
      winning_rank_ = card_rank;
      winning_player_ = player;
    }
  }
  else if (card_suit == static_cast<Suit>(trumps_)) {
    winning_suit_ = static_cast<Suit>(trumps_);
    winning_rank_ = card_rank;
    winning_player_ = player;
  }
}
} // namespace bridge_learning_env
