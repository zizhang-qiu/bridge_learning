#include "trick.h"
namespace bridge_learning_env {
Trick::Trick(Player leader, Denomination trumps, int card)
    : trumps_(trumps),
      led_suit_(CardSuit(card)),
      winning_suit_(CardSuit(card)),
      winning_rank_(CardRank(card)),
      leader_(leader),
      winning_player_(leader) {}

void Trick::Play(Player player, int card){
    Suit card_suit = CardSuit(card);
    int card_rank = CardRank(card);
    if(card_suit == winning_suit_){
        if(card_rank > winning_rank_){
            winning_rank_ = card_rank;
            winning_player_ = player;
        }
    }else if (card_suit == Suit(trumps_)) {
        winning_suit_ = Suit(trumps_);
        winning_rank_ = card_rank;
        winning_player_ = player;
    }
}
}  // namespace bridge