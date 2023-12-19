#ifndef BRIDGE_LEARNING_BRIDGE_LIB_TRICK_H
#define BRIDGE_LEARNING_BRIDGE_LIB_TRICK_H
#include "bridge_utils.h"

namespace bridge_learning_env {
class Trick {
 public:
  Trick(Player leader, Denomination trumps, int card);
  Trick() : Trick{kInvalidPlayer, kNoTrump, 0} {}
  void Play(Player player, int card);
  Suit LedSuit() const { return led_suit_; }
  Player Winner() const { return winning_player_; }
  Player Leader() const { return leader_; }

 private:
  Denomination trumps_;
  Suit led_suit_;
  Suit winning_suit_;
  int winning_rank_;
  Player leader_;
  Player winning_player_;
};
}  // namespace bridge

#endif // BRIDGE_LEARNING_BRIDGE_LIB_TRICK_H
