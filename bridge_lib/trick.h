#ifndef TRICK
#define TRICK
#include "bridge_utils.h"
namespace bridge {
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

#endif /* TRICK */
