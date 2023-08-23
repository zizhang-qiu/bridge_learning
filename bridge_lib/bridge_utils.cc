#include "bridge_utils.h"

namespace bridge {
int Bid(int level, Denomination denomination) {
  return (level - 1) * kNumDenominations + denomination + kFirstBid;
}

int BidLevel(int bid) { return 1 + (bid - kNumOtherCalls) / kNumDenominations; }

Denomination BidDenomination(int bid) {
  return Denomination((bid - kNumOtherCalls) % kNumDenominations);
}

std::string CallString(int call) {
  if (call == kPass) return "Pass";
  if (call == kDouble) return "Dbl";
  if (call == kRedouble) return "RDbl";
  return {kLevelChar[BidLevel(call)], kDenominationChar[BidDenomination(call)]};
}

Suit CardSuit(int card) { return Suit(card % kNumSuits); }
int CardRank(int card) { return card / kNumSuits; }
int Card(Suit suit, int rank) {
  return rank * kNumSuits + static_cast<int>(suit);
}

std::string CardString(int card) {
  return {kSuitChar[static_cast<int>(CardSuit(card))],
          kRankChar[CardRank(card)]};
}

int Partnership(Player player) { return player & 1; }
int Partner(Player player) { return player ^ 2; }

}  // namespace bridge
