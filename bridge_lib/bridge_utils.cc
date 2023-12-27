#include "bridge_utils.h"

namespace bridge_learning_env {
int BidIndex(const int level, const Denomination denomination) {
  return (level - 1) * kNumDenominations + denomination + kFirstBid;
}

int BidLevel(const int bid) { return 1 + (bid - kNumOtherCalls) / kNumDenominations; }

Denomination BidDenomination(const int bid) {
  return static_cast<Denomination>((bid - kNumOtherCalls) % kNumDenominations);
}

std::string CallString(const int call) {
  if (call == kPass) return "Pass";
  if (call == kDouble) return "Dbl";
  if (call == kRedouble) return "RDbl";
  return {kLevelChar[BidLevel(call)], kDenominationChar[BidDenomination(call)]};
}

Suit CardSuit(const int card) { return static_cast<Suit>(card % kNumSuits); }
int CardRank(const int card) { return card / kNumSuits; }
int CardIndex(const Suit suit, const int rank) { return rank * kNumSuits + suit; }

std::string CardString(const int card) { return CardString(CardSuit(card), CardRank(card)); }

std::string CardString(const Suit suit, const int rank) { return {SuitIndexToChar(suit), RankIndexToChar(rank)}; }

int Partnership(const Player player) { return player & 1; }
int Partner(const Player player) { return player ^ 2; }
int OpponentPartnership(const int partnership) { return 1 - partnership; }

int SuitToDDSSuit(const Suit suit) { return 3 - static_cast<int>(suit); }

Denomination DDSStrainToDenomination(const int strain) {
  if (strain == kNoTrump) {
    return kNoTrump;
  }
  return static_cast<Denomination>(3 - strain);
}
int DenominationToDDSStrain(const Denomination denomination) {
  return denomination == kNoTrump ? denomination : 3 - denomination;
}
char SuitIndexToChar(const Suit suit) {
  if (suit >= kClubsSuit && suit <= kSpadesSuit) {
    return kSuitChar[suit];
  }
  // 'I' for Invalid.
  return 'I';
}

char RankIndexToChar(const int rank) {
  if (rank >= 0 && rank < kNumCardsPerSuit) {
    return kRankChar[rank];
  }
  return 'I';
}

char DenominationIndexToChar(const Denomination denomination) {
  if (denomination >= kClubsTrump && denomination <= kNoTrump) {
    return kDenominationChar[denomination];
  }
  return 'I';
}

char LevelIndexToChar(const int level) {
  if (level >= 1 && level <= kNumBidLevels) {
    return kLevelChar[level];
  }
  return 'I';
}

std::string OtherCallsToString(const OtherCalls call) {
  switch (call) {
    case kPass:
      return "Pass";
    case kDouble:
      return "Dbl";
    case kRedouble:
      return "RDbl";
    default:
      return "I";
  }
}

int RankToDDSRank(const int rank) { return rank + 2; }
Suit DDSSuitToSuit(const int suit) { return static_cast<Suit>(3 - suit); }
int DDSRankToRank(const int dds_rank) { return dds_rank - 2; }
Suit SuitCharToSuit(const char suit_char) {
  const char upper_char = std::toupper(suit_char);
  switch (upper_char) {
    case 'C':
      return kClubsSuit;
    case 'D':
      return kDiamondsSuit;
    case 'H':
      return kHeartsSuit;
    case 'S':
      return kSpadesSuit;
    default:
      return kInvalidSuit;
  }
}
int RankCharToRank(const char rank_char) {
  switch (rank_char) {
    case '2':
      return 0;
    case '3':
      return 1;
    case '4':
      return 2;
    case '5':
      return 3;
    case '6':
      return 4;
    case '7':
      return 5;
    case '8':
      return 6;
    case '9':
      return 7;
    case 'T':
      return 8;
    case 'J':
      return 9;
    case 'Q':
      return 10;
    case 'K':
      return 11;
    case 'A':
      return 12;
    default:
      return -1;
  }
}

} // namespace bridge_learning_env
