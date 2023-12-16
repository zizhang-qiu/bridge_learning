//
// Created by qzz on 2023/9/21.
//

#include "bridge_move.h"
#include "utils.h"

namespace bridge_learning_env {

bool BridgeMove::IsBid() const {
  return denomination_ >= Denomination::kClubsTrump && level_ >= 1;
}
int BridgeMove::BidLevel() const {
  return level_;
}
Denomination BridgeMove::BidDenomination() const {
  return denomination_;
}
std::string BridgeMove::ToString() const {
  switch (MoveType()) {
    case kPlay:return StrCat("(Play ", CardString(suit_, rank_), ")");
    case kAuction:return StrCat("(Call ", AuctionToString(), ")");
    case kDeal:return StrCat("(Deal ", CardString(suit_, rank_), ")");
    default:return "(INVALID)";
  }
}

bool BridgeMove::operator==(const BridgeMove &other_move) const {
  if (MoveType() != other_move.MoveType()) {
    return false;
  }
  switch (MoveType()) {
    case kPlay:return CardRank() == other_move.CardRank() && CardSuit() == other_move.CardSuit();
    case kAuction:
      // On of other calls
      if (level_ == -1) {
        return OtherCall() == other_move.OtherCall();
      }
      return BidLevel() == other_move.BidLevel() && BidDenomination() == other_move.BidDenomination();
    case kDeal:return CardRank() == other_move.CardRank() && CardSuit() == other_move.CardSuit();
    default:return true;
  }
}

std::string BridgeMove::AuctionToString() const {
  if (other_call_ != OtherCalls::kNotOtherCall) {
    return OtherCallsToString(other_call_);
  }
  return {LevelIndexToChar(level_), DenominationIndexToChar(denomination_)};
}

std::ostream &operator<<(std::ostream &stream, const BridgeMove &move) {
  stream << move.ToString();
  return stream;
}

} // namespace bridge