//
// Created by qzz on 2023/9/21.
//

#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_MOVE_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_MOVE_H_
#include "bridge_card.h"
#include "bridge_utils.h"
namespace bridge_learning_env {

class BridgeMove {
 public:
  enum Type { kInvalid=-1, kAuction, kPlay, kDeal };
  BridgeMove(Type move_type, Suit suit, int rank,
             Denomination denomination, int level, OtherCalls other_call)
      : move_type_(move_type), suit_(suit),
        rank_(rank), denomination_(denomination), level_(level),
        other_call_(other_call) {}

  bool operator==(const BridgeMove &other_move) const;

  Type MoveType() const { return move_type_; }

  std::string ToString() const;

  bool IsBid() const;

  int BidLevel() const;

  Denomination BidDenomination() const;

  Suit CardSuit() const{return suit_;}

  int CardRank() const{return rank_;}

  OtherCalls OtherCall() const{return other_call_;}



 private:
  Suit suit_ = kInvalidSuit;
  int rank_ = -1;
  Denomination denomination_ = kInvalidDenomination;
  int level_ = -1;
  OtherCalls other_call_ = OtherCalls::kNotOtherCall;
  Type move_type_ = kInvalid;

  std::string AuctionToString() const;
};

} // namespace bridge

#endif // BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_MOVE_H_
