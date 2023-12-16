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
  enum Type { kInvalid = -1, kAuction, kPlay, kDeal };

  BridgeMove() : move_type_(kInvalid), suit_(kInvalidSuit), rank_(-1),
                 denomination_(kInvalidDenomination), level_(-1), other_call_(kNotOtherCall) {}

  BridgeMove(Type move_type, Suit suit, int rank,
             Denomination denomination, int level, OtherCalls other_call)
      : move_type_(move_type), suit_(suit),
        rank_(rank), denomination_(denomination), level_(level),
        other_call_(other_call) {}

  BridgeMove(const BridgeMove &) = default;

  bool operator==(const BridgeMove &other_move) const;

  [[nodiscard]] Type MoveType() const { return move_type_; }

  [[nodiscard]] std::string ToString() const;

  [[nodiscard]] bool IsBid() const;

  [[nodiscard]] int BidLevel() const;

  [[nodiscard]] Denomination BidDenomination() const;

  [[nodiscard]] Suit CardSuit() const { return suit_; }

  [[nodiscard]] int CardRank() const { return rank_; }

  [[nodiscard]] OtherCalls OtherCall() const { return other_call_; }

  [[nodiscard]] std::string AuctionToString() const;

 private:
  Suit suit_;
  int rank_;
  Denomination denomination_;
  int level_;
  OtherCalls other_call_;
  Type move_type_;

};

std::ostream &operator<<(std::ostream &stream, const BridgeMove& move);

} // namespace bridge

#endif // BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_MOVE_H_
