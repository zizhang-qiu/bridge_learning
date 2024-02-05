//
// Created by qzz on 2024/1/30.
//

#include "utils.h"

namespace sayc {
bool HasOpeningBidBeenMade(const ble::BridgeObservation& obs) {
  const auto& auction_history = obs.AuctionHistory();
  for (const auto& call : auction_history) {
    if (call.move.IsBid()) {
      return true;
    }
  }
  return false;
}

ble::BridgeMove ConstructBidMove(ble::Suit suit, int level) {
  return {ble::BridgeMove::Type::kAuction,
          ble::Suit::kInvalidSuit,
          -1,
          static_cast<ble::Denomination>(suit),
          level,
          ble::OtherCalls::kNotOtherCall
  };
}

const std::vector<std::string> kPassStrings = {"pass", "p"};
const std::vector<std::string> kDoubleStrings = {"double", "dbl", "x", "d"};
const std::vector<std::string> kRedoubleStrings = {
    "redouble", "rdbl", "xx", "r"};

ble::BridgeMove
ConstructAuctionMoveFromString(const std::string& call_string) {
  const std::string call_string_lower = absl::AsciiStrToLower(call_string);
  if (std::find(kPassStrings.begin(), kPassStrings.end(), call_string_lower) !=
      kPassStrings.end()) {
    return {ble::BridgeMove::kAuction,
            ble::Suit::kInvalidSuit,
            -1,
            ble::Denomination::kInvalidDenomination,
            -1,
            ble::OtherCalls::kPass};
  }
  if (std::find(kDoubleStrings.begin(), kDoubleStrings.end(),
                call_string_lower) !=
      kDoubleStrings.end()) {
    return {ble::BridgeMove::kAuction,
            ble::Suit::kInvalidSuit,
            -1,
            ble::Denomination::kInvalidDenomination,
            -1,
            ble::OtherCalls::kDouble};
  }
  if (std::find(kRedoubleStrings.begin(), kRedoubleStrings.end(),
                call_string_lower) !=
      kRedoubleStrings.end()) {
    return {ble::BridgeMove::kAuction,
            ble::Suit::kInvalidSuit,
            -1,
            ble::Denomination::kInvalidDenomination,
            -1,
            ble::OtherCalls::kRedouble};
  }

  // A bid.
  SPIEL_CHECK_GE(call_string_lower.size(), 2);
  SPIEL_CHECK_LE(call_string_lower.size(), 3);

  if (std::isdigit(call_string_lower[0]) == 0) {
    SpielFatalError(absl::StrCat(
        "The first char of a bid should be a digit, but got ", call_string));
  }

  // A bid is of size 3 only if it ends with "NT".
  if (call_string_lower.size() == 3) {
    if (call_string_lower.substr(1) != "nt") {
      SpielFatalError(absl::StrCat(
          "A bid of size 3 can only ends with nt(NT), but got ", call_string));
    }
  }

  int level = call_string[0] - '0';
  ble::Denomination denomination{ble::Denomination::kNoTrump};
  if (call_string_lower.size() == 2) {
    switch (call_string_lower[1]) {
      case 'c':
        denomination = ble::Denomination::kClubsTrump;
        break;
      case 'd':
        denomination = ble::Denomination::kDiamondsTrump;
        break;
      case 'h':
        denomination = ble::Denomination::kHeartsTrump;
        break;
      case 's':
        denomination = ble::Denomination::kSpadesTrump;
        break;
      case 'n':
        denomination = ble::Denomination::kNoTrump;
        break;
      default:
        SpielFatalError(absl::StrCat(
            "Get wrong denomination, check your input: ",
            call_string));
    }
  }

  return {ble::BridgeMove::kAuction,
          ble::Suit::kInvalidSuit,
          -1,
          denomination,
          level,
          ble::OtherCalls::kNotOtherCall};

}

}