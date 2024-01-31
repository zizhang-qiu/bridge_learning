//
// Created by qzz on 24-2-1.
//
#include "../common_utils/log_utils.h"
#include "playcc/sayc/utils.h"

namespace sayc {
void ConstructAuctionMoveFromStringTest() {
  // Pass
  const std::vector<std::string> pass_strings = {
      "Pass", "pass", "P", "p"};
  const ble::BridgeMove pass_move{ble::BridgeMove::kAuction,
                                  ble::Suit::kInvalidSuit,
                                  -1, ble::Denomination::kInvalidDenomination,
                                  -1,
                                  ble::OtherCalls::kPass};
  for (const auto& str : pass_strings) {
    const auto move = ConstructAuctionMoveFromString(str);
    SPIEL_CHECK_EQ(move, pass_move);
  }

  // Double
  const std::vector<std::string> double_strings = {
      "Double", "double", "Dbl", "dbl", "X", "x"};
  const ble::BridgeMove double_move{ble::BridgeMove::kAuction,
                                    ble::Suit::kInvalidSuit,
                                    -1, ble::Denomination::kInvalidDenomination,
                                    -1,
                                    ble::OtherCalls::kDouble};
  for (const auto& str : double_strings) {
    const auto move = ConstructAuctionMoveFromString(str);
    SPIEL_CHECK_EQ(move, double_move);
  }

  // Redouble
  const std::vector<std::string> redouble_strings = {
      "Redouble", "redouble", "RDbl", "rdbl", "XX", "xx", "R", "r"};
  const ble::BridgeMove redouble_move{ble::BridgeMove::kAuction,
                                      ble::Suit::kInvalidSuit,
                                      -1,
                                      ble::Denomination::kInvalidDenomination,
                                      -1,
                                      ble::OtherCalls::kRedouble};
  for (const auto& str : redouble_strings) {
    const auto move = ConstructAuctionMoveFromString(str);
    SPIEL_CHECK_EQ(move, redouble_move);
  }

  // Bids.

  for (int level = 1; level <= ble::kNumBidLevels; ++level) {
    for (const ble::Denomination denomination : ble::kAllDenominations) {
      const std::string bid_string{ble::kLevelChar[level],
                                   ble::kDenominationChar[denomination]};
      const ble::BridgeMove expected_bid{ble::BridgeMove::kAuction,
                                         ble::Suit::kInvalidSuit, -1,
                                         denomination, level,
                                         ble::OtherCalls::kNotOtherCall};
      const auto bid = ConstructAuctionMoveFromString(bid_string);
      SPIEL_CHECK_EQ(bid, expected_bid);
    }
  }

  // Special case: XNT
  for (int level = 1; level <= ble::kNumBidLevels; ++level) {
    const std::string bid_string = absl::StrCat(level, "NT");
    const ble::BridgeMove expected_bid{ble::BridgeMove::kAuction,
                                       ble::Suit::kInvalidSuit, -1,
                                       ble::Denomination::kNoTrump, level,
                                       ble::OtherCalls::kNotOtherCall};
    const auto bid = ConstructAuctionMoveFromString(bid_string);
    SPIEL_CHECK_EQ(bid, expected_bid);
  }

}
}

int main(int argc, char* argv[]) {
  sayc::ConstructAuctionMoveFromStringTest();
}