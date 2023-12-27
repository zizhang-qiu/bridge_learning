//
// Created by qzz on 2023/9/25.
//

#include "auction_tracker.h"

#include "utils.h"
namespace bridge_learning_env {

void AuctionTracker::ApplyAuction(const BridgeMove &move, Player current_player) {
  REQUIRE(move.MoveType() == BridgeMove::kAuction);
  const OtherCalls other_call = move.OtherCall();
  if (other_call == kPass) {
    ++num_passes_;
  }
  else {
    num_passes_ = 0;
  }
  const int partnership = Partnership(current_player);
  if (other_call == kDouble) {
    REQUIRE(Partnership(contract_.declarer) != partnership);
    REQUIRE(contract_.double_status == kUndoubled);
    REQUIRE(contract_.level > 0);
    contract_.double_status = kDoubled;
  } else if (other_call == kRedouble) {
    REQUIRE(Partnership(contract_.declarer) == partnership);
    REQUIRE(contract_.double_status == kDoubled);
    contract_.double_status = kRedoubled;
  } else if (other_call == kPass) {
    if (num_passes_ == kNumPlayers || (num_passes_ == 3 && contract_.level > 0)) {
      is_auction_terminated_ = true;
      return;
    }
  } else {
    REQUIRE(move.BidLevel() > contract_.level
                || (move.BidLevel() == contract_.level && move.BidDenomination() > contract_.denomination));
    contract_.level = move.BidLevel();
    contract_.denomination = move.BidDenomination();
    contract_.double_status = kUndoubled;
    if (!first_bidder_[partnership][contract_.denomination].has_value()) {
      first_bidder_[partnership][contract_.denomination] = current_player;
    }
    contract_.declarer = first_bidder_[partnership][contract_.denomination].value();
  }
}
bool AuctionTracker::AuctionIsLegal(const BridgeMove &move, const Player current_player) const {
  if (IsAuctionTerminated() || move.MoveType() != BridgeMove::kAuction) {
    return false;
  }
  if (move.OtherCall() == kPass) {
    return true;
  }
  const int partnership = Partnership(current_player);
  if (move.OtherCall() == kDouble) {
    if (contract_.level == 0 || contract_.double_status != kUndoubled
        || Partnership(contract_.declarer) == partnership) {
      return false;
    }
    return true;
  }
  if (move.OtherCall() == kRedouble) {
    if (contract_.double_status != kDoubled || Partnership(contract_.declarer) != partnership) {
      return false;
    }
    return true;
  }

  // A bid
  if (move.BidLevel() > contract_.level
      || (move.BidLevel() == contract_.level && move.BidDenomination() > contract_.denomination)) {
    return true;
  }
  return false;
}
}


