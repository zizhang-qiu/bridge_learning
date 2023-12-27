//
// Created by qzz on 2023/9/25.
//

#ifndef BRIDGE_LEARNING_BRIDGE_LIB_AUCTION_TRACKER_H_
#define BRIDGE_LEARNING_BRIDGE_LIB_AUCTION_TRACKER_H_
#include <optional>

#include "bridge_scoring.h"
#include "bridge_utils.h"
#include "bridge_move.h"

namespace bridge_learning_env {
class AuctionTracker {
 public:
  AuctionTracker() : num_passes_(0), first_bidder_(), is_auction_terminated_(false), contract_() {}
  void ApplyAuction(const BridgeMove &move, Player current_player);
  bool AuctionIsLegal(const BridgeMove &move, Player current_player) const;
  bool IsAuctionTerminated() const { return is_auction_terminated_; };

  Contract GetContract() const { return contract_; }

  Player Declarer() const { return contract_.declarer; }
 private:
  // Tracks number of consecutive passes.
  int num_passes_;
  // Tracks for each denomination and partnership, who bid first, in order to
  // determine the declarer.
  std::array<std::array<std::optional<Player>, kNumDenominations>,
             kNumPartnerships> first_bidder_;

  bool is_auction_terminated_;

  Contract contract_;
};
}

#endif //BRIDGE_LEARNING_BRIDGE_LIB_AUCTION_TRACKER_H_
