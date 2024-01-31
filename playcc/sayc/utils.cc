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
}
