//
// Created by 13738 on 2024/8/8.
//

#include <numeric>
#include "detailed_encoder.h"
#include "rela/utils.h"

int FlatLength(const std::vector<int> &shape) {
  return std::accumulate(std::begin(shape), std::end(shape), 1,
                         std::multiplies<>());
}

int EncodeAuctionDetailed(const ble::BridgeObservation &obs, int start_offset, std::vector<int> *encoding) {
  int offset = start_offset;
  const auto &history = obs.AuctionHistory();
  int idx = 0;
  // Opening pass.
  for (; idx < history.size(); idx++) {
    if (history[idx].move.IsBid()) {
      break;
    }
    if (history[idx].move.OtherCall() == ble::kPass) {
      (*encoding)[offset + history[idx].player] = 1;
    }
  }

  offset += ble::kNumPlayers;

  // For each bid, a 4 * 6 = 24 bits block is used to track whether a player
  // makes/passes/doubles/passes/redoubles/passes the bid.
  int last_bid = 0;
  bool last_bid_doubled = false;
  bool last_bid_redoubled = false;
  for (; idx < history.size(); idx++) {
    const auto &item = history[idx];
    if (item.other_call == ble::kPass) {
      const int pass_idx = 1 + 2 * (int(last_bid_doubled) + int(last_bid_redoubled));
      (*encoding)[offset + (last_bid - ble::kFirstBid) * kSingleBidTensorSize + ble::kNumPlayers * pass_idx
          + item.player] = 1;
    } else if (item.other_call == ble::kDouble) {
      last_bid_doubled = true;
      (*encoding)[offset + (last_bid - ble::kFirstBid) * kSingleBidTensorSize + ble::kNumPlayers * 2
          + item.player] = 1;
    } else if (item.other_call == ble::kRedouble) {
      last_bid_redoubled = true;
      (*encoding)[offset + (last_bid - ble::kFirstBid) * kSingleBidTensorSize + ble::kNumPlayers * 4
          + item.player] = 1;
    } else {
      // Should be a bid.
      const int bid_index = BidIndex(item.level, item.denomination);
      (*encoding)[offset + (bid_index - ble::kFirstBid) * kSingleBidTensorSize +
          item.player] = 1;
      last_bid = bid_index;
      last_bid_doubled = false;
      last_bid_redoubled = false;
    }
  }
  offset += kSingleBidTensorSize * ble::kNumBids;

  return offset - start_offset;
}

std::vector<int> DetailedEncoder::Encode(const ble::BridgeObservation &obs) const {
  if (obs.NumCardsPlayed() > 0){
    rela::utils::RelaFatalError("This encoder doesn't support playing phase.");
  }
  std::vector<int> encoding(FlatLength(Shape()), 0);
  int offset = 0;

  offset += ble::EncodeVulnerabilityBoth(obs, parent_game_, offset, &encoding);
  offset += EncodeAuctionDetailed(obs, offset, &encoding);
  offset += ble::EncodePlayerHand(obs, offset, &encoding, /*relative_player=*/0);

  REQUIRE_EQ(offset, kDetailedFeatureSize);
  return encoding;
}

