#include "dnns_encoder.h"

#include "jps_encoder.h"

namespace bridge_learning_env {

constexpr int pass_index[] = {1, 2, 4, 5, 7, 8};

int EncodeHandRankMajorClubsStartStyle(const BridgeObservation& obs,
                                       int start_offset,
                                       std::vector<int>* encoding,
                                       int relative_player) {
  int offset = start_offset;
  // Rank major, C2, C3, ..CA, D2, ..., DA, ...SA
  for (const auto& card : obs.Hands()[relative_player].Cards()) {
    const int card_index = card.CardSuit() * kNumCardsPerSuit + card.Rank();
    (*encoding)[card_index] = 1;
  }
  offset += kNumCards;
  REQUIRE_EQ(offset - start_offset, kNumCards);

  return offset - start_offset;
}

int EncodeDNNsBiddingHistory(const BridgeObservation& obs, int start_offset,
                             std::vector<int>* encoding) {
  int offset = start_offset;

  const auto& history = obs.AuctionHistory();
  int idx = 0;
  // Opening pass
  for (; idx < history.size(); idx++) {
    if (history[idx].move.IsBid()) {
      break;
    }
    if (history[idx].move.OtherCall() == kPass) {
      // Use absolute index here.
      int player = (obs.ObservingPlayer() + history[idx].player) % kNumPlayers;
      (*encoding)[offset + player] = 1;
    }
  }
  offset += kNumPlayers - 1;

  // For each bid, a 9-bit block is used for representing
  // bid-pass-pass-double-pass-pass-redouble-pass-pass
  int last_bid = 0;
  int num_passes_on_current_bid = 0;
  for (; idx < history.size(); idx++) {
    const auto& item = history[idx];
    if (item.other_call == kPass) {
      num_passes_on_current_bid++;
      (*encoding)[offset + last_bid * 9 +
                  pass_index[num_passes_on_current_bid - 1]] = 1;
    } else if (item.other_call == kDouble) {
      (*encoding)[offset + last_bid * 9 + 3] = 1;
    } else if (item.other_call == kRedouble) {
      (*encoding)[offset + last_bid * 9 + 6] = 1;
    } else {
      // A bid was made.
      const int bid_index =
          (item.level - 1) * kNumDenominations + item.denomination;
      (*encoding)[offset + bid_index * 9] = 1;
      last_bid = bid_index;
      num_passes_on_current_bid = 0;
    }
  }
  offset += 9 * kNumBids;

  REQUIRE_EQ(offset - start_offset, kDNNsBiddingHistoryTensorSize);
  return offset - start_offset;
}

std::vector<int> DNNsEncoder::Encode(const BridgeObservation& obs) const {
  int offset = 0;
  std::vector<int> encoding(Shape()[0]);

  // Own hand
  // std::cout << "Before encode hand." << std::endl;
  offset += EncodeHandRankMajorClubsStartStyle(obs, offset, &encoding,
                                               /*relative_player=*/0);
  // Vulnerability
  // std::cout << "Before encode vul." << std::endl;
  offset += EncodeJPSVulnerability(obs, offset, &encoding);

  // Bidding history.
  // std::cout << "Before encode history." << std::endl;
  offset += EncodeDNNsBiddingHistory(obs, offset, &encoding);

  // std::cout << "After encode hand." << std::endl;

  REQUIRE_EQ(offset, Shape()[0]);
  return encoding;
}

}  // namespace bridge_learning_env