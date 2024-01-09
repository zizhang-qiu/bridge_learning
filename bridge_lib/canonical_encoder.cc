#include "canonical_encoder.h"

#include <numeric>

#include "bridge_observation.h"

namespace bridge_learning_env {
// Computes the product of dimensions in shape, i.e. how many individual
// pieces of data the encoded observation requires.
int FlatLength(const std::vector<int> &shape) {
  return std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
}

int EncodeVulnerability(const BridgeObservation &obs,
                        const std::shared_ptr<BridgeGame> &game,
                        const int start_offset,
                        std::vector<int> *encoding) {
  int offset = start_offset;
  (*encoding)[offset + obs.IsPlayerVulnerable()] = 1;
  offset += kNumVulnerabilities;
  (*encoding)[offset + obs.IsOpponentVulnerable()] = 1;
  offset += kNumVulnerabilities;
  REQUIRE(offset - start_offset == kNumPartnerships * kNumVulnerabilities);
  return offset - start_offset;
}

int EncodeAuction(const BridgeObservation &obs, const int start_offset, std::vector<int> *encoding) {
  int offset = start_offset;
  const auto &history = obs.AuctionHistory();
  int idx = 0;
  // Opening pass
  for (; idx < history.size(); idx++) {
    if (history[idx].move.IsBid()) {
      break;
    }
    if (history[idx].move.OtherCall() == kPass) {
      (*encoding)[offset + history[idx].player] = 1;
    }
  }
  //  std::cout << "idx: " << idx << "\n";
  offset += kNumPlayers;
  // For each bid, a 4 * 3 = 12 bits block is used to track whether a player
  // makes/doubles/redoubles the bid.
  int last_bid = 0;
  for (; idx < history.size(); idx++) {
    const auto &item = history[idx];
    if (item.other_call == kDouble) {
      (*encoding)[offset + (last_bid - kFirstBid) * kSingleBidTensorSize + kNumPlayers + item.player] = 1;
    }
    else if (item.other_call == kRedouble) {
      (*encoding)[offset + (last_bid - kFirstBid) * kSingleBidTensorSize + kNumPlayers * 2 + item.player] = 1;
    }
    else if (item.move.IsBid()) {
      // Should be a bid.
      const int bid_index = BidIndex(item.level, item.denomination);
      (*encoding)[offset + (bid_index - kFirstBid) * kSingleBidTensorSize + item.player] = 1;
      last_bid = bid_index;
    }
  }
  offset += kBiddingHistoryTensorSize;
  REQUIRE(offset - start_offset == kOpeningPassTensorSize + kBiddingHistoryTensorSize);
  return offset - start_offset;
}

int EncodePlayerHand(const BridgeObservation &obs, const int start_offset, std::vector<int> *encoding) {
  int offset = start_offset;
  const auto &cards = obs.Hands()[0].Cards();
  REQUIRE(cards.size() == kNumCardsPerHand);
  for (const BridgeCard &card : cards) {
    REQUIRE(card.IsValid());
    (*encoding)[offset + card.Index()] = 1;
  }
  offset += kNumCards;
  return offset - start_offset;
}

std::vector<int> CanonicalEncoder::Shape() const {
  return {kVulnerabilityTensorSize + kOpeningPassTensorSize + kBiddingHistoryTensorSize + kCardTensorSize};
}
std::vector<int> CanonicalEncoder::Encode(const BridgeObservation &obs) const {
  std::vector<int> encoding(FlatLength(Shape()), 0);

  int offset = 0;
  offset += EncodeVulnerability(obs, parent_game_, offset, &encoding);
  offset += EncodeAuction(obs, offset, &encoding);
  offset += EncodePlayerHand(obs, offset, &encoding);

  REQUIRE(offset == encoding.size());
  return encoding;
}

} // namespace bridge_learning_env
