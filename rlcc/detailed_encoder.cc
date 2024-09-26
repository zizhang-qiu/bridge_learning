//
// Created by 13738 on 2024/8/8.
//

#include <numeric>
#include "detailed_encoder.h"
#include "rela/utils.h"
#include "encoder_registerer.h"

int FlatLength(const std::vector<int> &shape) {
  return std::accumulate(std::begin(shape), std::end(shape), 1,
                         std::multiplies<>());
}

constexpr int VulnerabilitySectionLength() {
  return ble::kNumVulnerabilities * ble::kNumPartnerships;
}

constexpr int OpeningPassSectionLength() {
  return ble::kNumPlayers;
}

constexpr int SingleBidSectionLength(){
  // For each bid and player, a player can make it, pass after it is made,
  // double it, pass after it is doubled, redouble it and pass after it is redoubled.
  return 6 * ble::kNumPlayers;
}

constexpr int BiddingSectionLength() {
  return SingleBidSectionLength() * ble::kNumBids;
}

constexpr int HandSectionLength() {
  return ble::kNumPlayers * ble::kNumCards;
}

std::vector<int> DetailedEncoder::Shape() const {
  int l = VulnerabilitySectionLength() + OpeningPassSectionLength() + BiddingSectionLength() + HandSectionLength();
  return {l};
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
      (*encoding)[offset + (last_bid - ble::kFirstBid) * SingleBidSectionLength() + ble::kNumPlayers * pass_idx
          + item.player] = 1;
    } else if (item.other_call == ble::kDouble) {
      last_bid_doubled = true;
      (*encoding)[offset + (last_bid - ble::kFirstBid) * SingleBidSectionLength() + ble::kNumPlayers * 2
          + item.player] = 1;
    } else if (item.other_call == ble::kRedouble) {
      last_bid_redoubled = true;
      (*encoding)[offset + (last_bid - ble::kFirstBid) * SingleBidSectionLength() + ble::kNumPlayers * 4
          + item.player] = 1;
    } else {
      // Should be a bid.
      const int bid_index = BidIndex(item.level, item.denomination);
      (*encoding)[offset + (bid_index - ble::kFirstBid) * SingleBidSectionLength() +
          item.player] = 1;
      last_bid = bid_index;
      last_bid_doubled = false;
      last_bid_redoubled = false;
    }
  }
  offset += SingleBidSectionLength() * ble::kNumBids;

  return offset - start_offset;
}
int EncoderTurn(const ble::BridgeObservation &obs, int start_offset, std::vector<int> *encoding) {
  int offset = start_offset;
  (*encoding)[offset] = !obs.LegalMoves().empty();
  offset += 1;
  return offset - start_offset;
}

std::vector<int> DetailedEncoder::Encode(const ble::BridgeObservation &obs,
                                         const std::unordered_map<std::string, std::any> &kwargs) const {
  if (obs.NumCardsPlayed() > 0) {
    rela::utils::RelaFatalError("Detailed encoder doesn't support playing phase.");
  }
  std::vector<int> encoding(FlatLength(Shape()), 0);
  int offset = 0;

  offset += ble::EncodeVulnerabilityBoth(obs, parent_game_, offset, &encoding);
  offset += EncodeAuctionDetailed(obs, offset, &encoding);
  bool show_all_hands = true;
  if (kwargs.count("show_all_hands")) {
    show_all_hands = std::any_cast<bool>(kwargs.at("show_all_hands"));
  }
  offset += ble::EncodeAllHands(obs, offset, &encoding, /*show_all_hands=*/show_all_hands);
  if (turn_) {
    offset += EncoderTurn(obs, offset, &encoding);
  }

  REQUIRE_EQ(offset, FlatLength(Shape()));
  return encoding;
}

class DetailedEncoderFactory : public rlcc::ObservationEncoderFactory {
 public:
  std::unique_ptr<ble::ObservationEncoder> Create(const std::shared_ptr<ble::BridgeGame> &game,
                                                  const bridge_learning_env::GameParameters &encoder_params) override {
    const bool turn = ble::ParameterValue<bool>(encoder_params, "turn", false);
    return std::make_unique<DetailedEncoder>(game, turn);
  }
};

rlcc::REGISTER_OBSERVATION_ENCODER("detailed", DetailedEncoderFactory);

