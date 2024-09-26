#include "pbe_encoder.h"

namespace bridge_learning_env {

int EncodeHandRankMajorStyle(const BridgeObservation& obs, int start_offset,
                             std::vector<int>* encoding, int relative_player) {
  int offset = start_offset;
  // Rank major, S2, S3, ..SA, H2, ..., HA, ...CA
  for (const auto& card : obs.Hands()[relative_player].Cards()) {
    const int rank_major_card_index =
        (3 - card.CardSuit()) * kNumCardsPerSuit + card.Rank();
    (*encoding)[offset + rank_major_card_index] = 1;
  }
  offset += kNumCards;
  REQUIRE_EQ(offset - start_offset, kNumCards);

  return offset - start_offset;
}

int EncodePBEBiddingHistory(const BridgeObservation& obs, int start_offset,
                            std::vector<int>* encoding) {
  int offset = start_offset;
  for (int i = kPBEBiddingHistoryTensorSize - 7;
       i < kPBEBiddingHistoryTensorSize - 2; ++i) {
    (*encoding)[offset + i] = 1;
  }
  const auto& bidding_history = obs.AuctionHistory();
  for (const auto& item : bidding_history) {
    if (item.player == 0 || item.player == 2) {
      (*encoding)[offset + kPBEBiddingHistoryTensorSize - 2] += 1;
      if ((*encoding)[offset + kPBEBiddingHistoryTensorSize - 2] > 3) {
        (*encoding)[offset + kPBEBiddingHistoryTensorSize - 1] = 1;
        break;
      }
      if (item.other_call == OtherCalls::kPass) {
        (*encoding)[offset + 0] = 1;
        if ((*encoding)[offset + kPBEBiddingHistoryTensorSize - 2] > 1) {
          (*encoding)[offset + kPBEBiddingHistoryTensorSize - 1] = 1;
          break;
        }
      } else {
        (*encoding)[offset + (item.level - 1) * kNumDenominations +
                    item.denomination] = 1;
      }
    }
  }
  offset += kPBEBiddingHistoryTensorSize;
  REQUIRE_EQ(offset - start_offset, kPBEBiddingHistoryTensorSize);

  return offset - start_offset;
}

std::vector<int> PBEEncoder::Encode(const BridgeObservation& obs,
                                    const std::unordered_map<std::string, std::any> &kwargs) const {
  REQUIRE(obs.CurrentPhase() == bridge_learning_env::Phase::kAuction);
  int offset = 0;
  std::vector<int> encoding(kNumCards + kPBEBiddingHistoryTensorSize, 0);
  // Encode player hand.
  offset +=
      EncodeHandRankMajorStyle(obs, offset, &encoding, /*relative_player=*/0);

  // Encode bidding history.
  offset += EncodePBEBiddingHistory(obs, offset, &encoding);

  REQUIRE_EQ(offset, kPBEBiddingHistoryTensorSize + kNumCards);
  return encoding;
}
}  // namespace bridge_learning_env