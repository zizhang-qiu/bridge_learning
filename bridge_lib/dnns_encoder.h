#ifndef BRIDGE_LIB_DNNS_ENCODER_H
#define BRIDGE_LIB_DNNS_ENCODER_H
#include "observation_encoder.h"

namespace bridge_learning_env {
inline constexpr int kDNNsBiddingHistoryTensorSize = kMaxAuctionLength;

int EncodeHandRankMajorClubsStartStyle(const BridgeObservation& obs,
                                       int start_offset,
                                       std::vector<int>* encoding,
                                       int relative_player);

int EncodeDNNsBiddingHistory(const BridgeObservation& obs, int start_offset,
                             std::vector<int>* encoding);

class DNNsEncoder : public ObservationEncoder {
 public:
  std::vector<int> Shape() const override {
    return {kNumCards + kNumPartnerships + kDNNsBiddingHistoryTensorSize};
  }

  Type type() const override { return kDNNS; }

  std::vector<int> Encode(const BridgeObservation& obs,
                          const std::unordered_map<std::string, std::any> &kwargs = {}) const override;

  ObservationEncoder::EncoderPhase EncodingPhase() const override{
    return kAuction;
  }
};
}  // namespace bridge_learning_env
#endif /* BRIDGE_LIB_DNNS_ENCODER_H */
