#ifndef BRIDGE_LIB_PBE_ENCODER_H
#define BRIDGE_LIB_PBE_ENCODER_H
#include "observation_encoder.h"

namespace bridge_learning_env {

inline constexpr int kPBEBiddingHistoryTensorSize = kNumBids + 1  // Pass
                                                    + 5 + 1 + 1;

int EncodeHandRankMajorStyle(const BridgeObservation& obs, int start_offset,
                             std::vector<int>* encoding, int relative_player);

int EncodePBEBiddingHistory(const BridgeObservation& obs, int start_offset,
                            std::vector<int>* encoding);

class PBEEncoder : public ObservationEncoder {
 public:
  PBEEncoder(const std::shared_ptr<BridgeGame>& game) : parent_game_(game) {}

  [[nodiscard]] std::vector<int> Shape() const override {
    return {kNumCards + kPBEBiddingHistoryTensorSize};
  }

  [[nodiscard]] std::vector<int> Encode(
      const BridgeObservation& obs,
      const std::unordered_map<std::string, std::any> &kwargs = {}) const override;

  [[nodiscard]] ObservationEncoder::Type type() const override {
    return ObservationEncoder::Type::kPBE;
  }

  ObservationEncoder::EncoderPhase EncodingPhase() const override{
    return kAuction;
  }

 private:
  const std::shared_ptr<BridgeGame> parent_game_;
};
}  // namespace bridge_learning_env

#endif /* BRIDGE_LIB_PBE_ENCODER_H */
