//
// Created by 13738 on 2024/8/8.
//

#ifndef BRIDGE_LEARNING_RLCC_DETAILED_ENCODER_H_
#define BRIDGE_LEARNING_RLCC_DETAILED_ENCODER_H_
#include "bridge_lib/observation_encoder.h"
#include "bridge_lib/canonical_encoder.h"
namespace ble = bridge_learning_env;

// For each bid and player, a player can make it, pass after it is made,
// double it, pass after it is doubled, redouble it and pass after it is redoubled.
inline static constexpr int kSingleBidTensorSize = 6 * ble::kNumPlayers;
inline static constexpr int kDetailedFeatureSize =
    ble::kNumPartnerships * ble::kNumVulnerabilities // Vulnerabilities.
        + ble::kNumPlayers                           // Opening pass.
        + ble::kNumCards                             // Hand.
        + ble::kNumBids * kSingleBidTensorSize;      // Encoding for each bid.

int EncodeAuctionDetailed(const ble::BridgeObservation &obs, int start_offset, std::vector<int> *encoding);

class DetailedEncoder : public ble::ObservationEncoder {
 public:
  DetailedEncoder(const std::shared_ptr<ble::BridgeGame> &game) : parent_game_(game) {}
  std::vector<int> Shape() const override { return {kDetailedFeatureSize}; }
  std::vector<int> Encode(const ble::BridgeObservation &obs) const override;
  Type type() const override { return kDetailed; }
 private:

  std::shared_ptr<ble::BridgeGame> parent_game_;
};
#endif //BRIDGE_LEARNING_RLCC_DETAILED_ENCODER_H_
