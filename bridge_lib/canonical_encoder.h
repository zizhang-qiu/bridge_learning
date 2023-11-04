#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_ENCODER_H
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_ENCODER_H

#include "bridge_game.h"
#include "bridge_observation.h"
#include "bridge_utils.h"
#include "observation_encoder.h"

namespace bridge_learning_env {
inline constexpr int kBitsPerCard = kNumSuits * kNumCardsPerSuit;
inline constexpr int kVulnerabilityTensorSize =
    kNumVulnerabilities * kNumPartnerships;
inline constexpr int kOpeningPassTensorSize = kNumPlayers;
inline constexpr int kSingleBidTensorSize =
    kNumPlayers * 3; // For a single bid, it can be made, doubled or redoubled
inline constexpr int kBiddingHistoryTensorSize =
    kSingleBidTensorSize * kNumBidLevels * kNumDenominations;
inline constexpr int kCardTensorSize = kNumCards;
inline constexpr int kPlayerSeatTensorSize = kNumPlayers;
inline constexpr int kBiddingTensorSize =
    kVulnerabilityTensorSize + kOpeningPassTensorSize +
    kBiddingHistoryTensorSize + kCardTensorSize + kPlayerSeatTensorSize;



int EncodeVulnerability(const BridgeObservation &obs,
                        const std::shared_ptr<BridgeGame>& game,
                        int start_offset, std::vector<int> *encoding);

int EncodeAuction(const BridgeObservation &obs, int start_offset,
                  std::vector<int> *encoding);

int EncodePlayerHand(const BridgeObservation &obs, int start_offset,
                     std::vector<int> *encoding);

class CanonicalEncoder : public ObservationEncoder {
public:
  explicit CanonicalEncoder(std::shared_ptr<BridgeGame> game)
      : parent_game_(std::move(game)) {}

  [[nodiscard]] std::vector<int> Encode(const BridgeObservation &obs) const override;

  [[nodiscard]] std::vector<int> Shape() const override;

  [[nodiscard]] ObservationEncoder::Type type() const override {
    return ObservationEncoder::Type::kCanonical;
  }

private:
  const std::shared_ptr<BridgeGame> parent_game_;
};
} // namespace bridge

#endif // BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_ENCODER_H
