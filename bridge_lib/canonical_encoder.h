#ifndef BRIDGE_LIB_CANONICAL_ENCODER_H
#define BRIDGE_LIB_CANONICAL_ENCODER_H

#include <vector>
#include "bridge_game.h"
#include "bridge_observation.h"
#include "bridge_utils.h"
#include "observation_encoder.h"

namespace bridge_learning_env {
inline constexpr int kBitsPerCard = kNumSuits * kNumCardsPerSuit;
inline constexpr int kVulnerabilityTensorSize =
    kNumVulnerabilities * kNumPartnerships;
inline constexpr int kOpeningPassTensorSize = kNumPlayers;
inline constexpr int kSingleBidTensorSize = kNumPlayers * 3;
// For a single bid, it can be made, doubled or redoubled
inline constexpr int kBiddingHistoryTensorSize =
    kSingleBidTensorSize * kNumBidLevels * kNumDenominations;
inline constexpr int kCardTensorSize = kNumCards;
inline constexpr int kPlayerSeatTensorSize = kNumPlayers;
inline constexpr int kAuctionTensorSize =
    kVulnerabilityTensorSize + kOpeningPassTensorSize +
        kBiddingHistoryTensorSize + kCardTensorSize;
inline constexpr int kHCPTensorSize = 38;      // AKQJAKQAKQAKQ = 37
inline constexpr int kControlTensorSize = 13;  // AAAAKKKKxxxxx = 12
// Each suit may have [0, 13] cards.
inline constexpr int kSuitLengthTensorSize = kNumSuits * (kNumCardsPerSuit + 1);
inline constexpr int kHandEvaluationOneHotTensorSize =
    kHCPTensorSize + kControlTensorSize + kSuitLengthTensorSize;
inline constexpr int kHandEvaluationTensorSize = 6;

int EncodeVulnerabilityBoth(const BridgeObservation &obs,
                            const std::shared_ptr<BridgeGame> &game,
                            int start_offset, std::vector<int> *encoding);

int EncodeAuction(const BridgeObservation &obs, int start_offset,
                  std::vector<int> *encoding);

int EncodePlayerHand(const BridgeObservation &obs, int start_offset,
                     std::vector<int> *encoding, int relative_player);

int EncodeAllHands(const BridgeObservation &obs, const int start_offset,
                   std::vector<int> *encoding, const bool show_all_hands);

int EncodeContract(const BridgeObservation &obs, int start_offset,
                   std::vector<int> *encoding);

int EncodeVulnerabilityDeclarer(const BridgeObservation &obs, int start_offset,
                                std::vector<int> *encoding);

int EncodeDummyHand(const BridgeObservation &obs, int start_offset,
                    std::vector<int> *encoding);

int EncodePlayedTricks(const BridgeObservation &obs, int start_offset,
                       std::vector<int> *encoding, int num_tricks);

int EncodeNumTricksWon(const BridgeObservation &obs, int start_offset,
                       std::vector<int> *encoding);

int EncodeHandEvaluationOneHot(const BridgeObservation &obs, int start_offset,
                               std::vector<int> *encoding, int relative_player);

int EncodeHandEvaluation(const BridgeObservation &obs, int start_offset,
                         std::vector<int> *encoding, int relative_player);

class CanonicalEncoder : public ObservationEncoder {
 public:
  explicit CanonicalEncoder(const std::shared_ptr<BridgeGame> &game)
      : parent_game_(game){}

  [[nodiscard]] std::vector<int> Encode(
      const BridgeObservation &obs,
      const std::unordered_map<std::string, std::any> &kwargs = {}) const override;

  [[nodiscard]] std::vector<int> Shape() const override;

  std::vector<int> EncodeMyHand(const BridgeObservation &obs) const;

  std::vector<int> EncodeOtherHands(const BridgeObservation &obs) const;

  std::vector<int> EncodeOtherHandEvaluationsOneHot(
      const BridgeObservation &obs) const;

  std::vector<int> EncodeOtherHandEvaluations(
      const BridgeObservation &obs) const;

  // std::vector<int> EncodePBE(const BridgeObservation& obs) const;

  ObservationEncoder::EncoderPhase EncodingPhase() const override {
    return kAuction;
  }

  [[nodiscard]] ObservationEncoder::Type type() const override {
    return ObservationEncoder::Type::kCanonical;
  }

 private:
  const std::shared_ptr<BridgeGame> parent_game_;
};
}  // namespace bridge_learning_env

#endif /* BRIDGE_LIB_CANONICAL_ENCODER_H */
