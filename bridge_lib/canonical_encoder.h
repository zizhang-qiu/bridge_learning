#ifndef BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_ENCODER_H
#define BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_ENCODER_H

#include "bridge_game.h"
#include "bridge_observation.h"
#include "bridge_utils.h"
#include "observation_encoder.h"

namespace bridge_learning_env {
inline constexpr int kBitsPerCard = kNumSuits * kNumCardsPerSuit;
inline constexpr int kVulnerabilityTensorSize = kNumVulnerabilities * kNumPartnerships;
inline constexpr int kOpeningPassTensorSize = kNumPlayers;
inline constexpr int kSingleBidTensorSize = kNumPlayers * 3; // For a single bid, it can be made, doubled or redoubled
inline constexpr int kBiddingHistoryTensorSize = kSingleBidTensorSize * kNumBidLevels * kNumDenominations;
inline constexpr int kCardTensorSize = kNumCards;
inline constexpr int kPlayerSeatTensorSize = kNumPlayers;
inline constexpr int kBiddingTensorSize =
    kVulnerabilityTensorSize
    + kOpeningPassTensorSize
    + kBiddingHistoryTensorSize
    + kCardTensorSize;

int EncodeVulnerabilityBoth(const BridgeObservation& obs,
                            const std::shared_ptr<BridgeGame>& game,
                            int start_offset,
                            std::vector<int>* encoding);

int EncodeAuction(const BridgeObservation& obs,
                  int start_offset,
                  std::vector<int>* encoding);

int EncodePlayerHand(const BridgeObservation& obs,
                     int start_offset,
                     std::vector<int>* encoding);

int EncodeContract(const BridgeObservation& obs,
                   int start_offset,
                   std::vector<int>* encoding);

int EncodeVulnerabilityDeclarer(const BridgeObservation& obs,
                                int start_offset,
                                std::vector<int>* encoding);

int EncodeDummyHand(const BridgeObservation& obs,
                     int start_offset,
                     std::vector<int>* encoding);

int EncodePlayedTricks(const BridgeObservation& obs,
                     int start_offset,
                     std::vector<int>* encoding,
                     int num_tricks);

int EncodeNumTricksWon(const BridgeObservation& obs,
                     int start_offset,
                     std::vector<int>* encoding);

class CanonicalEncoder : public ObservationEncoder {
  public:
    explicit CanonicalEncoder(const std::shared_ptr<BridgeGame>& game,
                              const int num_tricks_in_observation = kNumTricks)
      : parent_game_(game),
        num_tricks_in_observation_(num_tricks_in_observation) {
    }

    [[nodiscard]] std::vector<int> Encode(const BridgeObservation& obs) const override;

    [[nodiscard]] std::vector<int> Shape() const override;

    [[nodiscard]] ObservationEncoder::Type type() const override {
      return ObservationEncoder::Type::kCanonical;
    }

    int GetPlayTensorSize() const {
      return kNumBidLevels // What the contract is
          + kNumDenominations // What trumps are
          + kNumOtherCalls // Undoubled / doubled / redoubled
          + kNumPlayers // Who declarer is
          + kNumVulnerabilities // Vulnerability of the declaring side
          + kNumCards // Our remaining cards
          + kNumCards // Dummy's remaining cards
          + num_tricks_in_observation_ * kNumPlayers * kNumCards // Number of played tricks
          + kNumTricks // Number of tricks we have won
          + kNumTricks; // Number of tricks they have won
    }

  private:
    const std::shared_ptr<BridgeGame> parent_game_;
    const int num_tricks_in_observation_;
};
} // namespace bridge

#endif // BRIDGE_LEARNING_BRIDGE_LIB_BRIDGE_ENCODER_H
