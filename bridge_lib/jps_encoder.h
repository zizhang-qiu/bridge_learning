#ifndef BRIDGE_LIB_JPS_ENCODER_H
#define BRIDGE_LIB_JPS_ENCODER_H
#include <memory>
#include "observation_encoder.h"

namespace bridge_learning_env {

inline constexpr int kJPSTensorSize =
    kNumCards                 // Cards of current player
    + kNumPlayers * kNumBids  // Four each bids, does the player make it?
    + kNumBids                // Four each bid, is the bid doubled?
    + kNumVulnerabilities     // Vulnerability.
    + kNumCalls               // Legal moves.
    + 1;                      // Padding.

int EncodeJPSBiddingHistory(const BridgeObservation& obs, int start_offset,
                            std::vector<int>* encoding);

int EncodeJPSVulnerability(const BridgeObservation& obs, int start_offset,
                           std::vector<int>* encoding);

int EncodeJPSLegalMoves(const BridgeObservation& obs, int start_offset,
                        std::vector<int>* encoding);

    class JPSEncoder : public ObservationEncoder {
 public:
  JPSEncoder(const std::shared_ptr<BridgeGame>& game) : parent_game_(game) {}

  std::vector<int> Shape() const override { return {kJPSTensorSize}; }

  Type type() const override { return ObservationEncoder::Type::kJPS; }

  std::vector<int> Encode(const BridgeObservation& obs) const override;

 private:
  const std::shared_ptr<BridgeGame> parent_game_;
};
}  // namespace bridge_learning_env

#endif /* BRIDGE_LIB_JPS_ENCODER_H */
